from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

import numpy as np
import pandas as pd
import xarray as xr
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.binning_helpers import BinningContainer, compute_spanning_bins
from non_local_detector.models.base import ClusterlessDetector
from pyphocorehelpers.print_helpers import WrappingMessagePrinter

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import (
    SpyglassClusterlessDecodingParameters,
    build_is_training_mask,
    nld_posterior_flat_p_x_given_n,
    nld_posterior_to_p_x_given_n,
    raise_if_posterior_exceeds_memory_limit,
    run_clusterless_decoder_in_memory,
)
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, non_serialized_field, serialized_field


@custom_define(slots=False, eq=False)
class SpyglassClusterlessDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder):
    """Clusterless position decoder using non_local_detector (Spyglass v1 logic) on PfND spatial grids.

        position_info : pd.DataFrame with time index and position_x / position_y columns
        spike_times : list of per-electrode spike time arrays
        spike_waveform_features : list of per-electrode waveform mark arrays

        Usage:

            from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_decoder import SpyglassClusterlessDecoder

    """

    position_upsample_hz: float = 500.0
    position_info: pd.DataFrame = serialized_field(default=None)
    spike_times: List[np.ndarray] = serialized_field(default=None)
    spike_waveform_features: List[np.ndarray] = serialized_field(default=None)
    encoding_interval: np.ndarray = serialized_field(default=None)
    decoding_interval: np.ndarray = serialized_field(default=None)
    position_variable_names: List[str] = serialized_field(default=None)
    spyglass_params: SpyglassClusterlessDecodingParameters = None
    is_training_mask: np.ndarray = serialized_field(default=None)
    classifier: ClusterlessDetector = non_serialized_field(default=None, repr=False)
    nld_results: xr.Dataset = non_serialized_field(default=None, repr=False)
    p_x_given_n: np.ndarray = non_serialized_field(default=None, repr=False)
    flat_p_x_given_n: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_positions: np.ndarray = non_serialized_field(default=None, repr=False)
    revised_most_likely_positions: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_position_flat_indicies: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_position_indicies: np.ndarray = non_serialized_field(default=None, repr=False)
    time_binning_container: BinningContainer = non_serialized_field(default=None, repr=False)
    decode_times: np.ndarray = non_serialized_field(default=None, repr=False)
    marginal: DynamicContainer = non_serialized_field(default=None, repr=False)


    @property
    def time_bin_size(self) -> float:
        if self.position_info is None or len(self.position_info.index) < 2:
            return 1.0 / float(self.position_upsample_hz)
        return float(np.median(np.diff(self.position_info.index.to_numpy(dtype=float))))


    @property
    def time_window_centers(self) -> np.ndarray:
        return self.decode_times if self.decode_times is not None else (self.position_info.index.to_numpy(dtype=float) if self.position_info is not None else np.asarray([]))


    @property
    def time_window_edges(self) -> np.ndarray:
        centers = self.time_window_centers
        if centers is None or len(centers) == 0:
            return np.asarray([])
        half_bin = self.time_bin_size / 2.0
        return np.concatenate([[centers[0] - half_bin], centers + half_bin])


    @property
    def time_window_edges_binning_info(self):
        return self.time_binning_container.edge_info if self.time_binning_container is not None else None


    @property
    def time_window_center_binning_info(self):
        return self.time_binning_container.center_info if self.time_binning_container is not None else None


    @property
    def num_time_windows(self) -> int:
        centers = self.time_window_centers
        return len(centers) if centers is not None else 0


    @property
    def active_time_windows(self):
        if self.time_binning_container is not None and self.time_binning_container.edges is not None:
            edges = self.time_binning_container.edges
            return list(zip(edges[:-1], edges[1:]))
        window_starts = self.time_window_centers - (self.time_bin_size / 2.0)
        window_ends = self.time_window_centers + (self.time_bin_size / 2.0)
        return list(zip(window_starts, window_ends))


    @property
    def active_time_window_centers(self):
        window_starts = self.time_window_centers - (self.time_bin_size / 2.0)
        window_ends = self.time_window_centers + (self.time_bin_size / 2.0)
        return window_starts + ((window_ends - window_starts) / 2.0)


    @property
    def total_spike_counts_per_window(self) -> np.ndarray:
        if self.spike_times is None or self.decode_times is None:
            return None
        decode_times = np.asarray(self.decode_times, dtype=float)
        counts = np.zeros(len(decode_times), dtype=int)
        for electrode_spike_times in self.spike_times:
            if len(electrode_spike_times) == 0:
                continue
            bin_indices = np.searchsorted(decode_times, np.asarray(electrode_spike_times, dtype=float))
            valid = (bin_indices >= 0) & (bin_indices < len(decode_times))
            counts[bin_indices[valid]] += 1
        return counts


    @property
    def is_non_firing_time_bin(self) -> np.ndarray:
        spike_counts = self.total_spike_counts_per_window
        if spike_counts is None:
            return np.zeros(self.num_time_windows, dtype=bool)
        return spike_counts == 0


    @property
    def P_x_given_n(self):
        return self.p_x_given_n


    @property
    def num_neurons(self) -> int:
        raise NotImplementedError("Spyglass clusterless decoding uses waveform features, not individual neurons (num_neurons).")


    def setup(self):
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        self._setup_computation_variables()


    def _setup_computation_variables(self):
        """Spyglass clusterless decoders do not build Zhang F/P_x matrices; fit NLD classifier when training data is available."""
        if self.position_info is not None and self.spike_times is not None and self.spike_waveform_features is not None:
            self._ensure_fitted_classifier(debug_print=self.debug_print)


    def get_by_id(self, ids, defer_compute_all:bool=False):
        raise NotImplementedError("SpyglassClusterlessDecoder does not support neuron-based slicing (get_by_id) because it relies on clusterless waveform features.")


    @classmethod
    def is_spyglass_clusterless_decoder(cls, decoder) -> bool:
        return isinstance(decoder, cls)


    @classmethod
    def is_clusterless_decoder(cls, decoder) -> bool:
        return isinstance(decoder, cls)


    def replacing_computation_epochs(self, epochs):
        """Return a train-epoch copy without dropping Spyglass clusterless state."""
        from neuropy.core.epoch import Epoch, ensure_dataframe

        new_epochs_obj = Epoch(ensure_dataframe(deepcopy(epochs)).epochs.get_valid_df()).get_non_overlapping()
        updated_decoder = deepcopy(self)
        updated_decoder.pf = self.pf.replacing_computation_epochs(deepcopy(new_epochs_obj))
        for attr_name in ('classifier', 'nld_results', 'p_x_given_n', 'flat_p_x_given_n', 'most_likely_positions', 'revised_most_likely_positions', 'most_likely_position_flat_indicies', 'most_likely_position_indicies', 'time_binning_container', 'is_training_mask', 'decode_times', 'marginal'):
            setattr(updated_decoder, attr_name, None)
        return updated_decoder


    @property
    def flat_position_size(self) -> int:
        return int(np.prod(self.original_position_data_shape))


    def _reshape_output(self, flat_p_x_given_n: np.ndarray) -> np.ndarray:
        return np.reshape(flat_p_x_given_n, (*self.original_position_data_shape, flat_p_x_given_n.shape[-1]), order='F')


    def _flatten_output(self, p_x_given_n: np.ndarray) -> np.ndarray:
        return np.reshape(p_x_given_n, (self.flat_position_size, p_x_given_n.shape[-1]), order='F')


    def _format_decoder_posterior_outputs(self, flat_p_x_given_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_x_given_n = self._reshape_output(flat_p_x_given_n)
        most_likely_position_flat_indicies, most_likely_position_indicies = self.perform_compute_most_likely_positions(flat_p_x_given_n, self.original_position_data_shape)
        if self.ndim > 1:
            most_likely_positions = np.vstack((self.pf.xbin_centers[most_likely_position_indicies[0, :]], self.pf.ybin_centers[most_likely_position_indicies[1, :]])).T
        else:
            most_likely_positions = np.squeeze(self.pf.xbin_centers[most_likely_position_indicies[0, :]])
        return p_x_given_n, flat_p_x_given_n, most_likely_positions, most_likely_position_flat_indicies, most_likely_position_indicies


    def _resolved_position_variable_names(self) -> List[str]:
        if self.position_variable_names is not None:
            return list(self.position_variable_names)
        params = self.spyglass_params if self.spyglass_params is not None else SpyglassClusterlessDecodingParameters(position_upsample_hz=self.position_upsample_hz)
        return params.resolved_position_variable_names(self.pf)


    def decode(self, unit_specific_time_binned_spike_counts=None, time_bin_size: float = None, output_flat_versions=False, debug_print=True, spike_times=None, spike_waveform_features=None, position_info=None, decoding_interval=None, encoding_interval=None):
        """Decode clusterless waveform features via non_local_detector; does not alter internal decoder state.

        ``unit_specific_time_binned_spike_counts`` is intentionally ignored. Pass spike_times / spike_waveform_features
        directly or rely on values stored on the decoder.
        """
        active_spike_times = spike_times if spike_times is not None else self.spike_times
        active_spike_waveform_features = spike_waveform_features if spike_waveform_features is not None else self.spike_waveform_features
        active_position_info = position_info if position_info is not None else self.position_info
        active_decoding_interval = decoding_interval if decoding_interval is not None else self.decoding_interval
        active_encoding_interval = encoding_interval if encoding_interval is not None else self.encoding_interval
        if active_spike_times is None or active_spike_waveform_features is None or active_position_info is None or active_decoding_interval is None:
            raise ValueError("SpyglassClusterlessDecoder.decode() requires spike_times, spike_waveform_features, position_info, and decoding_interval on the decoder or passed as kwargs.")
        position_variable_names = self._resolved_position_variable_names()
        n_decode_times = 0
        for interval_start, interval_end in np.asarray(active_decoding_interval):
            interval_time = active_position_info.loc[interval_start:interval_end].index
            n_decode_times += len(interval_time)
        if debug_print:
            print(f"SpyglassClusterlessDecoder.decode(): n_decode_times≈{n_decode_times}, n_electrodes={len(active_spike_times)}")
        with WrappingMessagePrinter(f"decode(...) called. Computing ~{n_decode_times} windows for p_x_given_n...", begin_line_ending="... ", finished_message="decode completed.", enable_print=(debug_print or self.debug_print)):
            p_x_given_n_out, flat_p_x_given_n, most_likely_positions, most_likely_position_flat_indicies, most_likely_position_indicies = self._predict_spyglass_posterior(active_spike_times, active_spike_waveform_features, active_position_info, active_decoding_interval, encoding_interval_for_fit=active_encoding_interval, debug_print=debug_print)
            flat_outputs_container = DynamicContainer(flat_p_x_given_n=flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies) if output_flat_versions else None
            if debug_print:
                print(f"p_x_given_n_out.shape: {p_x_given_n_out.shape}")
            return most_likely_positions, p_x_given_n_out, most_likely_position_indicies, flat_outputs_container


    def decode_specific_epochs(self, spikes_df: pd.DataFrame, filter_epochs, decoding_time_bin_size: float = 0.05, use_single_time_bin_per_epoch: bool = False, slideby: Optional[float] = None, debug_print=False) -> "DecodedFilterEpochsResult":
        """Decode clusterless waveform features for each provided epoch.

        ``spikes_df`` is intentionally ignored: Spyglass clusterless decoding uses ``self.spike_times`` and ``self.position_info``.
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        assert (spikes_df is None), f"spikes_df MUST be None, it will not be used for Spyglass clusterless decoding."

        if self.position_info is None or self.spike_times is None or self.spike_waveform_features is None:
            raise ValueError("SpyglassClusterlessDecoder.decode_specific_epochs() requires self.position_info, self.spike_times, and self.spike_waveform_features.")

        filter_epochs_df = filter_epochs.copy() if isinstance(filter_epochs, pd.DataFrame) else filter_epochs.to_dataframe()
        if not {'start', 'stop'}.issubset(filter_epochs_df.columns):
            raise ValueError("filter_epochs must provide 'start' and 'stop' columns.")

        num_filter_epochs = len(filter_epochs_df)
        most_likely_positions_list = []
        p_x_given_n_list = []
        marginal_x_list = []
        marginal_y_list = []
        marginal_z_list = []
        most_likely_position_indicies_list = []
        decoded_spike_lists = []
        time_bin_edges = []
        time_bin_containers = []
        nbins = []
        ndim = getattr(self, 'ndim', getattr(self.pf, 'ndim', 1))

        for epoch_idx in range(num_filter_epochs):
            epoch_start = float(filter_epochs_df.iloc[epoch_idx]['start'])
            epoch_stop = float(filter_epochs_df.iloc[epoch_idx]['stop'])
            epoch_decoding_interval = np.array([[epoch_start, epoch_stop]], dtype=float)
            epoch_position_info = self.position_info.loc[epoch_start:epoch_stop]
            curr_epoch_num_time_bins = len(epoch_position_info.index)
            nbins.append(curr_epoch_num_time_bins)
            decoded_spike_lists.append(self.spike_times)
            curr_time_bin_container = BinningContainer(edges=np.asarray([epoch_start, epoch_stop]))
            time_bin_containers.append(curr_time_bin_container)
            time_bin_edges.append(np.asarray([epoch_start, epoch_stop]))

            if curr_epoch_num_time_bins > 0:
                most_likely_positions, p_x_given_n, most_likely_position_indicies, _flat_outputs_container = self.decode(spike_times=self.spike_times, spike_waveform_features=self.spike_waveform_features, position_info=self.position_info, decoding_interval=epoch_decoding_interval, encoding_interval=self.encoding_interval, output_flat_versions=False, debug_print=debug_print)
            else:
                pf_occupancy = getattr(self.pf, 'occupancy', None)
                empty_position_shape = tuple(np.shape(pf_occupancy)) if pf_occupancy is not None else (0,)
                p_x_given_n = np.empty((*empty_position_shape, 0), dtype=float)
                most_likely_positions = np.empty((0, ndim), dtype=float) if ndim > 1 else np.asarray([], dtype=float)
                most_likely_position_indicies = np.empty((ndim, 0), dtype=int) if ndim > 1 else np.asarray([], dtype=int)

            most_likely_positions = np.atleast_1d(most_likely_positions)
            p_x_given_n = np.atleast_1d(p_x_given_n)
            curr_unit_marginal_x, curr_unit_marginal_y = self.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
            most_likely_positions_list.append(most_likely_positions)
            p_x_given_n_list.append(p_x_given_n)
            most_likely_position_indicies_list.append(np.atleast_1d(most_likely_position_indicies))
            marginal_x_list.append(curr_unit_marginal_x)
            marginal_y_list.append(curr_unit_marginal_y)
            marginal_z_list.append(None)
        ## END for epoch_idx in range(num_filter_epochs)...


        if debug_print:
            print(f"SpyglassClusterlessDecoder.decode_specific_epochs(): decoded {num_filter_epochs} epochs, nbins={nbins}")

        result_kwargs = dict(decoding_time_bin_size=decoding_time_bin_size, slideby=slideby, filter_epochs=filter_epochs_df, num_filter_epochs=num_filter_epochs,
                             most_likely_positions_list=most_likely_positions_list, p_x_given_n_list=p_x_given_n_list,
                             marginal_x_list=marginal_x_list, marginal_y_list=marginal_y_list, marginal_z_list=marginal_z_list,
                             most_likely_position_indicies_list=most_likely_position_indicies_list, spkcount=decoded_spike_lists, nbins=nbins,
                             time_bin_containers=time_bin_containers, time_bin_edges=time_bin_edges, pos_bin_edges=getattr(self, 'xbin', None))
        return DecodedFilterEpochsResult(**result_kwargs)


    def _ensure_fitted_classifier(self, spike_times_for_fit=None, spike_waveform_features_for_fit=None, position_info_for_fit=None, encoding_interval_for_fit=None, debug_print: bool = False) -> ClusterlessDetector:
        if self.classifier is not None:
            return self.classifier
        training_spike_times = self.spike_times if self.spike_times is not None else spike_times_for_fit
        training_spike_waveform_features = self.spike_waveform_features if self.spike_waveform_features is not None else spike_waveform_features_for_fit
        training_position_info = self.position_info if self.position_info is not None else position_info_for_fit
        training_encoding_interval = self.encoding_interval if self.encoding_interval is not None else encoding_interval_for_fit
        if training_spike_times is None or training_spike_waveform_features is None or training_position_info is None or training_encoding_interval is None:
            raise ValueError("SpyglassClusterlessDecoder requires spike_times, spike_waveform_features, position_info, and encoding_interval before fitting the classifier.")
        params = self.spyglass_params if self.spyglass_params is not None else SpyglassClusterlessDecodingParameters(position_upsample_hz=self.position_upsample_hz)
        position_variable_names = self._resolved_position_variable_names()
        self.is_training_mask = build_is_training_mask(training_position_info, training_encoding_interval, position_variable_names)
        decoding_kwargs = {'is_training': self.is_training_mask}
        key = {'estimate_decoding_params': bool(params.estimate_decoding_params)}
        self.classifier, _nld_results = run_clusterless_decoder_in_memory(key=key, decoding_params=params.resolved_decoding_params(), decoding_kwargs=decoding_kwargs, position_info=training_position_info, position_variable_names=position_variable_names, spike_times=training_spike_times, spike_waveform_features=training_spike_waveform_features, decoding_interval=np.asarray(training_encoding_interval, dtype=float))
        return self.classifier


    def _predict_spyglass_posterior(self, spike_times, spike_waveform_features, position_info, decoding_interval, encoding_interval_for_fit=None, debug_print=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = self.spyglass_params if self.spyglass_params is not None else SpyglassClusterlessDecodingParameters(position_upsample_hz=self.position_upsample_hz)
        position_variable_names = self._resolved_position_variable_names()
        decoding_interval = np.asarray(decoding_interval, dtype=float)
        n_decode_times = sum(len(position_info.loc[interval_start:interval_end].index) for interval_start, interval_end in decoding_interval)
        n_position_bins = int(np.prod(self.original_position_data_shape))
        raise_if_posterior_exceeds_memory_limit(n_time=n_decode_times, n_position_bins=n_position_bins, max_memory_gib=params.max_posterior_memory_gib)
        active_encoding_interval = encoding_interval_for_fit if encoding_interval_for_fit is not None else self.encoding_interval
        decoding_kwargs = {'is_training': build_is_training_mask(position_info, active_encoding_interval, position_variable_names)} if active_encoding_interval is not None else {}
        key = {'estimate_decoding_params': bool(params.estimate_decoding_params)}
        self.classifier, self.nld_results = run_clusterless_decoder_in_memory(key=key, decoding_params=params.resolved_decoding_params(), decoding_kwargs=decoding_kwargs, position_info=position_info, position_variable_names=position_variable_names, spike_times=spike_times, spike_waveform_features=spike_waveform_features, decoding_interval=decoding_interval)
        self.decode_times = np.asarray(self.nld_results.time.values, dtype=float)
        flat_p_x_given_n = nld_posterior_flat_p_x_given_n(self.nld_results, self.pf, should_match_pf_grid=params.should_match_pf_grid)
        if flat_p_x_given_n.shape[0] != self.flat_position_size:
            p_x_given_n = nld_posterior_to_p_x_given_n(self.nld_results, self.pf, should_match_pf_grid=params.should_match_pf_grid)
            if p_x_given_n.ndim == (self.pf.ndim + 1):
                flat_p_x_given_n = self._flatten_output(p_x_given_n)
            else:
                flat_p_x_given_n = nld_posterior_flat_p_x_given_n(self.nld_results, self.pf, should_match_pf_grid=params.should_match_pf_grid)
        return self._format_decoder_posterior_outputs(flat_p_x_given_n)


    def compute_all(self, debug_print: bool = True) -> None:
        """ main pre-compute function """
        if self.position_info is None or self.spike_times is None or self.spike_waveform_features is None or self.decoding_interval is None:
            raise ValueError("SpyglassClusterlessDecoder requires position_info, spike_times, spike_waveform_features, and decoding_interval before compute_all().")
        self.p_x_given_n, self.flat_p_x_given_n, self.most_likely_positions, self.most_likely_position_flat_indicies, self.most_likely_position_indicies = self._predict_spyglass_posterior(self.spike_times, self.spike_waveform_features, self.position_info, self.decoding_interval, encoding_interval_for_fit=self.encoding_interval, debug_print=(debug_print or self.debug_print))
        self.revised_most_likely_positions = self.most_likely_positions.copy()
        curr_unit_marginal_x, curr_unit_marginal_y = self.perform_build_marginals(self.p_x_given_n, self.most_likely_positions, debug_print=(debug_print or self.debug_print))
        self.marginal = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
        if self.decode_times is not None and len(self.decode_times) > 0:
            time_window_edges, time_window_edges_binning_info = compute_spanning_bins(self.decode_times, bin_size=self.time_bin_size)
            self.time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        if debug_print or self.debug_print:
            print(f"SpyglassClusterlessDecoder.compute_all(): p_x_given_n.shape={self.p_x_given_n.shape}, most_likely_positions.shape={np.shape(self.most_likely_positions)}")


    @classmethod
    def overwrite_standard_decoders(cls, curr_active_pipeline, enable_force_overwrite: bool = False, include_includelist=None, debug_print=True):
        """ helper for the pipeline computations.
            if the typical pf1D/pf2D Decoders from units are missing, replace them with the Spyglass clusterless ones if they exist.

        Usage:

            from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_decoder import SpyglassClusterlessDecoder

            SpyglassClusterlessDecoder.overwrite_standard_decoders(curr_active_pipeline, enable_force_overwrite=False)

        """
        if include_includelist is None:
            include_includelist = list(curr_active_pipeline.computation_results.keys())

        for filter_name, comp_r in curr_active_pipeline.computation_results.items():
            if filter_name in include_includelist:
                cd = comp_r.computed_data

                if enable_force_overwrite or (cd.get('pf1D_Decoder', None) is None):
                    a_pf1D_SpyglassClusterlessDecoder = cd.get('pf1D_SpyglassClusterlessDecoder', None)
                    if a_pf1D_SpyglassClusterlessDecoder is not None:
                        cd['pf1D_Decoder'] = a_pf1D_SpyglassClusterlessDecoder
                        if debug_print:
                            print(filter_name, a_pf1D_SpyglassClusterlessDecoder.p_x_given_n.shape if a_pf1D_SpyglassClusterlessDecoder else None)

                if enable_force_overwrite or (cd.get('pf2D_Decoder', None) is None):
                    a_pf2D_SpyglassClusterlessDecoder = cd.get('pf2D_SpyglassClusterlessDecoder', None)
                    if a_pf2D_SpyglassClusterlessDecoder is not None:
                        cd['pf2D_Decoder'] = a_pf2D_SpyglassClusterlessDecoder
                        if debug_print:
                            print(filter_name, a_pf2D_SpyglassClusterlessDecoder.p_x_given_n.shape if a_pf2D_SpyglassClusterlessDecoder else None)
            ## END if filter_name in include_includelist:...

        ## END for filter_name, comp_r in curr_active_...
