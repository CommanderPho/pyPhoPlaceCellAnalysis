from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

import numpy as np
import pandas as pd
import xarray as xr
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo, compute_spanning_bins
from non_local_detector.models.base import ClusterlessDetector
from pyphocorehelpers.print_helpers import WrappingMessagePrinter

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import (
    SpyglassClusterlessDecodingParameters,
    build_is_training_mask,
    build_unitfree_pfnd_from_session,
    clusterless_events_to_spyglass_spike_lists,
    epochs_from_pfnd,
    fit_clusterless_detector_in_memory,
    nld_posterior_flat_p_x_given_n,
    nld_posterior_to_p_x_given_n,
    pfnd_to_spyglass_position_info,
    predict_clusterless_epochs_in_memory,
    raise_if_posterior_exceeds_memory_limit,
    run_clusterless_decoder_in_memory,
)
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, non_serialized_field, serialized_field


def _build_epoch_window_spec(epoch_start: float, epoch_stop: float, decoding_time_bin_size: float, slideby: Optional[float] = None, use_single_time_bin_per_epoch: bool = False) -> Tuple[np.ndarray, np.ndarray, BinningContainer]:
    """Build (window_starts, window_stops, BinningContainer) for a single epoch, mirroring epochs_spkcount."""
    t0, t1 = float(epoch_start), float(epoch_stop)
    dur = t1 - t0
    if use_single_time_bin_per_epoch:
        edges = np.array([t0, t1], dtype=float)
        step = dur if dur > 0 else 0.01
        edge_info = BinningInfo(variable_extents=(t0, t1), step=step, num_bins=2)
        return edges[:1], edges[1:], BinningContainer.init_from_edges(edges=edges, edge_info=edge_info)

    W = float(decoding_time_bin_size)
    H = float(slideby) if slideby is not None else W
    assert H > 0, f"slideby must be > 0, got {H}"
    assert H <= W + 1e-9, f"slideby ({H}) must be <= decoding_time_bin_size ({W})"
    use_sliding = not np.isclose(H, W, rtol=0.0, atol=1e-12)
    if use_sliding:
        if dur <= 0:
            starts = np.array([t0], dtype=float)
            stops = np.array([t0 + W], dtype=float)
        elif dur < W:
            starts = np.array([t0], dtype=float)
            stops = np.array([t1], dtype=float)
        else:
            n_windows = max(1, int(np.floor((dur - W) / H)) + 1)
            starts = t0 + np.arange(n_windows, dtype=float) * H
            stops = starts + W
        container = BinningContainer.from_sliding_windows(window_start_edges=starts, window_stop_edges=stops, epoch_variable_extents=(t0, t1), window_width=W, hop=H)
        return starts, stops, container

    if dur < W:
        if dur <= 0:
            edges = np.array([t0, t0 + W], dtype=float)
            edge_info = BinningInfo(variable_extents=(t0, t0 + W), step=W, num_bins=2)
        else:
            edges = np.array([t0, t1], dtype=float)
            edge_info = BinningInfo(variable_extents=(t0, t1), step=dur, num_bins=2)
        return edges[:1], edges[1:], BinningContainer.init_from_edges(edges=edges, edge_info=edge_info)

    edges, edge_info = compute_spanning_bins(variable_values=None, bin_size=W, variable_start_value=t0, variable_end_value=t1)
    return edges[:-1], edges[1:], BinningContainer.init_from_edges(edges=edges, edge_info=edge_info)


def _count_electrode_spikes_in_windows(spike_times: List[np.ndarray], window_starts: np.ndarray, window_stops: np.ndarray) -> np.ndarray:
    """Count spikes per electrode in each [window_start, window_stop) window -> shape (n_electrodes, n_windows)."""
    n_electrodes = len(spike_times)
    n_windows = len(window_starts)
    counts = np.zeros((n_electrodes, n_windows), dtype=int)
    if n_windows == 0 or n_electrodes == 0:
        return counts
    for e_idx, st in enumerate(spike_times):
        st_arr = np.asarray(st, dtype=float)
        if len(st_arr) == 0:
            continue
        left = np.searchsorted(st_arr, window_starts, side="left")
        right = np.searchsorted(st_arr, window_stops, side="right")
        counts[e_idx, :] = np.maximum(0, right - left)
    ## END for e_idx, st in enumerate(spike_times)...

    return counts


@custom_define(slots=False, eq=False)
class SpyglassClusterlessDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder):
    """Clusterless position decoder using non_local_detector (Spyglass v1 logic) on PfND spatial grids (1D & 2D).

        position_info : pd.DataFrame with time index and position_x [/ position_y] columns
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
    kept_electrode_indices: np.ndarray = serialized_field(default=None)
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


    def __getstate__(self):
        state = self.__dict__.copy()
        state['nld_results'] = None
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)


    @property
    def time_bin_size(self) -> float:
        if self.decode_times is not None and len(self.decode_times) >= 2:
            return float(np.median(np.diff(np.asarray(self.decode_times, dtype=float))))
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
        edges = self.time_window_edges
        if len(edges) < 2:
            return np.zeros(0, dtype=int)
        per_electrode = _count_electrode_spikes_in_windows(self.spike_times, edges[:-1], edges[1:])
        return np.sum(per_electrode, axis=0)


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
    def spikes_df(self) -> Optional[pd.DataFrame]:
        return None


    @property
    def num_neurons(self) -> int:
        return 0


    @property
    def num_electrodes(self) -> int:
        if self.kept_electrode_indices is not None:
            return int(len(self.kept_electrode_indices))
        return int(len(self.spike_times)) if self.spike_times is not None else 0


    def setup(self):
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        self._setup_computation_variables()


    def _setup_computation_variables(self):
        """Spyglass clusterless decoders do not build Zhang F/P_x matrices; fit NLD classifier once when training data is available."""
        if self.position_info is not None and self.spike_times is not None and self.spike_waveform_features is not None and self.encoding_interval is not None:
            self._ensure_fitted_classifier(debug_print=self.debug_print)


    def get_by_id(self, ids, defer_compute_all:bool=False):
        raise NotImplementedError("SpyglassClusterlessDecoder does not support neuron-based slicing (get_by_id) because it relies on clusterless waveform features.")


    @classmethod
    def is_spyglass_clusterless_decoder(cls, decoder) -> bool:
        return isinstance(decoder, cls)


    @classmethod
    def is_clusterless_decoder(cls, decoder) -> bool:
        return isinstance(decoder, cls)


    @classmethod
    def from_session(cls, sess, ndim: int = 2, pf=None, clusterless_spike_events=None, spyglass_params: Optional[SpyglassClusterlessDecodingParameters] = None, computation_config=None, included_epochs=None, setup_on_init: bool = True, should_compute_all: bool = False, debug_print: bool = False) -> "SpyglassClusterlessDecoder":
        """Build a standalone SpyglassClusterlessDecoder directly from a NeuroPy DataSession (works with 0 sorted units)."""
        from neuropy.core.clusterless_spike_events import default_clusterless_spike_events_path, load_clusterless_spike_events

        if clusterless_spike_events is None:
            clusterless_spike_events = getattr(sess, 'clusterless_spike_events', None)
        if clusterless_spike_events is None:
            session_basedir = Path(getattr(sess, 'basepath', None) or Path(sess.filePrefix).parent)
            session_name = getattr(getattr(sess, 'config', None), 'session_name', None) or getattr(sess, 'name', None) or Path(sess.filePrefix).stem
            clusterless_events_path = default_clusterless_spike_events_path(session_basedir, session_name)
            if clusterless_events_path.is_file():
                clusterless_spike_events = load_clusterless_spike_events(clusterless_events_path)
                sess.clusterless_spike_events = clusterless_spike_events
        if clusterless_spike_events is None:
            raise ValueError("SpyglassClusterlessDecoder.from_session() requires clusterless_spike_events on sess or on disk.")
        active_events = load_clusterless_spike_events(clusterless_spike_events) if isinstance(clusterless_spike_events, (str, Path)) else clusterless_spike_events
        if pf is None:
            pf = build_unitfree_pfnd_from_session(sess, ndim=ndim, computation_config=computation_config, included_epochs=included_epochs)
        params = deepcopy(spyglass_params) if spyglass_params is not None else SpyglassClusterlessDecodingParameters()
        spike_times, spike_waveform_features = clusterless_events_to_spyglass_spike_lists(active_events)
        position_variable_names = params.resolved_position_variable_names(pf)
        position_info = pfnd_to_spyglass_position_info(pf, upsample_hz=params.position_upsample_hz, position_variable_names=position_variable_names)
        encoding_interval, decoding_interval = epochs_from_pfnd(pf)
        is_training_mask = build_is_training_mask(position_info, encoding_interval, position_variable_names, pf=pf, use_pf_speed_thresh=params.use_pf_speed_thresh_for_training)
        decoder = cls(pf=pf, position_upsample_hz=params.position_upsample_hz, position_info=position_info, spike_times=spike_times, spike_waveform_features=spike_waveform_features, encoding_interval=encoding_interval, decoding_interval=decoding_interval, position_variable_names=position_variable_names, spyglass_params=params, is_training_mask=is_training_mask, setup_on_init=setup_on_init, post_load_on_init=False, debug_print=debug_print)
        if should_compute_all:
            decoder.compute_all(debug_print=debug_print)
        return decoder


    def replacing_computation_epochs(self, epochs, preserve_source_neuron_IDs: bool = False):
        """Return a train-epoch copy without dropping Spyglass clusterless state."""
        from neuropy.core.epoch import Epoch, ensure_dataframe

        new_epochs_obj = Epoch(ensure_dataframe(deepcopy(epochs)).epochs.get_valid_df()).get_non_overlapping()
        updated_decoder = deepcopy(self)
        updated_decoder.pf = self.pf.replacing_computation_epochs(deepcopy(new_epochs_obj), preserve_source_neuron_IDs=preserve_source_neuron_IDs)
        updated_decoder.encoding_interval, updated_decoder.decoding_interval = epochs_from_pfnd(updated_decoder.pf)
        for attr_name in ('classifier', 'nld_results', 'p_x_given_n', 'flat_p_x_given_n', 'most_likely_positions', 'revised_most_likely_positions', 'most_likely_position_flat_indicies', 'most_likely_position_indicies', 'time_binning_container', 'is_training_mask', 'kept_electrode_indices', 'decode_times', 'marginal'):
            setattr(updated_decoder, attr_name, None)
        ## END for attr_name in ('classifier', 'nld_results', 'p_x_given_n', 'flat_p_x_given_n', 'most_likely_positions', 'revised_most_likely_positions', 'most_likely_position_flat_indicies', 'most_likely_position_indicies', 'time_binning_container', 'is_training_mask', 'kept_electrode_indices', 'decode_times', 'marginal')...

        return updated_decoder


    @property
    def flat_position_size(self) -> int:
        return int(np.prod(self.original_position_data_shape))


    def _reshape_output(self, flat_p_x_given_n: np.ndarray) -> np.ndarray:
        return np.reshape(flat_p_x_given_n, (*self.original_position_data_shape, flat_p_x_given_n.shape[-1]), order='C')


    def _flatten_output(self, p_x_given_n: np.ndarray) -> np.ndarray:
        return np.reshape(p_x_given_n, (self.flat_position_size, p_x_given_n.shape[-1]), order='C')


    def _format_decoder_posterior_outputs(self, flat_p_x_given_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_x_given_n = self._reshape_output(flat_p_x_given_n)
        most_likely_position_flat_indicies, most_likely_position_indicies = self.perform_compute_most_likely_positions(flat_p_x_given_n, self.original_position_data_shape)
        most_likely_positions = self.perform_lookup_most_likely_positions(most_likely_position_indicies, xbin_centers=self.pf.xbin_centers, ybin_centers=(self.pf.ybin_centers if int(self.ndim) > 1 else None), ndim=int(self.ndim))
        return p_x_given_n, flat_p_x_given_n, most_likely_positions, most_likely_position_flat_indicies, most_likely_position_indicies


    def _resolved_params(self) -> SpyglassClusterlessDecodingParameters:
        return self.spyglass_params if self.spyglass_params is not None else SpyglassClusterlessDecodingParameters(position_upsample_hz=self.position_upsample_hz)


    def _resolved_position_variable_names(self) -> List[str]:
        if self.position_variable_names is not None:
            return list(self.position_variable_names)
        return self._resolved_params().resolved_position_variable_names(self.pf)


    def decode(self, unit_specific_time_binned_spike_counts=None, time_bin_size: Optional[float] = None, output_flat_versions=False, debug_print=True, spike_times=None, spike_waveform_features=None, position_info=None, decoding_interval=None, encoding_interval=None):
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
        active_decoding_interval = np.atleast_2d(np.asarray(active_decoding_interval, dtype=float))
        n_decode_times = 0
        for interval_start, interval_end in active_decoding_interval:
            if time_bin_size is not None:
                n_decode_times += max(1, int(np.ceil((float(interval_end) - float(interval_start)) / float(time_bin_size))))
            else:
                interval_time = active_position_info.loc[interval_start:interval_end].index
                n_decode_times += len(interval_time)
        ## END for interval_start, interval_end in active_decoding_interval...

        if debug_print:
            print(f"SpyglassClusterlessDecoder.decode(): n_decode_times≈{n_decode_times}, n_electrodes={len(active_spike_times)}")
        with WrappingMessagePrinter(f"decode(...) called. Computing ~{n_decode_times} windows for p_x_given_n...", begin_line_ending="... ", finished_message="decode completed.", enable_print=(debug_print or self.debug_print)):
            p_x_given_n_out, flat_p_x_given_n, most_likely_positions, most_likely_position_flat_indicies, most_likely_position_indicies, _decoded_times = self._predict_spyglass_posterior(active_spike_times, active_spike_waveform_features, active_position_info, active_decoding_interval, encoding_interval_for_fit=active_encoding_interval, time_bin_size=time_bin_size, mutate_state=False, debug_print=debug_print)
            flat_outputs_container = DynamicContainer(flat_p_x_given_n=flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies) if output_flat_versions else None
            if debug_print:
                print(f"p_x_given_n_out.shape: {p_x_given_n_out.shape}")
            return most_likely_positions, p_x_given_n_out, most_likely_position_indicies, flat_outputs_container


    def decode_specific_epochs(self, spikes_df: Optional[pd.DataFrame] = None, filter_epochs=None, decoding_time_bin_size: float = 0.05, use_single_time_bin_per_epoch: bool = False, slideby: Optional[float] = None, debug_print=False) -> "DecodedFilterEpochsResult":
        """Decode clusterless waveform features for each provided epoch.

        ``spikes_df`` is accepted for drop-in compatibility with ``BasePositionDecoder`` and intentionally ignored:
        Spyglass clusterless decoding uses ``self.spike_times`` and ``self.spike_waveform_features``.
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        if self.position_info is None or self.spike_times is None or self.spike_waveform_features is None:
            raise ValueError("SpyglassClusterlessDecoder.decode_specific_epochs() requires self.position_info, self.spike_times, and self.spike_waveform_features.")

        filter_epochs_df = filter_epochs.copy() if isinstance(filter_epochs, pd.DataFrame) else filter_epochs.to_dataframe()
        if not {'start', 'stop'}.issubset(filter_epochs_df.columns):
            raise ValueError("filter_epochs must provide 'start' and 'stop' columns.")

        if decoding_time_bin_size is None:
            decoding_time_bin_size = self.time_bin_size

        num_filter_epochs = len(filter_epochs_df)
        params = self._resolved_params()
        classifier = self._ensure_fitted_classifier(debug_print=debug_print)

        epoch_intervals = np.column_stack([filter_epochs_df['start'].to_numpy(dtype=float), filter_epochs_df['stop'].to_numpy(dtype=float)]) if num_filter_epochs > 0 else np.empty((0, 2), dtype=float)
        target_windows_per_epoch: List[Tuple[np.ndarray, np.ndarray]] = []
        time_bin_containers: List[BinningContainer] = []
        time_bin_edges: List[np.ndarray] = []
        decoded_spike_lists: List[np.ndarray] = []
        nbins = np.zeros(num_filter_epochs, dtype=int)

        active_spike_times_for_counts = [self.spike_times[int(i)] for i in self.kept_electrode_indices] if self.kept_electrode_indices is not None else self.spike_times

        for epoch_idx in range(num_filter_epochs):
            epoch_start = float(filter_epochs_df.iloc[epoch_idx]['start'])
            epoch_stop = float(filter_epochs_df.iloc[epoch_idx]['stop'])
            if epoch_stop < epoch_start:
                raise ValueError(f"filter_epochs row {epoch_idx} has stop < start: start={epoch_start}, stop={epoch_stop}")
            w_starts, w_stops, curr_time_bin_container = _build_epoch_window_spec(epoch_start, epoch_stop, decoding_time_bin_size=decoding_time_bin_size, slideby=slideby, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch)
            target_windows_per_epoch.append((w_starts, w_stops))
            time_bin_containers.append(curr_time_bin_container)
            time_bin_edges.append(np.atleast_1d(curr_time_bin_container.edges))
            nbins[epoch_idx] = len(curr_time_bin_container.centers)
            decoded_spike_lists.append(_count_electrode_spikes_in_windows(active_spike_times_for_counts, w_starts, w_stops))
        ## END for epoch_idx in range(num_filter_epochs)...

        n_position_bins = int(np.prod(self.original_position_data_shape))
        dt_fine = 1.0 / float(classifier.sampling_frequency)
        max_epoch_fine = max((max(1, int(np.ceil(max(0.0, float(s1) - float(s0)) / dt_fine))) for s0, s1 in epoch_intervals), default=0)
        raise_if_posterior_exceeds_memory_limit(n_time=max_epoch_fine, n_position_bins=n_position_bins, max_memory_gib=params.max_posterior_memory_gib)

        flat_posteriors_list: List[np.ndarray] = []
        if num_filter_epochs > 0:
            flat_posteriors_list, _fine_centers_list = predict_clusterless_epochs_in_memory(classifier=classifier, spike_times=self.spike_times, spike_waveform_features=self.spike_waveform_features, pf=self.pf, epoch_intervals=epoch_intervals, target_windows_per_epoch=target_windows_per_epoch, state_index=params.state_index_for_posterior, should_match_pf_grid=params.should_match_pf_grid, kept_electrode_indices=self.kept_electrode_indices)

        most_likely_positions_list = []
        p_x_given_n_list = []
        marginal_x_list = []
        marginal_y_list = []
        marginal_z_list = []
        most_likely_position_indicies_list = []

        for epoch_idx in range(num_filter_epochs):
            flat_p_x_given_n = flat_posteriors_list[epoch_idx]
            p_x_given_n, _flat, most_likely_positions, _flat_idxs, most_likely_position_indicies = self._format_decoder_posterior_outputs(flat_p_x_given_n)
            most_likely_positions = np.atleast_1d(most_likely_positions)
            p_x_given_n = np.atleast_1d(p_x_given_n)
            curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = self.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
            most_likely_positions_list.append(most_likely_positions)
            p_x_given_n_list.append(p_x_given_n)
            most_likely_position_indicies_list.append(np.atleast_1d(most_likely_position_indicies))
            marginal_x_list.append(curr_unit_marginal_x)
            marginal_y_list.append(curr_unit_marginal_y)
            marginal_z_list.append(curr_unit_marginal_z)
        ## END for epoch_idx in range(num_filter_epochs)...

        if debug_print:
            print(f"SpyglassClusterlessDecoder.decode_specific_epochs(): decoded {num_filter_epochs} epochs, nbins={nbins}")

        result_kwargs = dict(decoding_time_bin_size=decoding_time_bin_size, decoding_slideby=slideby, filter_epochs=filter_epochs_df, num_filter_epochs=num_filter_epochs,
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
        params = self._resolved_params()
        position_variable_names = self._resolved_position_variable_names()
        self.is_training_mask = build_is_training_mask(training_position_info, training_encoding_interval, position_variable_names, pf=self.pf, use_pf_speed_thresh=params.use_pf_speed_thresh_for_training)
        decoding_kwargs = {'is_training': self.is_training_mask}
        self.classifier, self.kept_electrode_indices = fit_clusterless_detector_in_memory(decoding_params=params.resolved_decoding_params(pf=self.pf), position_info=training_position_info, position_variable_names=position_variable_names, spike_times=training_spike_times, spike_waveform_features=training_spike_waveform_features, decoding_kwargs=decoding_kwargs)
        return self.classifier


    def _predict_spyglass_posterior(self, spike_times, spike_waveform_features, position_info, decoding_interval, encoding_interval_for_fit=None, time_bin_size: Optional[float] = None, mutate_state: bool = False, debug_print=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = self._resolved_params()
        position_variable_names = self._resolved_position_variable_names()
        decoding_interval = np.atleast_2d(np.asarray(decoding_interval, dtype=float))
        n_position_bins = int(np.prod(self.original_position_data_shape))

        if params.estimate_decoding_params:
            n_decode_times = sum(len(position_info.loc[interval_start:interval_end].index) for interval_start, interval_end in decoding_interval)
            raise_if_posterior_exceeds_memory_limit(n_time=n_decode_times, n_position_bins=n_position_bins, max_memory_gib=params.max_posterior_memory_gib)
            active_encoding_interval = encoding_interval_for_fit if encoding_interval_for_fit is not None else self.encoding_interval
            decoding_kwargs = {'is_training': build_is_training_mask(position_info, active_encoding_interval, position_variable_names, pf=self.pf, use_pf_speed_thresh=params.use_pf_speed_thresh_for_training)} if active_encoding_interval is not None else {}
            classifier, nld_results = run_clusterless_decoder_in_memory(key={'estimate_decoding_params': True}, decoding_params=params.resolved_decoding_params(pf=self.pf), decoding_kwargs=decoding_kwargs, position_info=position_info, position_variable_names=position_variable_names, spike_times=spike_times, spike_waveform_features=spike_waveform_features, decoding_interval=decoding_interval)
            decoded_times = np.asarray(nld_results.time.values, dtype=float)
            flat_p_x_given_n = nld_posterior_flat_p_x_given_n(nld_results, self.pf, should_match_pf_grid=params.should_match_pf_grid, state_index=params.state_index_for_posterior)
            if mutate_state:
                self.classifier = classifier
                self.nld_results = nld_results
                self.decode_times = decoded_times
            p_x_given_n, flat_p_x_given_n, mlp, mlp_flat, mlp_idx = self._format_decoder_posterior_outputs(flat_p_x_given_n)
            return p_x_given_n, flat_p_x_given_n, mlp, mlp_flat, mlp_idx, decoded_times

        classifier = self._ensure_fitted_classifier(spike_times_for_fit=spike_times, spike_waveform_features_for_fit=spike_waveform_features, position_info_for_fit=position_info, encoding_interval_for_fit=encoding_interval_for_fit, debug_print=debug_print)
        target_windows_per_epoch = None
        decoded_times_parts: List[np.ndarray] = []
        if time_bin_size is not None:
            target_windows_per_epoch = []
            for iv_start, iv_stop in decoding_interval:
                w_starts, w_stops, container = _build_epoch_window_spec(iv_start, iv_stop, decoding_time_bin_size=float(time_bin_size))
                target_windows_per_epoch.append((w_starts, w_stops))
                decoded_times_parts.append(np.asarray(container.centers, dtype=float))
            ## END for iv_start, iv_stop in decoding_interval...

            total_decode_times = sum(len(t_arr) for t_arr in decoded_times_parts)
        else:
            dt_fine = 1.0 / float(classifier.sampling_frequency)
            total_decode_times = sum(max(1, int(np.ceil(max(0.0, float(s1) - float(s0)) / dt_fine))) for s0, s1 in decoding_interval)
        raise_if_posterior_exceeds_memory_limit(n_time=total_decode_times, n_position_bins=n_position_bins, max_memory_gib=params.max_posterior_memory_gib)

        kept_indices = self.kept_electrode_indices if (self.kept_electrode_indices is not None and len(spike_times) > int(np.max(self.kept_electrode_indices))) else None
        flat_list, fine_centers_list = predict_clusterless_epochs_in_memory(classifier=classifier, spike_times=spike_times, spike_waveform_features=spike_waveform_features, pf=self.pf, epoch_intervals=decoding_interval, target_windows_per_epoch=target_windows_per_epoch, state_index=params.state_index_for_posterior, should_match_pf_grid=params.should_match_pf_grid, kept_electrode_indices=kept_indices)
        flat_p_x_given_n = np.concatenate(flat_list, axis=1) if len(flat_list) > 1 else flat_list[0]
        decoded_times = np.concatenate(decoded_times_parts) if target_windows_per_epoch is not None else np.concatenate(fine_centers_list)
        if mutate_state:
            self.decode_times = decoded_times
        p_x_given_n, flat_p_x_given_n, mlp, mlp_flat, mlp_idx = self._format_decoder_posterior_outputs(flat_p_x_given_n)
        return p_x_given_n, flat_p_x_given_n, mlp, mlp_flat, mlp_idx, decoded_times


    def compute_all(self, decoding_time_bin_size: Optional[float] = None, chunk_duration_sec: float = 300.0, debug_print: bool = True) -> None:
        """ main pre-compute function (chunks long decoding intervals to bound peak memory). """
        if self.position_info is None or self.spike_times is None or self.spike_waveform_features is None or self.decoding_interval is None:
            raise ValueError("SpyglassClusterlessDecoder requires position_info, spike_times, spike_waveform_features, and decoding_interval before compute_all().")
        chunked_intervals: List[List[float]] = []
        for iv_start, iv_stop in np.atleast_2d(np.asarray(self.decoding_interval, dtype=float)):
            t_cur = float(iv_start)
            t_end = float(iv_stop)
            if chunk_duration_sec is None or chunk_duration_sec <= 0 or (t_end - t_cur) <= float(chunk_duration_sec):
                chunked_intervals.append([t_cur, t_end])
            else:
                while t_cur < t_end:
                    t_next = min(t_end, t_cur + float(chunk_duration_sec))
                    chunked_intervals.append([t_cur, t_next])
                    t_cur = t_next
        ## END for iv_start, iv_stop in np.atleast_2d(np.asarray(self.decoding_interval, dtype=float))...

        chunked_arr = np.asarray(chunked_intervals, dtype=float)
        self.p_x_given_n, self.flat_p_x_given_n, self.most_likely_positions, self.most_likely_position_flat_indicies, self.most_likely_position_indicies, self.decode_times = self._predict_spyglass_posterior(self.spike_times, self.spike_waveform_features, self.position_info, chunked_arr, encoding_interval_for_fit=self.encoding_interval, time_bin_size=decoding_time_bin_size, mutate_state=True, debug_print=(debug_print or self.debug_print))
        self.revised_most_likely_positions = self.perform_compute_forward_filled_positions(self.most_likely_positions, self.is_non_firing_time_bin)
        curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = self.perform_build_marginals(self.p_x_given_n, self.most_likely_positions, debug_print=(debug_print or self.debug_print))
        if self.ndim == 1:
            curr_unit_marginal_x.revised_most_likely_positions_1D = np.atleast_1d(self.revised_most_likely_positions).T
        elif self.ndim >= 2:
            curr_unit_marginal_x.revised_most_likely_positions_1D = self.revised_most_likely_positions[:, 0].T
            if curr_unit_marginal_y is not None:
                curr_unit_marginal_y.revised_most_likely_positions_1D = self.revised_most_likely_positions[:, 1].T
        self.marginal = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y, z=curr_unit_marginal_z)
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
                            print(filter_name, a_pf1D_SpyglassClusterlessDecoder.p_x_given_n.shape if a_pf1D_SpyglassClusterlessDecoder.p_x_given_n is not None else None)

                if enable_force_overwrite or (cd.get('pf2D_Decoder', None) is None):
                    a_pf2D_SpyglassClusterlessDecoder = cd.get('pf2D_SpyglassClusterlessDecoder', None)
                    if a_pf2D_SpyglassClusterlessDecoder is not None:
                        cd['pf2D_Decoder'] = a_pf2D_SpyglassClusterlessDecoder
                        if debug_print:
                            print(filter_name, a_pf2D_SpyglassClusterlessDecoder.p_x_given_n.shape if a_pf2D_SpyglassClusterlessDecoder.p_x_given_n is not None else None)
            ## END if filter_name in include_includelist:...

        ## END for filter_name, comp_r in curr_active_pipeline.computation_results.items()...

