from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

import nptyping as ND
import numpy as np
import pandas as pd
import xarray as xr
from nptyping import NDArray
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.binning_helpers import BinningInfo, BinningContainer, compute_spanning_bins
from replay_trajectory_classification import ClusterlessClassifier, ClusterlessDecoder, RandomWalk, estimate_movement_var
from pyphocorehelpers.print_helpers import WrappingMessagePrinter

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import (
    ClusterlessDecodingParameters,
    _get_multiunit_electrode_keep_mask,
    build_clusterless_training_data_from_pfnd,
    build_multiunits_from_spike_events,
    build_rtc_environment_from_pfnd,
    epochs_multiunits,
    rtc_posterior_flat_p_x_given_n,
    rtc_posterior_to_p_x_given_n,
)
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, non_serialized_field, serialized_field


@custom_define(slots=False, eq=False)
class ClusterlessRTCPositionDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder):
    """Clusterless position decoder using replay_trajectory_classification on PfND spatial grids.

        position : array_like, shape (n_time, n_position_dims)
        multiunits : array_like, shape (n_time, n_marks, n_electrodes)
        is_training : None or array_like, shape (n_time,)

        Usage:

            from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder import ClusterlessRTCPositionDecoder

    """

    sampling_frequency_hz: float = 1000.0
    multiunits: NDArray[ND.Shape["N_TIME_BINS, N_MARKS, N_ELECTRODES"], np.floating] = serialized_field(default=None, metadata={'shape': ('N_TIME_BINS', 'N_MARKS', 'N_ELECTRODES')})
    rtc_time: np.ndarray = None
    clusterless_params: ClusterlessDecodingParameters = None
    classifier: ClusterlessClassifier = non_serialized_field(default=None, repr=False)
    rtc_results: xr.Dataset = non_serialized_field(default=None, repr=False)
    p_x_given_n: np.ndarray = non_serialized_field(default=None, repr=False)
    flat_p_x_given_n: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_positions: np.ndarray = non_serialized_field(default=None, repr=False)
    revised_most_likely_positions: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_position_flat_indicies: np.ndarray = non_serialized_field(default=None, repr=False)
    most_likely_position_indicies: np.ndarray = non_serialized_field(default=None, repr=False)
    time_binning_container: BinningContainer = non_serialized_field(default=None, repr=False)
    is_training_mask: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = serialized_field(default=None, metadata={'shape': ('N_TIME_BINS',)})
    multiunit_electrode_keep_mask: np.ndarray = non_serialized_field(default=None, repr=False)
    rtc_position_bin_centers: np.ndarray = non_serialized_field(default=None, repr=False)
    estimated_log_likelihood_memory_bytes: int = non_serialized_field(default=None, repr=False)
    marginal: DynamicContainer = non_serialized_field(default=None, repr=False)


    @property
    def time_bin_size(self) -> float:
        return 1.0 / float(self.sampling_frequency_hz)


    @property
    def time_window_centers(self) -> np.ndarray:
        return self.rtc_time


    @property
    def time_window_edges(self) -> np.ndarray:
        half_bin = self.time_bin_size / 2.0
        return np.concatenate([[self.rtc_time[0] - half_bin], self.rtc_time + half_bin])


    @property
    def time_window_edges_binning_info(self):
        return self.time_binning_container.edge_info if self.time_binning_container is not None else None


    @property
    def time_window_center_binning_info(self):
        return self.time_binning_container.center_info if self.time_binning_container is not None else None


    @property
    def num_time_windows(self) -> int:
        return len(self.rtc_time) if self.rtc_time is not None else 0


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
        if self.multiunits is None:
            return None
        return np.sum(np.any(np.isfinite(self.multiunits), axis=1), axis=1)


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
        if self.multiunit_electrode_keep_mask is not None:
            return int(np.sum(self.multiunit_electrode_keep_mask))
        if self.multiunits is not None and np.ndim(self.multiunits) == 3:
            return int(self.multiunits.shape[2])
        return 0


    def setup(self):
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        self._setup_computation_variables()


    def _setup_computation_variables(self):
        """Clusterless decoders do not build Zhang F/P_x matrices; fit RTC classifier when training data is available."""
        if self.multiunits is not None and self.rtc_time is not None:
            self._ensure_fitted_classifier(debug_print=self.debug_print)


    def get_by_id(self, ids, defer_compute_all:bool=False):
        raise NotImplementedError("ClusterlessRTCPositionDecoder does not support neuron-based slicing (get_by_id) because it relies on multiunits.")


    @classmethod
    def is_clusterless_decoder(cls, decoder) -> bool:
        return isinstance(decoder, cls)


    @classmethod
    def from_session(cls, sess, ndim: int = 2, pf=None, clusterless_spike_events=None, clusterless_params: Optional[ClusterlessDecodingParameters] = None, computation_config=None, included_epochs=None, setup_on_init: bool = True, should_compute_all: bool = False, debug_print: bool = False) -> "ClusterlessRTCPositionDecoder":
        """Build a standalone ClusterlessRTCPositionDecoder directly from a NeuroPy DataSession (works with 0 sorted units)."""
        from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
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
            raise ValueError("ClusterlessRTCPositionDecoder.from_session() requires clusterless_spike_events on sess or on disk.")
        active_events = load_clusterless_spike_events(clusterless_spike_events) if isinstance(clusterless_spike_events, (str, Path)) else clusterless_spike_events
        if pf is None:
            if computation_config is None:
                computation_config = PlacefieldComputationParameters()
            empty_spikes_df = pd.DataFrame({'aclu': pd.Series(dtype=np.int64), 't_seconds': pd.Series(dtype=np.float64), 't': pd.Series(dtype=np.float64), 't_rel_seconds': pd.Series(dtype=np.float64), 'neuron_type': pd.Series(dtype=object)})
            active_pos = deepcopy(sess.position)
            if ndim == 1:
                if not active_pos.has_linear_pos:
                    active_pos.compute_linearized_position(method=getattr(computation_config, 'linearization_method', 'isomap') or 'isomap')
                target_pos = deepcopy(active_pos.linear_pos_obj)
            else:
                active_pos.drop_dimensions_above(desired_ndim=2)
                target_pos = active_pos
            pf = PfND.from_config_values(empty_spikes_df, position=target_pos, epochs=included_epochs, speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh, grid_bin=computation_config.grid_bin, grid_bin_bounds=computation_config.grid_bin_bounds, smooth=computation_config.smooth)
        pos_sampling_rate_Hz = float((getattr(sess.position, 'metadata', None) or {}).get('sampling_rate', getattr(sess, 'position_sampling_rate', 120.0)))
        params = deepcopy(clusterless_params) if clusterless_params is not None else ClusterlessDecodingParameters(position_sampling_frequency_Hz=pos_sampling_rate_Hz)
        pos_df = pf.filtered_pos_df
        source_times = pos_df['t'].to_numpy(dtype=float) if 't' in pos_df.columns else pos_df['t_seconds'].to_numpy(dtype=float)
        pf_multiunits, pf_rtc_time = build_multiunits_from_spike_events(active_events, t_start=float(source_times.min()), t_end=float(source_times.max()), sampling_frequency_hz=params.clusterless_sampling_frequency_hz)
        decoder = cls(pf=pf, sampling_frequency_hz=params.clusterless_sampling_frequency_hz, multiunits=pf_multiunits, rtc_time=pf_rtc_time, clusterless_params=params, setup_on_init=setup_on_init, post_load_on_init=False, debug_print=debug_print)
        if should_compute_all:
            decoder.compute_all(debug_print=debug_print)
        return decoder


    def replacing_computation_epochs(self, epochs, preserve_source_neuron_IDs: bool = False):
        """Return a train-epoch copy without dropping clusterless multiunit state."""
        from neuropy.core.epoch import Epoch, ensure_dataframe

        new_epochs_obj = Epoch(ensure_dataframe(deepcopy(epochs)).epochs.get_valid_df()).get_non_overlapping()
        updated_decoder = deepcopy(self)
        updated_decoder.pf = self.pf.replacing_computation_epochs(deepcopy(new_epochs_obj), preserve_source_neuron_IDs=preserve_source_neuron_IDs)
        for attr_name in ('classifier', 'rtc_results', 'p_x_given_n', 'flat_p_x_given_n', 'most_likely_positions', 'revised_most_likely_positions', 'most_likely_position_flat_indicies', 'most_likely_position_indicies', 'time_binning_container', 'is_training_mask', 'multiunit_electrode_keep_mask', 'rtc_position_bin_centers', 'estimated_log_likelihood_memory_bytes', 'marginal'):
            setattr(updated_decoder, attr_name, None)
        ## END for attr_name in ('classifier', 'rtc_results', 'p_x_given_n', 'flat_p_x_given_n', 'most_likely_positions', 'revised_most_likely_positions', 'most_likely_position_flat_indicies', 'most_likely_position_indicies', 'time_binning_container', 'is_training_mask', 'multiunit_electrode_keep_mask', 'rtc_position_bin_centers', 'estimated_log_likelihood_memory_bytes', 'marginal')...

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


    def decode(self,
                unit_specific_time_binned_spike_counts,
                time_bin_size: float, output_flat_versions=False, debug_print=True, multiunits=None, rtc_time=None, is_compute_acausal=True, use_gpu=None):
        """Decode clusterless multiunits via RTC; does not alter internal decoder state.

        Accepts multiunits with shape (n_time, n_marks, n_electrodes) as the first argument or via multiunits=.
        Binned per-neuron spike counts (n_neurons, n_time_bins) are not supported.
        """
        active_multiunits = multiunits if multiunits is not None else unit_specific_time_binned_spike_counts
        active_multiunits = np.asarray(active_multiunits, dtype=float)
        if active_multiunits.ndim != 3:
            raise ValueError("ClusterlessRTCPositionDecoder.decode() requires multiunits with shape (n_time, n_marks, n_electrodes). Binned spike counts (n_neurons, n_time_bins) are not supported.")
        n_time = active_multiunits.shape[0]
        if rtc_time is None:
            active_rtc_time = (np.arange(n_time, dtype=float) + 0.5) * float(time_bin_size)
        else:
            active_rtc_time = np.asarray(rtc_time, dtype=float)
        if len(active_rtc_time) != n_time:
            raise ValueError(f"rtc_time length {len(active_rtc_time)} != multiunits n_time {n_time}")
        if debug_print:
            print(f"ClusterlessRTCPositionDecoder.decode(): n_time={n_time}, n_marks={active_multiunits.shape[1]}, n_electrodes={active_multiunits.shape[2]}")
        with WrappingMessagePrinter(f"decode(...) called. Computing {n_time} windows for p_x_given_n...", begin_line_ending="... ", finished_message="decode completed.", enable_print=(debug_print or self.debug_print)):
            p_x_given_n_out, flat_p_x_given_n, most_likely_positions, most_likely_position_flat_indicies, most_likely_position_indicies = self._predict_clusterless_posterior(active_multiunits, active_rtc_time, multiunits_for_fit=active_multiunits, rtc_time_for_fit=active_rtc_time, is_compute_acausal=is_compute_acausal, use_gpu=use_gpu, debug_print=debug_print)
            flat_outputs_container = DynamicContainer(flat_p_x_given_n=flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies) if output_flat_versions else None
            if debug_print:
                print(f"p_x_given_n_out.shape: {p_x_given_n_out.shape}")
            return most_likely_positions, p_x_given_n_out, most_likely_position_indicies, flat_outputs_container


    def decode_specific_epochs(self, spikes_df: Optional[pd.DataFrame] = None, filter_epochs=None, decoding_time_bin_size: float = 0.05, use_single_time_bin_per_epoch: bool = False, slideby: Optional[float] = None, debug_print=False) -> "DecodedFilterEpochsResult":
        """Decode clusterless multiunits for each provided epoch.

        ``spikes_df`` is accepted for drop-in compatibility with ``BasePositionDecoder`` and intentionally ignored:
        clusterless RTC decoding uses ``self.multiunits`` and ``self.rtc_time``.
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        if self.multiunits is None or self.rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder.decode_specific_epochs() requires self.multiunits and self.rtc_time.")

        active_multiunits = np.asarray(self.multiunits, dtype=float)
        active_rtc_time = np.asarray(self.rtc_time, dtype=float)
        if active_multiunits.ndim != 3:
            raise ValueError("ClusterlessRTCPositionDecoder.decode_specific_epochs() requires self.multiunits with shape (n_time, n_marks, n_electrodes).")
        if len(active_rtc_time) != active_multiunits.shape[0]:
            raise ValueError(f"rtc_time length {len(active_rtc_time)} != multiunits n_time {active_multiunits.shape[0]}")

        filter_epochs_df = filter_epochs.copy() if isinstance(filter_epochs, pd.DataFrame) else filter_epochs.to_dataframe()
        if not {'start', 'stop'}.issubset(filter_epochs_df.columns):
            raise ValueError("filter_epochs must provide 'start' and 'stop' columns.")

        if decoding_time_bin_size is None:
            decoding_time_bin_size = self.time_bin_size

        coarse_multiunits_list, coarse_rtc_time_list, nbins, time_bin_containers = epochs_multiunits(active_multiunits, active_rtc_time, filter_epochs_df, bin_size=decoding_time_bin_size, fine_bin_size=self.time_bin_size, slideby=slideby, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)
        num_filter_epochs = len(filter_epochs_df)
        most_likely_positions_list = []
        p_x_given_n_list = []
        marginal_x_list = []
        marginal_y_list = []
        marginal_z_list = []
        most_likely_position_indicies_list = []
        decoded_multiunits_list = []
        time_bin_edges = []
        ndim = getattr(self, 'ndim', getattr(self.pf, 'ndim', 1))

        for epoch_idx, (selected_multiunits, selected_rtc_time, curr_time_bin_container) in enumerate(zip(coarse_multiunits_list, coarse_rtc_time_list, time_bin_containers)):
            curr_epoch_num_time_bins = int(nbins[epoch_idx])
            decoded_multiunits_list.append(selected_multiunits)

            if curr_epoch_num_time_bins > 0:
                most_likely_positions, p_x_given_n, most_likely_position_indicies, _flat_outputs_container = self.decode(selected_multiunits, time_bin_size=decoding_time_bin_size, rtc_time=selected_rtc_time, output_flat_versions=False, debug_print=debug_print)
            else:
                epoch_start = float(filter_epochs_df.iloc[epoch_idx]['start'])
                epoch_stop = float(filter_epochs_df.iloc[epoch_idx]['stop'])
                pf_occupancy = getattr(self.pf, 'occupancy', None)
                empty_position_shape = tuple(np.shape(pf_occupancy)) if pf_occupancy is not None else (0,)
                p_x_given_n = np.empty((*empty_position_shape, 0), dtype=float)
                most_likely_positions = np.empty((0, ndim), dtype=float) if ndim > 1 else np.asarray([], dtype=float)
                most_likely_position_indicies = np.empty((ndim, 0), dtype=int) if ndim > 1 else np.asarray([], dtype=int)

            most_likely_positions = np.atleast_1d(most_likely_positions)
            p_x_given_n = np.atleast_1d(p_x_given_n)
            curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = self.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
            most_likely_positions_list.append(most_likely_positions)
            p_x_given_n_list.append(p_x_given_n)
            most_likely_position_indicies_list.append(np.atleast_1d(most_likely_position_indicies))
            marginal_x_list.append(curr_unit_marginal_x)
            marginal_y_list.append(curr_unit_marginal_y)
            marginal_z_list.append(curr_unit_marginal_z)
            time_bin_edges.append(np.atleast_1d(curr_time_bin_container.edges))
        ## END for epoch_idx, (selected_multiunits, selected_rtc_time, curr_time_bin_container) in enumerate(...)


        if debug_print:
            print(f"ClusterlessRTCPositionDecoder.decode_specific_epochs(): decoded {num_filter_epochs} epochs, nbins={nbins}")

        result_kwargs = dict(decoding_time_bin_size=decoding_time_bin_size, decoding_slideby=slideby, filter_epochs=filter_epochs_df, num_filter_epochs=num_filter_epochs,
                             most_likely_positions_list=most_likely_positions_list, p_x_given_n_list=p_x_given_n_list,
                             marginal_x_list=marginal_x_list, marginal_y_list=marginal_y_list, marginal_z_list=marginal_z_list,
                             most_likely_position_indicies_list=most_likely_position_indicies_list, spkcount=decoded_multiunits_list, nbins=nbins,
                             time_bin_containers=time_bin_containers, time_bin_edges=time_bin_edges, pos_bin_edges=getattr(self, 'xbin', None))
        return DecodedFilterEpochsResult(**result_kwargs)


    @staticmethod
    def estimate_log_likelihood_memory_bytes(n_time: int, n_position_bins: int, dtype=np.float32) -> int:
        return int(n_time) * int(n_position_bins) * int(np.dtype(dtype).itemsize)


    @classmethod
    def raise_if_log_likelihood_exceeds_memory_limit(cls, n_time: int, n_position_bins: int, max_memory_gib: Optional[float]) -> int:
        estimated_bytes = cls.estimate_log_likelihood_memory_bytes(n_time=n_time, n_position_bins=n_position_bins, dtype=np.float32)
        if max_memory_gib is not None:
            max_memory_bytes = int(float(max_memory_gib) * (1024 ** 3))
            if estimated_bytes > max_memory_bytes:
                estimated_gib = estimated_bytes / float(1024 ** 3)
                raise MemoryError(f"Clusterless likelihood would allocate {estimated_gib:.2f} GiB for shape ({int(n_time)}, {int(n_position_bins)}) float32, exceeding max_log_likelihood_memory_gib={float(max_memory_gib):.2f}. Reduce clusterless_sampling_frequency_hz, increase rtc_2d_place_bin_size_override, or raise the limit explicitly.")
        return estimated_bytes


    @classmethod
    def _is_rtc_gpu_acceleration_available(cls) -> bool:
        """Return True when CuPy can run RTC-style GPU posterior kernels (device + NVRTC)."""
        try:
            import cupy as cp
            from cupy_backends.cuda.libs import nvrtc
            if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                return False
            nvrtc.getVersion()
            _ = float(cp.arange(4, dtype=cp.float32).sum())
            return True
        except Exception:
            return False


    def _resolve_use_gpu(self, use_gpu: Optional[bool], debug_print: bool = False) -> bool:
        if use_gpu is None:
            use_gpu = self._is_rtc_gpu_acceleration_available()
            if not use_gpu:
                print("Warning: ClusterlessRTCPositionDecoder GPU acceleration unavailable; falling back to CPU decoding.")
        elif use_gpu and not self._is_rtc_gpu_acceleration_available():
            print("Warning: ClusterlessRTCPositionDecoder GPU requested but unavailable; falling back to CPU decoding.")
            use_gpu = False
        elif use_gpu and (debug_print or self.debug_print):
            print("ClusterlessRTCPositionDecoder: using GPU acceleration (CuPy).")
        return use_gpu


    def _ensure_fitted_classifier(self, multiunits_for_fit=None, rtc_time_for_fit=None, debug_print: bool = False) -> Tuple[Union[ClusterlessDecoder, ClusterlessClassifier], np.ndarray]:
        if self.classifier is not None and self.rtc_position_bin_centers is not None:
            return self.classifier, self.rtc_position_bin_centers
        training_multiunits = self.multiunits if self.multiunits is not None else multiunits_for_fit
        training_rtc_time = self.rtc_time if self.rtc_time is not None else rtc_time_for_fit
        if training_multiunits is None or training_rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder requires multiunits and rtc_time on the decoder (or passed to decode()) before fitting the classifier.")
        params = self.clusterless_params if self.clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=self.sampling_frequency_hz)
        place_bin_size_override = params.rtc_place_bin_size_override if self.ndim == 1 else (params.rtc_2d_place_bin_size_override if params.rtc_2d_place_bin_size_override is not None else params.rtc_place_bin_size_override)
        environment = build_rtc_environment_from_pfnd(self.pf, environment_name=params.rtc_environment_name, place_bin_size_override=place_bin_size_override)
        position_train, multiunits_train, is_training = build_clusterless_training_data_from_pfnd(self.pf, training_multiunits, training_rtc_time, self.sampling_frequency_hz)
        self.is_training_mask = is_training
        self.multiunit_electrode_keep_mask = _get_multiunit_electrode_keep_mask(multiunits_train, time_mask=is_training)
        multiunits_train = multiunits_train[:, :, self.multiunit_electrode_keep_mask]
        if params.rtc_model_type == "classifier":
            self.classifier = ClusterlessClassifier(environments=[environment], clusterless_algorithm="multiunit_likelihood", clusterless_algorithm_params={"mark_std": params.rtc_mark_std, "position_std": params.rtc_position_std})
        else:
            movement_var = params.rtc_movement_var
            if movement_var is None:
                try:
                    pos_for_var = position_train[is_training] if np.any(is_training) else position_train
                    est_var = float(estimate_movement_var(pos_for_var, sampling_frequency=self.sampling_frequency_hz))
                    movement_var = est_var if (np.isfinite(est_var) and est_var > 0.0) else 6.0
                except Exception:
                    movement_var = 6.0
            self.classifier = ClusterlessDecoder(environment=environment, transition_type=RandomWalk(movement_var=float(movement_var)), clusterless_algorithm="multiunit_likelihood", clusterless_algorithm_params={"mark_std": params.rtc_mark_std, "position_std": params.rtc_position_std})
        self.classifier.fit(position_train, multiunits_train, is_training=is_training)
        fitted_environment = self.classifier.environments[0] if hasattr(self.classifier, 'environments') else self.classifier.environment
        n_position_bins = int(np.prod(self.original_position_data_shape))
        self.rtc_position_bin_centers = np.asarray(fitted_environment.place_bin_centers_)
        self.estimated_log_likelihood_memory_bytes = self.estimate_log_likelihood_memory_bytes(n_time=len(multiunits_train), n_position_bins=n_position_bins, dtype=np.float32)
        return self.classifier, self.rtc_position_bin_centers


    def _predict_clusterless_posterior(self, multiunits, rtc_time, multiunits_for_fit=None, rtc_time_for_fit=None, is_compute_acausal=True, use_gpu=None, debug_print=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        use_gpu = self._resolve_use_gpu(use_gpu=use_gpu, debug_print=debug_print)
        classifier, _rtc_position_bin_centers = self._ensure_fitted_classifier(multiunits_for_fit=multiunits_for_fit, rtc_time_for_fit=rtc_time_for_fit, debug_print=debug_print)
        params = self.clusterless_params if self.clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=self.sampling_frequency_hz)
        multiunits = np.asarray(multiunits, dtype=float)
        if self.multiunit_electrode_keep_mask is not None and multiunits.shape[2] == len(self.multiunit_electrode_keep_mask):
            multiunits = multiunits[:, :, self.multiunit_electrode_keep_mask]
        rtc_time = np.asarray(rtc_time, dtype=float)
        n_position_bins = int(np.prod(self.original_position_data_shape))
        self.raise_if_log_likelihood_exceeds_memory_limit(n_time=len(multiunits), n_position_bins=n_position_bins, max_memory_gib=params.max_log_likelihood_memory_gib)
        self.rtc_results = classifier.predict(multiunits, time=rtc_time[:len(multiunits)], is_compute_acausal=is_compute_acausal, use_gpu=use_gpu)
        flat_p_x_given_n = rtc_posterior_flat_p_x_given_n(self.rtc_results, self.pf, state_index=params.state_index_for_posterior)
        if flat_p_x_given_n.shape[0] != self.flat_position_size:
            p_x_given_n = rtc_posterior_to_p_x_given_n(self.rtc_results, self.pf, state_index=params.state_index_for_posterior, should_match_pf_grid=params.should_match_pf_grid)
            if p_x_given_n.ndim == (self.pf.ndim + 1):
                flat_p_x_given_n = self._flatten_output(p_x_given_n)
            else:
                flat_p_x_given_n = rtc_posterior_flat_p_x_given_n(self.rtc_results, self.pf, state_index=params.state_index_for_posterior)
        return self._format_decoder_posterior_outputs(flat_p_x_given_n)


    def compute_all(self, is_compute_acausal=True, use_gpu: Optional[bool] = None, debug_print: bool = True) -> None:
        """ main pre-compute function """
        if self.multiunits is None or self.rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder requires multiunits and rtc_time before compute_all().")
        self.p_x_given_n, self.flat_p_x_given_n, self.most_likely_positions, self.most_likely_position_flat_indicies, self.most_likely_position_indicies = self._predict_clusterless_posterior(self.multiunits, self.rtc_time, multiunits_for_fit=self.multiunits, rtc_time_for_fit=self.rtc_time, is_compute_acausal=is_compute_acausal, use_gpu=use_gpu, debug_print=(debug_print or self.debug_print))
        self.revised_most_likely_positions = self.perform_compute_forward_filled_positions(self.most_likely_positions, self.is_non_firing_time_bin)
        curr_unit_marginal_x, curr_unit_marginal_y, curr_unit_marginal_z = self.perform_build_marginals(self.p_x_given_n, self.most_likely_positions, debug_print=(debug_print or self.debug_print))
        if self.ndim == 1:
            curr_unit_marginal_x.revised_most_likely_positions_1D = np.atleast_1d(self.revised_most_likely_positions).T
        elif self.ndim >= 2:
            curr_unit_marginal_x.revised_most_likely_positions_1D = self.revised_most_likely_positions[:, 0].T
            if curr_unit_marginal_y is not None:
                curr_unit_marginal_y.revised_most_likely_positions_1D = self.revised_most_likely_positions[:, 1].T
        self.marginal = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y, z=curr_unit_marginal_z)
        time_window_edges, time_window_edges_binning_info = compute_spanning_bins(self.rtc_time, bin_size=self.time_bin_size)
        self.time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        if debug_print or self.debug_print:
            print(f"ClusterlessRTCPositionDecoder.compute_all(): p_x_given_n.shape={self.p_x_given_n.shape}, most_likely_positions.shape={np.shape(self.most_likely_positions)}")


    @classmethod
    def overwrite_standard_decoders(cls, curr_active_pipeline, enable_force_overwrite: bool = False, include_includelist=None, debug_print=True):
        """ helper for the pipeline computations.
            if the typical pf1D/pf2D Decoders from units are missing, replace them with the clusterless ones if they exist.

        Usage:

            from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder import ClusterlessRTCPositionDecoder

            ClusterlessRTCPositionDecoder.overwrite_standard_decoders(curr_active_pipeline, enable_force_overwrite=False)


        """
        if include_includelist is None:
            include_includelist = list(curr_active_pipeline.computation_results.keys())

        for filter_name, comp_r in curr_active_pipeline.computation_results.items():
            if filter_name in include_includelist:
                # cd = curr_active_pipeline.computation_results[filter_name].computed_data
                cd = comp_r.computed_data

                if enable_force_overwrite or (cd.get('pf1D_Decoder', None) is None):
                    a_pf1D_ClusterlessDecoder = cd.get('pf1D_ClusterlessDecoder', None)
                    if a_pf1D_ClusterlessDecoder is not None:
                        cd['pf1D_Decoder'] = a_pf1D_ClusterlessDecoder
                        if debug_print:
                            print(filter_name, a_pf1D_ClusterlessDecoder.p_x_given_n.shape if a_pf1D_ClusterlessDecoder else None)


                if enable_force_overwrite or (cd.get('pf2D_Decoder', None) is None):
                    a_pf2D_ClusterlessDecoder = cd.get('pf2D_ClusterlessDecoder', None)
                    if a_pf2D_ClusterlessDecoder is not None:
                        cd['pf2D_Decoder'] = a_pf2D_ClusterlessDecoder
                        if debug_print:
                            print(filter_name, a_pf2D_ClusterlessDecoder.p_x_given_n.shape if a_pf2D_ClusterlessDecoder else None)
            ## END if filter_name in include_includelist:...

        ## END for filter_name, comp_r in curr_active_...

