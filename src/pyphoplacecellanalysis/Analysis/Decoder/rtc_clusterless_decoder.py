from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

import nptyping as ND
import numpy as np
import pandas as pd
import xarray as xr
from nptyping import NDArray
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.mixins.binning_helpers import BinningInfo, BinningContainer, compute_spanning_bins
from replay_trajectory_classification import ClusterlessClassifier
from pyphocorehelpers.print_helpers import WrappingMessagePrinter

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import (
    ClusterlessDecodingParameters,
    build_clusterless_training_data_from_pfnd,
    build_rtc_environment_from_pfnd,
    most_likely_positions_from_posterior,
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
    time_binning_container: BinningContainer = non_serialized_field(default=None, repr=False)
    is_training_mask: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = serialized_field(default=None, metadata={'shape': ('N_TIME_BINS',)})
    rtc_position_bin_centers: np.ndarray = non_serialized_field(default=None, repr=False)
    estimated_log_likelihood_memory_bytes: int = non_serialized_field(default=None, repr=False)


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
    def num_neurons(self) -> int:
        raise NotImplementedError("Clusterless decoding uses multiunits, not individual neurons (num_neurons).")


    def setup(self):
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None


    def _setup_computation_variables(self):
        """Clusterless decoders do not build Zhang F/P_x matrices."""
        pass


    def get_by_id(self, ids, defer_compute_all:bool=False):
        raise NotImplementedError("ClusterlessRTCPositionDecoder does not support neuron-based slicing (get_by_id) because it relies on multiunits.")


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
            p_x_given_n, flat_p_x_given_n, place_bin_centers, _rtc_results = self._predict_clusterless_posterior(active_multiunits, active_rtc_time, multiunits_for_fit=active_multiunits, rtc_time_for_fit=active_rtc_time, is_compute_acausal=is_compute_acausal, use_gpu=use_gpu, debug_print=debug_print)
            num_time_windows = flat_p_x_given_n.shape[1]
            pf_flat_size = int(np.prod(self.original_position_data_shape))
            if pf_flat_size == flat_p_x_given_n.shape[0]:
                p_x_given_n_out = np.reshape(flat_p_x_given_n, (*self.original_position_data_shape, num_time_windows))
                position_shape_for_unravel = self.original_position_data_shape
            else:
                p_x_given_n_out = p_x_given_n
                position_shape_for_unravel = (flat_p_x_given_n.shape[0],)
            most_likely_position_flat_indicies, most_likely_position_indicies = self.perform_compute_most_likely_positions(flat_p_x_given_n, position_shape_for_unravel)
            posterior_for_positions = p_x_given_n_out if (p_x_given_n_out.ndim > 1 and p_x_given_n_out.shape[0] == pf_flat_size) else flat_p_x_given_n
            most_likely_positions = most_likely_positions_from_posterior(posterior_for_positions, self.pf, place_bin_centers=place_bin_centers)
            flat_outputs_container = DynamicContainer(flat_p_x_given_n=flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies) if output_flat_versions else None
            if debug_print:
                print(f"p_x_given_n_out.shape: {p_x_given_n_out.shape}")
            return most_likely_positions, p_x_given_n_out, most_likely_position_indicies, flat_outputs_container


    def decode_specific_epochs(self, spikes_df: pd.DataFrame, filter_epochs, decoding_time_bin_size: float = 0.05, use_single_time_bin_per_epoch: bool = False, slideby: Optional[float] = None, debug_print=False) -> "DecodedFilterEpochsResult":
        """Decode clusterless multiunits for each provided epoch.

        ``spikes_df`` is intentionally ignored: clusterless RTC decoding uses ``self.multiunits`` and ``self.rtc_time``.
        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        assert (spikes_df is None), f"spikes_df MUST be None, it will not be used for Clusterless-decoding."

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

        half_bin = self.time_bin_size / 2.0
        num_filter_epochs = len(filter_epochs_df)
        most_likely_positions_list = []
        p_x_given_n_list = []
        marginal_x_list = []
        marginal_y_list = []
        marginal_z_list = []
        most_likely_position_indicies_list = []
        decoded_multiunits_list = []
        nbins = np.zeros(num_filter_epochs, dtype=int)
        time_bin_containers = []
        time_bin_edges = []
        ndim = getattr(self, 'ndim', getattr(self.pf, 'ndim', 1))

        for epoch_idx, epoch_row in enumerate(filter_epochs_df.itertuples(index=False)):
            epoch_start = float(getattr(epoch_row, 'start'))
            epoch_stop = float(getattr(epoch_row, 'stop'))
            if epoch_stop < epoch_start:
                raise ValueError(f"filter_epochs row {epoch_idx} has stop < start: start={epoch_start}, stop={epoch_stop}")

            if use_single_time_bin_per_epoch:
                epoch_midpoint = epoch_start + ((epoch_stop - epoch_start) / 2.0)
                epoch_indices = np.asarray([int(np.argmin(np.abs(active_rtc_time - epoch_midpoint)))], dtype=int) if len(active_rtc_time) > 0 else np.asarray([], dtype=int)
            else:
                epoch_indices = np.flatnonzero((active_rtc_time >= epoch_start) & (active_rtc_time <= epoch_stop))
                if len(epoch_indices) == 0 and len(active_rtc_time) > 0:
                    epoch_midpoint = epoch_start + ((epoch_stop - epoch_start) / 2.0)
                    nearest_idx = int(np.argmin(np.abs(active_rtc_time - epoch_midpoint)))
                    nearest_time = active_rtc_time[nearest_idx]
                    if (epoch_start - half_bin) <= nearest_time <= (epoch_stop + half_bin):
                        epoch_indices = np.asarray([nearest_idx], dtype=int)

            selected_rtc_time = active_rtc_time[epoch_indices]
            selected_multiunits = active_multiunits[epoch_indices, :, :]
            curr_epoch_num_time_bins = len(selected_rtc_time)
            nbins[epoch_idx] = curr_epoch_num_time_bins
            decoded_multiunits_list.append(selected_multiunits)

            if curr_epoch_num_time_bins > 0:
                curr_time_bin_edges = np.concatenate([[selected_rtc_time[0] - half_bin], selected_rtc_time + half_bin])
                curr_edge_info = BinningInfo(**dict(variable_extents=(curr_time_bin_edges[0], curr_time_bin_edges[-1]), step=self.time_bin_size, num_bins=len(curr_time_bin_edges)))
                curr_center_info = BinningInfo(**dict(variable_extents=(curr_time_bin_edges[0], curr_time_bin_edges[-1]), step=self.time_bin_size, num_bins=curr_epoch_num_time_bins))
                curr_time_bin_container = BinningContainer(edges=curr_time_bin_edges, centers=selected_rtc_time, edge_info=curr_edge_info, center_info=curr_center_info)
                ## main call: calls self.decode(...) 
                most_likely_positions, p_x_given_n, most_likely_position_indicies, _flat_outputs_container = self.decode(selected_multiunits, time_bin_size=self.time_bin_size, rtc_time=selected_rtc_time, output_flat_versions=False,
                                                                                                                            debug_print=debug_print)
            else:
                ## empty epoch (not t bins)
                curr_time_bin_edges = np.asarray([], dtype=float)
                empty_info = BinningInfo(**dict(variable_extents=(epoch_start, epoch_stop), step=self.time_bin_size, num_bins=0))
                curr_time_bin_container = BinningContainer(edges=curr_time_bin_edges, centers=np.asarray([], dtype=float), edge_info=empty_info, center_info=empty_info)
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
            time_bin_containers.append(curr_time_bin_container)
            time_bin_edges.append(curr_time_bin_edges)
        ## END for epoch_idx, epoch_row in enumerate(filter_epochs_df.itertuples(index=False))..


        if debug_print:
            print(f"ClusterlessRTCPositionDecoder.decode_specific_epochs(): decoded {num_filter_epochs} epochs, nbins={nbins}")

        result_kwargs = dict(decoding_time_bin_size=decoding_time_bin_size, slideby=slideby, filter_epochs=filter_epochs_df, num_filter_epochs=num_filter_epochs,
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


    def _ensure_fitted_classifier(self, multiunits_for_fit=None, rtc_time_for_fit=None, debug_print: bool = False) -> Tuple[ClusterlessClassifier, np.ndarray]:
        if self.classifier is not None and self.rtc_position_bin_centers is not None:
            return self.classifier, self.rtc_position_bin_centers
        training_multiunits = self.multiunits if self.multiunits is not None else multiunits_for_fit
        training_rtc_time = self.rtc_time if self.rtc_time is not None else rtc_time_for_fit
        if training_multiunits is None or training_rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder requires multiunits and rtc_time on the decoder (or passed to decode()) before fitting the classifier.")
        params = self.clusterless_params if self.clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=self.sampling_frequency_hz)
        place_bin_size_override = params.rtc_place_bin_size_override
        if (place_bin_size_override is None) and (self.ndim > 1):
            place_bin_size_override = params.rtc_2d_place_bin_size_override
        environment = build_rtc_environment_from_pfnd(self.pf, environment_name=params.rtc_environment_name, place_bin_size_override=place_bin_size_override)
        self.classifier = ClusterlessClassifier(environments=[environment], clusterless_algorithm="multiunit_likelihood", clusterless_algorithm_params={"mark_std": params.rtc_mark_std, "position_std": params.rtc_position_std})
        position_train, multiunits_train, is_training = build_clusterless_training_data_from_pfnd(self.pf, training_multiunits, training_rtc_time, self.sampling_frequency_hz)
        self.is_training_mask = is_training
        self.classifier.fit(position_train, multiunits_train, is_training=is_training)
        fitted_environment = self.classifier.environments[0]
        n_position_bins = int(np.asarray(fitted_environment.is_track_interior_).size)
        self.estimated_log_likelihood_memory_bytes = self.raise_if_log_likelihood_exceeds_memory_limit(n_time=len(multiunits_train), n_position_bins=n_position_bins, max_memory_gib=params.max_log_likelihood_memory_gib)
        self.rtc_position_bin_centers = np.asarray(fitted_environment.place_bin_centers_)
        return self.classifier, self.rtc_position_bin_centers


    def _predict_clusterless_posterior(self, multiunits, rtc_time, multiunits_for_fit=None, rtc_time_for_fit=None, is_compute_acausal=True, use_gpu=None, debug_print=False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], xr.Dataset]:
        use_gpu = self._resolve_use_gpu(use_gpu=use_gpu, debug_print=debug_print)
        classifier, rtc_position_bin_centers = self._ensure_fitted_classifier(multiunits_for_fit=multiunits_for_fit, rtc_time_for_fit=rtc_time_for_fit, debug_print=debug_print)
        params = self.clusterless_params if self.clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=self.sampling_frequency_hz)
        multiunits = np.asarray(multiunits, dtype=float)
        rtc_time = np.asarray(rtc_time, dtype=float)
        n_position_bins = int(np.asarray(classifier.environments[0].is_track_interior_).size)
        self.raise_if_log_likelihood_exceeds_memory_limit(n_time=len(multiunits), n_position_bins=n_position_bins, max_memory_gib=params.max_log_likelihood_memory_gib)
        rtc_results = classifier.predict(multiunits, time=rtc_time[:len(multiunits)], is_compute_acausal=is_compute_acausal, use_gpu=use_gpu)
        p_x_given_n = rtc_posterior_to_p_x_given_n(rtc_results, self.pf, state_index=params.state_index_for_posterior, should_match_pf_grid=params.should_match_pf_grid)
        flat_p_x_given_n = p_x_given_n.reshape(int(np.prod(p_x_given_n.shape[:-1])), p_x_given_n.shape[-1]) if p_x_given_n.ndim > 2 else p_x_given_n
        place_bin_centers = rtc_position_bin_centers if (rtc_position_bin_centers is not None and p_x_given_n.shape[0] == len(rtc_position_bin_centers)) else None
        return p_x_given_n, flat_p_x_given_n, place_bin_centers, rtc_results


    def compute_all(self, is_compute_acausal=True, use_gpu: Optional[bool] = None, debug_print: bool = True) -> None:
        """ main pre-compute function """
        if self.multiunits is None or self.rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder requires multiunits and rtc_time before compute_all().")
        p_x_given_n, flat_p_x_given_n, active_position_bin_centers, self.rtc_results = self._predict_clusterless_posterior(self.multiunits, self.rtc_time, multiunits_for_fit=self.multiunits, rtc_time_for_fit=self.rtc_time, is_compute_acausal=is_compute_acausal, use_gpu=use_gpu, debug_print=(debug_print or self.debug_print))
        self.p_x_given_n = p_x_given_n
        self.flat_p_x_given_n = flat_p_x_given_n
        self.most_likely_positions = most_likely_positions_from_posterior(self.p_x_given_n, self.pf, place_bin_centers=active_position_bin_centers)
        self.revised_most_likely_positions = self.most_likely_positions.copy()
        self.most_likely_position_flat_indicies = np.argmax(self.p_x_given_n, axis=0)
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

