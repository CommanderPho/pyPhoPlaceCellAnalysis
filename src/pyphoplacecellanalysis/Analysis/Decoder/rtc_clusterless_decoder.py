from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from neuropy.utils.mixins.binning_helpers import BinningContainer, compute_spanning_bins
from replay_trajectory_classification import ClusterlessClassifier

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import (
    ClusterlessDecodingParameters,
    build_clusterless_training_data_from_pfnd,
    build_rtc_environment_from_pfnd,
    most_likely_positions_from_posterior,
    rtc_posterior_to_p_x_given_n,
)
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, non_serialized_field


@custom_define(slots=False, eq=False)
class ClusterlessRTCPositionDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder):
    """Clusterless position decoder using replay_trajectory_classification on PfND spatial grids."""

    sampling_frequency_hz: float = 1000.0
    multiunits: np.ndarray = None
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
    is_training_mask: np.ndarray = non_serialized_field(default=None, repr=False)
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
    def P_x(self):
        raise NotImplementedError("ClusterlessRTCPositionDecoder does not compute P_x independently. It directly computes p_x_given_n.")


    @property
    def neuron_IDXs(self):
        raise NotImplementedError("Clusterless decoding uses multiunits, not individual neurons (neuron_IDXs).")


    @property
    def neuron_IDs(self):
        raise NotImplementedError("Clusterless decoding uses multiunits, not individual neurons (neuron_IDs).")


    @property
    def num_neurons(self) -> int:
        raise NotImplementedError("Clusterless decoding uses multiunits, not individual neurons (num_neurons).")


    @property
    def F(self):
        raise NotImplementedError("ClusterlessRTCPositionDecoder does not compute standard place fields (F).")


    def get_by_id(self, ids, defer_compute_all:bool=False):
        raise NotImplementedError("ClusterlessRTCPositionDecoder does not support neuron-based slicing (get_by_id) because it relies on multiunits.")


    def decode(self, *args, **kwargs):
        raise NotImplementedError("ClusterlessRTCPositionDecoder uses `compute_all()` natively and does not support the generalized `decode` method.")


    def decode_specific_epochs(self, *args, **kwargs):
        raise NotImplementedError("ClusterlessRTCPositionDecoder uses `compute_all()` natively and does not currently support `decode_specific_epochs`.")


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


    def compute_all(self, is_compute_acausal=True, use_gpu: Optional[bool] = None, debug_print: bool = True) -> None:
        if self.multiunits is None or self.rtc_time is None:
            raise ValueError("ClusterlessRTCPositionDecoder requires multiunits and rtc_time before compute_all().")
        if use_gpu is None:
            use_gpu = self._is_rtc_gpu_acceleration_available()
            if not use_gpu:
                print("Warning: ClusterlessRTCPositionDecoder GPU acceleration unavailable; falling back to CPU decoding.")
        elif use_gpu and not self._is_rtc_gpu_acceleration_available():
            print("Warning: ClusterlessRTCPositionDecoder GPU requested but unavailable; falling back to CPU decoding.")
            use_gpu = False
        elif use_gpu and (debug_print or self.debug_print):
            print("ClusterlessRTCPositionDecoder.compute_all(): using GPU acceleration (CuPy).")
        params = self.clusterless_params if self.clusterless_params is not None else ClusterlessDecodingParameters(clusterless_sampling_frequency_hz=self.sampling_frequency_hz)
        place_bin_size_override = params.rtc_place_bin_size_override
        if (place_bin_size_override is None) and (self.ndim > 1):
            place_bin_size_override = params.rtc_2d_place_bin_size_override
        environment = build_rtc_environment_from_pfnd(self.pf, environment_name=params.rtc_environment_name, place_bin_size_override=place_bin_size_override)
        self.classifier = ClusterlessClassifier(environments=[environment], clusterless_algorithm="multiunit_likelihood", clusterless_algorithm_params={"mark_std": params.rtc_mark_std, "position_std": params.rtc_position_std})
        position_train, multiunits_train, is_training = build_clusterless_training_data_from_pfnd(self.pf, self.multiunits, self.rtc_time, self.sampling_frequency_hz)
        self.is_training_mask = is_training
        self.classifier.fit(position_train, multiunits_train, is_training=is_training)
        fitted_environment = self.classifier.environments[0]
        n_position_bins = int(np.asarray(fitted_environment.is_track_interior_).size)
        self.estimated_log_likelihood_memory_bytes = self.raise_if_log_likelihood_exceeds_memory_limit(n_time=len(multiunits_train), n_position_bins=n_position_bins, max_memory_gib=params.max_log_likelihood_memory_gib)
        self.rtc_position_bin_centers = np.asarray(fitted_environment.place_bin_centers_)
        self.rtc_results = self.classifier.predict(multiunits_train, time=self.rtc_time[:len(multiunits_train)], is_compute_acausal=is_compute_acausal, use_gpu=use_gpu)
        self.p_x_given_n = rtc_posterior_to_p_x_given_n(self.rtc_results, self.pf, state_index=params.state_index_for_posterior, should_match_pf_grid=params.should_match_pf_grid)
        self.flat_p_x_given_n = self.p_x_given_n.reshape(self.flat_position_size, self.num_time_windows) if self.p_x_given_n.ndim > 2 else self.p_x_given_n
        active_position_bin_centers = self.rtc_position_bin_centers if (self.rtc_position_bin_centers is not None and self.p_x_given_n.shape[0] == len(self.rtc_position_bin_centers)) else None
        self.most_likely_positions = most_likely_positions_from_posterior(self.p_x_given_n, self.pf, place_bin_centers=active_position_bin_centers)
        self.revised_most_likely_positions = self.most_likely_positions.copy()
        self.most_likely_position_flat_indicies = np.argmax(self.p_x_given_n, axis=0)
        time_window_edges, time_window_edges_binning_info = compute_spanning_bins(self.rtc_time, bin_size=self.time_bin_size)
        self.time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        if debug_print or self.debug_print:
            print(f"ClusterlessRTCPositionDecoder.compute_all(): p_x_given_n.shape={self.p_x_given_n.shape}, most_likely_positions.shape={np.shape(self.most_likely_positions)}")
