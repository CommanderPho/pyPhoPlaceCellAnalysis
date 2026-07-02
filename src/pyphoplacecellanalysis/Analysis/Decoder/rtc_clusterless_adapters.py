"""Adapters between NeuroPy PfND / session data and replay_trajectory_classification clusterless decoders."""
from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from neuropy.analyses.placefields import PfND
from neuropy.core.clusterless_spike_events import ClusterlessSpikeEvents, CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION, default_clusterless_spike_events_path, load_clusterless_spike_events, save_clusterless_spike_events
from neuropy.utils.efficient_interval_search import determine_event_interval_is_included
from pyphocorehelpers.function_helpers import function_attributes
from replay_trajectory_classification.core import atleast_2d, get_centers
from replay_trajectory_classification.environments import Environment, get_track_boundary, get_track_interior


@dataclass
class ClusterlessDecodingParameters:
    clusterless_sampling_frequency_hz: float = 1000.0
    position_sampling_frequency_Hz: float = 120.0
    rtc_place_bin_size_override: Optional[float] = None
    rtc_2d_place_bin_size_override: Optional[float] = None
    rtc_mark_std: float = 24.0
    rtc_position_std: float = 6.0
    rtc_environment_name: str = ""
    state_index_for_posterior: Optional[int] = None
    max_log_likelihood_memory_gib: Optional[float] = 36.0 ## 36 GB
    should_match_pf_grid: bool = True


@dataclass
class PfNDSyncedEnvironment(Environment):
    """RTC Environment whose spatial grid is aligned to PfND xbin/ybin edges when available.

    from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import PfNDSyncedEnvironment, ClusterlessDecodingParameters

    """

    pf: Any = None


    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.environment_name == other
        if isinstance(other, Environment):
            return self.environment_name == other.environment_name
        return NotImplemented


    def fit_place_grid(self, position: Optional[np.ndarray] = None, infer_track_interior: bool = True):
        pf_xbin = getattr(self.pf, 'xbin', None) if self.pf is not None else None
        if pf_xbin is not None and len(np.asarray(pf_xbin)) >= 2:
            if position is None:
                raise ValueError("Must provide position when fitting PfNDSyncedEnvironment.")
            position = atleast_2d(np.asarray(position, dtype=float))
            is_nan = np.any(np.isnan(position), axis=1)
            position = position[~is_nan]
            x_edges = np.asarray(pf_xbin, dtype=float)
            if self.pf.ndim == 1:
                self.edges_ = (x_edges,)
                x_centers = get_centers(x_edges)
                self.centers_shape_ = (len(x_centers),)
                self.place_bin_centers_ = x_centers[:, np.newaxis]
                mesh_edges = np.meshgrid(x_edges, indexing='ij')
                self.place_bin_edges_ = np.stack([mesh_edges[0].ravel(), mesh_edges[0].ravel()], axis=1)
            else:
                y_edges = np.asarray(self.pf.ybin, dtype=float)
                self.edges_ = (x_edges, y_edges)
                mesh_centers = np.meshgrid(get_centers(x_edges), get_centers(y_edges))
                self.centers_shape_ = mesh_centers[0].T.shape
                self.place_bin_centers_ = np.stack([center.ravel() for center in mesh_centers], axis=1)
                mesh_edges = np.meshgrid(x_edges, y_edges)
                self.place_bin_edges_ = np.stack([edge.ravel() for edge in mesh_edges], axis=1)
            self.infer_track_interior = infer_track_interior
            if self.is_track_interior is None and infer_track_interior:
                self.is_track_interior_ = get_track_interior(position, bins=self.centers_shape_, fill_holes=self.fill_holes, dilate=self.dilate, bin_count_threshold=self.bin_count_threshold)
            elif self.is_track_interior is None and not infer_track_interior:
                self.is_track_interior_ = np.ones(self.centers_shape_, dtype=bool)
            if len(self.edges_) > 1:
                self.is_track_boundary_ = get_track_boundary(self.is_track_interior_, connectivity=1)
            else:
                self.is_track_boundary_ = None
            return self
        return super().fit_place_grid(position, infer_track_interior=infer_track_interior)

def _pfnd_place_bin_size(pf: PfND, place_bin_size_override: Optional[float] = None) -> float:
    if place_bin_size_override is not None:
        return float(place_bin_size_override)
    pos_bin_size = pf.pos_bin_size
    if isinstance(pos_bin_size, (tuple, list, np.ndarray)):
        return float(np.mean(pos_bin_size))
    return float(pos_bin_size)


def _pfnd_position_range(pf: PfND) -> Optional[np.ndarray]:
    grid_bin_bounds = pf.config.grid_bin_bounds
    if grid_bin_bounds is None:
        return None
    if pf.ndim == 1:
        bounds_1d = getattr(pf.config, 'grid_bin_bounds_1D', None)
        if bounds_1d is None:
            return None
        bounds_1d = np.asarray(bounds_1d, dtype=float)
        if bounds_1d.ndim == 0:
            bounds_1d = np.asarray(grid_bin_bounds, dtype=float)
        return bounds_1d.reshape(1, 2)
    grid_bin_bounds = np.asarray(grid_bin_bounds, dtype=float)
    if grid_bin_bounds.shape == (4,):
        xmin, ymin, xmax, ymax = grid_bin_bounds
        return np.array([[xmin, xmax], [ymin, ymax]], dtype=float)
    return grid_bin_bounds.reshape(2, 2)


def position_array_from_pfnd(pf: PfND) -> np.ndarray:
    pos_df = pf.filtered_pos_df
    if pf.ndim == 1:
        if 'lin_pos' in pos_df.columns:
            return pos_df['lin_pos'].to_numpy(dtype=float)[:, np.newaxis]
        return pos_df['x'].to_numpy(dtype=float)[:, np.newaxis]
    return np.column_stack([pos_df['x'].to_numpy(dtype=float), pos_df['y'].to_numpy(dtype=float)])


def build_rtc_environment_from_pfnd(pf: PfND, environment_name: str = "", place_bin_size_override: Optional[float] = None) -> PfNDSyncedEnvironment:
    position_range = _pfnd_position_range(pf)
    place_bin_size = _pfnd_place_bin_size(pf, place_bin_size_override=place_bin_size_override)
    if position_range is not None:
        return PfNDSyncedEnvironment(environment_name=environment_name, place_bin_size=place_bin_size, position_range=position_range, infer_track_interior=True, pf=pf)
    return PfNDSyncedEnvironment(environment_name=environment_name, place_bin_size=place_bin_size, infer_track_interior=True, pf=pf)


def resample_position_to_rtc_clock(position_array: np.ndarray, source_times: np.ndarray, t_start: float, t_end: float, sampling_frequency_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    position_array = np.asarray(position_array, dtype=float)
    if position_array.ndim == 1:
        position_array = position_array[:, np.newaxis]
    n_time = max(1, int(np.round((t_end - t_start) * sampling_frequency_hz)))
    rtc_time = t_start + (np.arange(n_time, dtype=float) + 0.5) / sampling_frequency_hz
    rtc_time = rtc_time[rtc_time <= t_end]
    resampled_position = np.empty((len(rtc_time), position_array.shape[1]), dtype=float)
    for dim_idx in range(position_array.shape[1]):
        resampled_position[:, dim_idx] = np.interp(rtc_time, source_times, position_array[:, dim_idx])
    return rtc_time, resampled_position


def _pfnd_epoch_filtered_position_df(pf: PfND) -> pd.DataFrame:
    pos_df = pf.position.to_dataframe()
    if pf.epochs is not None:
        epoch_filtered_pos_df = pos_df.position.time_sliced(pf.epochs.starts, pf.epochs.stops)
    else:
        epoch_filtered_pos_df = pos_df.position.time_sliced(pf.position.t_start, pf.position.t_stop)
    pos_non_na_column_labels = ['x', 'y'] if pf.ndim > 1 else ['x']
    return epoch_filtered_pos_df.dropna(axis=0, how='any', subset=pos_non_na_column_labels).copy()


def _pfnd_speed_filtered_training_intervals(pf: PfND, position_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Return [start, stop] intervals used by PfND.setup for occupancy-preserving speed filtering."""
    if position_df is None:
        position_df = _pfnd_epoch_filtered_position_df(pf)
    speed_column_name = 'speed'
    if pf.should_smooth_speed and (pf.config.smooth is not None) and (pf.config.smooth[0] > 0.0):
        from scipy.ndimage import gaussian_filter1d
        position_df = position_df.copy()
        position_df['speed_smooth'] = gaussian_filter1d(position_df.speed.to_numpy(), sigma=pf.config.smooth[0])
        speed_column_name = 'speed_smooth'
    if pf.epochs is not None:
        epochs_df = deepcopy(pf.epochs.to_dataframe())
    else:
        epochs_df = pd.DataFrame({'start': [float(pf.position.t_start)], 'stop': [float(pf.position.t_stop)], 'label': [0], 'duration': [float(pf.position.t_stop - pf.position.t_start)]})
    speed_filtered_epochs_df = PfND.filtered_by_speed(epochs_df, position_df=position_df, speed_thresh=pf.config.speed_thresh, speed_column_override_name=speed_column_name, debug_print=False)
    if speed_filtered_epochs_df is None or len(speed_filtered_epochs_df) == 0:
        return np.empty((0, 2), dtype=float)
    return speed_filtered_epochs_df[['start', 'stop']].to_numpy(dtype=float)


def build_is_training_mask_from_pfnd(pf: PfND, rtc_time: np.ndarray) -> np.ndarray:
    """Match PfND.setup inclusion: epoch filter, NaN position drop, optional speed smoothing, occupancy-preserving speed epochs."""
    included_intervals = _pfnd_speed_filtered_training_intervals(pf)
    if included_intervals.size == 0:
        return np.zeros(len(rtc_time), dtype=bool)
    return determine_event_interval_is_included(np.asarray(rtc_time, dtype=float), included_intervals)


def _get_multiunit_electrode_keep_mask(multiunits: np.ndarray, time_mask: Optional[np.ndarray] = None) -> np.ndarray:
    active_multiunits = multiunits[time_mask, ...] if time_mask is not None else multiunits
    has_spikes_by_electrode = np.any(np.any(np.isfinite(active_multiunits), axis=1), axis=0)
    if not np.any(has_spikes_by_electrode):
        raise ValueError("Clusterless decoding requires at least one electrode with finite waveform marks.")
    return has_spikes_by_electrode


def _drop_empty_multiunit_electrodes(multiunits: np.ndarray, time_mask: Optional[np.ndarray] = None) -> np.ndarray:
    return multiunits[:, :, _get_multiunit_electrode_keep_mask(multiunits, time_mask=time_mask)]


def build_multiunits_from_array(multiunits: np.ndarray, time: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    multiunits = np.asarray(multiunits, dtype=float)
    if multiunits.ndim != 3:
        raise ValueError(f"multiunits must have shape (n_time, n_marks, n_electrodes); got {multiunits.shape}")
    return _drop_empty_multiunit_electrodes(multiunits), time


def build_multiunits_from_rtc_simulation(n_runs: int = 5, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from replay_trajectory_classification.clusterless_simulation import make_simulated_run_data
    time, position, _sampling_frequency, multiunits, _multiunits_spikes = make_simulated_run_data(n_runs=n_runs, **kwargs)
    return time, position[:, np.newaxis], multiunits, position


def _assign_spike_marks_to_multiunits(multiunits: np.ndarray, time_bin_indices: np.ndarray, electrode_indices: np.ndarray, mark_features: np.ndarray) -> None:
    n_marks = multiunits.shape[1]
    mark_features = np.asarray(mark_features, dtype=float)
    if mark_features.ndim == 1:
        mark_features = mark_features[np.newaxis, :]
    n_features = min(n_marks, mark_features.shape[-1])
    for spike_idx in range(len(time_bin_indices)):
        t_idx = int(time_bin_indices[spike_idx])
        e_idx = int(electrode_indices[spike_idx])
        if 0 <= t_idx < multiunits.shape[0] and 0 <= e_idx < multiunits.shape[2]:
            multiunits[t_idx, :n_features, e_idx] = mark_features[spike_idx, :n_features]


@function_attributes(short_name=None, tags=['BAD', 'BUG', 'ISSUE'], input_requires=[], output_provides=[], uses=['_assign_spike_marks_to_multiunits'], used_by=[], creation_date='2026-07-01 08:52', related_items=[])
def build_multiunits_from_session(sess, sampling_frequency_hz: float, t_start: float, t_end: float, spikes_df: Optional[pd.DataFrame] = None, n_mark_dims: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """ uses the `sess.neurons` object, so it is not true clusterless data as they have already been filtered by pyrmaidal/sua/etc 
    """
    neurons = sess.neurons
    if neurons.waveforms is None:
        raise ValueError("Clusterless decoding requires session.neurons.waveforms to be populated with per-neuron waveform mark features, or pass a pre-built multiunits array via the pipeline multiunits= kwarg. Expected RTC tensor shape: (n_time, n_marks, n_electrodes) with NaN for no spike and non-NaN mark features for spikes. See replay_trajectory_classification notebook 03-Decoding_with_Clusterless_Spikes.")
    if spikes_df is None:
        spikes_df = sess.spikes_df
    spikes_df = spikes_df.copy()
    time_variable_name = sess.spikes.time_variable_name if hasattr(sess, 'spikes') else 't'
    if time_variable_name not in spikes_df.columns and 't_seconds' in spikes_df.columns:
        time_variable_name = 't_seconds'
    spike_times = spikes_df[time_variable_name].to_numpy(dtype=float)
    valid_spikes = (spike_times >= t_start) & (spike_times <= t_end)
    spike_times = spike_times[valid_spikes]
    spikes_df = spikes_df.loc[valid_spikes].reset_index(drop=True)
    n_time = max(1, int(np.round((t_end - t_start) * sampling_frequency_hz)))
    rtc_time = t_start + (np.arange(n_time, dtype=float) + 0.5) / sampling_frequency_hz
    rtc_time = rtc_time[rtc_time <= t_end]
    if neurons.shank_ids is not None:
        n_electrodes = int(np.max(np.asarray(neurons.shank_ids, dtype=int))) + 1
    elif neurons.peak_channels is not None:
        n_electrodes = int(np.max(np.asarray(neurons.peak_channels, dtype=int))) + 1
    else:
        n_electrodes = int(neurons.n_neurons)
    multiunits = np.full((len(rtc_time), n_mark_dims, n_electrodes), np.nan, dtype=float)
    aclu_column = 'aclu' if 'aclu' in spikes_df.columns else ('cluster' if 'cluster' in spikes_df.columns else None)
    if aclu_column is None:
        raise ValueError("spikes_df must contain 'aclu' or 'cluster' to map spikes to waveform marks.")
    reverse_map = neurons.reverse_cellID_index_map
    time_bin_indices = np.clip(np.searchsorted(rtc_time, spike_times), 0, len(rtc_time) - 1)
    electrode_indices = []
    mark_features = []
    waveforms = np.asarray(neurons.waveforms)
    for spike_row_idx, aclu in enumerate(spikes_df[aclu_column].to_numpy()):
        if aclu not in reverse_map:
            continue
        neuron_idx = reverse_map[int(aclu)]
        neuron_waveform = waveforms[neuron_idx]
        if neuron_waveform.ndim == 1:
            feature_vector = neuron_waveform
        elif neuron_waveform.ndim == 2:
            feature_vector = neuron_waveform[0] if neuron_waveform.shape[0] == 1 else neuron_waveform.mean(axis=0)
        else:
            feature_vector = neuron_waveform.reshape(-1)
        if neurons.shank_ids is not None:
            electrode_idx = int(neurons.shank_ids[neuron_idx])
        elif neurons.peak_channels is not None:
            electrode_idx = int(neurons.peak_channels[neuron_idx])
        else:
            electrode_idx = int(neuron_idx)
        electrode_indices.append(electrode_idx)
        mark_features.append(feature_vector)
    if len(mark_features) == 0:
        raise ValueError("No spikes could be mapped to waveform marks for clusterless decoding.")
    _assign_spike_marks_to_multiunits(multiunits, time_bin_indices[:len(mark_features)], np.asarray(electrode_indices, dtype=int), np.asarray(mark_features, dtype=float))
    return _drop_empty_multiunit_electrodes(multiunits), rtc_time


@function_attributes(short_name=None, tags=['MAIN', 'GOOD', 'load', 'phy', 'clusterless'], input_requires=[], output_provides=[], uses=[], used_by=['build_multiunits_from_phy_folder'], creation_date='2026-07-01 10:12', related_items=[])
def extract_clusterless_spike_events_from_phy_folder(phy_path: Union[str, Path], t_start: Optional[float] = None, t_end: Optional[float] = None, electrode_mode: str = "shank", n_mark_dims: int = 4, chunk_size: int = 100_000, sampling_frequency_hz: float = 1000.0) -> ClusterlessSpikeEvents:
    """Extract sparse clusterless spike events from a Phy/Kilosort folder without allocating dense multiunits.

    Saves portable spike times, electrode indices, and PC marks for later epoch-local binning.
    Do not materialize full-session dense multiunits at 1 kHz on long recordings.

    When ``t_start`` and/or ``t_end`` are omitted, the session span is inferred from ``params.py``
    (``t_start``/``tmin``, ``t_end``/``tmax``/``duration``, ``n_samples_dat``, or ``dat_path`` file size)
    and falls back to the last spike time if no recording duration is available.
    """
    return ClusterlessSpikeEvents.from_phy_folder(phy_path, t_start=t_start, t_end=t_end, electrode_mode=electrode_mode, n_mark_dims=n_mark_dims, chunk_size=chunk_size, sampling_frequency_hz=sampling_frequency_hz)


@function_attributes(short_name=None, tags=['spikes'], input_requires=[], output_provides=[], uses=['_assign_spike_marks_to_multiunits'], used_by=[], creation_date='2026-07-01 10:11', related_items=[])
def build_multiunits_from_spike_events(events: ClusterlessSpikeEvents, t_start: Optional[float] = None, t_end: Optional[float] = None, sampling_frequency_hz: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize dense RTC multiunits for a time window from sparse clusterless spike events.

    When ``t_start`` and/or ``t_end`` are omitted, missing bounds are taken from ``events.t_start`` and ``events.t_end``.
    """
    if t_start is None:
        t_start = float(events.t_start)
    if t_end is None:
        t_end = float(events.t_end)

    sampling_frequency_hz = float(sampling_frequency_hz if sampling_frequency_hz is not None else events.sampling_frequency_hz)
    valid_spikes = (events.spike_times_sec >= t_start) & (events.spike_times_sec <= t_end)
    spike_times_sec = events.spike_times_sec[valid_spikes]
    electrode_indices = events.electrode_indices[valid_spikes]
    marks = events.marks[valid_spikes]
    if len(spike_times_sec) == 0:
        raise ValueError(f"No clusterless spike events found for epoch t=[{t_start}, {t_end}] seconds.")
    n_time = max(1, int(np.round((t_end - t_start) * sampling_frequency_hz)))
    rtc_time = t_start + (np.arange(n_time, dtype=float) + 0.5) / sampling_frequency_hz
    rtc_time = rtc_time[rtc_time <= t_end]
    n_mark_dims = int(events.marks.shape[1])
    n_electrodes = int(np.max(electrode_indices)) + 1
    multiunits = np.full((len(rtc_time), n_mark_dims, n_electrodes), np.nan, dtype=float)
    time_bin_indices = np.clip(np.searchsorted(rtc_time, spike_times_sec.astype(float)), 0, len(rtc_time) - 1)
    _assign_spike_marks_to_multiunits(multiunits, time_bin_indices, electrode_indices.astype(int), marks.astype(float))
    return _drop_empty_multiunit_electrodes(multiunits), rtc_time


@function_attributes(tags=['correct', 'mua', 'clusterless', 'correct'], input_requires=[], output_provides=[], uses=['extract_clusterless_spike_events_from_phy_folder','build_multiunits_from_spike_events'], used_by=[], creation_date='2026-07-01 08:55', related_items=[])
def build_multiunits_from_phy_folder(phy_path: Union[str, Path], t_start: float, t_end: float, sampling_frequency_hz: float, electrode_mode: str = "shank", n_mark_dims: int = 4, chunk_size: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """Build RTC clusterless multiunits from a Phy/Kilosort export folder.

    Reads per-spike PC marks from pc_features.npy (all detected spikes; ignores spike_clusters).
    Epoch filtering is applied before binning to limit memory use. Full-session clusterless
    decoding at 1 kHz can still require large RAM for long epochs — prefer
    extract_clusterless_spike_events_from_phy_folder + save_clusterless_spike_events for transfer.

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import build_multiunits_from_phy_folder

        multiunits, rtc_time = build_multiunits_from_phy_folder(
            phy_path, t_start=11510.0, t_end=14693.0, sampling_frequency_hz=1000.0, electrode_mode="channel"
        )
        # Pass to pipeline: perform_specific_computation(..., multiunits=multiunits, rtc_time=rtc_time)

    """
    events = extract_clusterless_spike_events_from_phy_folder(phy_path, t_start=t_start, t_end=t_end, electrode_mode=electrode_mode, n_mark_dims=n_mark_dims, chunk_size=chunk_size, sampling_frequency_hz=sampling_frequency_hz)
    return build_multiunits_from_spike_events(events, t_start=t_start, t_end=t_end, sampling_frequency_hz=sampling_frequency_hz)





# ==================================================================================================================================================================================================================================================================================== #
# Post Compute                                                                                                                                                                                                                                                                         #
# ==================================================================================================================================================================================================================================================================================== #
def _rtc_posterior_spatial_time_values(rtc_results: xr.Dataset, state_index: Optional[int] = None) -> np.ndarray:
    posterior = rtc_results.acausal_posterior
    if state_index is None:
        posterior_time_bins = posterior.sum(dim="state") if "state" in posterior.dims else posterior
    else:
        posterior_time_bins = posterior.isel(state=state_index)
    spatial_dims = [a_dim for a_dim in posterior_time_bins.dims if a_dim != "time"]
    return posterior_time_bins.transpose(*spatial_dims, "time").values


def rtc_posterior_flat_p_x_given_n(rtc_results: xr.Dataset, pf: PfND, state_index: Optional[int] = None) -> np.ndarray:
    posterior_values = _rtc_posterior_spatial_time_values(rtc_results, state_index=state_index)
    return posterior_values.reshape((int(np.prod(posterior_values.shape[:-1])), posterior_values.shape[-1]), order="F")


def rtc_posterior_to_p_x_given_n(rtc_results: xr.Dataset, pf: PfND, state_index: Optional[int] = None, should_match_pf_grid: bool = True) -> np.ndarray:
    posterior_values = _rtc_posterior_spatial_time_values(rtc_results, state_index=state_index)
    position_shape = tuple(np.shape(pf.occupancy))
    n_time = posterior_values.shape[-1]
    flat_size = int(np.prod(posterior_values.shape[:-1]))
    pf_flat_size = int(np.prod(position_shape))
    if flat_size == pf_flat_size:
        return posterior_values.reshape((*position_shape, n_time), order="F")
    flat_p_x_given_n = posterior_values.reshape((flat_size, n_time), order="F")
    if not should_match_pf_grid:
        warnings.warn(f"RTC posterior flat size {flat_size} != PfND occupancy size {pf_flat_size}; returning RTC spatial shape {posterior_values.shape}.", stacklevel=2)
        return posterior_values
    if flat_size > pf_flat_size:
        warnings.warn(f"RTC posterior has {flat_size} bins but PfND occupancy has {pf_flat_size}; truncating to PfND grid.", stacklevel=2)
        flat_p_x_given_n = flat_p_x_given_n[:pf_flat_size, :]
    else:
        warnings.warn(f"RTC posterior has {flat_size} bins but PfND occupancy has {pf_flat_size}; zero-padding to PfND grid.", stacklevel=2)
        padded = np.zeros((pf_flat_size, n_time), dtype=posterior_values.dtype)
        padded[:flat_size, :] = flat_p_x_given_n
        flat_p_x_given_n = padded
    return flat_p_x_given_n.reshape((*position_shape, n_time), order="F")


def most_likely_positions_from_posterior(p_x_given_n: np.ndarray, pf: PfND, place_bin_centers: Optional[np.ndarray] = None) -> np.ndarray:
    most_likely_flat_indices = np.argmax(p_x_given_n, axis=0)
    if place_bin_centers is not None:
        place_bin_centers = np.asarray(place_bin_centers)
        most_likely_flat_indices = np.clip(most_likely_flat_indices, 0, len(place_bin_centers) - 1)
        return np.squeeze(place_bin_centers[most_likely_flat_indices])
    if pf.ndim == 1:
        x_centers = pf.xbin_centers
        return x_centers[np.clip(most_likely_flat_indices, 0, len(x_centers) - 1)]
    x_centers = pf.xbin_centers
    y_centers = pf.ybin_centers
    n_x = len(x_centers)
    x_idx = most_likely_flat_indices // len(y_centers)
    y_idx = most_likely_flat_indices % len(y_centers)
    return np.column_stack([x_centers[np.clip(x_idx, 0, n_x - 1)], y_centers[np.clip(y_idx, 0, len(y_centers) - 1)]])


def build_clusterless_training_data_from_pfnd(pf: PfND, multiunits: np.ndarray, rtc_time: np.ndarray, sampling_frequency_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 

    """
    from replay_trajectory_classification import ClusterlessClassifier, Environment, RandomWalk, Uniform, Identity, estimate_movement_var

    source_pos_df = pf.filtered_pos_df
    pos_sampling_rate_Hz: float = source_pos_df.metadata.metadata.get('sampling_rate', 120.0) ## Hz
    pos_sampling_rate_Hz


    source_times = source_pos_df['t'].to_numpy(dtype=float) if 't' in source_pos_df.columns else source_pos_df['t_seconds'].to_numpy(dtype=float)
    t_start = float(rtc_time[0] - 0.5 / sampling_frequency_hz)
    t_end = float(rtc_time[-1] + 0.5 / sampling_frequency_hz)

    

    # movement_var = estimate_movement_var(source_pos_df[['x', 'y']].to_numpy(), sampling_frequency=)

    position_source = position_array_from_pfnd(pf)
    _, resampled_position = resample_position_to_rtc_clock(position_source, source_times, t_start, t_end, sampling_frequency_hz)
    if len(resampled_position) != len(rtc_time):
        min_len = min(len(resampled_position), len(rtc_time))
        resampled_position = resampled_position[:min_len]
        multiunits = multiunits[:min_len]
        rtc_time = rtc_time[:min_len]
    is_training = build_is_training_mask_from_pfnd(pf, rtc_time)
    return resampled_position, multiunits, is_training
