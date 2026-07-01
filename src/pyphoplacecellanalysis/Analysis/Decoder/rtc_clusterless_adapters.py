"""Adapters between NeuroPy PfND / session data and replay_trajectory_classification clusterless decoders."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from neuropy.analyses.placefields import PfND
from replay_trajectory_classification.environments import Environment


@dataclass
class ClusterlessDecodingParameters:
    clusterless_sampling_frequency_hz: float = 1000.0
    rtc_place_bin_size_override: Optional[float] = None
    rtc_2d_place_bin_size_override: Optional[float] = 16.0
    rtc_mark_std: float = 24.0
    rtc_position_std: float = 6.0
    rtc_environment_name: str = ""
    state_index_for_posterior: Optional[int] = None
    max_log_likelihood_memory_gib: Optional[float] = 8.0
    should_match_pf_grid: bool = False


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


def build_rtc_environment_from_pfnd(pf: PfND, environment_name: str = "", place_bin_size_override: Optional[float] = None) -> Environment:
    position_range = _pfnd_position_range(pf)
    if position_range is None:
        return Environment(environment_name=environment_name, place_bin_size=_pfnd_place_bin_size(pf, place_bin_size_override=place_bin_size_override), infer_track_interior=True)
    return Environment(environment_name=environment_name, place_bin_size=_pfnd_place_bin_size(pf, place_bin_size_override=place_bin_size_override), position_range=position_range, infer_track_interior=True)


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


def build_is_training_mask_from_pfnd(pf: PfND, rtc_time: np.ndarray, source_times: np.ndarray, source_speed: np.ndarray) -> np.ndarray:
    speed_at_rtc = np.interp(rtc_time, source_times, source_speed)
    return speed_at_rtc >= float(pf.config.speed_thresh)


def _drop_empty_multiunit_electrodes(multiunits: np.ndarray) -> np.ndarray:
    has_spikes_by_electrode = np.any(np.any(np.isfinite(multiunits), axis=1), axis=0)
    if not np.any(has_spikes_by_electrode):
        raise ValueError("Clusterless decoding requires at least one electrode with finite waveform marks.")
    return multiunits[:, :, has_spikes_by_electrode]


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


def build_multiunits_from_session(sess, sampling_frequency_hz: float, t_start: float, t_end: float, spikes_df: Optional[pd.DataFrame] = None, n_mark_dims: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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


_PHY_CLUSTERLESS_REQUIRED_FILES = ("params.py", "spike_times.npy", "spike_templates.npy", "pc_features.npy", "pc_feature_ind.npy")


def _subfn_read_phy_params(phy_path: Path) -> float:
    params: dict[str, str] = {}
    with (phy_path / "params.py").open("r", encoding="utf-8") as params_file:
        for line in params_file:
            line_values = line.replace("\n", "").replace('r"', '"').replace('"', "").split("=")
            if len(line_values) >= 2:
                params[line_values[0].strip()] = line_values[1].strip()
    if "sample_rate" not in params:
        raise ValueError(f"params.py in {phy_path} is missing sample_rate.")
    return float(params["sample_rate"])


def _subfn_resolve_channel_shanks(phy_path: Path) -> Optional[np.ndarray]:
    candidate_paths = [phy_path / "channel_shanks.npy", phy_path.parent / "sorter_output" / "channel_shanks.npy"]
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return np.load(candidate_path)
    return None


def _subfn_get_epoch_spike_slice(spike_times: np.ndarray, sample_rate_hz: float, t_start: float, t_end: float) -> slice:
    sample_start = int(np.floor(float(t_start) * sample_rate_hz))
    sample_end = int(np.ceil(float(t_end) * sample_rate_hz))
    spike_start = int(np.searchsorted(spike_times, sample_start, side="left"))
    spike_end = int(np.searchsorted(spike_times, sample_end, side="right"))
    return slice(spike_start, spike_end)


def _subfn_build_channel_inverse_map(channel_map: np.ndarray) -> np.ndarray:
    channel_map = np.asarray(channel_map, dtype=int)
    inverse_map = np.full(int(channel_map.max()) + 1, -1, dtype=int)
    for recording_idx, probe_channel in enumerate(channel_map):
        inverse_map[int(probe_channel)] = int(recording_idx)
    return inverse_map


def _subfn_extract_peak_channel_marks(pc_features: np.ndarray, pc_feature_ind: np.ndarray, spike_templates: np.ndarray, spike_indices: np.ndarray, n_mark_dims: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    n_spikes = len(spike_indices)
    n_slots = int(pc_features.shape[2])
    channels = np.empty(n_spikes, dtype=int)
    marks = np.empty((n_spikes, n_mark_dims), dtype=float)
    for spike_offset, spike_index in enumerate(spike_indices):
        template_index = int(spike_templates[spike_index])
        template_channels = pc_feature_ind[template_index]
        spike_pcs = pc_features[spike_index]
        slot_norms = np.array([np.linalg.norm(spike_pcs[:, slot_idx]) if template_channels[slot_idx] >= 0 else -1.0 for slot_idx in range(n_slots)], dtype=float)
        peak_slot = int(np.argmax(slot_norms))
        channels[spike_offset] = int(template_channels[peak_slot])
        marks[spike_offset, :] = spike_pcs[:n_mark_dims, peak_slot]
    return channels, marks


def _subfn_map_channels_to_electrodes(channels: np.ndarray, electrode_mode: str, channel_map: Optional[np.ndarray], channel_shanks: Optional[np.ndarray]) -> np.ndarray:
    channels = np.asarray(channels, dtype=int)
    if electrode_mode == "shank":
        if channel_shanks is None:
            raise ValueError("channel_shanks is required for electrode_mode='shank'.")
        inverse_map = _subfn_build_channel_inverse_map(channel_map) if channel_map is not None else None
        electrode_indices = np.empty(len(channels), dtype=int)
        for spike_idx, probe_channel in enumerate(channels):
            recording_idx = int(inverse_map[probe_channel]) if inverse_map is not None and probe_channel < len(inverse_map) and inverse_map[probe_channel] >= 0 else int(probe_channel)
            electrode_indices[spike_idx] = int(channel_shanks[recording_idx])
        return electrode_indices
    if electrode_mode != "channel":
        raise ValueError(f"electrode_mode must be 'shank' or 'channel'; got {electrode_mode!r}")
    if channel_map is not None:
        inverse_map = _subfn_build_channel_inverse_map(channel_map)
        return np.array([int(inverse_map[probe_channel]) if probe_channel < len(inverse_map) and inverse_map[probe_channel] >= 0 else int(probe_channel) for probe_channel in channels], dtype=int)
    return channels.astype(int, copy=False)


def _subfn_bin_spikes_to_multiunits(multiunits: np.ndarray, spike_times_sec: np.ndarray, marks: np.ndarray, electrode_indices: np.ndarray, rtc_time: np.ndarray) -> None:
    time_bin_indices = np.clip(np.searchsorted(rtc_time, spike_times_sec), 0, len(rtc_time) - 1)
    _assign_spike_marks_to_multiunits(multiunits, time_bin_indices, electrode_indices, marks)


def build_multiunits_from_phy_folder(phy_path: Union[str, Path], t_start: float, t_end: float, sampling_frequency_hz: float, electrode_mode: str = "shank", n_mark_dims: int = 4, chunk_size: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """Build RTC clusterless multiunits from a Phy/Kilosort export folder.

    Reads per-spike PC marks from pc_features.npy (all detected spikes; ignores spike_clusters).
    Epoch filtering is applied before binning to limit memory use. Full-session clusterless
    decoding at 1 kHz can still require large RAM for long epochs.

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import build_multiunits_from_phy_folder

        multiunits, rtc_time = build_multiunits_from_phy_folder(
            phy_path, t_start=11510.0, t_end=14693.0, sampling_frequency_hz=1000.0, electrode_mode="channel"
        )
        # Pass to pipeline: perform_specific_computation(..., multiunits=multiunits, rtc_time=rtc_time)

    """
    phy_path = Path(phy_path)
    missing_files = [a_file for a_file in _PHY_CLUSTERLESS_REQUIRED_FILES if not (phy_path / a_file).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Phy folder {phy_path} is missing required files: {missing_files}")
    sample_rate_hz = _subfn_read_phy_params(phy_path)
    spike_times = np.asarray(np.load(phy_path / "spike_times.npy", mmap_mode="r")).reshape(-1)
    spike_templates = np.asarray(np.load(phy_path / "spike_templates.npy", mmap_mode="r")).reshape(-1)
    pc_features = np.load(phy_path / "pc_features.npy", mmap_mode="r")
    pc_feature_ind = np.load(phy_path / "pc_feature_ind.npy")
    channel_map = np.load(phy_path / "channel_map.npy") if (phy_path / "channel_map.npy").is_file() else None
    channel_shanks = _subfn_resolve_channel_shanks(phy_path)
    epoch_slice = _subfn_get_epoch_spike_slice(spike_times, sample_rate_hz, t_start, t_end)
    if epoch_slice.start >= epoch_slice.stop:
        raise ValueError(f"No spikes found in Phy folder for epoch t=[{t_start}, {t_end}] seconds.")
    n_time = max(1, int(np.round((t_end - t_start) * sampling_frequency_hz)))
    rtc_time = t_start + (np.arange(n_time, dtype=float) + 0.5) / sampling_frequency_hz
    rtc_time = rtc_time[rtc_time <= t_end]
    effective_electrode_mode = electrode_mode
    if electrode_mode == "shank" and (channel_shanks is None or len(np.unique(channel_shanks)) <= 1):
        warnings.warn(f"channel_shanks missing or degenerate in {phy_path}; falling back to electrode_mode='channel'.", stacklevel=2)
        effective_electrode_mode = "channel"
    if effective_electrode_mode == "shank":
        assert channel_shanks is not None
        n_electrodes = int(np.max(channel_shanks)) + 1
    elif channel_map is not None:
        n_electrodes = len(channel_map)
    else:
        n_electrodes = int(pc_feature_ind[pc_feature_ind >= 0].max()) + 1
    multiunits = np.full((len(rtc_time), n_mark_dims, n_electrodes), np.nan, dtype=float)
    for chunk_start in range(epoch_slice.start, epoch_slice.stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, epoch_slice.stop)
        spike_indices = np.arange(chunk_start, chunk_stop, dtype=int)
        channels, marks = _subfn_extract_peak_channel_marks(pc_features, pc_feature_ind, spike_templates, spike_indices, n_mark_dims=n_mark_dims)
        electrode_indices = _subfn_map_channels_to_electrodes(channels, effective_electrode_mode, channel_map, channel_shanks)
        spike_times_sec = np.asarray(spike_times[spike_indices], dtype=float) / sample_rate_hz
        _subfn_bin_spikes_to_multiunits(multiunits, spike_times_sec, marks, electrode_indices, rtc_time)
    return _drop_empty_multiunit_electrodes(multiunits), rtc_time


def rtc_posterior_to_p_x_given_n(rtc_results: xr.Dataset, pf: PfND, state_index: Optional[int] = None, should_match_pf_grid: bool = False) -> np.ndarray:
    posterior = rtc_results.acausal_posterior
    if state_index is None:
        posterior_time_bins = posterior.sum(dim="state")
    else:
        posterior_time_bins = posterior.isel(state=state_index)
    spatial_dims = [a_dim for a_dim in posterior_time_bins.dims if a_dim != "time"]
    posterior_time_bins = posterior_time_bins.transpose(*spatial_dims, "time")
    posterior_values = posterior_time_bins.values
    p_x_given_n_rtc = posterior_values.reshape((int(np.prod(posterior_values.shape[:-1])), posterior_values.shape[-1]), order="F")
    if not should_match_pf_grid:
        return p_x_given_n_rtc
    n_pf_bins = int(np.prod(np.shape(pf.occupancy)))
    n_rtc_bins = p_x_given_n_rtc.shape[0]
    if n_rtc_bins == n_pf_bins:
        return p_x_given_n_rtc
    n_time = p_x_given_n_rtc.shape[1]
    if n_rtc_bins > n_pf_bins:
        return p_x_given_n_rtc[:n_pf_bins, :]
    padded = np.zeros((n_pf_bins, n_time), dtype=float)
    padded[:n_rtc_bins, :] = p_x_given_n_rtc
    return padded


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
    source_pos_df = pf.filtered_pos_df
    source_times = source_pos_df['t'].to_numpy(dtype=float) if 't' in source_pos_df.columns else source_pos_df['t_seconds'].to_numpy(dtype=float)
    t_start = float(rtc_time[0] - 0.5 / sampling_frequency_hz)
    t_end = float(rtc_time[-1] + 0.5 / sampling_frequency_hz)
    position_source = position_array_from_pfnd(pf)
    _, resampled_position = resample_position_to_rtc_clock(position_source, source_times, t_start, t_end, sampling_frequency_hz)
    if len(resampled_position) != len(rtc_time):
        min_len = min(len(resampled_position), len(rtc_time))
        resampled_position = resampled_position[:min_len]
        multiunits = multiunits[:min_len]
        rtc_time = rtc_time[:min_len]
    source_speed = pf.speed
    is_training = build_is_training_mask_from_pfnd(pf, rtc_time, source_times, source_speed)
    return resampled_position, multiunits, is_training
