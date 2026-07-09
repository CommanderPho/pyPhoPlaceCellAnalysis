"""Adapters between NeuroPy PfND / clusterless spike events and Spyglass v1 non_local_detector decoders."""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from neuropy.analyses.placefields import PfND
from neuropy.core.clusterless_spike_events import ClusterlessSpikeEvents
from non_local_detector.models import ContFragClusterlessClassifier
from non_local_detector.models.base import ClusterlessDetector
from scipy.ndimage import label as scipy_label


@dataclass
class SpyglassClusterlessDecodingParameters:
    position_upsample_hz: float = 500.0
    position_variable_names: Optional[List[str]] = None
    decoding_params: Optional[dict] = None
    estimate_decoding_params: bool = False
    should_match_pf_grid: bool = True
    max_posterior_memory_gib: Optional[float] = 36.0


    def resolved_decoding_params(self) -> dict:
        if self.decoding_params is not None:
            return dict(self.decoding_params)
        return dict(vars(ContFragClusterlessClassifier()))


    def resolved_position_variable_names(self, pf: PfND) -> List[str]:
        if self.position_variable_names is not None:
            return list(self.position_variable_names)
        if pf.ndim >= 2:
            return ['position_x', 'position_y']
        return ['position_x']


def upsample_position_for_decoding(position_df: pd.DataFrame, upsampling_sampling_rate: float, upsampling_interpolation_method: str = "linear", position_variable_names: List[str] = None) -> pd.DataFrame:
    """Mirror spyglass.decoding.v1.core.PositionGroup._upsample (500 Hz default in tutorial 41)."""
    upsampling_start_time = position_df.index[0]
    upsampling_end_time = position_df.index[-1]
    n_samples = int(np.ceil((upsampling_end_time - upsampling_start_time) * upsampling_sampling_rate)) + 1
    new_time = np.linspace(upsampling_start_time, upsampling_end_time, n_samples)
    new_index = pd.Index(np.unique(np.concatenate((position_df.index, new_time))), name="time")
    nan_intervals = {}
    if position_variable_names is None:
        position_variable_names = list(position_df.columns)
    for column in position_variable_names:
        is_nan = position_df[column].isna().to_numpy().astype(int)
        st = np.where(np.diff(is_nan) == 1)[0] + 1
        en = np.where(np.diff(is_nan) == -1)[0]
        if is_nan[0]:
            st = np.insert(st, 0, 0)
        if is_nan[-1]:
            en = np.append(en, len(is_nan) - 1)
        st = position_df.index[st].to_numpy()
        en = position_df.index[en].to_numpy()
        nan_intervals[column] = list(zip(st, en))
    position_df = position_df.reindex(index=new_index).interpolate(method=upsampling_interpolation_method).reindex(index=new_time)
    for column, intervals in nan_intervals.items():
        for st, en in intervals:
            position_df.loc[st:en, column] = np.nan
    return position_df


def _get_valid_fit_predict_kwargs(classifier, decoding_kwargs: dict) -> Tuple[dict, dict]:
    """Mirror spyglass.decoding.v1.utils.get_valid_kwargs."""
    valid_fit_kwargs = set(inspect.signature(classifier.fit).parameters.keys())
    valid_predict_kwargs = set(inspect.signature(classifier.predict).parameters.keys())
    if decoding_kwargs:
        ignored_kwargs = set(decoding_kwargs.keys()) - (valid_fit_kwargs | valid_predict_kwargs)
        if ignored_kwargs:
            warnings.warn(f"Ignoring decoding_kwargs not valid for fit/predict: {sorted(ignored_kwargs)}")
    fit_kwargs = {k: v for k, v in decoding_kwargs.items() if k in valid_fit_kwargs}
    predict_kwargs = {k: v for k, v in decoding_kwargs.items() if k in valid_predict_kwargs}
    return fit_kwargs, predict_kwargs


def _concatenate_interval_results(interval_results: List[xr.Dataset]) -> xr.Dataset:
    """Mirror spyglass.decoding.v1.utils.concatenate_interval_results."""
    if not interval_results:
        raise ValueError("No interval results to concatenate")
    total_length = sum(len(result.time) for result in interval_results)
    interval_labels = np.empty(total_length, dtype=np.intp)
    offset = 0
    for interval_idx, result in enumerate(interval_results):
        n_times = len(result.time)
        interval_labels[offset: offset + n_times] = interval_idx
        offset += n_times
    concatenated = xr.concat(interval_results, dim="time")
    return concatenated.assign_coords(interval_labels=("time", interval_labels))


def run_clusterless_decoder_in_memory(key: dict, decoding_params: dict, decoding_kwargs: dict, position_info: pd.DataFrame, position_variable_names: List[str], spike_times: List[np.ndarray], spike_waveform_features: List[np.ndarray], decoding_interval: np.ndarray) -> Tuple[ClusterlessDetector, xr.Dataset]:
    """Mirror spyglass.decoding.v1.clusterless.ClusterlessDecodingV1._run_decoder (no DB / no file I/O)."""
    classifier = ClusterlessDetector(**decoding_params)
    if key.get("estimate_decoding_params", False):
        is_missing = np.ones(len(position_info), dtype=bool)
        for interval_start, interval_end in decoding_interval:
            is_missing[(position_info.index >= interval_start) & (position_info.index <= interval_end)] = False
        if np.all(is_missing):
            raise ValueError(f"All decoding intervals empty: {decoding_interval.tolist()}")
        if "is_missing" not in decoding_kwargs:
            decoding_kwargs = {**decoding_kwargs, "is_missing": is_missing}
        results = classifier.estimate_parameters(position_time=position_info.index.to_numpy(), position=position_info[position_variable_names].to_numpy(), spike_times=spike_times, spike_waveform_features=spike_waveform_features, time=position_info.index.to_numpy(), **decoding_kwargs)
        raw_labels, _ = scipy_label(~decoding_kwargs["is_missing"])
        results = results.assign_coords(interval_labels=("time", raw_labels - 1))
    else:
        fit_kwargs, predict_kwargs = _get_valid_fit_predict_kwargs(classifier, decoding_kwargs)
        classifier.fit(position_time=position_info.index.to_numpy(), position=position_info[position_variable_names].to_numpy(), spike_times=spike_times, spike_waveform_features=spike_waveform_features, **fit_kwargs)
        interval_results = []
        for interval_start, interval_end in decoding_interval:
            interval_time = position_info.loc[interval_start:interval_end].index.to_numpy()
            if interval_time.size == 0:
                warnings.warn(f"Decoding interval {interval_start}:{interval_end} is empty")
                continue
            interval_results.append(classifier.predict(position_time=interval_time, position=position_info.loc[interval_start:interval_end][position_variable_names].to_numpy(), spike_times=spike_times, spike_waveform_features=spike_waveform_features, time=interval_time, **predict_kwargs))
        if not interval_results:
            raise ValueError(f"All decoding intervals empty: {decoding_interval.tolist()}")
        results = _concatenate_interval_results(interval_results)
    state_names = results.coords["states"].values
    results["initial_conditions"] = xr.DataArray(classifier.initial_conditions_, dims=("state_bins",), coords={"state_bins": results.coords["state_bins"]}, name="initial_conditions")
    results["discrete_state_transitions"] = xr.DataArray(classifier.discrete_state_transitions_, dims=("states_from", "states_to"), coords={"states_from": state_names, "states_to": state_names}, name="discrete_state_transitions")
    if vars(classifier).get("discrete_transition_coefficients_") is not None:
        results["discrete_transition_coefficients"] = classifier.discrete_transition_coefficients_
    return classifier, results


def clusterless_events_to_spyglass_spike_lists(events: ClusterlessSpikeEvents) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert ClusterlessSpikeEvents to per-electrode spike_times + waveform feature lists.

    Only electrodes with at least one spike in ``events`` are included (compact list).
    Do not pad to max electrode index with empty arrays — empty electrodes poison
    non_local_detector's ground-process-intensity fit with NaN.
    """
    if len(events.spike_times_sec) == 0:
        raise ValueError('No clusterless spike events to decode.')
    spike_times: List[np.ndarray] = []
    spike_waveform_features: List[np.ndarray] = []
    for electrode_idx in np.unique(events.electrode_indices):
        electrode_idx = int(electrode_idx)
        mask = events.electrode_indices == electrode_idx
        spike_times.append(events.spike_times_sec[mask].astype(float))
        spike_waveform_features.append(events.marks[mask].astype(float))
    return spike_times, spike_waveform_features


def pfnd_to_spyglass_position_info(pf: PfND, upsample_hz: float = 500.0, position_variable_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Build upsampled Spyglass position_info DataFrame from PfND filtered position."""
    pos_df = pf.filtered_pos_df
    time_col = 't' if 't' in pos_df.columns else 't_seconds'
    if position_variable_names is None:
        position_variable_names = ['position_x', 'position_y'] if (pf.ndim >= 2 and 'y' in pos_df.columns) else ['position_x']
    position_columns = {}
    if 'position_x' in position_variable_names:
        position_columns['position_x'] = pos_df['x'].to_numpy(dtype=float)
    if 'position_y' in position_variable_names:
        if 'y' not in pos_df.columns:
            raise ValueError("pfnd_to_spyglass_position_info() requires 'y' in filtered_pos_df for 2D position_variable_names.")
        position_columns['position_y'] = pos_df['y'].to_numpy(dtype=float)
    position_info = pd.DataFrame(position_columns, index=pos_df[time_col].to_numpy(dtype=float))
    position_info.index.name = 'time'
    return upsample_position_for_decoding(position_info, upsampling_sampling_rate=upsample_hz, position_variable_names=list(position_columns.keys()))


def build_is_training_mask(position_info: pd.DataFrame, encoding_interval: np.ndarray, position_variable_names: List[str] = None) -> np.ndarray:
    """Mirror ClusterlessDecodingV1.make_fetch is_training logic."""
    if position_variable_names is None:
        position_variable_names = list(position_info.columns)
    is_training = np.zeros(len(position_info), dtype=bool)
    for interval_start, interval_end in encoding_interval:
        is_training[(position_info.index >= interval_start) & (position_info.index <= interval_end)] = True
    is_training[position_info[position_variable_names].isna().to_numpy().max(axis=1)] = False
    return is_training


def epochs_from_pfnd(pf: PfND) -> Tuple[np.ndarray, np.ndarray]:
    """Return encoding and decoding interval arrays (n_intervals, 2) from PfND epochs."""
    epochs_df = pf.epochs.to_dataframe() if hasattr(pf.epochs, 'to_dataframe') else pf.epochs
    if not {'start', 'stop'}.issubset(epochs_df.columns):
        raise ValueError("pf.epochs must provide 'start' and 'stop' columns.")
    encoding_interval = np.column_stack([epochs_df['start'].to_numpy(dtype=float), epochs_df['stop'].to_numpy(dtype=float)])
    decoding_interval = encoding_interval.copy()
    return encoding_interval, decoding_interval


def estimate_posterior_memory_bytes(n_time: int, n_position_bins: int, dtype=np.float32) -> int:
    return int(n_time) * int(n_position_bins) * int(np.dtype(dtype).itemsize)


def raise_if_posterior_exceeds_memory_limit(n_time: int, n_position_bins: int, max_memory_gib: Optional[float]) -> int:
    estimated_bytes = estimate_posterior_memory_bytes(n_time=n_time, n_position_bins=n_position_bins, dtype=np.float32)
    if max_memory_gib is not None:
        max_memory_bytes = int(float(max_memory_gib) * (1024 ** 3))
        if estimated_bytes > max_memory_bytes:
            estimated_gib = estimated_bytes / float(1024 ** 3)
            raise MemoryError(f"Spyglass clusterless posterior would allocate {estimated_gib:.2f} GiB for shape ({int(n_time)}, {int(n_position_bins)}) float32, exceeding max_posterior_memory_gib={float(max_memory_gib):.2f}. Reduce position_upsample_hz or raise the limit explicitly.")
    return estimated_bytes


def nld_spatial_posterior(results: xr.Dataset) -> xr.DataArray:
    """Unstack state_bins and marginalize over discrete states -> (time, x_position, [y_position])."""
    posterior = results.acausal_posterior.unstack("state_bins")
    state_dim = "states" if "states" in posterior.dims else "state"
    if state_dim in posterior.dims:
        posterior = posterior.sum(state_dim)
    return posterior


def _nearest_pf_bin_indices(nld_centers: np.ndarray, pf_centers: np.ndarray) -> np.ndarray:
    nld_centers = np.asarray(nld_centers, dtype=float)
    pf_centers = np.asarray(pf_centers, dtype=float)
    return np.clip(np.searchsorted(nld_centers, pf_centers), 0, len(nld_centers) - 1)


def _scatter_nld_posterior_to_pf_grid(spatial_posterior: xr.DataArray, pf: PfND) -> np.ndarray:
    posterior_values = np.asarray(spatial_posterior.values, dtype=float)
    n_time = posterior_values.shape[0]
    position_shape = tuple(np.shape(pf.occupancy))
    pf_flat_size = int(np.prod(position_shape))
    if pf.ndim == 1:
        x_centers_nld = np.asarray(spatial_posterior.x_position.values, dtype=float)
        x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
        flat_p_x_given_n = np.zeros((pf_flat_size, n_time), dtype=float)
        for nld_bin_idx, pf_bin_idx in enumerate(x_idx):
            flat_p_x_given_n[pf_bin_idx, :] += posterior_values[:, nld_bin_idx]
    else:
        x_centers_nld = np.asarray(spatial_posterior.x_position.values, dtype=float)
        y_centers_nld = np.asarray(spatial_posterior.y_position.values, dtype=float)
        if posterior_values.ndim == 2:
            x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
            flat_p_x_given_n = np.zeros((pf_flat_size, n_time), dtype=float)
            for nld_bin_idx, pf_bin_idx in enumerate(x_idx):
                flat_p_x_given_n[pf_bin_idx, :] += posterior_values[:, nld_bin_idx]
        else:
            x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
            y_idx = _nearest_pf_bin_indices(y_centers_nld, pf.ybin_centers)
            flat_p_x_given_n = np.zeros((pf_flat_size, n_time), dtype=float)
            for nld_x_idx in range(len(x_centers_nld)):
                for nld_y_idx in range(len(y_centers_nld)):
                    pf_flat_idx = int(x_idx[nld_x_idx] * len(pf.ybin_centers) + y_idx[nld_y_idx])
                    flat_p_x_given_n[pf_flat_idx, :] += posterior_values[:, nld_x_idx, nld_y_idx]
    row_sums = flat_p_x_given_n.sum(axis=0, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    flat_p_x_given_n = flat_p_x_given_n / row_sums
    return flat_p_x_given_n.reshape((*position_shape, n_time), order='F')


def nld_posterior_to_p_x_given_n(results: xr.Dataset, pf: PfND, should_match_pf_grid: bool = True) -> np.ndarray:
    spatial_posterior = nld_spatial_posterior(results)
    posterior_values = np.asarray(spatial_posterior.values, dtype=float)
    n_time = posterior_values.shape[0]
    position_shape = tuple(np.shape(pf.occupancy))
    pf_flat_size = int(np.prod(position_shape))
    spatial_shape = tuple(posterior_values.shape[1:])
    nld_flat_size = int(np.prod(spatial_shape))
    if nld_flat_size == pf_flat_size and spatial_shape == position_shape:
        return posterior_values.reshape((*position_shape, n_time), order='F')
    if not should_match_pf_grid:
        warnings.warn(f"NLD posterior spatial shape {spatial_shape} != PfND occupancy shape {position_shape}; returning NLD spatial shape.", stacklevel=2)
        return posterior_values
    return _scatter_nld_posterior_to_pf_grid(spatial_posterior, pf)


def nld_posterior_flat_p_x_given_n(results: xr.Dataset, pf: PfND, should_match_pf_grid: bool = True) -> np.ndarray:
    p_x_given_n = nld_posterior_to_p_x_given_n(results, pf, should_match_pf_grid=should_match_pf_grid)
    return np.reshape(p_x_given_n, (int(np.prod(p_x_given_n.shape[:-1])), p_x_given_n.shape[-1]), order='F')
