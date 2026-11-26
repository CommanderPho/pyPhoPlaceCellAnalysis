"""Adapters between NeuroPy PfND / clusterless spike events and Spyglass v1 non_local_detector decoders."""
from __future__ import annotations

from copy import deepcopy
import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import jax.numpy as jnp
import jax.ops
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from neuropy.analyses.placefields import PfND, PlacefieldComputationParameters
from neuropy.core.clusterless_spike_events import ClusterlessSpikeEvents
from non_local_detector.environment import Environment, get_centers, get_track_boundary, get_track_interior, make_nD_track_graph_from_environment
from non_local_detector.likelihoods.clusterless_kde import block_estimate_log_joint_mark_intensity, kde_distance
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood
from non_local_detector.models import ClusterlessDecoder, ContFragClusterlessClassifier
from non_local_detector.models.base import ClusterlessDetector
from scipy.ndimage import label as scipy_label


@dataclass
class PfNDSyncedNLDEnvironment(Environment):
    """non_local_detector Environment aligned directly to PfND xbin/ybin edges (1D or 2D, C-order)."""

    pf: Any = None


    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.environment_name == other
        if isinstance(other, Environment):
            return self.environment_name == other.environment_name
        return NotImplemented


    def fit_place_grid(self, position: Optional[np.ndarray] = None, infer_track_interior: bool = True) -> "PfNDSyncedNLDEnvironment":
        pf_xbin = getattr(self.pf, 'xbin', None) if self.pf is not None else None
        if pf_xbin is not None and len(np.asarray(pf_xbin)) >= 2:
            if position is None:
                raise ValueError("Must provide position when fitting PfNDSyncedNLDEnvironment.")
            position = np.asarray(position, dtype=float)
            if position.ndim == 1:
                position = position[:, np.newaxis]
            is_nan = np.any(np.isnan(position), axis=1)
            position = position[~is_nan]
            x_edges = np.asarray(pf_xbin, dtype=float)
            if self.pf.ndim == 1:
                self.edges_ = (x_edges,)
                x_centers = get_centers(x_edges)
                self.centers_shape_ = (len(x_centers),)
                self.place_bin_centers_ = x_centers[:, np.newaxis]
                self.place_bin_edges_ = x_edges[:, np.newaxis]
            else:
                y_edges = np.asarray(self.pf.ybin, dtype=float)
                self.edges_ = (x_edges, y_edges)
                mesh_edges = np.meshgrid(x_edges, y_edges, indexing="ij")
                self.place_bin_edges_ = np.stack([edge.ravel() for edge in mesh_edges], axis=1)
                mesh_centers = np.meshgrid(get_centers(x_edges), get_centers(y_edges), indexing="ij")
                self.place_bin_centers_ = np.stack([center.ravel() for center in mesh_centers], axis=1)
                self.centers_shape_ = (len(x_edges) - 1, len(y_edges) - 1)

            self.infer_track_interior = infer_track_interior
            if self.is_track_interior is not None:
                self.is_track_interior_ = np.asarray(self.is_track_interior, dtype=bool)
            elif self.infer_track_interior:
                self.is_track_interior_ = get_track_interior(position, self.edges_, self.fill_holes, self.dilate, self.bin_count_threshold)
                if not np.any(self.is_track_interior_):
                    self.is_track_interior_ = np.ones(self.centers_shape_, dtype=bool)
            else:
                self.is_track_interior_ = np.ones(self.centers_shape_, dtype=bool)

            if len(self.edges_) > 1:
                self.is_track_boundary_ = get_track_boundary(self.is_track_interior_, n_position_dims=len(self.edges_), connectivity=1)
            else:
                self.is_track_boundary_ = None

            self.track_graphDD = make_nD_track_graph_from_environment(self)
            node_positions = nx.get_node_attributes(self.track_graphDD, "pos")
            node_positions = np.asarray(list(node_positions.values()))
            distance = np.full((len(node_positions), len(node_positions)), np.inf)
            for to_node_id, from_node_id in nx.shortest_path_length(self.track_graphDD, weight="distance"):
                distance[to_node_id, list(from_node_id.keys())] = list(from_node_id.values())
            ## END for to_node_id, from_node_id in nx.shortest_path_length(self.track_graphDD, weight="distance")...

            self.distance_between_nodes_ = distance
            return self
        return super().fit_place_grid(position, infer_track_interior=infer_track_interior)


def _pfnd_nld_place_bin_size(pf: PfND) -> Union[float, Tuple[float, ...]]:
    pos_bin_size = pf.pos_bin_size
    if isinstance(pos_bin_size, (tuple, list, np.ndarray)):
        vals = tuple(float(v) for v in pos_bin_size)
        return vals[0] if pf.ndim == 1 else vals[:pf.ndim]
    return float(pos_bin_size)


def build_nld_environment_from_pfnd(pf: PfND, environment_name: str = "", infer_track_interior: bool = True) -> PfNDSyncedNLDEnvironment:
    """Construct a PfNDSyncedNLDEnvironment aligned to `pf.xbin` (and `pf.ybin` in 2D)."""
    return PfNDSyncedNLDEnvironment(environment_name=environment_name, place_bin_size=_pfnd_nld_place_bin_size(pf), infer_track_interior=infer_track_interior, pf=pf)


def build_unitfree_pfnd_from_session(sess, ndim: int = 2, computation_config: Optional[PlacefieldComputationParameters] = None, included_epochs=None) -> PfND:
    """Build a valid 0-neuron PfND (spatial grid, speed-filtered occupancy, filtered_pos_df) without sorted units."""
    if computation_config is None:
        computation_config = PlacefieldComputationParameters()
    empty_spikes_df = pd.DataFrame({'aclu': pd.Series(dtype=np.int64), 't_seconds': pd.Series(dtype=np.float64), 't': pd.Series(dtype=np.float64), 't_rel_seconds': pd.Series(dtype=np.float64), 'neuron_type': pd.Series(dtype=object)})
    active_pos = deepcopy(sess.position)
    if ndim == 1:
        if not active_pos.has_linear_pos:
            linearization_method = getattr(computation_config, 'linearization_method', 'isomap') or 'isomap'
            try:
                active_pos.compute_linearized_position(method=linearization_method)
            except Exception as lin_err:
                if linearization_method != 'isomap':
                    warnings.warn(f"compute_linearized_position(method={linearization_method!r}) failed ({lin_err}); falling back to 'isomap'.", stacklevel=2)
                    active_pos.compute_linearized_position(method='isomap')
                else:
                    raise
        target_pos = deepcopy(active_pos.linear_pos_obj)
    else:
        active_pos.drop_dimensions_above(desired_ndim=2)
        target_pos = active_pos
    if included_epochs is None:
        included_epochs = getattr(sess, 'epochs', None)
        if included_epochs is None:
            included_epochs = getattr(sess, 'paradigm', None)
        if included_epochs is None:
            from neuropy.core.epoch import Epoch
            pos_t = np.asarray(target_pos.time, dtype=float)
            pos_t = pos_t[np.isfinite(pos_t)]
            if len(pos_t) >= 2:
                included_epochs = Epoch.from_starts_stops_arrays(starts=np.array([float(pos_t[0])]), stops=np.array([float(pos_t[-1])]), labels=np.array(['session']))
    grid_bin_bounds = computation_config.grid_bin_bounds
    if grid_bin_bounds is None:
        pos_df = target_pos.to_dataframe()
        x_vals = np.asarray(pos_df['x'], dtype=float)
        x_vals = x_vals[np.isfinite(x_vals)]
        x_bounds = (float(np.min(x_vals)), float(np.max(x_vals))) if len(x_vals) > 0 else (0.0, 1.0)
        if ndim == 1:
            grid_bin_bounds = (x_bounds, x_bounds)
        else:
            y_vals = np.asarray(pos_df['y'], dtype=float)
            y_vals = y_vals[np.isfinite(y_vals)]
            y_bounds = (float(np.min(y_vals)), float(np.max(y_vals))) if len(y_vals) > 0 else (0.0, 1.0)
            grid_bin_bounds = (x_bounds, y_bounds)
    return PfND.from_config_values(empty_spikes_df, position=target_pos, epochs=included_epochs, speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh, grid_bin=computation_config.grid_bin, grid_bin_bounds=grid_bin_bounds, smooth=computation_config.smooth)


@dataclass
class SpyglassClusterlessDecodingParameters:
    position_upsample_hz: float = 500.0
    position_variable_names: Optional[List[str]] = None
    decoding_params: Optional[dict] = None
    estimate_decoding_params: bool = False
    should_match_pf_grid: bool = True
    max_posterior_memory_gib: Optional[float] = 36.0
    model_type: str = "cont_frag"  # "cont_frag" (ContFragClusterlessClassifier) or "continuous" (ClusterlessDecoder)
    waveform_std: float = 24.0
    position_std: float = 6.0
    block_size: int = 1000
    use_pf_speed_thresh_for_training: bool = True
    sync_environment_to_pf: bool = True
    state_index_for_posterior: Optional[int] = None


    def resolved_decoding_params(self, pf: Optional[PfND] = None) -> dict:
        if self.decoding_params is not None:
            params = dict(self.decoding_params)
        else:
            model_cls = ClusterlessDecoder if self.model_type == "continuous" else ContFragClusterlessClassifier
            algo_params = {"waveform_std": float(self.waveform_std), "position_std": float(self.position_std), "block_size": int(self.block_size)}
            params = dict(vars(model_cls(clusterless_algorithm_params=algo_params, sampling_frequency=float(self.position_upsample_hz))))
        if self.sync_environment_to_pf and (pf is not None):
            params["environments"] = (build_nld_environment_from_pfnd(pf, infer_track_interior=bool(params.get("infer_track_interior", True))),)
        return params


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


def filter_active_training_electrodes(spike_times: List[np.ndarray], spike_waveform_features: List[np.ndarray], training_time_bounds: Tuple[float, float], is_training_intervals: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Filter out electrodes with zero spikes inside the training bounds (avoids 0/0 NaN in NLD KDE GPI)."""
    t_min, t_max = float(training_time_bounds[0]), float(training_time_bounds[1])
    kept_times: List[np.ndarray] = []
    kept_features: List[np.ndarray] = []
    kept_indices: List[int] = []
    for e_idx, (st, wf) in enumerate(zip(spike_times, spike_waveform_features)):
        st_arr = np.asarray(st, dtype=float)
        if len(st_arr) == 0:
            continue
        in_bounds = (st_arr >= t_min) & (st_arr <= t_max)
        if is_training_intervals is not None and len(is_training_intervals) > 0:
            in_train = np.zeros(len(st_arr), dtype=bool)
            for iv_start, iv_stop in is_training_intervals:
                in_train |= (st_arr >= float(iv_start)) & (st_arr <= float(iv_stop))
            ## END for iv_start, iv_stop in is_training_intervals...

            in_bounds = in_bounds & in_train
        if np.any(in_bounds):
            kept_times.append(st_arr)
            kept_features.append(np.asarray(wf, dtype=float))
            kept_indices.append(int(e_idx))
    ## END for e_idx, (st, wf) in enumerate(zip(spike_times, spike_waveform_features))...

    if not kept_times:
        raise ValueError("Clusterless decoding requires at least one electrode with spikes inside the training interval.")
    return kept_times, kept_features, np.asarray(kept_indices, dtype=int)


def fit_clusterless_detector_in_memory(decoding_params: dict, position_info: pd.DataFrame, position_variable_names: List[str], spike_times: List[np.ndarray], spike_waveform_features: List[np.ndarray], decoding_kwargs: Optional[dict] = None) -> Tuple[ClusterlessDetector, np.ndarray]:
    """Fit a non_local_detector ClusterlessDetector once in memory without running predict()."""
    classifier = ClusterlessDetector(**decoding_params)
    decoding_kwargs = dict(decoding_kwargs) if decoding_kwargs is not None else {}
    fit_kwargs, _predict_kwargs = _get_valid_fit_predict_kwargs(classifier, decoding_kwargs)
    pos_time = position_info.index.to_numpy(dtype=float)
    pos_vals = position_info[position_variable_names].to_numpy(dtype=float)
    is_training = fit_kwargs.get("is_training", None)
    if is_training is not None:
        train_mask = np.asarray(is_training, dtype=bool)
        if not np.any(train_mask):
            raise ValueError("is_training mask has zero True samples; cannot fit clusterless detector.")
        valid_pos_times = pos_time[train_mask]
        train_bounds = (float(valid_pos_times[0]), float(valid_pos_times[-1]))
    else:
        train_bounds = (float(pos_time[0]), float(pos_time[-1]))
    filtered_times, filtered_features, kept_indices = filter_active_training_electrodes(spike_times, spike_waveform_features, training_time_bounds=train_bounds)
    classifier.fit(position_time=pos_time, position=pos_vals, spike_times=filtered_times, spike_waveform_features=filtered_features, **fit_kwargs)
    return classifier, kept_indices


def _pool_fine_posterior_to_windows(fine_flat_posterior: np.ndarray, fine_edges: np.ndarray, window_starts: np.ndarray, window_stops: np.ndarray) -> np.ndarray:
    """Pool fine-step posterior (n_flat_bins, n_fine) onto target [window_starts, window_stops] by overlap duration."""
    n_flat, n_fine = fine_flat_posterior.shape
    n_windows = len(window_starts)
    if n_windows == 0:
        return np.empty((n_flat, 0), dtype=float)
    if n_fine == 0:
        return np.full((n_flat, n_windows), np.nan, dtype=float)
    fine_starts = fine_edges[:-1]
    fine_stops = fine_edges[1:]
    pooled = np.empty((n_flat, n_windows), dtype=float)
    for w_idx, (w_start, w_stop) in enumerate(zip(window_starts, window_stops)):
        overlap = np.maximum(0.0, np.minimum(fine_stops, float(w_stop)) - np.maximum(fine_starts, float(w_start)))
        total_overlap = float(np.sum(overlap))
        if total_overlap <= 0.0:
            mid = 0.5 * (float(w_start) + float(w_stop))
            nearest = int(np.clip(np.searchsorted(fine_edges, mid, side="right") - 1, 0, n_fine - 1))
            col = fine_flat_posterior[:, nearest].copy()
        else:
            weights = overlap / total_overlap
            col = np.nansum(fine_flat_posterior * weights[np.newaxis, :], axis=1)
        col_sum = float(np.nansum(col))
        if np.isfinite(col_sum) and col_sum > 0.0:
            col = col / col_sum
        else:
            col = np.full(n_flat, np.nan, dtype=float)
        pooled[:, w_idx] = col
    ## END for w_idx, (w_start, w_stop) in enumerate(zip(window_starts, window_stops))...

    return pooled


def _unstack_nld_interior_posterior_to_flat_c_order(acausal_posterior: np.ndarray, classifier: ClusterlessDetector, state_index: Optional[int] = None) -> np.ndarray:
    """Map NLD interior-only acausal_posterior (n_time, n_interior_state_bins) -> (n_flat_spatial_bins, n_time) C-order."""
    acausal_posterior = np.asarray(acausal_posterior, dtype=float)
    if acausal_posterior.ndim == 1:
        acausal_posterior = acausal_posterior[np.newaxis, :]
    n_time = acausal_posterior.shape[0]
    environment = classifier.environments[0]
    is_track_interior = np.asarray(environment.is_track_interior_, dtype=bool).ravel(order="C")
    n_flat_spatial = len(is_track_interior)
    interior_state_ind = np.asarray(classifier.state_ind_[classifier.is_track_interior_state_bins_], dtype=int)
    flat_spatial = np.zeros((n_flat_spatial, n_time), dtype=float)
    has_any_spatial_state = False
    for state_id, obs in enumerate(classifier.observation_models):
        if obs.is_local or obs.is_no_spike:
            continue
        if (state_index is not None) and (state_id != int(state_index)):
            continue
        state_cols = (interior_state_ind == state_id)
        if np.sum(state_cols) == int(np.sum(is_track_interior)):
            flat_spatial[is_track_interior, :] += acausal_posterior[:, state_cols].T
            has_any_spatial_state = True
    ## END for state_id, obs in enumerate(classifier.observation_models)...

    if not has_any_spatial_state:
        return np.full((n_flat_spatial, n_time), np.nan, dtype=float)
    col_sums = np.nansum(flat_spatial, axis=0, keepdims=True)
    valid_cols = np.isfinite(col_sums) & (col_sums > 0.0)
    flat_spatial[:, valid_cols[0]] = flat_spatial[:, valid_cols[0]] / col_sums[:, valid_cols[0]]
    flat_spatial[:, ~valid_cols[0]] = np.nan
    return flat_spatial


def _map_nld_env_flat_to_pf_flat(flat_env_posterior: np.ndarray, environment: Environment, pf: PfND, should_match_pf_grid: bool = True) -> np.ndarray:
    """Align C-order flat NLD posterior (n_env_flat, n_time) to PfND occupancy grid (n_pf_flat, n_time) in C-order."""
    position_shape = tuple(np.shape(pf.occupancy))
    env_shape = tuple(environment.centers_shape_)
    if env_shape == position_shape:
        return flat_env_posterior
    if not should_match_pf_grid:
        warnings.warn(f"NLD posterior spatial shape {env_shape} != PfND occupancy shape {position_shape}; returning NLD spatial shape.", stacklevel=2)
        return flat_env_posterior
    n_time = flat_env_posterior.shape[1]
    pf_flat_size = int(np.prod(position_shape))
    out = np.zeros((pf_flat_size, n_time), dtype=float)
    env_edges = environment.edges_
    if pf.ndim == 1:
        x_centers_nld = get_centers(np.asarray(env_edges[0], dtype=float))
        x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
        for nld_x_idx, pf_x_idx in enumerate(x_idx):
            out[int(pf_x_idx), :] += np.nan_to_num(flat_env_posterior[nld_x_idx, :], nan=0.0)
        ## END for nld_x_idx, pf_x_idx in enumerate(x_idx)...

    else:
        x_centers_nld = get_centers(np.asarray(env_edges[0], dtype=float))
        y_centers_nld = get_centers(np.asarray(env_edges[1], dtype=float))
        x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
        y_idx = _nearest_pf_bin_indices(y_centers_nld, pf.ybin_centers)
        env_3d = flat_env_posterior.reshape((*env_shape, n_time), order="C")
        n_y_pf = len(pf.ybin_centers)
        for nld_x_idx in range(len(x_centers_nld)):
            for nld_y_idx in range(len(y_centers_nld)):
                pf_flat_idx = int(x_idx[nld_x_idx]) * n_y_pf + int(y_idx[nld_y_idx])
                out[pf_flat_idx, :] += np.nan_to_num(env_3d[nld_x_idx, nld_y_idx, :], nan=0.0)
            ## END for nld_y_idx in range(len(y_centers_nld))...

        ## END for nld_x_idx in range(len(x_centers_nld))...

    col_sums = out.sum(axis=0, keepdims=True)
    valid = col_sums > 0.0
    out[:, valid[0]] = out[:, valid[0]] / col_sums[:, valid[0]]
    out[:, ~valid[0]] = np.nan
    return out


def predict_clusterless_epochs_in_memory(classifier: ClusterlessDetector, spike_times: List[np.ndarray], spike_waveform_features: List[np.ndarray], pf: PfND, epoch_intervals: np.ndarray, target_windows_per_epoch: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None, state_index: Optional[int] = None, should_match_pf_grid: bool = True, kept_electrode_indices: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Predict clusterless posteriors for one or more intervals using an already-fitted ClusterlessDetector.

    Computes per-electrode spatial KDE distances once across all intervals, bins spikes onto exact
    fine edges [t0, t1] (fixing NLD's last-bin drop), runs per-epoch HMM smoothing, and pools fine
    posteriors onto optional target (window_starts, window_stops) per epoch.

    Returns (flat_posteriors_list, fine_centers_list) where each flat posterior has C-order shape
    (n_flat_position_bins, n_windows_in_epoch).
    """
    epoch_intervals = np.atleast_2d(np.asarray(epoch_intervals, dtype=float))
    n_epochs = len(epoch_intervals)
    if kept_electrode_indices is not None:
        active_spike_times = [np.asarray(spike_times[int(i)], dtype=float) for i in kept_electrode_indices]
        active_spike_features = [np.asarray(spike_waveform_features[int(i)], dtype=float) for i in kept_electrode_indices]
    else:
        active_spike_times = [np.asarray(st, dtype=float) for st in spike_times]
        active_spike_features = [np.asarray(wf, dtype=float) for wf in spike_waveform_features]

    dt_fine = 1.0 / float(classifier.sampling_frequency)
    fine_edges_list: List[np.ndarray] = []
    fine_centers_list: List[np.ndarray] = []
    gpi_scales: List[float] = []
    for epoch_start, epoch_stop in epoch_intervals:
        dur = float(epoch_stop) - float(epoch_start)
        if dur <= 0.0:
            fine_edges = np.array([float(epoch_start), float(epoch_start) + dt_fine], dtype=float)
        else:
            n_fine = max(1, int(np.ceil(dur / dt_fine)))
            fine_edges = np.linspace(float(epoch_start), float(epoch_stop), n_fine + 1, dtype=float)
        fine_edges_list.append(fine_edges)
        fine_centers_list.append(0.5 * (fine_edges[:-1] + fine_edges[1:]))
        gpi_scales.append(float(fine_edges[1] - fine_edges[0]) * float(classifier.sampling_frequency))
    ## END for epoch_start, epoch_stop in epoch_intervals...

    # Precompute spatial log-likelihoods for each unique (environment_name, encoding_group) across all epochs in one electrode pass
    spatial_ll_by_key: dict = {}
    unique_keys = []
    for obs in classifier.observation_models:
        if obs.is_local or obs.is_no_spike:
            continue
        lk = (obs.environment_name, obs.encoding_group)
        if lk not in unique_keys:
            unique_keys.append(lk)
    ## END for obs in classifier.observation_models...

    for lk in unique_keys:
        enc = classifier.encoding_model_[lk]
        env = enc["environment"]
        is_track_interior = np.asarray(env.is_track_interior_, dtype=bool).ravel(order="C")
        interior_centers = jnp.asarray(env.place_bin_centers_[is_track_interior])
        occupancy = enc["occupancy"]
        summed_gpi = enc["summed_ground_process_intensity"]
        position_std = enc["position_std"]
        waveform_std = enc["waveform_std"]
        block_size = int(enc["block_size"])
        enc_wf_list = enc["encoding_spike_waveform_features"]
        enc_pos_list = enc["encoding_positions"]
        mean_rates = enc["mean_rates"]

        epoch_lls = [(-1.0 * float(gpi_scales[k]) * summed_gpi) * jnp.ones((len(fine_centers_list[k]), 1), dtype=jnp.float32) for k in range(n_epochs)]

        for e_enc_wf, e_enc_pos, e_mean_rate, e_dec_wf, e_dec_st in zip(enc_wf_list, enc_pos_list, mean_rates, active_spike_features, active_spike_times):
            # Collect spikes across all epochs so kde_distance(interior_centers, e_enc_pos) runs only once per electrode
            epoch_spike_slices: List[Tuple[int, np.ndarray, np.ndarray]] = []
            for k in range(n_epochs):
                t0, t1 = float(fine_edges_list[k][0]), float(fine_edges_list[k][-1])
                i0 = int(np.searchsorted(e_dec_st, t0, side="left"))
                i1 = int(np.searchsorted(e_dec_st, t1, side="right"))
                if i1 > i0:
                    st_k = e_dec_st[i0:i1]
                    wf_k = e_dec_wf[i0:i1]
                    bin_ind_k = np.clip(np.searchsorted(fine_edges_list[k], st_k, side="right") - 1, 0, len(fine_centers_list[k]) - 1)
                    epoch_spike_slices.append((k, wf_k, bin_ind_k))
            ## END for k in range(n_epochs)...

            if not epoch_spike_slices:
                continue
            position_distance = kde_distance(interior_centers, e_enc_pos, std=position_std)
            for k, wf_k, bin_ind_k in epoch_spike_slices:
                spike_log_intensity = block_estimate_log_joint_mark_intensity(jnp.asarray(wf_k), e_enc_wf, waveform_std, occupancy, e_mean_rate, position_distance, block_size)
                epoch_lls[k] = epoch_lls[k] + jax.ops.segment_sum(spike_log_intensity, jnp.asarray(bin_ind_k, dtype=jnp.int32), indices_are_sorted=True, num_segments=len(fine_centers_list[k]))
            ## END for k, wf_k, bin_ind_k in epoch_spike_slices...

        ## END for e_enc_wf, e_enc_pos, e_mean_rate, e_dec_wf, e_dec_st in zip(enc_wf_list, enc_pos_list, mean_rates, active_spike_features, active_spike_times)...

        spatial_ll_by_key[lk] = epoch_lls
    ## END for lk in unique_keys...

    interior_state_ind = np.asarray(classifier.state_ind_[classifier.is_track_interior_state_bins_], dtype=int)
    n_total_interior_state_bins = int(np.sum(classifier.is_track_interior_state_bins_))
    flat_posteriors_list: List[np.ndarray] = []

    for k in range(n_epochs):
        fine_centers_k = fine_centers_list[k]
        fine_edges_k = fine_edges_list[k]
        n_fine_k = len(fine_centers_k)
        full_ll_k = jnp.zeros((n_fine_k, n_total_interior_state_bins), dtype=jnp.float32)
        for state_id, obs in enumerate(classifier.observation_models):
            is_state_bin = (interior_state_ind == state_id)
            if obs.is_no_spike:
                full_ll_k = full_ll_k.at[:, is_state_bin].set(predict_no_spike_log_likelihood(fine_edges_k[:-1], active_spike_times, classifier.no_spike_rate))
            elif obs.is_local:
                raise NotImplementedError("Local observation models require position at predict time; use run_clusterless_decoder_in_memory.")
            else:
                lk = (obs.environment_name, obs.encoding_group)
                full_ll_k = full_ll_k.at[:, is_state_bin].set(spatial_ll_by_key[lk][k])
        ## END for state_id, obs in enumerate(classifier.observation_models)...

        acausal_posterior, _acausal_state_probs, _mll, _causal, _pred, _ll = classifier._predict(time=fine_centers_k, log_likelihoods=full_ll_k, cache_likelihood=False, n_chunks=1)
        flat_env = _unstack_nld_interior_posterior_to_flat_c_order(np.asarray(acausal_posterior, dtype=float), classifier, state_index=state_index)
        flat_pf = _map_nld_env_flat_to_pf_flat(flat_env, classifier.environments[0], pf, should_match_pf_grid=should_match_pf_grid)
        if target_windows_per_epoch is not None:
            w_starts, w_stops = target_windows_per_epoch[k]
            flat_pf = _pool_fine_posterior_to_windows(flat_pf, fine_edges_k, w_starts, w_stops)
        flat_posteriors_list.append(flat_pf)
    ## END for k in range(n_epochs)...

    return flat_posteriors_list, fine_centers_list


def run_clusterless_decoder_in_memory(key: dict, decoding_params: dict, decoding_kwargs: dict, position_info: pd.DataFrame, position_variable_names: List[str], spike_times: List[np.ndarray], spike_waveform_features: List[np.ndarray], decoding_interval: np.ndarray) -> Tuple[ClusterlessDetector, xr.Dataset]:
    """Mirror spyglass.decoding.v1.clusterless.ClusterlessDecodingV1._run_decoder (no DB / no file I/O)."""
    classifier = ClusterlessDetector(**decoding_params)
    if key.get("estimate_decoding_params", False):
        is_missing = np.ones(len(position_info), dtype=bool)
        for interval_start, interval_end in decoding_interval:
            is_missing[(position_info.index >= interval_start) & (position_info.index <= interval_end)] = False
        ## END for interval_start, interval_end in decoding_interval...

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
        ## END for interval_start, interval_end in decoding_interval...

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
    ## END for electrode_idx in np.unique(events.electrode_indices)...

    return spike_times, spike_waveform_features


def pfnd_to_spyglass_position_info(pf: PfND, upsample_hz: float = 500.0, position_variable_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Build upsampled Spyglass position_info DataFrame from PfND filtered position (1D or 2D)."""
    pos_df = pf.filtered_pos_df
    time_col = 't' if 't' in pos_df.columns else 't_seconds'
    if position_variable_names is None:
        position_variable_names = ['position_x', 'position_y'] if (pf.ndim >= 2 and 'y' in pos_df.columns) else ['position_x']
    position_columns = {}
    if 'position_x' in position_variable_names:
        x_col = 'lin_pos' if (pf.ndim == 1 and 'lin_pos' in pos_df.columns) else 'x'
        position_columns['position_x'] = pos_df[x_col].to_numpy(dtype=float)
    if 'position_y' in position_variable_names:
        if 'y' not in pos_df.columns:
            raise ValueError("pfnd_to_spyglass_position_info() requires 'y' in filtered_pos_df for 2D position_variable_names.")
        position_columns['position_y'] = pos_df['y'].to_numpy(dtype=float)
    position_info = pd.DataFrame(position_columns, index=pos_df[time_col].to_numpy(dtype=float))
    position_info.index.name = 'time'
    return upsample_position_for_decoding(position_info, upsampling_sampling_rate=upsample_hz, position_variable_names=list(position_columns.keys()))


def _pfnd_speed_filtered_training_intervals_nld(pf: PfND) -> np.ndarray:
    """Return [start, stop] intervals matching PfND.setup speed filtering without importing RTC."""
    pos_df = pf.position.to_dataframe()
    if pf.epochs is not None:
        epoch_filtered_pos_df = pos_df.position.time_sliced(pf.epochs.starts, pf.epochs.stops)
    else:
        epoch_filtered_pos_df = pos_df.position.time_sliced(pf.position.t_start, pf.position.t_stop)
    pos_non_na_cols = ['x', 'y'] if pf.ndim > 1 else ['x']
    position_df = epoch_filtered_pos_df.dropna(axis=0, how='any', subset=pos_non_na_cols).copy()
    speed_column_name = 'speed'
    if getattr(pf, 'should_smooth_speed', False) and (pf.config.smooth is not None) and (pf.config.smooth[0] > 0.0) and ('speed' in position_df.columns):
        from scipy.ndimage import gaussian_filter1d
        position_df['speed_smooth'] = gaussian_filter1d(position_df.speed.to_numpy(), sigma=pf.config.smooth[0])
        speed_column_name = 'speed_smooth'
    if 'speed' not in position_df.columns and speed_column_name not in position_df.columns:
        return np.empty((0, 2), dtype=float)
    if pf.epochs is not None:
        epochs_df = deepcopy(pf.epochs.to_dataframe())
    else:
        epochs_df = pd.DataFrame({'start': [float(pf.position.t_start)], 'stop': [float(pf.position.t_stop)], 'label': [0], 'duration': [float(pf.position.t_stop - pf.position.t_start)]})
    speed_filtered_epochs_df = PfND.filtered_by_speed(epochs_df, position_df=position_df, minimum_speed_thresh=pf.config.speed_thresh, speed_column_override_name=speed_column_name, debug_print=False)
    if speed_filtered_epochs_df is None or len(speed_filtered_epochs_df) == 0:
        return np.empty((0, 2), dtype=float)
    return speed_filtered_epochs_df[['start', 'stop']].to_numpy(dtype=float)


def build_is_training_mask(position_info: pd.DataFrame, encoding_interval: np.ndarray, position_variable_names: List[str] = None, pf: Optional[PfND] = None, use_pf_speed_thresh: bool = False) -> np.ndarray:
    """Mirror ClusterlessDecodingV1.make_fetch is_training logic, optionally intersected with PfND speed intervals."""
    if position_variable_names is None:
        position_variable_names = list(position_info.columns)
    is_training = np.zeros(len(position_info), dtype=bool)
    pos_times = position_info.index.to_numpy(dtype=float)
    for interval_start, interval_end in encoding_interval:
        is_training[(pos_times >= interval_start) & (pos_times <= interval_end)] = True
    ## END for interval_start, interval_end in encoding_interval...

    if use_pf_speed_thresh and (pf is not None) and (getattr(pf, 'position', None) is not None):
        from neuropy.utils.efficient_interval_search import determine_event_interval_is_included
        speed_intervals = _pfnd_speed_filtered_training_intervals_nld(pf)
        if speed_intervals.size > 0:
            speed_mask = determine_event_interval_is_included(pos_times, speed_intervals)
            if np.any(is_training & speed_mask):
                is_training = is_training & speed_mask
    is_training[position_info[position_variable_names].isna().to_numpy().max(axis=1)] = False
    return is_training


def epochs_from_pfnd(pf: PfND) -> Tuple[np.ndarray, np.ndarray]:
    """Return encoding and decoding interval arrays (n_intervals, 2) from PfND epochs (or filtered_pos_df fallback)."""
    if getattr(pf, 'epochs', None) is None:
        pos_df = pf.filtered_pos_df
        time_col = 't' if 't' in pos_df.columns else 't_seconds'
        t_vals = pos_df[time_col].to_numpy(dtype=float)
        encoding_interval = np.array([[float(np.min(t_vals)), float(np.max(t_vals))]], dtype=float)
        return encoding_interval, encoding_interval.copy()
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


def nld_spatial_posterior(results: xr.Dataset, state_index: Optional[int] = None) -> xr.DataArray:
    """Unstack state_bins and marginalize (or select) over discrete states -> (time, position) or (time, x_position, y_position)."""
    posterior = results.acausal_posterior.unstack("state_bins")
    if "time" not in posterior.dims:
        posterior = posterior.expand_dims("time")
    state_dim = "states" if "states" in posterior.dims else ("state" if "state" in posterior.dims else None)
    if state_dim is not None:
        if state_index is None:
            posterior = posterior.sum(state_dim)
        else:
            posterior = posterior.isel({state_dim: int(state_index)})
    return posterior


def _nearest_pf_bin_indices(nld_centers: np.ndarray, pf_centers: np.ndarray) -> np.ndarray:
    """Map each NLD bin center to the nearest PfND bin center index."""
    nld_centers = np.asarray(nld_centers, dtype=float)
    pf_centers = np.asarray(pf_centers, dtype=float)
    return np.argmin(np.abs(nld_centers[:, np.newaxis] - pf_centers[np.newaxis, :]), axis=1)


def _scatter_nld_posterior_to_pf_grid(spatial_posterior: xr.DataArray, pf: PfND) -> np.ndarray:
    """Scatter NLD spatial_posterior (time, *spatial) onto PfND occupancy grid (*pf_shape, time) in C-order."""
    spatial_dims = [d for d in spatial_posterior.dims if d != "time"]
    posterior_values = np.asarray(spatial_posterior.transpose("time", *spatial_dims).values, dtype=float)
    n_time = posterior_values.shape[0]
    position_shape = tuple(np.shape(pf.occupancy))
    pf_flat_size = int(np.prod(position_shape))
    if pf.ndim == 1 or len(spatial_dims) == 1:
        x_dim = "position" if "position" in spatial_posterior.coords else "x_position"
        x_centers_nld = np.asarray(spatial_posterior.coords[x_dim].values, dtype=float)
        x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
        flat_p_x_given_n = np.zeros((pf_flat_size, n_time), dtype=float)
        for nld_bin_idx, pf_bin_idx in enumerate(x_idx):
            flat_p_x_given_n[int(pf_bin_idx), :] += np.nan_to_num(posterior_values[:, nld_bin_idx], nan=0.0)
        ## END for nld_bin_idx, pf_bin_idx in enumerate(x_idx)...

    else:
        x_centers_nld = np.asarray(spatial_posterior.coords["x_position"].values, dtype=float)
        y_centers_nld = np.asarray(spatial_posterior.coords["y_position"].values, dtype=float)
        x_idx = _nearest_pf_bin_indices(x_centers_nld, pf.xbin_centers)
        y_idx = _nearest_pf_bin_indices(y_centers_nld, pf.ybin_centers)
        flat_p_x_given_n = np.zeros((pf_flat_size, n_time), dtype=float)
        n_y_pf = len(pf.ybin_centers)
        for nld_x_idx in range(len(x_centers_nld)):
            for nld_y_idx in range(len(y_centers_nld)):
                pf_flat_idx = int(x_idx[nld_x_idx]) * n_y_pf + int(y_idx[nld_y_idx])
                flat_p_x_given_n[pf_flat_idx, :] += np.nan_to_num(posterior_values[:, nld_x_idx, nld_y_idx], nan=0.0)
            ## END for nld_y_idx in range(len(y_centers_nld))...

        ## END for nld_x_idx in range(len(x_centers_nld))...

    row_sums = flat_p_x_given_n.sum(axis=0, keepdims=True)
    valid = row_sums > 0.0
    flat_p_x_given_n[:, valid[0]] = flat_p_x_given_n[:, valid[0]] / row_sums[:, valid[0]]
    flat_p_x_given_n[:, ~valid[0]] = np.nan
    return flat_p_x_given_n.reshape((*position_shape, n_time), order="C")


def nld_posterior_to_p_x_given_n(results: xr.Dataset, pf: PfND, should_match_pf_grid: bool = True, state_index: Optional[int] = None) -> np.ndarray:
    """Convert NLD xr.Dataset posterior to (*pf.occupancy.shape, n_time) with time on the last axis."""
    spatial_posterior = nld_spatial_posterior(results, state_index=state_index)
    spatial_dims = [d for d in spatial_posterior.dims if d != "time"]
    posterior_space_time = np.asarray(spatial_posterior.transpose(*spatial_dims, "time").values, dtype=float)
    position_shape = tuple(np.shape(pf.occupancy))
    spatial_shape = tuple(posterior_space_time.shape[:-1])
    if spatial_shape == position_shape:
        return posterior_space_time
    if not should_match_pf_grid:
        warnings.warn(f"NLD posterior spatial shape {spatial_shape} != PfND occupancy shape {position_shape}; returning NLD spatial shape.", stacklevel=2)
        return posterior_space_time
    return _scatter_nld_posterior_to_pf_grid(spatial_posterior, pf)


def nld_posterior_flat_p_x_given_n(results: xr.Dataset, pf: PfND, should_match_pf_grid: bool = True, state_index: Optional[int] = None) -> np.ndarray:
    """Convert NLD xr.Dataset posterior to C-order flat (n_flat_position_bins, n_time)."""
    p_x_given_n = nld_posterior_to_p_x_given_n(results, pf, should_match_pf_grid=should_match_pf_grid, state_index=state_index)
    return np.reshape(p_x_given_n, (int(np.prod(p_x_given_n.shape[:-1])), p_x_given_n.shape[-1]), order="C")
