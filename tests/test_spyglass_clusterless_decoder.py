import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neuropy.core.clusterless_spike_events import ClusterlessSpikeEvents
from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import (
    SpyglassClusterlessDecodingParameters,
    _concatenate_interval_results,
    build_is_training_mask,
    clusterless_events_to_spyglass_spike_lists,
    nld_posterior_flat_p_x_given_n,
    nld_posterior_to_p_x_given_n,
    nld_spatial_posterior,
    upsample_position_for_decoding,
)
from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_decoder import SpyglassClusterlessDecoder


class _MockPfConfig:
    def __init__(self, grid_bin_bounds=None):
        self.grid_bin_bounds = grid_bin_bounds


    @property
    def grid_bin_bounds_1D(self):
        if self.grid_bin_bounds is None:
            return None
        if np.isscalar(self.grid_bin_bounds):
            return self.grid_bin_bounds
        if len(self.grid_bin_bounds) == 2 and not isinstance(self.grid_bin_bounds[0], (tuple, list, np.ndarray)):
            return self.grid_bin_bounds
        return self.grid_bin_bounds[0]


class _MockPfND:
    def __init__(self, n_bins: int, ndim: int = 1, grid_bin_bounds=None, pos_bin_size: float = 2.0):
        self.ndim = ndim
        self.config = _MockPfConfig(grid_bin_bounds=grid_bin_bounds)
        self.pos_bin_size = pos_bin_size if ndim == 1 else (pos_bin_size, pos_bin_size)
        self.occupancy = np.ones(n_bins if ndim == 1 else (n_bins, n_bins))
        if ndim == 1:
            self.xbin = np.linspace(0.0, float(n_bins * pos_bin_size), n_bins + 1, dtype=float)
            self.ybin = None
        else:
            self.xbin = np.linspace(0.0, float(n_bins * pos_bin_size), n_bins + 1, dtype=float)
            self.ybin = np.linspace(0.0, float(n_bins * pos_bin_size), n_bins + 1, dtype=float)
        self.xbin_centers = self.xbin[:-1] + np.diff(self.xbin) / 2.0
        self.ybin_centers = (self.ybin[:-1] + np.diff(self.ybin) / 2.0) if ndim == 2 else None
        self.epochs = None
        self.filtered_pos_df = None


    def replacing_computation_epochs(self, epochs, preserve_source_neuron_IDs: bool = False):
        replaced_pf = _MockPfND(n_bins=len(self.xbin_centers), ndim=self.ndim, grid_bin_bounds=self.config.grid_bin_bounds, pos_bin_size=float(np.mean(np.atleast_1d(self.pos_bin_size))))
        replaced_pf.epochs = epochs
        return replaced_pf


def _make_mock_nld_results(n_time: int, n_x: int, n_y: int) -> xr.Dataset:
    states = np.array(['Continuous', 'Fragmented'], dtype=object)
    x_positions = np.linspace(0.0, float(n_x * 10), n_x)
    y_positions = np.linspace(0.0, float(n_y * 10), n_y)
    state_bins_index = pd.MultiIndex.from_product([states, x_positions, y_positions], names=['state', 'x_position', 'y_position'])
    posterior = np.random.rand(n_time, len(state_bins_index)).astype(np.float32)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return xr.Dataset({
        'acausal_posterior': (('time', 'state_bins'), posterior),
        'acausal_state_probabilities': (('time', 'states'), np.ones((n_time, len(states)), dtype=np.float32) / len(states)),
    }, coords={'time': np.linspace(0.0, 1.0, n_time), 'state_bins': state_bins_index, 'states': states})


def _make_position_info(n_samples: int = 50, t_start: float = 0.0, t_stop: float = 1.0) -> pd.DataFrame:
    times = np.linspace(t_start, t_stop, n_samples)
    return pd.DataFrame({'position_x': np.linspace(0.0, 80.0, n_samples), 'position_y': np.linspace(0.0, 80.0, n_samples)}, index=times)


def _make_clusterless_events() -> ClusterlessSpikeEvents:
    spike_times_sec = np.array([0.1, 0.2, 0.5, 0.8], dtype=float)
    electrode_indices = np.array([0, 0, 1, 1], dtype=int)
    marks = np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [5.0, 6.0, 7.0, 8.0], [5.1, 6.1, 7.1, 8.1]], dtype=float)
    return ClusterlessSpikeEvents(spike_times_sec=spike_times_sec, electrode_indices=electrode_indices, marks=marks, sampling_frequency_hz=1000.0, t_start=0.0, t_stop=1.0)


def test_upsample_position_for_decoding_increases_samples():
    position_info = _make_position_info(n_samples=10, t_start=0.0, t_stop=1.0)
    upsampled = upsample_position_for_decoding(position_info, upsampling_sampling_rate=100.0)
    assert len(upsampled) > len(position_info)
    assert list(upsampled.columns) == ['position_x', 'position_y']


def test_build_is_training_mask_respects_encoding_interval():
    position_info = _make_position_info(n_samples=100, t_start=0.0, t_stop=1.0)
    encoding_interval = np.array([[0.0, 0.5]])
    is_training = build_is_training_mask(position_info, encoding_interval, ['position_x', 'position_y'])
    assert is_training.sum() > 0
    assert is_training.sum() < len(position_info)


def test_concatenate_interval_results_labels():
    ds1 = xr.Dataset({'acausal_posterior': (('time',), np.array([1.0, 2.0]))}, coords={'time': [0.0, 0.1]})
    ds2 = xr.Dataset({'acausal_posterior': (('time',), np.array([3.0]))}, coords={'time': [0.2]})
    concatenated = _concatenate_interval_results([ds1, ds2])
    assert len(concatenated.time) == 3
    np.testing.assert_array_equal(concatenated.interval_labels.values, np.array([0, 0, 1]))


def test_clusterless_events_to_spyglass_spike_lists_compact_electrodes():
    events = _make_clusterless_events()
    spike_times, spike_waveform_features = clusterless_events_to_spyglass_spike_lists(events)
    assert len(spike_times) == 2
    assert len(spike_waveform_features) == 2
    assert len(spike_times[0]) == 2
    assert len(spike_times[1]) == 2


def test_nld_posterior_to_p_x_given_n_2d_shape():
    n_time, n_x, n_y = 12, 6, 7
    mock_pf = _MockPfND(n_bins=n_x, ndim=2)
    mock_pf.occupancy = np.ones((n_x, n_y))
    results = _make_mock_nld_results(n_time=n_time, n_x=n_x, n_y=n_y)
    spatial = nld_spatial_posterior(results)
    assert 'x_position' in spatial.dims
    assert 'y_position' in spatial.dims
    p_x_given_n = nld_posterior_to_p_x_given_n(results, mock_pf)
    assert p_x_given_n.shape == (n_x, n_y, n_time)
    flat_p_x_given_n = nld_posterior_flat_p_x_given_n(results, mock_pf)
    assert flat_p_x_given_n.shape == (n_x * n_y, n_time)


def test_spyglass_decode_roundtrip_with_mocked_run_decoder():
    n_time, n_x, n_y = 10, 6, 7
    mock_pf = _MockPfND(n_bins=n_x, ndim=2)
    mock_pf.occupancy = np.ones((n_x, n_y))
    mock_pf.ybin = np.linspace(0.0, 10.0, n_y + 1)
    mock_pf.ybin_centers = 0.5 * (mock_pf.ybin[:-1] + mock_pf.ybin[1:])
    position_info = _make_position_info(n_samples=20, t_start=0.0, t_stop=1.0)
    events = _make_clusterless_events()
    spike_times, spike_waveform_features = clusterless_events_to_spyglass_spike_lists(events)
    encoding_interval = np.array([[0.0, 1.0]])
    decoding_interval = np.array([[0.0, 1.0]])
    mock_results = _make_mock_nld_results(n_time=n_time, n_x=n_x, n_y=n_y)
    mock_classifier = MagicMock()
    decoder = SpyglassClusterlessDecoder(pf=mock_pf, position_info=position_info, spike_times=spike_times, spike_waveform_features=spike_waveform_features, encoding_interval=encoding_interval, decoding_interval=decoding_interval, spyglass_params=SpyglassClusterlessDecodingParameters(estimate_decoding_params=True), setup_on_init=False, post_load_on_init=False, debug_print=False)
    with patch('pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_decoder.run_clusterless_decoder_in_memory', return_value=(mock_classifier, mock_results)):
        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = decoder.decode(spike_times=spike_times, spike_waveform_features=spike_waveform_features, position_info=position_info, decoding_interval=decoding_interval, encoding_interval=encoding_interval, output_flat_versions=True, debug_print=False)
    assert decoder.p_x_given_n is None
    assert p_x_given_n.shape == (n_x, n_y, n_time)
    assert most_likely_positions.shape == (n_time, 2)
    assert flat_outputs_container is not None
    assert flat_outputs_container.flat_p_x_given_n.shape == (n_x * n_y, n_time)


def test_spyglass_is_clusterless_decoder_type_check():
    mock_pf = _MockPfND(n_bins=6, ndim=2)
    decoder = SpyglassClusterlessDecoder(pf=mock_pf, setup_on_init=False, post_load_on_init=False, debug_print=False)
    assert SpyglassClusterlessDecoder.is_clusterless_decoder(decoder)
    assert SpyglassClusterlessDecoder.is_spyglass_clusterless_decoder(decoder)


def test_spyglass_replacing_computation_epochs_refreshes_encoding_interval():
    from neuropy.core.epoch import Epoch

    mock_pf = _MockPfND(n_bins=6, ndim=2)
    full_epochs_df = pd.DataFrame({'start': [0.0, 0.5], 'stop': [0.4, 1.0], 'label': ['a', 'b']})
    train_epochs_df = pd.DataFrame({'start': [0.0], 'stop': [0.4], 'label': ['a']})
    mock_pf.epochs = Epoch(full_epochs_df)
    encoding_interval = np.array([[0.0, 1.0]])
    decoding_interval = np.array([[0.0, 1.0]])
    decoder = SpyglassClusterlessDecoder(pf=mock_pf, encoding_interval=encoding_interval, decoding_interval=decoding_interval, setup_on_init=False, post_load_on_init=False, debug_print=False)
    train_decoder = decoder.replacing_computation_epochs(Epoch(train_epochs_df))
    np.testing.assert_array_equal(train_decoder.encoding_interval, np.array([[0.0, 0.4]]))
    np.testing.assert_array_equal(train_decoder.decoding_interval, np.array([[0.0, 0.4]]))
    assert train_decoder.classifier is None


def test_position_decoding_spyglass_clusterless_registered_in_default_computation_functions():
    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "General" / "Pipeline" / "Stages" / "ComputationFunctions" / "DefaultComputationFunctions.py"
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text)
    method_names = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "DefaultComputationFunctions":
            method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
    ## END for node in module.body...

    assert "_perform_spyglass_clusterless_position_decoding_computation" in method_names
    assert "position_decoding_spyglass_clusterless" in source_text


def _make_synthetic_place_selective_session(ndim: int = 1, duration_s: float = 4.0, pos_fs_hz: float = 50.0, seed: int = 42):
    """Build a synthetic 0-neuron DataSession with place-selective clusterless spikes in 1D or non-square 2D."""
    from types import SimpleNamespace
    from neuropy.core.epoch import Epoch
    from neuropy.core.position import Position
    from neuropy.core.session.dataSession import DataSession

    rng = np.random.default_rng(seed)
    t_pos = np.linspace(0.0, duration_s, int(duration_s * pos_fs_hz) + 1)
    if ndim == 1:
        x_pos = 50.0 + 45.0 * np.sin(2.0 * np.pi * t_pos / 2.0)
        y_pos = np.zeros_like(x_pos)
        pos_df = pd.DataFrame({"t": t_pos, "x": x_pos, "y": y_pos, "lin_pos": x_pos, "speed": np.full_like(t_pos, 10.0)})
        # 4 place-selective electrodes across [5, 95]
        centers = np.array([15.0, 40.0, 65.0, 85.0])
    else:
        # Non-square 2D arena: x in [0, 60] (6 bins at 10 cm), y in [0, 80] (8 bins at 10 cm)
        x_pos = 30.0 + 25.0 * np.sin(2.0 * np.pi * t_pos / 2.0)
        y_pos = 40.0 + 35.0 * np.cos(2.0 * np.pi * t_pos / 2.5)
        pos_df = pd.DataFrame({"t": t_pos, "x": x_pos, "y": y_pos, "lin_pos": x_pos, "speed": np.full_like(t_pos, 10.0)})
        centers = np.array([[15.0, 20.0], [15.0, 65.0], [45.0, 20.0], [45.0, 65.0]])

    all_times = []
    all_elecs = []
    all_marks = []
    dt = 0.004
    fine_t = np.arange(0.0, duration_s, dt)
    x_fine = np.interp(fine_t, t_pos, x_pos)
    y_fine = np.interp(fine_t, t_pos, y_pos)

    for elec_idx, center in enumerate(centers):
        if ndim == 1:
            dist_sq = (x_fine - center) ** 2
        else:
            dist_sq = (x_fine - center[0]) ** 2 + (y_fine - center[1]) ** 2
        rate_hz = 1.0 + 60.0 * np.exp(-0.5 * dist_sq / (12.0 ** 2))
        spike_mask = rng.random(len(fine_t)) < (rate_hz * dt)
        elec_times = fine_t[spike_mask]
        if len(elec_times) == 0:
            elec_times = np.array([0.5])
        elec_marks = rng.normal(loc=float(elec_idx * 4.0), scale=0.4, size=(len(elec_times), 4)).astype(np.float32)
        all_times.append(elec_times.astype(np.float32))
        all_elecs.append(np.full(len(elec_times), elec_idx, dtype=np.int16))
        all_marks.append(elec_marks)
    ## END for elec_idx, center in enumerate(centers)...

    spike_times_sec = np.concatenate(all_times)
    electrode_indices = np.concatenate(all_elecs)
    marks = np.vstack(all_marks)
    order = np.argsort(spike_times_sec)
    events = ClusterlessSpikeEvents(spike_times_sec=spike_times_sec[order], electrode_indices=electrode_indices[order], marks=marks[order], sampling_frequency_hz=500.0, electrode_mode="channel", t_start=0.0, t_stop=duration_s)
    paradigm = Epoch.from_starts_stops_arrays(starts=np.array([0.0]), stops=np.array([duration_s]), labels=np.array(["maze"]))
    sess = DataSession(config=SimpleNamespace(session_name=f"synth_zero_neuron_{ndim}d"), neurons=None, position=Position(pos_df), paradigm=paradigm, flattened_spiketrains=None, clusterless_spike_events=events)
    return sess, pos_df


def test_spyglass_live_1d_fit_once_predict_many_and_epoch_modes():
    from neuropy.analyses.placefields import PlacefieldComputationParameters
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import SpyglassClusterlessDecodingParameters

    sess, pos_df = _make_synthetic_place_selective_session(ndim=1, duration_s=4.0, seed=11)
    params = SpyglassClusterlessDecodingParameters(position_upsample_hz=250.0, model_type="continuous", waveform_std=2.0, position_std=10.0, use_pf_speed_thresh_for_training=False, sync_environment_to_pf=True)
    pf_params = PlacefieldComputationParameters(speed_thresh=0.0, grid_bin=(10.0,), smooth=(0.0,), frate_thresh=0.0)
    decoder = SpyglassClusterlessDecoder.from_session(sess, ndim=1, computation_config=pf_params, spyglass_params=params, setup_on_init=False, debug_print=False)
    assert decoder.ndim == 1
    assert decoder.num_neurons == 0
    assert decoder.spikes_df is None
    assert decoder.pf.ratemap.n_neurons == 0

    # Fit once and verify classifier is reused without refitting across decode() and decode_specific_epochs()
    decoder.compute_all(decoding_time_bin_size=0.05, debug_print=False)
    fitted_clf = decoder.classifier
    assert fitted_clf is not None
    assert decoder.p_x_given_n.shape[0] == len(decoder.xbin_centers)
    assert decoder.revised_most_likely_positions is not None
    assert decoder.marginal.x.revised_most_likely_positions_1D is not None

    # Verify accuracy against true 1D trajectory
    decoded_times = np.asarray(decoder.time_window_centers, dtype=float)
    true_x = np.interp(decoded_times, pos_df["t"].to_numpy(), pos_df["x"].to_numpy())
    corr = float(np.corrcoef(true_x, np.asarray(decoder.most_likely_positions, dtype=float))[0, 1])
    assert corr > 0.5, f"Expected positive 1D decoding correlation, got {corr:.3f}"

    # Epoch decoding: non-None spikes_df accepted, contiguous, sliding, and single-bin modes
    filter_epochs = pd.DataFrame({"start": [0.5, 2.0], "stop": [1.0, 2.5]})
    res_contig = decoder.decode_specific_epochs(pd.DataFrame({"t_rel_seconds": []}), filter_epochs, decoding_time_bin_size=0.05, debug_print=False)
    assert decoder.classifier is fitted_clf
    assert isinstance(res_contig, DecodedFilterEpochsResult)
    assert res_contig.num_filter_epochs == 2
    assert np.all(res_contig.nbins == 10)
    assert all(p.shape == (len(decoder.xbin_centers), 10) for p in res_contig.p_x_given_n_list)

    res_sliding = decoder.decode_specific_epochs(None, filter_epochs, decoding_time_bin_size=0.10, slideby=0.05, debug_print=False)
    assert decoder.classifier is fitted_clf
    assert res_sliding.decoding_slideby == 0.05
    assert np.all(res_sliding.nbins == 9)

    res_single = decoder.decode_specific_epochs(None, filter_epochs, decoding_time_bin_size=0.05, use_single_time_bin_per_epoch=True, debug_print=False)
    assert decoder.classifier is fitted_clf
    assert np.all(res_single.nbins == 1)
    assert all(p.shape == (len(decoder.xbin_centers), 1) for p in res_single.p_x_given_n_list)


def test_spyglass_live_2d_nonsquare_coordinate_accuracy_and_c_order():
    from neuropy.analyses.placefields import PlacefieldComputationParameters
    from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import SpyglassClusterlessDecodingParameters

    sess, pos_df = _make_synthetic_place_selective_session(ndim=2, duration_s=4.0, seed=23)
    params = SpyglassClusterlessDecodingParameters(position_upsample_hz=200.0, model_type="continuous", waveform_std=2.0, position_std=10.0, use_pf_speed_thresh_for_training=False, sync_environment_to_pf=True)
    pf_params = PlacefieldComputationParameters(speed_thresh=0.0, grid_bin=(10.0, 10.0), smooth=(0.0, 0.0), frate_thresh=0.0)
    decoder = SpyglassClusterlessDecoder.from_session(sess, ndim=2, computation_config=pf_params, spyglass_params=params, setup_on_init=False, debug_print=False)
    n_x, n_y = len(decoder.xbin_centers), len(decoder.ybin_centers)
    assert n_x != n_y, f"Expected non-square 2D grid, got ({n_x}, {n_y})"

    decoder.compute_all(decoding_time_bin_size=0.05, debug_print=False)
    assert decoder.p_x_given_n.shape[:2] == (n_x, n_y)
    assert decoder.most_likely_positions.shape[1] == 2
    assert decoder.marginal.x.revised_most_likely_positions_1D is not None
    assert decoder.marginal.y.revised_most_likely_positions_1D is not None

    decoded_times = np.asarray(decoder.time_window_centers, dtype=float)
    true_x = np.interp(decoded_times, pos_df["t"].to_numpy(), pos_df["x"].to_numpy())
    true_y = np.interp(decoded_times, pos_df["t"].to_numpy(), pos_df["y"].to_numpy())
    corr_x = float(np.corrcoef(true_x, decoder.most_likely_positions[:, 0])[0, 1])
    corr_y = float(np.corrcoef(true_y, decoder.most_likely_positions[:, 1])[0, 1])
    assert corr_x > 0.5 and corr_y > 0.5, f"Expected strong 2D correlations without axis swap, got corr_x={corr_x:.3f}, corr_y={corr_y:.3f}"


def test_spyglass_standalone_pipeline_computation_on_zero_neuron_session():
    from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
    from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import DefaultComputationFunctions
    from pyphoplacecellanalysis.Analysis.Decoder.spyglass_clusterless_adapters import SpyglassClusterlessDecodingParameters

    sess, _pos_df = _make_synthetic_place_selective_session(ndim=2, duration_s=2.0, seed=31)
    comp_result = ComputationResult(sess=sess, computation_config=None, computed_data=DynamicParameters())
    params = SpyglassClusterlessDecodingParameters(position_upsample_hz=200.0, model_type="continuous", waveform_std=2.0, position_std=10.0, use_pf_speed_thresh_for_training=False, sync_environment_to_pf=True)

    out = DefaultComputationFunctions._perform_spyglass_clusterless_position_decoding_computation(
        comp_result, spyglass_params=params, run_1d=True, run_2d=True, compute_full_session_decoding=False, populate_standard_decoder_keys_if_missing=True, debug_print=False
    )
    assert "pf1D" in out.computed_data and out.computed_data["pf1D"].ratemap.n_neurons == 0
    assert "pf2D" in out.computed_data and out.computed_data["pf2D"].ratemap.n_neurons == 0
    assert isinstance(out.computed_data["pf1D_SpyglassClusterlessDecoder"], SpyglassClusterlessDecoder)
    assert isinstance(out.computed_data["pf2D_SpyglassClusterlessDecoder"], SpyglassClusterlessDecoder)
    assert out.computed_data["pf1D_Decoder"] is out.computed_data["pf1D_SpyglassClusterlessDecoder"]
    assert out.computed_data["pf2D_Decoder"] is out.computed_data["pf2D_SpyglassClusterlessDecoder"]
