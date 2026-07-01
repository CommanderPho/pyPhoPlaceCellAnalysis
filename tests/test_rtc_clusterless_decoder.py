import ast
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from replay_trajectory_classification import ClusterlessClassifier
from replay_trajectory_classification.environments import Environment

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import _pfnd_position_range, build_multiunits_from_array, build_multiunits_from_phy_folder, build_multiunits_from_rtc_simulation, build_multiunits_from_spike_events, build_rtc_environment_from_pfnd, extract_clusterless_spike_events_from_phy_folder, load_clusterless_spike_events, rtc_posterior_to_p_x_given_n, save_clusterless_spike_events
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ClusterlessRTCPositionDecoder, DecodedFilterEpochsResult


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
        self.xbin_centers = np.arange(n_bins, dtype=float)
        self.ybin_centers = np.arange(n_bins, dtype=float) if ndim == 2 else None
        self.epochs = None


    def replacing_computation_epochs(self, epochs):
        replaced_pf = _MockPfND(n_bins=len(self.xbin_centers), ndim=self.ndim, grid_bin_bounds=self.config.grid_bin_bounds, pos_bin_size=float(np.mean(np.atleast_1d(self.pos_bin_size))))
        replaced_pf.epochs = epochs
        return replaced_pf


def test_pfnd_position_range_shapes():
    range_1d_nested = _pfnd_position_range(_MockPfND(n_bins=10, ndim=1, grid_bin_bounds=((0.0, 100.0), (-20.0, 20.0))))
    assert range_1d_nested.shape == (1, 2)
    np.testing.assert_allclose(range_1d_nested, np.array([[0.0, 100.0]]))
    assert np.diff(range_1d_nested, axis=1).shape == (1, 1)

    range_1d_flat = _pfnd_position_range(_MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0)))
    assert range_1d_flat.shape == (1, 2)
    np.testing.assert_allclose(range_1d_flat, np.array([[0.0, 100.0]]))
    assert np.diff(range_1d_flat, axis=1).shape == (1, 1)

    range_2d_nested = _pfnd_position_range(_MockPfND(n_bins=10, ndim=2, grid_bin_bounds=((0.0, 100.0), (-20.0, 20.0))))
    assert range_2d_nested.shape == (2, 2)
    np.testing.assert_allclose(range_2d_nested, np.array([[0.0, 100.0], [-20.0, 20.0]]))
    assert np.diff(range_2d_nested, axis=1).shape == (2, 1)


def test_build_rtc_environment_fit_place_grid():
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    environment = build_rtc_environment_from_pfnd(mock_pf)
    position = np.linspace(0.0, 100.0, 50)[:, np.newaxis]
    environment.fit_place_grid(position)
    assert environment.place_bin_centers_.shape[1] == 1


def test_build_multiunits_from_rtc_simulation_shapes():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    assert len(time) == multiunits.shape[0]
    assert position.shape[0] == multiunits.shape[0]
    assert multiunits.ndim == 3


def test_build_multiunits_from_array_drops_empty_electrodes():
    multiunits = np.full((10, 4, 3), np.nan, dtype=float)
    multiunits[2, :, 1] = np.array([1.0, 2.0, 3.0, 4.0])
    filtered_multiunits, filtered_time = build_multiunits_from_array(multiunits)
    assert filtered_time is None
    assert filtered_multiunits.shape == (10, 4, 1)
    np.testing.assert_allclose(filtered_multiunits[2, :, 0], np.array([1.0, 2.0, 3.0, 4.0]))


def test_drop_empty_multiunit_electrodes_respects_training_mask():
    from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import _drop_empty_multiunit_electrodes
    multiunits = np.full((100, 4, 3), np.nan, dtype=float)
    multiunits[10:20, :, 0] = np.array([1.0, 2.0, 3.0, 4.0])
    multiunits[80:90, :, 2] = np.array([5.0, 6.0, 7.0, 8.0])
    is_training = np.zeros(100, dtype=bool)
    is_training[10:20] = True
    filtered_multiunits = _drop_empty_multiunit_electrodes(multiunits, time_mask=is_training)
    assert filtered_multiunits.shape == (100, 4, 1)
    np.testing.assert_allclose(filtered_multiunits[10:20, :, 0], multiunits[10:20, :, 0])


def _write_synthetic_phy_folder(phy_folder: Path, sample_rate_hz: float = 30000.0) -> None:
    phy_folder.mkdir(parents=True, exist_ok=True)
    (phy_folder / "params.py").write_text(f"sample_rate = {sample_rate_hz}\n", encoding="utf-8")
    spike_times = np.array([int(1.0 * sample_rate_hz), int(1.001 * sample_rate_hz), int(1.002 * sample_rate_hz), int(1.5 * sample_rate_hz), int(2.0 * sample_rate_hz)], dtype=np.int64)
    spike_templates = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    pc_feature_ind = np.array([[0, 1, -1, -1], [1, 2, -1, -1]], dtype=np.int64)
    pc_features = np.zeros((len(spike_times), 4, 4), dtype=np.float32)
    pc_features[0, :, 0] = np.array([1.0, 0.1, 0.2, 0.3], dtype=np.float32)
    pc_features[1, :, 1] = np.array([2.0, 0.4, 0.5, 0.6], dtype=np.float32)
    pc_features[2, :, 0] = np.array([3.0, 0.7, 0.8, 0.9], dtype=np.float32)
    pc_features[3, :, 1] = np.array([4.0, 1.0, 1.1, 1.2], dtype=np.float32)
    pc_features[4, :, 0] = np.array([5.0, 1.3, 1.4, 1.5], dtype=np.float32)
    channel_map = np.array([0, 1, 2], dtype=np.int32)
    np.save(phy_folder / "spike_times.npy", spike_times)
    np.save(phy_folder / "spike_templates.npy", spike_templates)
    np.save(phy_folder / "pc_feature_ind.npy", pc_feature_ind)
    np.save(phy_folder / "pc_features.npy", pc_features)
    np.save(phy_folder / "channel_map.npy", channel_map)


def test_build_multiunits_from_phy_folder_synthetic_roundtrip(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    multiunits, rtc_time = build_multiunits_from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, sampling_frequency_hz=1000.0, electrode_mode="channel")
    assert multiunits.ndim == 3
    assert multiunits.shape[1] == 4
    assert multiunits.shape[0] == len(rtc_time)
    assert multiunits.shape[2] >= 1
    assert np.isfinite(multiunits).any()
    filtered_multiunits, filtered_rtc_time = build_multiunits_from_array(multiunits, rtc_time)
    assert filtered_multiunits.shape[0] == len(filtered_rtc_time)
    assert filtered_multiunits.shape[1] == 4
    assert filtered_multiunits.shape[2] >= 1


def test_extract_clusterless_spike_events_synthetic_phy(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    events = extract_clusterless_spike_events_from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, electrode_mode="channel")
    assert events.spike_times_sec.dtype == np.float32
    assert events.electrode_indices.dtype == np.int16
    assert events.marks.dtype == np.float32
    assert events.marks.shape[1] == 4
    assert len(events.spike_times_sec) == 5
    assert np.all((events.spike_times_sec >= 1.0) & (events.spike_times_sec <= 2.0))


def test_extract_clusterless_spike_events_infers_session_times_from_params(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder, sample_rate_hz=30000.0)
    (phy_folder / "params.py").write_text("sample_rate = 30000.0\nn_samples_dat = 90000\n", encoding="utf-8")
    events = extract_clusterless_spike_events_from_phy_folder(phy_folder, electrode_mode="channel")
    assert events.t_start == 0.0
    assert events.t_end == pytest.approx(3.0)
    assert len(events.spike_times_sec) == 5


def test_extract_clusterless_spike_events_infers_missing_t_end_from_params(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder, sample_rate_hz=30000.0)
    (phy_folder / "params.py").write_text("sample_rate = 30000.0\nn_samples_dat = 90000\n", encoding="utf-8")
    events = extract_clusterless_spike_events_from_phy_folder(phy_folder, t_start=1.0, electrode_mode="channel")
    assert events.t_start == 1.0
    assert events.t_end == pytest.approx(3.0)
    assert len(events.spike_times_sec) == 5


def test_save_load_clusterless_spike_events_roundtrip(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    events = extract_clusterless_spike_events_from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, electrode_mode="channel")
    events_path = tmp_path / "test.clusterless_spikes.npz"
    save_clusterless_spike_events(events_path, events)
    loaded_events = load_clusterless_spike_events(events_path)
    np.testing.assert_allclose(loaded_events.spike_times_sec, events.spike_times_sec)
    np.testing.assert_array_equal(loaded_events.electrode_indices, events.electrode_indices)
    np.testing.assert_allclose(loaded_events.marks, events.marks)
    assert loaded_events.electrode_mode == events.electrode_mode
    assert loaded_events.n_mark_dims == events.n_mark_dims


def test_build_multiunits_from_spike_events_uses_event_times_when_omitted(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    events = extract_clusterless_spike_events_from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, electrode_mode="channel", sampling_frequency_hz=1000.0)
    explicit_multiunits, explicit_rtc_time = build_multiunits_from_spike_events(events, t_start=1.0, t_end=2.0, sampling_frequency_hz=1000.0)
    inferred_multiunits, inferred_rtc_time = build_multiunits_from_spike_events(events, sampling_frequency_hz=1000.0)
    np.testing.assert_allclose(inferred_rtc_time, explicit_rtc_time)
    np.testing.assert_allclose(inferred_multiunits, explicit_multiunits)


def test_build_multiunits_from_spike_events_matches_phy_folder(tmp_path):
    phy_folder = tmp_path / "phy"
    _write_synthetic_phy_folder(phy_folder)
    events = extract_clusterless_spike_events_from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, electrode_mode="channel", sampling_frequency_hz=1000.0)
    from_events, rtc_time_from_events = build_multiunits_from_spike_events(events, t_start=1.0, t_end=2.0, sampling_frequency_hz=1000.0)
    from_phy, rtc_time_from_phy = build_multiunits_from_phy_folder(phy_folder, t_start=1.0, t_end=2.0, sampling_frequency_hz=1000.0, electrode_mode="channel")
    np.testing.assert_allclose(rtc_time_from_events, rtc_time_from_phy)
    assert from_events.shape == from_phy.shape
    finite_from_events = np.isfinite(from_events)
    finite_from_phy = np.isfinite(from_phy)
    np.testing.assert_array_equal(finite_from_events, finite_from_phy)
    np.testing.assert_allclose(from_events[finite_from_events], from_phy[finite_from_phy])


def test_rtc_clusterless_classifier_simulation_roundtrip():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    classifier = ClusterlessClassifier(environments=[Environment(place_bin_size=6.0)])
    classifier.fit(position, multiunits)
    results = classifier.predict(multiunits)
    posterior = results.acausal_posterior.values
    assert posterior.ndim >= 2
    assert np.isfinite(posterior).any()


def test_rtc_posterior_to_p_x_given_n_shape():
    n_time, n_states, n_bins = 20, 2, 10
    mock_pf = _MockPfND(n_bins=n_bins, ndim=1)
    posterior = np.random.rand(n_time, n_states, n_bins)
    posterior /= posterior.sum(axis=-1, keepdims=True)
    rtc_results = xr.Dataset({"acausal_posterior": (("time", "state", "position"), posterior)})
    p_x_given_n = rtc_posterior_to_p_x_given_n(rtc_results, mock_pf)
    assert p_x_given_n.shape == (n_bins, n_time)


def test_position_decoding_clusterless_registered_in_default_computation_functions():
    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "General" / "Pipeline" / "Stages" / "ComputationFunctions" / "DefaultComputationFunctions.py"
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text)
    method_names = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "DefaultComputationFunctions":
            method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
    assert "_perform_clusterless_position_decoding_computation" in method_names
    assert "position_decoding_clusterless" in source_text


def test_clusterless_decode_multiunits_roundtrip():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    is_training = np.ones(len(time), dtype=bool)
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = decoder.decode(multiunits, time_bin_size=0.001, rtc_time=time, output_flat_versions=True, debug_print=False)
    assert decoder.p_x_given_n is None
    assert len(most_likely_positions) == multiunits.shape[0]
    assert p_x_given_n.ndim >= 2
    assert most_likely_position_indicies.ndim >= 1
    assert flat_outputs_container is not None
    assert flat_outputs_container.flat_p_x_given_n.shape[1] == multiunits.shape[0]


def test_clusterless_fit_drops_electrodes_without_training_spikes():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    multiunits = np.asarray(multiunits, dtype=float).copy()
    original_n_electrodes = multiunits.shape[2]
    multiunits[:, :, -1] = np.nan
    multiunits[500:520, :, -1] = 1.0
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    is_training = np.zeros(len(time), dtype=bool)
    is_training[:400] = True
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        decoder._ensure_fitted_classifier(multiunits_for_fit=multiunits, rtc_time_for_fit=time, debug_print=False)
    assert decoder.classifier is not None
    assert decoder.multiunit_electrode_keep_mask.shape[0] == original_n_electrodes
    assert not decoder.multiunit_electrode_keep_mask[-1]
    assert decoder.multiunits.shape[2] == int(np.sum(decoder.multiunit_electrode_keep_mask))


def test_clusterless_decode_rejects_spike_counts():
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, setup_on_init=False, post_load_on_init=False, debug_print=False)
    spike_counts = np.zeros((5, 20))
    with pytest.raises(ValueError, match="multiunits with shape"):
        decoder.decode(spike_counts, time_bin_size=0.05, debug_print=False)


def test_clusterless_replacing_computation_epochs_preserves_type_and_state():
    time, _position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    decoder.classifier = object()
    decoder.rtc_results = object()
    decoder.p_x_given_n = np.ones((10, 5))
    decoder.flat_p_x_given_n = np.ones((10, 5))
    decoder.most_likely_positions = np.ones(5)
    decoder.rtc_position_bin_centers = np.ones((10, 1))
    replacement_epochs = pd.DataFrame({"start": [float(time[100])], "stop": [float(time[200])], "label": ["train"]})

    replaced_decoder = decoder.replacing_computation_epochs(replacement_epochs)

    assert isinstance(replaced_decoder, ClusterlessRTCPositionDecoder)
    assert replaced_decoder is not decoder
    assert replaced_decoder.pf is not decoder.pf
    np.testing.assert_allclose(replaced_decoder.multiunits, decoder.multiunits)
    np.testing.assert_allclose(replaced_decoder.rtc_time, decoder.rtc_time)
    assert replaced_decoder.sampling_frequency_hz == decoder.sampling_frequency_hz
    assert replaced_decoder.classifier is None
    assert replaced_decoder.rtc_results is None
    assert replaced_decoder.p_x_given_n is None
    assert replaced_decoder.flat_p_x_given_n is None
    assert replaced_decoder.most_likely_positions is None
    assert replaced_decoder.rtc_position_bin_centers is None


def test_clusterless_decode_specific_epochs_uses_multiunits_for_epoch_windows():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    filter_epochs = pd.DataFrame({"start": [float(time[100]), float(time[250])], "stop": [float(time[119]), float(time[269])]})
    is_training = np.ones(len(time), dtype=bool)
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        result = decoder.decode_specific_epochs(None, filter_epochs, decoding_time_bin_size=0.001, debug_print=False)
    assert isinstance(result, DecodedFilterEpochsResult)
    assert result.num_filter_epochs == len(filter_epochs)
    assert np.all(result.nbins > 0)
    assert all(len(container.centers) == n_bins for container, n_bins in zip(result.time_bin_containers, result.nbins))
    assert all(posterior.shape[-1] == n_bins for posterior, n_bins in zip(result.p_x_given_n_list, result.nbins))
    assert all(len(positions) == n_bins for positions, n_bins in zip(result.most_likely_positions_list, result.nbins))


def test_clusterless_decode_specific_epochs_requires_none_spikes_df():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    filter_epochs = pd.DataFrame({"start": [float(time[100])], "stop": [float(time[119])]})
    is_training = np.ones(len(time), dtype=bool)
    with pytest.raises(AssertionError, match="spikes_df MUST be None"):
        decoder.decode_specific_epochs(pd.DataFrame(), filter_epochs, decoding_time_bin_size=0.001, debug_print=False)
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        result = decoder.decode_specific_epochs(None, filter_epochs, decoding_time_bin_size=0.001, debug_print=False)
    assert result.num_filter_epochs == 1
    assert result.nbins[0] > 0
    assert len(result.most_likely_positions_list[0]) == result.nbins[0]


def test_clusterless_decode_specific_epochs_single_time_bin_per_epoch():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    filter_epochs = pd.DataFrame({"start": [float(time[100]), float(time[250])], "stop": [float(time[119]), float(time[269])]})
    is_training = np.ones(len(time), dtype=bool)
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        result = decoder.decode_specific_epochs(None, filter_epochs, decoding_time_bin_size=0.001, use_single_time_bin_per_epoch=True, debug_print=False)
    assert np.all(result.nbins == 1)
    assert all(len(container.centers) == 1 for container in result.time_bin_containers)
    assert all(len(positions) == 1 for positions in result.most_likely_positions_list)


def test_decode_using_new_decoders_passes_none_spikes_df_for_clusterless():
    class _SpyClusterlessDecoder(ClusterlessRTCPositionDecoder):
        def __init__(self):
            pass


        def decode_specific_epochs(self, spikes_df, filter_epochs, decoding_time_bin_size: float = 0.05, use_single_time_bin_per_epoch: bool = False, slideby=None, debug_print=False):
            self.received_spikes_df = spikes_df
            self.received_filter_epochs = filter_epochs
            return "clusterless-result"

    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "General" / "Pipeline" / "Stages" / "ComputationFunctions" / "MultiContextComputationFunctions" / "DirectionalPlacefieldGlobalComputationFunctions.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(node for node in source_tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainTestLapsSplitting")
    method_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "decode_using_new_decoders")
    class_node = ast.ClassDef(name="TrainTestLapsSplitting", bases=[], keywords=[], body=[method_node], decorator_list=[])
    ast.fix_missing_locations(class_node)
    namespace = {"deepcopy": deepcopy, "Dict": dict, "DecodedFilterEpochsResult": object}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), filename=str(source_path), mode="exec"), namespace)
    TrainTestLapsSplitting = namespace["TrainTestLapsSplitting"]

    decoder = _SpyClusterlessDecoder()
    global_spikes_df = pd.DataFrame({"spike": [1]})
    test_epochs = pd.DataFrame({"start": [0.0], "stop": [1.0]})

    result = TrainTestLapsSplitting.decode_using_new_decoders(global_spikes_df=global_spikes_df, train_lap_specific_pf1D_Decoder_dict={"maze1": decoder}, test_epochs_dict={"maze1": test_epochs}, laps_decoding_time_bin_size=0.25)

    assert result == {"maze1": "clusterless-result"}
    assert decoder.received_spikes_df is None
