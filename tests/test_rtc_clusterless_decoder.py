import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from replay_trajectory_classification import ClusterlessClassifier
from replay_trajectory_classification.environments import Environment

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import _pfnd_position_range, build_multiunits_from_array, build_multiunits_from_rtc_simulation, build_rtc_environment_from_pfnd, rtc_posterior_to_p_x_given_n
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


def test_clusterless_decode_rejects_spike_counts():
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, setup_on_init=False, post_load_on_init=False, debug_print=False)
    spike_counts = np.zeros((5, 20))
    with pytest.raises(ValueError, match="multiunits with shape"):
        decoder.decode(spike_counts, time_bin_size=0.05, debug_print=False)


def test_clusterless_decode_specific_epochs_uses_multiunits_for_epoch_windows():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    filter_epochs = pd.DataFrame({"start": [float(time[100]), float(time[250])], "stop": [float(time[119]), float(time[269])]})
    is_training = np.ones(len(time), dtype=bool)
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        result = decoder.decode_specific_epochs(pd.DataFrame({"ignored": []}), filter_epochs, decoding_time_bin_size=0.001, debug_print=False)
    assert isinstance(result, DecodedFilterEpochsResult)
    assert result.num_filter_epochs == len(filter_epochs)
    assert np.all(result.nbins > 0)
    assert all(len(container.centers) == n_bins for container, n_bins in zip(result.time_bin_containers, result.nbins))
    assert all(posterior.shape[-1] == n_bins for posterior, n_bins in zip(result.p_x_given_n_list, result.nbins))
    assert all(len(positions) == n_bins for positions, n_bins in zip(result.most_likely_positions_list, result.nbins))


def test_clusterless_decode_specific_epochs_ignores_spikes_df():
    time, position, multiunits, _position_1d = build_multiunits_from_rtc_simulation(n_runs=2)
    mock_pf = _MockPfND(n_bins=10, ndim=1, grid_bin_bounds=(0.0, 100.0), pos_bin_size=10.0)
    decoder = ClusterlessRTCPositionDecoder(pf=mock_pf, sampling_frequency_hz=1000.0, multiunits=multiunits, rtc_time=time, setup_on_init=False, post_load_on_init=False, debug_print=False)
    filter_epochs = pd.DataFrame({"start": [float(time[100])], "stop": [float(time[119])]})
    is_training = np.ones(len(time), dtype=bool)
    with patch("pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_decoder.build_clusterless_training_data_from_pfnd", return_value=(position, multiunits, is_training)):
        result = decoder.decode_specific_epochs(pd.DataFrame(), filter_epochs, decoding_time_bin_size=0.001, debug_print=False)
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
        result = decoder.decode_specific_epochs(pd.DataFrame(), filter_epochs, decoding_time_bin_size=0.001, use_single_time_bin_per_epoch=True, debug_print=False)
    assert np.all(result.nbins == 1)
    assert all(len(container.centers) == 1 for container in result.time_bin_containers)
    assert all(len(positions) == 1 for positions in result.most_likely_positions_list)
