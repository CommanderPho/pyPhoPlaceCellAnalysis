import ast
from pathlib import Path

import numpy as np
import xarray as xr

from replay_trajectory_classification import ClusterlessClassifier
from replay_trajectory_classification.environments import Environment

from pyphoplacecellanalysis.Analysis.Decoder.rtc_clusterless_adapters import _pfnd_position_range, build_multiunits_from_array, build_multiunits_from_rtc_simulation, build_rtc_environment_from_pfnd, rtc_posterior_to_p_x_given_n


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
