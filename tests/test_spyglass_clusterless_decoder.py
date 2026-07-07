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


    def replacing_computation_epochs(self, epochs):
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
    n_time, n_x, n_y = 10, 6, 6
    mock_pf = _MockPfND(n_bins=n_x, ndim=2)
    mock_pf.occupancy = np.ones((n_x, n_y))
    position_info = _make_position_info(n_samples=20, t_start=0.0, t_stop=1.0)
    events = _make_clusterless_events()
    spike_times, spike_waveform_features = clusterless_events_to_spyglass_spike_lists(events)
    encoding_interval = np.array([[0.0, 1.0]])
    decoding_interval = np.array([[0.0, 1.0]])
    mock_results = _make_mock_nld_results(n_time=n_time, n_x=n_x, n_y=n_y)
    mock_classifier = MagicMock()
    decoder = SpyglassClusterlessDecoder(pf=mock_pf, position_info=position_info, spike_times=spike_times, spike_waveform_features=spike_waveform_features, encoding_interval=encoding_interval, decoding_interval=decoding_interval, setup_on_init=False, post_load_on_init=False, debug_print=False)
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


def test_position_decoding_spyglass_clusterless_registered_in_default_computation_functions():
    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "General" / "Pipeline" / "Stages" / "ComputationFunctions" / "DefaultComputationFunctions.py"
    source_text = source_path.read_text(encoding="utf-8")
    module = ast.parse(source_text)
    method_names = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "DefaultComputationFunctions":
            method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
    assert "_perform_spyglass_clusterless_position_decoding_computation" in method_names
    assert "position_decoding_spyglass_clusterless" in source_text
