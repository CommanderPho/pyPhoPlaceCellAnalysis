"""Unit tests for SingleEpochDecodedResult HDF5 round-trip (`to_hdf` / `read_hdf`)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import numpy as np

# Add package src to path as needed (matches other tests in this folder)
tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent

try:
    import pyphoplacecellanalysis
except ModuleNotFoundError:
    print(f'root_project_folder: {root_project_folder}')
    src_folder = root_project_folder.joinpath('src')
    print(f'pyphoplacecellanalysis src folder: {src_folder}')
    sys.path.insert(0, str(src_folder))
finally:
    import h5py
    # Import reconstruction first so NeuroPy circular-import chains resolve via the package load order.
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
    from neuropy.utils.dynamic_container import DynamicContainer
    from neuropy.utils.mixins.binning_helpers import BinningContainer


EpochTuple = namedtuple('EpochTuple', ['Index', 'start', 'stop', 'label', 'duration'])


def _build_normalized_posterior(*spatial_shape, n_t: int, rng: np.random.Generator) -> np.ndarray:
    """Build a posterior with last axis = time, normalized over spatial axes per time bin."""
    shape = (*spatial_shape, n_t)
    p = rng.random(shape).astype(float)
    # sum over all but last axis
    sum_axes = tuple(range(p.ndim - 1))
    p = p / np.sum(p, axis=sum_axes, keepdims=True)
    return p


def _build_1d_single_epoch_result(n_x: int = 5, n_t: int = 4, epoch_data_index: int = 3, include_spkcount: bool = True, seed: int = 0) -> SingleEpochDecodedResult:
    rng = np.random.default_rng(seed)
    p = _build_normalized_posterior(n_x, n_t=n_t, rng=rng)
    edges = np.linspace(10.0, 12.0, n_t + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tbc = BinningContainer(edges=edges, centers=centers)
    mlp = np.full(n_t, 1.5, dtype=float)
    mlpi = np.zeros(n_t, dtype=int)
    marginal_x = DynamicContainer(p_x_given_n=p.copy(), most_likely_positions_1D=mlp.copy())
    epoch_info = EpochTuple(Index=0, start=10.0, stop=12.0, label='0', duration=2.0)
    spkcount = np.arange(n_t, dtype=float) if include_spkcount else None
    return SingleEpochDecodedResult(
        p_x_given_n=p,
        epoch_info_tuple=epoch_info,
        most_likely_positions=mlp,
        most_likely_position_indicies=mlpi,
        nbins=n_t,
        time_bin_container=tbc,
        time_bin_edges=edges,
        marginal_x=marginal_x,
        marginal_y=None,
        marginal_z=None,
        epoch_data_index=epoch_data_index,
        spkcount=spkcount,
    )


def _build_2d_single_epoch_result(n_x: int = 4, n_y: int = 3, n_t: int = 5, epoch_data_index: int = 1, seed: int = 1) -> SingleEpochDecodedResult:
    rng = np.random.default_rng(seed)
    p = _build_normalized_posterior(n_x, n_y, n_t=n_t, rng=rng)
    edges = np.linspace(100.0, 101.0, n_t + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tbc = BinningContainer(edges=edges, centers=centers)
    # most_likely_positions: (n_t, 2)
    mlp = np.column_stack([np.full(n_t, 2.0), np.full(n_t, 3.0)])
    mlpi = np.zeros((2, n_t), dtype=int)
    # x-marginal for legacy export path
    marg_x_p = np.sum(p, axis=1)
    marg_x_p = marg_x_p / np.sum(marg_x_p, axis=0, keepdims=True)
    marg_y_p = np.sum(p, axis=0)
    marg_y_p = marg_y_p / np.sum(marg_y_p, axis=0, keepdims=True)
    marginal_x = DynamicContainer(p_x_given_n=marg_x_p, most_likely_positions_1D=mlp[:, 0].copy())
    marginal_y = DynamicContainer(p_x_given_n=marg_y_p, most_likely_positions_1D=mlp[:, 1].copy())
    epoch_info = EpochTuple(Index=1, start=100.0, stop=101.0, label='1', duration=1.0)
    return SingleEpochDecodedResult(
        p_x_given_n=p,
        epoch_info_tuple=epoch_info,
        most_likely_positions=mlp,
        most_likely_position_indicies=mlpi,
        nbins=n_t,
        time_bin_container=tbc,
        time_bin_edges=edges,
        marginal_x=marginal_x,
        marginal_y=marginal_y,
        marginal_z=None,
        epoch_data_index=epoch_data_index,
        spkcount=np.ones((n_x * n_y, n_t), dtype=float),
    )


def _build_overlapping_window_1d_result(n_x: int = 3, n_t: int = 4, seed: int = 2) -> SingleEpochDecodedResult:
    """1D result whose time_bin_edges are (n_t, 2) overlapping [start, stop] rows."""
    rng = np.random.default_rng(seed)
    p = _build_normalized_posterior(n_x, n_t=n_t, rng=rng)
    starts = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)[:n_t]
    stops = starts + 0.1
    edges_2d = np.column_stack([starts, stops])
    centers = 0.5 * (starts + stops)
    tbc = BinningContainer.from_sliding_windows(window_start_edges=starts, window_stop_edges=stops, epoch_variable_extents=(float(starts[0]), float(stops[-1])), window_width=0.1, hop=0.05)
    mlp = np.full(n_t, 0.5, dtype=float)
    mlpi = np.zeros(n_t, dtype=int)
    marginal_x = DynamicContainer(p_x_given_n=p.copy(), most_likely_positions_1D=mlp.copy())
    epoch_info = EpochTuple(Index=2, start=float(starts[0]), stop=float(stops[-1]), label='2', duration=float(stops[-1] - starts[0]))
    return SingleEpochDecodedResult(
        p_x_given_n=p,
        epoch_info_tuple=epoch_info,
        most_likely_positions=mlp,
        most_likely_position_indicies=mlpi,
        nbins=n_t,
        time_bin_container=tbc,
        time_bin_edges=edges_2d,
        marginal_x=marginal_x,
        marginal_y=None,
        marginal_z=None,
        epoch_data_index=2,
        spkcount=None,
    )


class TestSingleEpochDecodedResultHDFRoundtrip(unittest.TestCase):

    def test_full_object_roundtrip_1d(self):
        obj = _build_1d_single_epoch_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('single_epoch_1d.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_0', save_full_object=True)
                group = f['epoch_0']
                self.assertIn('p_x_given_n', group)
                self.assertIn('p_x_given_n_grey', group)
                self.assertIn('p_x_given_n_full', group)
                self.assertIn('spkcount', group)
                self.assertIn('t_bin_centers', group)
                self.assertIn('time_bin_edges', group)
                self.assertTrue(bool(group.attrs['save_full_object']))
                self.assertEqual(int(group.attrs['epoch_data_index']), 3)
                self.assertEqual(int(group.attrs['nbins']), obj.nbins)
            ## END with h5py.File(path, 'w') as f...

            restored = SingleEpochDecodedResult.read_hdf(path, key='epoch_0')
            self.assertEqual(restored.epoch_data_index, 3)
            self.assertEqual(restored.nbins, obj.nbins)
            np.testing.assert_allclose(restored.p_x_given_n, obj.p_x_given_n)
            np.testing.assert_allclose(restored.spkcount, obj.spkcount)
            np.testing.assert_allclose(restored.most_likely_positions, obj.most_likely_positions)
            np.testing.assert_array_equal(restored.most_likely_position_indicies, obj.most_likely_position_indicies)
            np.testing.assert_allclose(restored.time_bin_edges, obj.time_bin_edges)
            self.assertIsNotNone(restored.time_bin_container)
            self.assertEqual(len(restored.time_bin_container.centers), obj.nbins)
            self.assertIsNotNone(restored.marginal_x)
            np.testing.assert_allclose(restored.marginal_x.p_x_given_n, obj.p_x_given_n)
            self.assertEqual(restored.epoch_info_tuple.label, '0')
            self.assertAlmostEqual(float(restored.epoch_info_tuple.start), 10.0)
            self.assertAlmostEqual(float(restored.epoch_info_tuple.stop), 12.0)
        ## END with tempfile.TemporaryDirectory() as td...


    def test_full_object_roundtrip_2d(self):
        obj = _build_2d_single_epoch_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('single_epoch_2d.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_2d', save_full_object=True)
                group = f['epoch_2d']
                self.assertIn('p_x_given_n_full', group)
                # legacy dataset is x-marginal, not full 2D
                self.assertEqual(np.asarray(group['p_x_given_n']).ndim, 2)
                self.assertEqual(np.asarray(group['p_x_given_n_full']).ndim, 3)
                self.assertEqual(tuple(np.asarray(group['p_x_given_n_full']).shape), tuple(obj.p_x_given_n.shape))
            ## END with h5py.File(path, 'w') as f...

            restored = SingleEpochDecodedResult.read_hdf(path, key='epoch_2d')
            np.testing.assert_allclose(restored.p_x_given_n, obj.p_x_given_n)
            self.assertEqual(restored.p_x_given_n.ndim, 3)
            self.assertIsNotNone(restored.marginal_x)
            self.assertIsNotNone(restored.marginal_y)
            self.assertIsNone(restored.marginal_z)
            np.testing.assert_allclose(restored.most_likely_positions, obj.most_likely_positions)
            self.assertEqual(restored.epoch_data_index, 1)
        ## END with tempfile.TemporaryDirectory() as td...


    def test_legacy_mode_omits_full_object_fields(self):
        obj = _build_1d_single_epoch_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('legacy.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_0', save_full_object=False)
                group = f['epoch_0']
                self.assertIn('p_x_given_n', group)
                self.assertIn('p_x_given_n_grey', group)
                self.assertNotIn('p_x_given_n_full', group)
                self.assertNotIn('spkcount', group)
                self.assertFalse(bool(group.attrs.get('save_full_object', False)))
                self.assertNotIn('epoch_data_index', group.attrs)
            ## END with h5py.File(path, 'w') as f...
        ## END with tempfile.TemporaryDirectory() as td...


    def test_legacy_mode_still_readable_1d(self):
        """Legacy export stores marginal_x as `p_x_given_n`; for 1D that equals the full posterior."""
        obj = _build_1d_single_epoch_result(include_spkcount=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('legacy_readable.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_0', save_full_object=False)
            ## END with h5py.File(path, 'w') as f...

            restored = SingleEpochDecodedResult.read_hdf(path, key='epoch_0')
            self.assertIsNone(restored.epoch_data_index)
            self.assertIsNone(restored.spkcount)
            np.testing.assert_allclose(restored.p_x_given_n, obj.p_x_given_n)
            self.assertEqual(restored.nbins, obj.nbins)
            self.assertIsNotNone(restored.time_bin_container)
        ## END with tempfile.TemporaryDirectory() as td...


    def test_legacy_2d_loads_x_marginal_as_p_x_given_n(self):
        """Without p_x_given_n_full, 2D restore falls back to the exported x-marginal."""
        obj = _build_2d_single_epoch_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('legacy_2d.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_2d', save_full_object=False)
            ## END with h5py.File(path, 'w') as f...

            restored = SingleEpochDecodedResult.read_hdf(path, key='epoch_2d')
            self.assertEqual(restored.p_x_given_n.ndim, 2)
            np.testing.assert_allclose(restored.p_x_given_n, obj.marginal_x.p_x_given_n)
            # full 2D posterior is NOT recovered from legacy files
            self.assertNotEqual(restored.p_x_given_n.shape, obj.p_x_given_n.shape)
        ## END with tempfile.TemporaryDirectory() as td...


    def test_read_hdf_from_open_file_handle(self):
        obj = _build_1d_single_epoch_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('open_handle.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_0', save_full_object=True)
            ## END with h5py.File(path, 'w') as f...

            with h5py.File(path, 'r') as f:
                restored = SingleEpochDecodedResult.read_hdf(f, key='epoch_0')
                np.testing.assert_allclose(restored.p_x_given_n, obj.p_x_given_n)
                self.assertEqual(restored.epoch_data_index, obj.epoch_data_index)
            ## END with h5py.File(path, 'r') as f...
        ## END with tempfile.TemporaryDirectory() as td...


    def test_overlapping_window_edges_roundtrip(self):
        obj = _build_overlapping_window_1d_result()
        self.assertEqual(np.asarray(obj.time_bin_edges).ndim, 2)
        self.assertEqual(np.asarray(obj.time_bin_edges).shape[-1], 2)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('overlap.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_overlap', save_full_object=True)
            ## END with h5py.File(path, 'w') as f...

            restored = SingleEpochDecodedResult.read_hdf(path, key='epoch_overlap')
            np.testing.assert_allclose(restored.p_x_given_n, obj.p_x_given_n)
            np.testing.assert_allclose(restored.time_bin_edges, obj.time_bin_edges)
            self.assertIsNotNone(restored.time_bin_container)
            self.assertEqual(len(restored.time_bin_container.centers), obj.nbins)
            self.assertIsNotNone(restored.time_bin_container.window_start_edges)
            self.assertIsNotNone(restored.time_bin_container.window_stop_edges)
            np.testing.assert_allclose(restored.time_bin_container.window_start_edges, obj.time_bin_edges[:, 0])
            np.testing.assert_allclose(restored.time_bin_container.window_stop_edges, obj.time_bin_edges[:, 1])
        ## END with tempfile.TemporaryDirectory() as td...


    def test_full_object_without_spkcount(self):
        obj = _build_1d_single_epoch_result(include_spkcount=False)
        self.assertIsNone(obj.spkcount)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('no_spkcount.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, 'epoch_0', save_full_object=True)
                self.assertNotIn('spkcount', f['epoch_0'])
            ## END with h5py.File(path, 'w') as f...

            restored = SingleEpochDecodedResult.read_hdf(path, key='epoch_0')
            self.assertIsNone(restored.spkcount)
            np.testing.assert_allclose(restored.p_x_given_n, obj.p_x_given_n)
        ## END with tempfile.TemporaryDirectory() as td...


if __name__ == '__main__':
    unittest.main()
