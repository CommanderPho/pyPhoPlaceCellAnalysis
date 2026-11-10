"""Unit tests for DecodedFilterEpochsResult HDF5 round-trip (`to_hdf` / `read_hdf`)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
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
    import pandas as pd
    # Import reconstruction first so NeuroPy circular-import chains resolve via the package load order.
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
    from neuropy.utils.dynamic_container import DynamicContainer
    from neuropy.utils.mixins.binning_helpers import BinningContainer


def _build_normalized_posterior(n_x: int, n_t: int, rng: np.random.Generator) -> np.ndarray:
    p = rng.random((n_x, n_t)).astype(float)
    p = p / np.sum(p, axis=0, keepdims=True)
    return p


def _build_multi_epoch_1d_result(n_x: int = 5, epoch_n_ts=(3, 5, 2), decoding_time_bin_size: float = 0.05, decoding_slideby: float = 0.025, seed: int = 0) -> DecodedFilterEpochsResult:
    rng = np.random.default_rng(seed)
    n_epochs = len(epoch_n_ts)

    most_likely_positions_list = []
    p_x_given_n_list = []
    marginal_x_list = []
    marginal_y_list = []
    marginal_z_list = []
    most_likely_position_indicies_list = []
    spkcount_list = []
    time_bin_containers = []
    time_bin_edges_list = []
    nbins = np.asarray(list(epoch_n_ts), dtype=np.int_)
    epoch_rows = []
    t_cursor = 10.0

    for epoch_idx, n_t in enumerate(epoch_n_ts):
        p = _build_normalized_posterior(n_x=n_x, n_t=int(n_t), rng=rng)
        edges = np.linspace(t_cursor, t_cursor + float(n_t) * decoding_time_bin_size, int(n_t) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        tbc = BinningContainer(edges=edges, centers=centers)
        mlp = np.full(int(n_t), 1.5 + 0.1 * epoch_idx, dtype=float)
        mlpi = np.zeros(int(n_t), dtype=int)
        marginal_x = DynamicContainer(p_x_given_n=p.copy(), most_likely_positions_1D=mlp.copy())

        most_likely_positions_list.append(mlp)
        p_x_given_n_list.append(p)
        marginal_x_list.append(marginal_x)
        marginal_y_list.append(None)
        marginal_z_list.append(None)
        most_likely_position_indicies_list.append(mlpi)
        spkcount_list.append(np.arange(int(n_t), dtype=float) + epoch_idx)
        time_bin_containers.append(tbc)
        time_bin_edges_list.append(edges)
        epoch_rows.append({'start': float(edges[0]), 'stop': float(edges[-1]), 'label': str(epoch_idx), 'duration': float(edges[-1] - edges[0])})
        t_cursor = float(edges[-1]) + 0.5
    ## END for epoch_idx, n_t in enumerate(epoch_n_ts)...

    filter_epochs = pd.DataFrame(epoch_rows)
    pos_bin_edges = np.linspace(0.0, 10.0, n_x + 1)
    epoch_description_list = [f'epoch_{i}' for i in range(n_epochs)]

    return DecodedFilterEpochsResult(
        decoding_time_bin_size=decoding_time_bin_size,
        decoding_slideby=decoding_slideby,
        filter_epochs=filter_epochs,
        num_filter_epochs=n_epochs,
        most_likely_positions_list=most_likely_positions_list,
        p_x_given_n_list=p_x_given_n_list,
        marginal_x_list=marginal_x_list,
        marginal_y_list=marginal_y_list,
        marginal_z_list=marginal_z_list,
        most_likely_position_indicies_list=most_likely_position_indicies_list,
        spkcount=spkcount_list,
        nbins=nbins,
        time_bin_containers=time_bin_containers,
        time_bin_edges=time_bin_edges_list,
        epoch_description_list=epoch_description_list,
        pos_bin_edges=pos_bin_edges,
    )


class TestDecodedFilterEpochsResultHDFRoundtrip(unittest.TestCase):

    def test_full_object_roundtrip_1d_multi_epoch(self):
        obj = _build_multi_epoch_1d_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('decoded_epochs.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, key='decoded_epochs', save_full_object=True)
                group = f['decoded_epochs']
                self.assertTrue(bool(group.attrs['save_full_object']))
                self.assertEqual(int(group.attrs['num_filter_epochs']), obj.num_filter_epochs)
                self.assertAlmostEqual(float(group.attrs['decoding_time_bin_size']), obj.decoding_time_bin_size)
                self.assertAlmostEqual(float(group.attrs['decoding_slideby']), obj.decoding_slideby)
                self.assertIn('nbins', group)
                self.assertIn('pos_bin_edges', group)
                self.assertIn('epoch_description_list', group)
                self.assertIn('filter_epochs', group)
                self.assertIn('epochs', group)
                self.assertEqual(len(group['epochs'].keys()), obj.num_filter_epochs)
                # full-object fields present on first epoch
                first_epoch_name = sorted(group['epochs'].keys())[0]
                self.assertIn('p_x_given_n_full', group['epochs'][first_epoch_name])
                self.assertIn('spkcount', group['epochs'][first_epoch_name])
            ## END with h5py.File(path, 'w') as f...

            restored = DecodedFilterEpochsResult.read_hdf(path, key='decoded_epochs')
            self.assertEqual(restored.num_filter_epochs, obj.num_filter_epochs)
            self.assertAlmostEqual(restored.decoding_time_bin_size, obj.decoding_time_bin_size)
            self.assertAlmostEqual(restored.decoding_slideby, obj.decoding_slideby)
            np.testing.assert_array_equal(restored.nbins, obj.nbins)
            np.testing.assert_allclose(restored.pos_bin_edges, obj.pos_bin_edges)
            self.assertEqual(list(restored.epoch_description_list), list(obj.epoch_description_list))
            self.assertEqual(len(restored.p_x_given_n_list), obj.num_filter_epochs)

            for i in range(obj.num_filter_epochs):
                np.testing.assert_allclose(restored.p_x_given_n_list[i], obj.p_x_given_n_list[i])
                np.testing.assert_allclose(restored.most_likely_positions_list[i], obj.most_likely_positions_list[i])
                np.testing.assert_array_equal(restored.most_likely_position_indicies_list[i], obj.most_likely_position_indicies_list[i])
                np.testing.assert_allclose(restored.time_bin_edges[i], obj.time_bin_edges[i])
                np.testing.assert_allclose(restored.spkcount[i], obj.spkcount[i])
                self.assertIsNotNone(restored.time_bin_containers[i])
                self.assertEqual(len(restored.time_bin_containers[i].centers), int(obj.nbins[i]))
                self.assertIsNotNone(restored.marginal_x_list[i])
            ## END for i in range(obj.num_filter_epochs)...

            self.assertIn('start', restored.filter_epochs.columns)
            self.assertIn('stop', restored.filter_epochs.columns)
            np.testing.assert_allclose(restored.filter_epochs['start'].to_numpy(dtype=float), obj.filter_epochs['start'].to_numpy(dtype=float))
            np.testing.assert_allclose(restored.filter_epochs['stop'].to_numpy(dtype=float), obj.filter_epochs['stop'].to_numpy(dtype=float))
        ## END with tempfile.TemporaryDirectory() as td...


    def test_legacy_mode_omits_full_object_epoch_fields(self):
        obj = _build_multi_epoch_1d_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('legacy_decoded_epochs.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, key='decoded_epochs', save_full_object=False)
                group = f['decoded_epochs']
                self.assertFalse(bool(group.attrs.get('save_full_object', False)))
                first_epoch_name = sorted(group['epochs'].keys())[0]
                epoch_group = group['epochs'][first_epoch_name]
                self.assertIn('p_x_given_n', epoch_group)
                self.assertNotIn('p_x_given_n_full', epoch_group)
                self.assertNotIn('spkcount', epoch_group)
            ## END with h5py.File(path, 'w') as f...

            restored = DecodedFilterEpochsResult.read_hdf(path, key='decoded_epochs')
            self.assertEqual(restored.num_filter_epochs, obj.num_filter_epochs)
            # 1D legacy still recovers posterior via marginal export
            for i in range(obj.num_filter_epochs):
                np.testing.assert_allclose(restored.p_x_given_n_list[i], obj.p_x_given_n_list[i])
                self.assertIsNone(restored.spkcount[i])
            ## END for i in range(obj.num_filter_epochs)...
        ## END with tempfile.TemporaryDirectory() as td...


    def test_read_hdf_from_open_file_handle(self):
        obj = _build_multi_epoch_1d_result(epoch_n_ts=(2, 4))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('open_handle_decoded.h5')
            with h5py.File(path, 'w') as f:
                obj.to_hdf(f, key='decoded_epochs', save_full_object=True)
            ## END with h5py.File(path, 'w') as f...

            with h5py.File(path, 'r') as f:
                restored = DecodedFilterEpochsResult.read_hdf(f, key='decoded_epochs')
                self.assertEqual(restored.num_filter_epochs, obj.num_filter_epochs)
                np.testing.assert_allclose(restored.p_x_given_n_list[0], obj.p_x_given_n_list[0])
            ## END with h5py.File(path, 'r') as f...
        ## END with tempfile.TemporaryDirectory() as td...


    def test_to_hdf_path_argument_roundtrip(self):
        obj = _build_multi_epoch_1d_result(epoch_n_ts=(3, 3))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('path_arg_decoded.h5')
            obj.to_hdf(path, key='decoded_epochs', save_full_object=True, file_mode='w')
            restored = DecodedFilterEpochsResult.read_hdf(path, key='decoded_epochs')
            self.assertEqual(restored.num_filter_epochs, obj.num_filter_epochs)
            np.testing.assert_array_equal(restored.nbins, obj.nbins)
            np.testing.assert_allclose(restored.p_x_given_n_list[1], obj.p_x_given_n_list[1])
        ## END with tempfile.TemporaryDirectory() as td...


    def test_get_result_for_epoch_after_roundtrip(self):
        obj = _build_multi_epoch_1d_result()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td).joinpath('get_epoch_after.h5')
            obj.to_hdf(path, key='decoded_epochs', save_full_object=True, file_mode='w')
            restored = DecodedFilterEpochsResult.read_hdf(path, key='decoded_epochs')
            single: SingleEpochDecodedResult = restored.get_result_for_epoch(1)
            self.assertEqual(single.epoch_data_index, 1)
            np.testing.assert_allclose(single.p_x_given_n, obj.p_x_given_n_list[1])
            self.assertEqual(single.nbins, int(obj.nbins[1]))
        ## END with tempfile.TemporaryDirectory() as td...


if __name__ == '__main__':
    unittest.main()
