"""Cell-identity shuffle indexes `neuron_IDs`, not the longer ratemap."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

import neuropy.core.position  # finish neuropy init before the computation-function package import

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent

try:
    import pyphoplacecellanalysis
except ModuleNotFoundError:
    sys.path.insert(0, str(root_project_folder.joinpath('src')))
finally:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle


class _Ratemap:
    def __init__(self, neuron_ids):
        self.neuron_ids = np.asarray(neuron_ids)


class _Pf:
    def __init__(self, neuron_ids):
        self.ratemap = _Ratemap(neuron_ids)


class _Decoder:
    """Enough of a decoder for `WCorrShuffle._shuffle_pf1D_decoder`."""
    def __init__(self, neuron_IDs, ratemap_ids, n_pos: int = 4):
        self.neuron_IDs = np.asarray(neuron_IDs)
        self.neuron_IDXs = np.arange(len(self.neuron_IDs))
        self.F = np.arange(n_pos * len(self.neuron_IDs), dtype=float).reshape(n_pos, len(self.neuron_IDs))
        self.pf = _Pf(ratemap_ids)


class TestWCorrShuffleDecoderIndexing(unittest.TestCase):
    """The reported failure was `index 41 is out of bounds for axis 0 with size 33`."""

    def test_ratemap_longer_than_neuron_IDs_does_not_index_past_neuron_IDs(self):
        ratemap_ids = np.arange(42) # position 41 exists
        neuron_IDs = ratemap_ids[:33]
        self.assertEqual(len(neuron_IDs), 33)
        self.assertEqual(len(ratemap_ids), 42)
        self.assertGreaterEqual(int(np.where(ratemap_ids == 41)[0][0]), len(neuron_IDs))

        rng = np.random.default_rng(0)
        shuffle_aclus = rng.permutation(ratemap_ids)
        decoder = _Decoder(neuron_IDs, ratemap_ids)
        original_F = decoder.F.copy()

        shuffled = WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.arange(len(ratemap_ids)), shuffle_aclus=shuffle_aclus)

        expected_order = shuffle_aclus[np.isin(shuffle_aclus, neuron_IDs)]
        np.testing.assert_array_equal(shuffled.neuron_IDs, expected_order)
        self.assertTrue(np.all(shuffled.neuron_IDXs == np.arange(len(neuron_IDs))))
        np.testing.assert_array_equal(shuffled.F, original_F)
        np.testing.assert_array_equal(shuffled.pf.ratemap.neuron_ids, ratemap_ids)
        self.assertEqual(shuffled.F.shape[1], len(shuffled.neuron_IDs))


    def test_matched_lengths_permute_neuron_IDs_and_leave_F_columns(self):
        neuron_IDs = np.array([10, 20, 30, 40])
        shuffle_aclus = np.array([30, 10, 40, 20])
        decoder = _Decoder(neuron_IDs, neuron_IDs)
        original_F = decoder.F.copy()

        shuffled = WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.arange(len(neuron_IDs)), shuffle_aclus=shuffle_aclus)

        np.testing.assert_array_equal(shuffled.neuron_IDs, shuffle_aclus)
        np.testing.assert_array_equal(shuffled.F, original_F)
        self.assertFalse(np.array_equal(shuffled.neuron_IDs, neuron_IDs))


    def test_neurons_missing_from_shuffle_keep_original_relative_order(self):
        neuron_IDs = np.array([1, 2, 3, 4])
        shuffle_aclus = np.array([3, 1]) # 2 and 4 are absent
        decoder = _Decoder(neuron_IDs, neuron_IDs)

        shuffled = WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.array([2, 0]), shuffle_aclus=shuffle_aclus)

        np.testing.assert_array_equal(shuffled.neuron_IDs, np.array([3, 1, 2, 4]))
        self.assertEqual(shuffled.F.shape[1], 4)


    def test_duplicate_neuron_IDs_and_repeated_shuffle_aclus_are_rejected(self):
        decoder = _Decoder(np.array([10, 10, 20]), np.array([10, 20]))
        with self.assertRaises(ValueError):
            WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.arange(2), shuffle_aclus=np.array([20, 10]))

        decoder = _Decoder(np.array([10, 20, 30]), np.array([10, 20, 30]))
        original_ids = decoder.neuron_IDs.copy()
        with self.assertRaises(ValueError):
            WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.arange(3), shuffle_aclus=np.array([30, 10, 30, 20]))
        np.testing.assert_array_equal(decoder.neuron_IDs, original_ids)


    def test_F_neuron_axis_must_match_neuron_IDs(self):
        decoder = _Decoder(np.array([1, 2, 3]), np.arange(5))
        decoder.F = np.zeros((4, 5))
        with self.assertRaises(ValueError):
            WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.arange(5), shuffle_aclus=np.array([3, 1, 2]))


    def test_int32_shuffle_aclus_match_int64_neuron_IDs(self):
        neuron_IDs = np.array([10, 20, 30], dtype=np.int64)
        shuffle_aclus = np.array([30, 10, 20], dtype=np.int32)
        decoder = _Decoder(neuron_IDs, neuron_IDs)
        shuffled = WCorrShuffle._shuffle_pf1D_decoder(decoder, shuffle_IDXs=np.arange(3), shuffle_aclus=shuffle_aclus)
        np.testing.assert_array_equal(shuffled.neuron_IDs, np.array([30, 10, 20]))
