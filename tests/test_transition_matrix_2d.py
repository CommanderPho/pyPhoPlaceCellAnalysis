"""Tests for 2D (x, y) position transition helpers and ``TransitionMatrixComputations._compute_position_transition_matrix_2d``."""

import unittest
import sys
from pathlib import Path

import numpy as np

tests_folder = Path(__file__).parent.resolve()
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath("src")
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import (
    TransitionMatrixComputations,
    position_flat_index_from_xy,
    position_unravel_flat,
    reshape_square_tm_to_grid,
)


class TestTransitionMatrix2D(unittest.TestCase):

    def test_flat_index_roundtrip(self):
        n_x, n_y = 3, 4
        for ix in range(n_x):
            for iy in range(n_y):
                flat = position_flat_index_from_xy(ix, iy, n_y)
                self.assertEqual(flat.shape, ())
                ix_b, iy_b = position_unravel_flat(int(np.asarray(flat).item()), n_x, n_y)
                self.assertEqual(int(np.asarray(ix_b).item()), ix)
                self.assertEqual(int(np.asarray(iy_b).item()), iy)

    def test_flat_index_vectorized(self):
        n_y = 2
        ix = np.array([0, 1, 0])
        iy = np.array([0, 0, 1])
        flat = position_flat_index_from_xy(ix, iy, n_y)
        np.testing.assert_array_equal(flat, np.array([0, 2, 1]))

    def test_reshape_square_tm_to_grid(self):
        n_x, n_y = 2, 2
        T = np.arange(16, dtype=float).reshape(4, 4)
        G = reshape_square_tm_to_grid(T, n_x, n_y)
        self.assertEqual(G.shape, (2, 2, 2, 2))
        self.assertAlmostEqual(float(G[0, 0, 1, 0]), float(T[0, 2]))

    def test_compute_position_transition_matrix_2d_single_transition(self):
        x_labels = np.array([1, 2])
        y_labels = np.array([10, 20])
        bx = np.array([0, 1], dtype=int)
        by = np.array([0, 0], dtype=int)
        mats = TransitionMatrixComputations._compute_position_transition_matrix_2d(x_labels, y_labels, bx, by, n_powers=2, use_direct_observations_for_order=True, should_validate_normalization=True)
        self.assertEqual(len(mats), 2)
        t0 = mats[0]
        self.assertEqual(t0.shape, (4, 4))
        row0 = t0[0, :]
        self.assertAlmostEqual(float(row0[2]), 1.0)
        self.assertAlmostEqual(float(row0.sum()), 1.0)
        _rs = np.sum(t0, axis=1)
        self.assertTrue(np.allclose(_rs[np.nonzero(_rs)], 1.0))


if __name__ == "__main__":
    unittest.main()
