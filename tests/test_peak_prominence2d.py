"""Tests for peak_prominence2d: compute_2d_dt_posterior_peak_promenences return contract and caller compatibility."""
import os
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

from pyphoplacecellanalysis.External.peak_prominence2d import PeakPromenence


class TestCompute2dDtPosteriorPeakPromenences(unittest.TestCase):
    """Verify return type, shape, and caller-style usage of compute_2d_dt_posterior_peak_promenences."""

    def test_return_shape_and_type(self):
        """Return is (epoch_promenence_tuples, epoch_masks); masks are List[NDArray] each (x, y, t)."""
        rng = np.random.default_rng(42)
        n_x, n_y, n_t = 8, 10, 5
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_y, n_t)))
        alpha_list = [0.5, 0.9]
        epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list)
        self.assertIsInstance(epoch_masks_list, list)
        self.assertEqual(len(epoch_masks_list), len(alpha_list))
        for m in epoch_masks_list:
            self.assertIsInstance(m, np.ndarray)
            self.assertEqual(m.shape, (n_x, n_y, n_t))
            self.assertEqual(m.dtype, np.dtype(bool))
        self.assertEqual(len(epoch_promenence_tuples), n_t)

    def test_caller_compat_dict_zip_and_shape_assert(self):
        """Caller pattern: dict(zip(alpha_list, epoch_masks_list)) and shape == a_p_x_given_n.shape."""
        rng = np.random.default_rng(123)
        n_x, n_y, n_t = 6, 8, 4
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_y, n_t)))
        alpha_list = [0.8]
        epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list)
        epoch_masks_dict = dict(zip(alpha_list, epoch_masks_list))
        a_high_alpha = alpha_list[-1]
        an_alpha_epoch_masks = epoch_masks_dict[a_high_alpha]
        self.assertEqual(np.shape(an_alpha_epoch_masks), np.shape(a_p_x_given_n))

    def test_caller_compat_nansum_axis_01(self):
        """Caller pattern: np.nansum(an_alpha_epoch_masks, axis=(0, 1)) is well-defined and matches shape (n_t,)."""
        rng = np.random.default_rng(456)
        n_x, n_y, n_t = 5, 7, 3
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_y, n_t)))
        alpha_list = [0.9]
        _, epoch_masks_list = PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list)
        an_alpha_epoch_masks = epoch_masks_list[0]
        s = np.nansum(an_alpha_epoch_masks, axis=(0, 1))
        self.assertEqual(s.shape, (n_t,))
        self.assertTrue(np.issubdtype(s.dtype, np.integer))

    def test_memory_warn_emits_warning(self):
        """When memory_warn_bytes is set and estimate exceeds it, a warning is emitted (or MemoryError if strict)."""
        rng = np.random.default_rng(789)
        n_x, n_y, n_t = 4, 4, 10
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_y, n_t)))
        with self.assertWarns(UserWarning):
            PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9], memory_warn_bytes=1, memory_strict=False)

    def test_memory_strict_raises(self):
        """When memory_strict=True and estimate exceeds memory_warn_bytes, MemoryError is raised."""
        rng = np.random.default_rng(101)
        n_x, n_y, n_t = 4, 4, 10
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_y, n_t)))
        with self.assertRaises(MemoryError):
            PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9], memory_warn_bytes=1, memory_strict=True)


class TestCompute1dPeakProminence(unittest.TestCase):
    """Verify SciPy-backed 1D peak prominence core and dt/multi-epoch contracts."""

    def test_known_peaks_match_scipy(self):
        """Two-Gaussian curve: peak indices and prominences match SciPy reference."""
        from scipy.signal import find_peaks, peak_prominences
        x = np.linspace(0, 10, 201)
        Z_1d = np.exp(-0.5 * ((x - 3.0) / 0.4) ** 2) + 0.6 * np.exp(-0.5 * ((x - 7.0) / 0.5) ** 2)
        peak_coords, prominences = PeakPromenence.compute_1d_peak_prominence(Z_1d)
        expected_peaks, _ = find_peaks(Z_1d)
        expected_prominences, _, _ = peak_prominences(Z_1d, expected_peaks)
        self.assertEqual(peak_coords.ndim, 2)
        self.assertEqual(peak_coords.shape[1], 1)
        np.testing.assert_array_equal(peak_coords[:, 0], expected_peaks)
        np.testing.assert_allclose(prominences, expected_prominences)
        self.assertGreaterEqual(len(peak_coords), 2)

    def test_rejects_non_1d(self):
        with self.assertRaises(ValueError):
            PeakPromenence.compute_1d_peak_prominence(np.ones((4, 5)))

    def test_flat_signal_empty_peaks(self):
        peak_coords, prominences = PeakPromenence.compute_1d_peak_prominence(np.ones(20))
        self.assertEqual(peak_coords.shape, (0, 1))
        self.assertEqual(prominences.shape, (0,))


class TestCompute1dDtPosteriorPeakPromenences(unittest.TestCase):
    """Verify return type, shape, and caller-style usage of compute_1d_dt_posterior_peak_promenences."""

    def test_return_shape_and_type(self):
        """Return is (epoch_promenence_tuples, epoch_masks); masks are List[NDArray] each (x, t)."""
        rng = np.random.default_rng(42)
        n_x, n_t = 32, 5
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_t)))
        alpha_list = [0.5, 0.9]
        epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_1d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list)
        self.assertIsInstance(epoch_masks_list, list)
        self.assertEqual(len(epoch_masks_list), len(alpha_list))
        for m in epoch_masks_list:
            self.assertIsInstance(m, np.ndarray)
            self.assertEqual(m.shape, (n_x, n_t))
            self.assertEqual(m.dtype, np.dtype(bool))
        self.assertEqual(len(epoch_promenence_tuples), n_t)
        for peak_coords, prominences, peak_heights in epoch_promenence_tuples:
            self.assertEqual(peak_coords.ndim, 2)
            self.assertEqual(peak_coords.shape[1], 1)
            self.assertEqual(len(prominences), peak_coords.shape[0])
            self.assertEqual(len(peak_heights), peak_coords.shape[0])

    def test_caller_compat_dict_zip_and_shape_assert(self):
        """Caller pattern: dict(zip(alpha_list, epoch_masks_list)) and shape == a_p_x_given_n.shape."""
        rng = np.random.default_rng(123)
        n_x, n_t = 24, 4
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_t)))
        alpha_list = [0.8]
        epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_1d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=alpha_list)
        epoch_masks_dict = dict(zip(alpha_list, epoch_masks_list))
        a_high_alpha = alpha_list[-1]
        an_alpha_epoch_masks = epoch_masks_dict[a_high_alpha]
        self.assertEqual(np.shape(an_alpha_epoch_masks), np.shape(a_p_x_given_n))

    def test_empty_peak_time_bin(self):
        """Flat time bin yields empty peak arrays; other bins still processed."""
        n_x, n_t = 20, 3
        a_p_x_given_n = np.zeros((n_x, n_t), dtype=float)
        a_p_x_given_n[5, 1] = 1.0
        a_p_x_given_n[4, 1] = 0.5
        a_p_x_given_n[6, 1] = 0.5
        epoch_promenence_tuples, epoch_masks_list = PeakPromenence.compute_1d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9])
        peak_coords_0, prominences_0, peak_heights_0 = epoch_promenence_tuples[0]
        self.assertEqual(peak_coords_0.shape, (0, 1))
        self.assertEqual(prominences_0.shape, (0,))
        self.assertEqual(peak_heights_0.shape, (0,))
        peak_coords_1, _, _ = epoch_promenence_tuples[1]
        self.assertGreaterEqual(peak_coords_1.shape[0], 1)
        self.assertTrue(np.any(epoch_masks_list[0][:, 1]))

    def test_memory_warn_emits_warning(self):
        rng = np.random.default_rng(789)
        n_x, n_t = 16, 10
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_t)))
        with self.assertWarns(UserWarning):
            PeakPromenence.compute_1d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9], memory_warn_bytes=1, memory_strict=False)

    def test_memory_strict_raises(self):
        rng = np.random.default_rng(101)
        n_x, n_t = 16, 10
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_t)))
        with self.assertRaises(MemoryError):
            PeakPromenence.compute_1d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9], memory_warn_bytes=1, memory_strict=True)

    def test_multi_epoch_wrapper(self):
        rng = np.random.default_rng(202)
        p_x_given_n_list = [np.abs(rng.standard_normal((20, 3))), np.abs(rng.standard_normal((20, 4)))]
        peak_prominence_df, idx_list, tuples_dict, all_masks = PeakPromenence.compute_1d_posterior_peak_promenences(p_x_given_n_list=p_x_given_n_list, alpha=0.9)
        self.assertEqual(len(idx_list), 3 + 4)
        self.assertEqual(len(tuples_dict), 3 + 4)
        self.assertEqual(len(all_masks), 2)
        self.assertEqual(all_masks[0][0].shape, (20, 3))
        self.assertEqual(all_masks[1][0].shape, (20, 4))
        self.assertIn((0, 0), tuples_dict)
        self.assertIn((1, 3), tuples_dict)
        self.assertIsInstance(peak_prominence_df, pd.DataFrame)
        self.assertTrue({'neuron_IDX', 'time_bin_idx', 'summit_idx', 'peak_prominence', 'peak_height', 'peak_center_x', 'peak_center_binned_x'}.issubset(peak_prominence_df.columns))

    def test_build_df_optional_ids_and_xbins(self):
        """neuron_IDs/xbin_centers optional: neuron_IDX=IDX, peak_center_x=bin index; summit_idx ranks by height."""
        # Shorter peak first spatially (0.8), taller second (1.0) — summit_idx must reorder by height
        Z = np.array([0.0, 0.3, 0.8, 0.3, 0.0, 0.2, 1.0, 0.2, 0.0])
        peak_coords, prominences = PeakPromenence.compute_1d_peak_prominence(Z)
        peak_heights = Z[peak_coords[:, 0]]
        self.assertLess(peak_heights[0], peak_heights[1])  # spatial order: shorter then taller
        tuples_dict = {(0, 0): (peak_coords, prominences, peak_heights)}
        df = PeakPromenence._build_1d_peak_prominence_df(tuples_dict)
        self.assertEqual(df.loc[0, 'neuron_IDX'], 0)
        np.testing.assert_array_equal(df['peak_center_x'].to_numpy(), df['peak_center_binned_x'].to_numpy().astype(float))
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, 'summit_idx'], 0)
        self.assertEqual(df.loc[1, 'summit_idx'], 1)
        self.assertGreater(df.loc[0, 'peak_height'], df.loc[1, 'peak_height'])
        np.testing.assert_allclose(df['peak_height'].to_numpy(), np.sort(peak_heights)[::-1])
        self.assertEqual(df.loc[0, 'peak_center_binned_x'], 6)  # taller peak bin
        self.assertEqual(df.loc[1, 'peak_center_binned_x'], 2)  # shorter peak bin
        neuron_IDs = np.array([42])
        xbin_centers = np.arange(len(Z), dtype=float) * 2.0
        df2 = PeakPromenence._build_1d_peak_prominence_df(tuples_dict, neuron_IDs=neuron_IDs, xbin_centers=xbin_centers)
        self.assertEqual(df2.loc[0, 'aclu'], 42)
        np.testing.assert_allclose(df2['peak_center_x'].to_numpy(), xbin_centers[df2['peak_center_binned_x'].to_numpy()])


if __name__ == '__main__':
    unittest.main()
