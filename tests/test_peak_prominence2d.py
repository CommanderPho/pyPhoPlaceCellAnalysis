"""Tests for peak_prominence2d: compute_2d_dt_posterior_peak_promenences return contract and caller compatibility."""
import os
import sys
from pathlib import Path
import unittest
import numpy as np

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
            PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9], memory_warn_bytes=1)

    def test_memory_strict_raises(self):
        """When memory_strict=True and estimate exceeds memory_warn_bytes, MemoryError is raised."""
        rng = np.random.default_rng(101)
        n_x, n_y, n_t = 4, 4, 10
        a_p_x_given_n = np.abs(rng.standard_normal((n_x, n_y, n_t)))
        with self.assertRaises(MemoryError):
            PeakPromenence.compute_2d_dt_posterior_peak_promenences(a_p_x_given_n=a_p_x_given_n, alpha=[0.9], memory_warn_bytes=1, memory_strict=True)


if __name__ == '__main__':
    unittest.main()
