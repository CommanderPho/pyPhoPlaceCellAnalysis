"""1D tests for SparseGaussianVolume + DecodingLocalityMeasures.

Covers 1D peaked vs diffuse metrics, shape dispatch vs 2D path, and
n_total_pos_bins / mask_overlap normalization. Uses the same AST loader as
test_decoding_locality_measures_2d.py.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

import importlib.util as _ilu  # noqa: E402

_spec_2d = _ilu.spec_from_file_location('test_decoding_locality_measures_2d', tests_folder / 'test_decoding_locality_measures_2d.py')
_mod_2d = _ilu.module_from_spec(_spec_2d)
assert _spec_2d.loader is not None
_spec_2d.loader.exec_module(_mod_2d)
_load_sparse_and_decoding_locality = _mod_2d._load_sparse_and_decoding_locality


N_X, N_T, N_CONTEXTS = 20, 8, 2
EPOCH_NAMES = ['roam', 'sprinkle']


def _centers_edges(n: int, lo: float, hi: float):
    edges = np.linspace(lo, hi, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges


def _renorm_x(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=float).copy()
    if out.ndim == 2:
        s = np.nansum(out, axis=0, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        out = out / s
    elif out.ndim == 3:
        s = np.nansum(out, axis=0, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        out = out / s
    return out


def _make_peaked_posterior_1d(xbin_centers, peak_x_per_t, sigma_bins: float = 1.2) -> np.ndarray:
    n_t = len(peak_x_per_t)
    out = np.zeros((len(xbin_centers), n_t), dtype=float)
    x_step = float(np.mean(np.diff(xbin_centers)))
    for t, px in enumerate(peak_x_per_t):
        dist_sq = ((xbin_centers - px) / x_step) ** 2
        out[:, t] = np.exp(-0.5 * dist_sq / (sigma_bins ** 2))
    ## END for t, px in enumerate(peak_x_per_t)...
    return _renorm_x(out)


def _make_diffuse_posterior_1d(n_x, n_t) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = 0.85 + 0.15 * rng.random((n_x, n_t))
    return _renorm_x(out)


def _build_fixture_bundle_1d():
    xbin_centers, xbin = _centers_edges(N_X, 0.0, 200.0)
    t_centers = np.arange(N_T, dtype=float) * 0.25

    peak0 = [xbin_centers[2 + (t % 10)] for t in range(N_T)]
    peak1 = [xbin_centers[-(3 + (t % 8))] for t in range(N_T)]
    p0 = _make_peaked_posterior_1d(xbin_centers, peak0)
    p1 = _make_peaked_posterior_1d(xbin_centers, peak1)
    p_stack = np.stack([p0, p1], axis=1)  # (nx, 2, nt)
    p_diffuse = _make_diffuse_posterior_1d(N_X, N_T)

    new_positions = np.asarray(peak0, dtype=float)[:, np.newaxis]  # (n_t, 1)
    opposite = np.asarray(peak1, dtype=float)[:, np.newaxis]
    sigma = 2.0 * float(np.mean(np.diff(xbin)))

    pos_df = pd.DataFrame({
        't': t_centers,
        'x': new_positions[:, 0],
    })
    paradigm_epochs_df = pd.DataFrame({
        'start': [float(t_centers[0]) - 0.01, float(t_centers[N_T // 2])],
        'stop': [float(t_centers[N_T // 2]), float(t_centers[-1]) + 0.01],
        'label': EPOCH_NAMES,
    })
    return {
        'xbin_centers': xbin_centers, 'xbin': xbin, 't_centers': t_centers,
        'p0': p0, 'p1': p1, 'p_stack': p_stack, 'p_diffuse': p_diffuse,
        'new_positions': new_positions, 'opposite_positions': opposite,
        'sigma': sigma, 'pos_df': pos_df, 'paradigm_epochs_df': paradigm_epochs_df,
    }


class TestSparseGaussianVolume1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SparseGaussianVolume, cls.DecodingLocalityMeasures = _load_sparse_and_decoding_locality()
        cls.fx = _build_fixture_bundle_1d()

    def test_build_shape_and_peak_near_measured(self):
        fx = self.fx
        vol = self.SparseGaussianVolume.build_from_positions_1d(fx['xbin_centers'], fx['new_positions'], fx['sigma'])
        self.assertEqual(vol.spatial_ndim, 1)
        self.assertEqual(vol.shape, (N_X, N_T))
        dense = vol.to_dense()
        for t in range(N_T):
            ix = int(np.argmax(dense[:, t]))
            true_ix = int(np.argmin(np.abs(fx['xbin_centers'] - fx['new_positions'][t, 0])))
            self.assertLessEqual(abs(ix - true_ix), 2)
        ## END for t in range(N_T)...

    def test_marginal_sums_1d(self):
        fx = self.fx
        vol = self.SparseGaussianVolume.build_from_positions_1d(fx['xbin_centers'], fx['new_positions'], fx['sigma'])
        marg_x, marg_y = vol.marginal_sums()
        self.assertEqual(marg_x.shape, (N_X, N_T))
        self.assertIsNone(marg_y)
        dense = vol.to_dense()
        self.assertTrue(np.allclose(marg_x, dense))

    def test_multiply_mask_ops_1d(self):
        fx = self.fx
        vol = self.SparseGaussianVolume.build_from_positions_1d(fx['xbin_centers'], fx['new_positions'], fx['sigma'])
        ones = np.ones(vol.shape, dtype=float)
        zeros = np.zeros(vol.shape, dtype=float)
        int_ones = vol.multiply_mask_threshold_to_int(ones, 1e-9)
        int_zeros = vol.multiply_mask_threshold_to_int(zeros, 1e-9)
        self.assertEqual(int_ones.shape, vol.shape)
        self.assertTrue(np.all(int_zeros == 0))
        sum_spatial = vol.multiply_mask_threshold_sum_spatial(ones, 1e-9)
        self.assertEqual(sum_spatial.shape, (N_T,))
        self.assertTrue(np.allclose(sum_spatial, np.nansum(int_ones, axis=0)))


class TestDecodingLocalityMeasures1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SparseGaussianVolume, cls.DecodingLocalityMeasures = _load_sparse_and_decoding_locality()
        cls.fx = _build_fixture_bundle_1d()
        fx = cls.fx
        cls.vol_colocated = cls.SparseGaussianVolume.build_from_positions_1d(fx['xbin_centers'], fx['new_positions'], fx['sigma'])
        cls.vol_offset = cls.SparseGaussianVolume.build_from_positions_1d(fx['xbin_centers'], fx['opposite_positions'], fx['sigma'])
        cls._peaked = cls.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=None,
            gaussian_volume=cls.vol_colocated, alpha_list=[0.8], enable_debug_outputs=True, debug_print=False, spatial_ndim=1,
        )
        cls._diffuse = cls.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p_diffuse'], xbin_centers=fx['xbin_centers'], ybin_centers=None,
            gaussian_volume=cls.vol_colocated, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False, spatial_ndim=1,
        )

    def test_perform_build_normalized_outputs_1d(self):
        fx = self.fx
        out = self.DecodingLocalityMeasures.perform_build_normalized_outputs(fx['p_stack'], EPOCH_NAMES, spatial_ndim=1)
        self.assertEqual(set(out.keys()), set(EPOCH_NAMES))
        for i, name in enumerate(EPOCH_NAMES):
            self.assertEqual(out[name].shape, (N_X, N_T))
            sums = np.nansum(out[name], axis=0)
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-6))
        ## END for i, name in enumerate(EPOCH_NAMES)...

        # Single-context (n_x, n_t) wrap
        out_single = self.DecodingLocalityMeasures.perform_build_normalized_outputs(fx['p0'], ['only'], spatial_ndim=1)
        self.assertEqual(list(out_single.keys()), ['only'])
        self.assertEqual(out_single['only'].shape, (N_X, N_T))

    def test_shape_dispatch_1d_vs_2d(self):
        fx = self.fx
        # Explicit (n_x, n_t) → 1D (ybin_centers None)
        res_1d = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=None,
            gaussian_volume=None, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False,
        )
        self.assertEqual(res_1d['peak_prom_Focality'].shape, (N_T,))
        self.assertTrue(np.all((res_1d['peak_prom_Focality'] >= 0) & (res_1d['peak_prom_Focality'] <= 1)))
        # Focality = num_bins / n_x for 1D
        self.assertTrue(np.allclose(res_1d['peak_prom_Focality'], res_1d['peak_prom_num_bins'] / float(N_X), equal_nan=True))

        # (n_x, n_y, n_t) → 2D path: Focality = num_bins / (n_x * n_y)
        ybin_centers = np.linspace(0.0, 10.0, 5)
        n_y = len(ybin_centers)
        p_2d = np.zeros((N_X, n_y, N_T), dtype=float)
        for t in range(N_T):
            # peaked blob in 2D space
            X, Y = np.meshgrid(np.arange(N_X), np.arange(n_y), indexing='ij')
            p_2d[:, :, t] = np.exp(-0.5 * (((X - (2 + t % 6)) ** 2) + ((Y - 2) ** 2)) / (1.2 ** 2))
        ## END for t in range(N_T)...
        p_2d = p_2d / np.nansum(p_2d, axis=(0, 1), keepdims=True)
        res_2d = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=p_2d, xbin_centers=fx['xbin_centers'], ybin_centers=ybin_centers,
            gaussian_volume=None, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False, spatial_ndim=2,
        )
        self.assertEqual(res_2d['peak_prom_Focality'].shape, (N_T,))
        self.assertTrue(np.allclose(res_2d['peak_prom_Focality'], res_2d['peak_prom_num_bins'] / float(N_X * n_y), equal_nan=True))
        # Denominators differ: 1D uses n_x, 2D uses n_x*n_y
        self.assertNotEqual(N_X, N_X * n_y)

    def test_peaked_vs_diffuse_behavioral(self):
        peaked_f = float(np.nanmean(self._peaked['peak_prom_Focality']))
        diffuse_f = float(np.nanmean(self._diffuse['peak_prom_Focality']))
        peaked_p = float(np.nanmean(self._peaked['peak_prom_Peakiness']))
        diffuse_p = float(np.nanmean(self._diffuse['peak_prom_Peakiness']))
        self.assertLess(peaked_f, diffuse_f)
        self.assertGreaterEqual(peaked_p, diffuse_p - 1e-9)
        self.assertTrue(np.all(np.isfinite(self._peaked['peak_prom_num_peaks'])))

    def test_dist_colocated_vs_offset(self):
        fx = self.fx
        colocated = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=None,
            gaussian_volume=self.vol_colocated, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False, spatial_ndim=1,
        )
        offset = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=None,
            gaussian_volume=self.vol_offset, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False, spatial_ndim=1,
        )
        self.assertLess(float(np.nanmean(colocated['dist_to_highest_peak'])), float(np.nanmean(offset['dist_to_highest_peak'])))
        coloc_ov = float(np.nanmean(np.nansum(colocated['mask_overlap'], axis=0)))
        off_ov = float(np.nanmean(np.nansum(offset['mask_overlap'], axis=0)))
        self.assertGreaterEqual(coloc_ov, off_ov - 1e-9)

    def test_mask_overlap_df_normalization_1d(self):
        fx = self.fx
        peaked = self._peaked
        locality_measures_dict_dict = {
            'roam': {k: v for k, v in peaked.items() if k != 'debug'},
            'sprinkle': {k: v for k, v in peaked.items() if k != 'debug'},
        }
        df = self.DecodingLocalityMeasures.perform_build_locality_measures_df(
            locality_measures_dict_dict=locality_measures_dict_dict,
            time_window_centers=fx['t_centers'],
            paradigm_epochs_df=fx['paradigm_epochs_df'],
            xbin_centers=fx['xbin_centers'],
            ybin_centers=None,
            spatial_ndim=1,
        )
        self.assertEqual(len(df), N_T)
        self.assertIn('mask_overlap_roam', df.columns)
        self.assertTrue(np.all((df['mask_overlap_roam'] >= 0) & (df['mask_overlap_roam'] <= 1)))
        # Fraction uses n_x (not n_x * n_y)
        expected = np.nansum(peaked['mask_overlap'], 0) / float(N_X)
        self.assertTrue(np.allclose(df['mask_overlap_roam'].to_numpy(), expected))

    def test_instance_compute_smoke_1d(self):
        fx = self.fx
        obj = self.DecodingLocalityMeasures(
            defer_compute_on_init=True,
            spatial_ndim=1,
            time_window_centers=fx['t_centers'],
            pos_df=fx['pos_df'],
            xbin=fx['xbin'], ybin=None,
            xbin_centers=fx['xbin_centers'], ybin_centers=None,
            p_x_given_n=fx['p_stack'],
            epoch_names=list(EPOCH_NAMES),
            paradigm_epochs_df=fx['paradigm_epochs_df'],
            new_positions=fx['new_positions'],
            sigma=None,
        )
        self.assertEqual(obj.n_total_pos_bins, N_X)
        obj.perform_compute_on_load()
        x_step = float(np.nanmean(np.diff(fx['xbin'])))
        self.assertTrue(np.allclose(obj.sigma, x_step * 5.0))
        df = obj.compute()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        for name in EPOCH_NAMES:
            self.assertTrue(any(c.endswith(f'_{name}') for c in df.columns))
        ## END for name in EPOCH_NAMES...
        self.assertEqual(obj.gaussian_volume.spatial_ndim, 1)
        self.assertEqual(obj.gaussian_volume.shape, (N_X, N_T))


if __name__ == '__main__':
    unittest.main()
