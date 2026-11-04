"""2D regression tests for SparseGaussianVolume + DecodingLocalityMeasures.

Locks current 2D behavior before 1D generalization. Regenerate baseline constants
only intentionally when 2D math is deliberately changed.

Uses AST extraction of classes from PredictiveDecodingComputations.py to avoid
heavy import side-effects (awkward/pyvista/Qt chains).
"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as scipy_sparse
from scipy.interpolate import interp1d


tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))


SOURCE_PATH = root_project_folder / 'src' / 'pyphoplacecellanalysis' / 'General' / 'Pipeline' / 'Stages' / 'ComputationFunctions' / 'MultiContextComputationFunctions' / 'PredictiveDecodingComputations.py'


def _strip_to_classmethod_staticmethod_property(class_node: ast.ClassDef) -> None:
    class_node.decorator_list = []
    keep = {'classmethod', 'staticmethod', 'property'}
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kept = []
            for dec in item.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in keep:
                    kept.append(dec)
                elif isinstance(dec, ast.Attribute) and dec.attr in keep:
                    kept.append(dec)
            ## END for dec in item.decorator_list...
            item.decorator_list = kept
        ## END if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))...
    ## END for item in class_node.body...


def _keep_only_methods(class_node: ast.ClassDef) -> None:
    """Drop attrs field assignments; keep methods/properties/docstring."""
    kept_body = []
    for item in class_node.body:
        if isinstance(item, ast.Expr) and isinstance(getattr(item, 'value', None), ast.Constant):
            kept_body.append(item)  # docstring
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kept_body.append(item)
    ## END for item in class_node.body...
    class_node.body = kept_body


def _load_sparse_and_decoding_locality():
    source_tree = ast.parse(SOURCE_PATH.read_text(encoding='utf-8'))
    sparse_node = None
    dlm_node = None
    for node in source_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'SparseGaussianVolume':
            sparse_node = node
        elif isinstance(node, ast.ClassDef) and node.name == 'DecodingLocalityMeasures':
            dlm_node = node
    ## END for node in source_tree.body...
    assert sparse_node is not None
    assert dlm_node is not None

    _strip_to_classmethod_staticmethod_property(sparse_node)
    _strip_to_classmethod_staticmethod_property(dlm_node)
    _keep_only_methods(dlm_node)
    dlm_node.bases = []  # drop ComputedResult / attrs bases for isolated exec

    # Inject a simple __init__ for DecodingLocalityMeasures (attrs-free)
    init_src = '''
def __init__(self, defer_compute_on_init=True, **kwargs):
    defaults = dict(
        time_window_centers=None, pos_df=None, xbin=None, ybin=None,
        xbin_centers=None, ybin_centers=None, p_x_given_n=None,
        epoch_names=[], interpolator=None, paradigm_epochs_df=None,
        new_positions=None, sigma=None, gaussian_volume=None,
        p_x_given_n_dict=None, locality_measures_dict_dict={},
        debugging_dict_dict={}, locality_measures_df=None,
        non_local_PBE_non_moving_epochs_df=None,
        moving_avg_meas_pos_overlap_dict={},
        spatial_ndim=2,
    )
    defaults.update(kwargs)
    defaults['defer_compute_on_init'] = defer_compute_on_init
    for k, v in defaults.items():
        setattr(self, k, v)
    if not defer_compute_on_init:
        self.perform_compute_on_load()
'''
    init_node = ast.parse(init_src).body[0]
    dlm_node.body.insert(0, init_node)

    from neuropy.utils.efficient_interval_search import OverlappingIntervalsFallbackBehavior

    namespace: Dict[str, Any] = {
        'np': np,
        'pd': pd,
        'deepcopy': deepcopy,
        'Path': Path,
        'Optional': Optional,
        'List': List,
        'Dict': Dict,
        'Tuple': Tuple,
        'Any': Any,
        'Union': Union,
        'Callable': Callable,
        'NDArray': np.ndarray,
        'scipy_sparse': scipy_sparse,
        'interp1d': interp1d,
        'SparseGaussianVolume': None,  # filled after first exec pass
        'OverlappingIntervalsFallbackBehavior': OverlappingIntervalsFallbackBehavior,
        'ComputedResult': object,
        'function_attributes': (lambda **kw: (lambda f: f)),
        'metadata_attributes': (lambda **kw: (lambda c: c)),
    }
    future_import = ast.parse('from __future__ import annotations').body[0]
    exec(compile(ast.Module(body=[future_import, sparse_node], type_ignores=[]), filename=str(SOURCE_PATH), mode='exec'), namespace)
    namespace['SparseGaussianVolume'] = namespace['SparseGaussianVolume']
    exec(compile(ast.Module(body=[future_import, dlm_node], type_ignores=[]), filename=str(SOURCE_PATH), mode='exec'), namespace)
    return namespace['SparseGaussianVolume'], namespace['DecodingLocalityMeasures']


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

N_X, N_Y, N_T, N_CONTEXTS = 12, 10, 8, 2
EPOCH_NAMES = ['roam', 'sprinkle']


def _centers_edges(n: int, lo: float, hi: float):
    edges = np.linspace(lo, hi, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, edges


def _renorm_xy(p: np.ndarray) -> np.ndarray:
    out = np.asarray(p, dtype=float).copy()
    if out.ndim == 3:
        s = np.nansum(out, axis=(0, 1), keepdims=True)
        s = np.where(s == 0, 1.0, s)
        out = out / s
    elif out.ndim == 4:
        s = np.nansum(out, axis=(0, 1), keepdims=True)
        s = np.where(s == 0, 1.0, s)
        out = out / s
    return out


def _make_peaked_posterior(xbin_centers, ybin_centers, peak_xy_per_t, sigma_bins: float = 1.2) -> np.ndarray:
    X, Y = np.meshgrid(xbin_centers, ybin_centers, indexing='ij')
    n_t = len(peak_xy_per_t)
    out = np.zeros((len(xbin_centers), len(ybin_centers), n_t), dtype=float)
    x_step = float(np.mean(np.diff(xbin_centers)))
    y_step = float(np.mean(np.diff(ybin_centers)))
    for t, (px, py) in enumerate(peak_xy_per_t):
        dist_sq = ((X - px) / x_step) ** 2 + ((Y - py) / y_step) ** 2
        out[:, :, t] = np.exp(-0.5 * dist_sq / (sigma_bins ** 2))
    ## END for t, (px, py) in enumerate(peak_xy_per_t)...
    return _renorm_xy(out)


def _make_diffuse_posterior(n_x, n_y, n_t) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = 0.85 + 0.15 * rng.random((n_x, n_y, n_t))
    return _renorm_xy(out)


def _build_fixture_bundle():
    xbin_centers, xbin = _centers_edges(N_X, 0.0, 120.0)
    ybin_centers, ybin = _centers_edges(N_Y, 0.0, 100.0)
    t_centers = np.arange(N_T, dtype=float) * 0.25

    # Peaks drift slowly across the arena for context 0; context 1 peaks elsewhere
    peak0 = [(xbin_centers[2 + (t % 6)], ybin_centers[2 + (t % 5)]) for t in range(N_T)]
    peak1 = [(xbin_centers[-(3 + (t % 5))], ybin_centers[-(3 + (t % 4))]) for t in range(N_T)]
    p0 = _make_peaked_posterior(xbin_centers, ybin_centers, peak0)
    p1 = _make_peaked_posterior(xbin_centers, ybin_centers, peak1)
    p_stack = np.stack([p0, p1], axis=2)  # (nx, ny, 2, nt)
    p_diffuse = _make_diffuse_posterior(N_X, N_Y, N_T)

    new_positions = np.asarray(peak0, dtype=float)  # colocated with context-0 peaks
    opposite = np.asarray(peak1, dtype=float)
    sigma = 2.0 * float(np.mean([np.mean(np.diff(xbin)), np.mean(np.diff(ybin))]))

    pos_df = pd.DataFrame({
        't': t_centers,
        'x': new_positions[:, 0],
        'y': new_positions[:, 1],
    })
    paradigm_epochs_df = pd.DataFrame({
        'start': [float(t_centers[0]) - 0.01, float(t_centers[N_T // 2])],
        'stop': [float(t_centers[N_T // 2]), float(t_centers[-1]) + 0.01],
        'label': EPOCH_NAMES,
    })
    return {
        'xbin_centers': xbin_centers, 'ybin_centers': ybin_centers,
        'xbin': xbin, 'ybin': ybin, 't_centers': t_centers,
        'p0': p0, 'p1': p1, 'p_stack': p_stack, 'p_diffuse': p_diffuse,
        'new_positions': new_positions, 'opposite_positions': opposite,
        'sigma': sigma, 'pos_df': pos_df, 'paradigm_epochs_df': paradigm_epochs_df,
    }


# Frozen baseline summaries from current 2D math (seed/fixtures above).
# Regenerate intentionally only when 2D math is deliberately changed.
GOLDEN_BASELINE: Dict[str, float] = {
    'peaked_focality_mean': 0.008333333333333333,
    'peaked_peakiness_mean': 1.0,
    'peaked_dist_mean': 0.017184846625836116,
    'peaked_mask_overlap_sum_mean': 105.125,
    'diffuse_focality_mean': 1.0,
    'diffuse_peakiness_mean': 1.0,
}


class TestSparseGaussianVolume2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SparseGaussianVolume, cls.DecodingLocalityMeasures = _load_sparse_and_decoding_locality()
        cls.fix = _build_fixture_bundle()

    def test_build_shape_and_peak_near_measured(self):
        fx = self.fix
        vol = self.SparseGaussianVolume.build_from_positions(fx['xbin_centers'], fx['ybin_centers'], fx['new_positions'], fx['sigma'])
        self.assertEqual(vol.shape, (N_X, N_Y, N_T))
        dense = vol.to_dense()
        for t in range(N_T):
            flat = np.argmax(dense[:, :, t])
            ix, iy = np.unravel_index(flat, (N_X, N_Y))
            true_ix = int(np.argmin(np.abs(fx['xbin_centers'] - fx['new_positions'][t, 0])))
            true_iy = int(np.argmin(np.abs(fx['ybin_centers'] - fx['new_positions'][t, 1])))
            self.assertLessEqual(abs(ix - true_ix), 2)
            self.assertLessEqual(abs(iy - true_iy), 2)
        ## END for t in range(N_T)...

    def test_marginal_sums_and_compression(self):
        fx = self.fix
        vol = self.SparseGaussianVolume.build_from_positions(fx['xbin_centers'], fx['ybin_centers'], fx['new_positions'], fx['sigma'])
        marg_x, marg_y = vol.marginal_sums()
        self.assertEqual(marg_x.shape, (N_X, N_T))
        self.assertEqual(marg_y.shape, (N_Y, N_T))
        dense = vol.to_dense()
        for t in range(N_T):
            self.assertTrue(np.allclose(marg_x[:, t], dense[:, :, t].sum(axis=1)))
            self.assertTrue(np.allclose(marg_y[:, t], dense[:, :, t].sum(axis=0)))
        ## END for t in range(N_T)...
        # CSR overhead can exceed dense on tiny grids; just ensure properties are finite/positive
        self.assertGreater(vol.nbytes_sparse, 0)
        self.assertGreater(vol.nbytes_dense_equivalent, 0)
        self.assertTrue(np.isfinite(vol.compression_ratio))

    def test_multiply_mask_ops(self):
        fx = self.fix
        vol = self.SparseGaussianVolume.build_from_positions(fx['xbin_centers'], fx['ybin_centers'], fx['new_positions'], fx['sigma'])
        ones = np.ones(vol.shape, dtype=float)
        zeros = np.zeros(vol.shape, dtype=float)
        int_ones = vol.multiply_mask_threshold_to_int(ones, 1e-9)
        int_zeros = vol.multiply_mask_threshold_to_int(zeros, 1e-9)
        self.assertEqual(int_ones.shape, vol.shape)
        self.assertTrue(np.all(int_zeros == 0))
        sum_spatial = vol.multiply_mask_threshold_sum_spatial(ones, 1e-9)
        self.assertEqual(sum_spatial.shape, (N_T,))
        self.assertTrue(np.allclose(sum_spatial, np.nansum(int_ones, axis=(0, 1))))


class TestDecodingLocalityMeasures2DClassmethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SparseGaussianVolume, cls.DecodingLocalityMeasures = _load_sparse_and_decoding_locality()
        cls.fix = _build_fixture_bundle()
        fx = cls.fix
        cls.vol_colocated = cls.SparseGaussianVolume.build_from_positions(fx['xbin_centers'], fx['ybin_centers'], fx['new_positions'], fx['sigma'])
        cls.vol_offset = cls.SparseGaussianVolume.build_from_positions(fx['xbin_centers'], fx['ybin_centers'], fx['opposite_positions'], fx['sigma'])
        cls.alpha_list = [0.8]
        # Capture / assert frozen baselines
        peaked = cls.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=fx['ybin_centers'],
            gaussian_volume=cls.vol_colocated, alpha_list=cls.alpha_list, enable_debug_outputs=True, debug_print=False,
        )
        diffuse = cls.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p_diffuse'], xbin_centers=fx['xbin_centers'], ybin_centers=fx['ybin_centers'],
            gaussian_volume=cls.vol_colocated, alpha_list=cls.alpha_list, enable_debug_outputs=False, debug_print=False,
        )
        cls._peaked = peaked
        cls._diffuse = diffuse
        cls.BASELINE = {
            'peaked_focality_mean': float(np.nanmean(peaked['peak_prom_Focality'])),
            'peaked_peakiness_mean': float(np.nanmean(peaked['peak_prom_Peakiness'])),
            'peaked_dist_mean': float(np.nanmean(peaked['dist_to_highest_peak'])),
            'peaked_mask_overlap_sum_mean': float(np.nanmean(np.nansum(peaked['mask_overlap'], axis=(0, 1)))),
            'diffuse_focality_mean': float(np.nanmean(diffuse['peak_prom_Focality'])),
            'diffuse_peakiness_mean': float(np.nanmean(diffuse['peak_prom_Peakiness'])),
        }

    def test_perform_build_normalized_outputs(self):
        fx = self.fix
        out = self.DecodingLocalityMeasures.perform_build_normalized_outputs(fx['p_stack'], EPOCH_NAMES)
        self.assertEqual(set(out.keys()), set(EPOCH_NAMES))
        for i, name in enumerate(EPOCH_NAMES):
            self.assertEqual(out[name].shape, (N_X, N_Y, N_T))
            sums = np.nansum(out[name], axis=(0, 1))
            self.assertTrue(np.allclose(sums, 1.0, atol=1e-6))
            # Isolation: mass matches renormalized slice of that context
            expected = fx['p_stack'][:, :, i, :].copy()
            s = np.nansum(expected, axis=(0, 1), keepdims=True)
            s = np.where(s == 0, 1.0, s)
            expected = expected / s
            self.assertTrue(np.allclose(out[name], expected, atol=1e-6))
        ## END for i, name in enumerate(EPOCH_NAMES)...

    def test_compute_locality_keys_shapes_without_gaussian(self):
        fx = self.fix
        res = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=fx['ybin_centers'],
            gaussian_volume=None, alpha_list=[0.8], enable_debug_outputs=True, debug_print=False,
        )
        for key in ('peak_prom_num_bins', 'peak_prom_Focality', 'peak_prom_Peakiness', 'peak_prom_num_peaks'):
            self.assertIn(key, res)
            self.assertEqual(np.shape(res[key])[0], N_T)
        ## END for key in ...
        self.assertIn('debug', res)
        self.assertTrue(np.all((res['peak_prom_Focality'] >= 0) & (res['peak_prom_Focality'] <= 1)))

    def test_compute_locality_keys_with_gaussian(self):
        res = self._peaked
        for key in ('mask_overlap', 'peak_prom', 'dist_to_highest_peak', 'peak_prom_Focality'):
            self.assertIn(key, res)
        ## END for key in ...
        self.assertEqual(res['mask_overlap'].shape, (N_X, N_Y, N_T))
        self.assertEqual(res['dist_to_highest_peak'].shape, (N_T,))
        self.assertTrue(np.all((res['dist_to_highest_peak'] >= 0) & (res['dist_to_highest_peak'] <= 1 + 1e-9)))

    def test_peaked_vs_diffuse_behavioral(self):
        peaked_f = float(np.nanmean(self._peaked['peak_prom_Focality']))
        diffuse_f = float(np.nanmean(self._diffuse['peak_prom_Focality']))
        peaked_p = float(np.nanmean(self._peaked['peak_prom_Peakiness']))
        diffuse_p = float(np.nanmean(self._diffuse['peak_prom_Peakiness']))
        self.assertLess(peaked_f, diffuse_f)
        # Peakiness can saturate at 1.0 for both; require peaked >= diffuse when both finite
        self.assertGreaterEqual(peaked_p, diffuse_p - 1e-9)

    def test_dist_colocated_vs_offset(self):
        fx = self.fix
        colocated = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=fx['ybin_centers'],
            gaussian_volume=self.vol_colocated, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False,
        )
        offset = self.DecodingLocalityMeasures.compute_locality_measures_for_posterior(
            a_p_x_given_n=fx['p0'], xbin_centers=fx['xbin_centers'], ybin_centers=fx['ybin_centers'],
            gaussian_volume=self.vol_offset, alpha_list=[0.8], enable_debug_outputs=False, debug_print=False,
        )
        self.assertLess(float(np.nanmean(colocated['dist_to_highest_peak'])), float(np.nanmean(offset['dist_to_highest_peak'])))
        coloc_ov = float(np.nanmean(np.nansum(colocated['mask_overlap'], axis=(0, 1))))
        off_ov = float(np.nanmean(np.nansum(offset['mask_overlap'], axis=(0, 1))))
        self.assertGreaterEqual(coloc_ov, off_ov - 1e-9)

    def test_frozen_numeric_baseline(self):
        """Strict regression against hardcoded summaries from current 2D code."""
        b = self.BASELINE
        for k, v in GOLDEN_BASELINE.items():
            self.assertTrue(np.allclose(v, b[k], rtol=1e-5, atol=1e-6), msg=f'baseline drift on {k}: {b[k]} vs {v}')
        ## END for k, v in GOLDEN_BASELINE.items()...

    def test_perform_build_locality_measures_df(self):
        fx = self.fix
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
            ybin_centers=fx['ybin_centers'],
        )
        self.assertEqual(len(df), N_T)
        self.assertTrue(np.allclose(df['t'].to_numpy(), fx['t_centers']))
        self.assertNotIn('peak_prom_roam', df.columns)  # skipped raw peak_prom volume
        self.assertIn('peak_prom_Focality_roam', df.columns)
        self.assertIn('mask_overlap_roam', df.columns)
        self.assertTrue(np.all((df['mask_overlap_roam'] >= 0) & (df['mask_overlap_roam'] <= 1)))
        self.assertIn('correct_paradigm_epoch', df.columns)
        self.assertIn('is_non_local_period', df.columns)
        # Non-local rule for roam rows
        is_roam = (df['correct_paradigm_epoch'] == 'roam')
        expected_nl = np.logical_and(df.loc[is_roam, 'dist_to_highest_peak_roam'] >= 0.4, df.loc[is_roam, 'mask_overlap_roam'] < 0.1)
        self.assertTrue(np.array_equal(df.loc[is_roam, 'is_non_local_period'].to_numpy(), expected_nl.to_numpy()))


class TestDecodingLocalityMeasures2DInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SparseGaussianVolume, cls.DecodingLocalityMeasures = _load_sparse_and_decoding_locality()
        cls.fix = _build_fixture_bundle()

    def test_instance_compute_smoke(self):
        fx = self.fix
        obj = self.DecodingLocalityMeasures(
            defer_compute_on_init=True,
            time_window_centers=fx['t_centers'],
            pos_df=fx['pos_df'],
            xbin=fx['xbin'], ybin=fx['ybin'],
            xbin_centers=fx['xbin_centers'], ybin_centers=fx['ybin_centers'],
            p_x_given_n=fx['p_stack'],
            epoch_names=list(EPOCH_NAMES),
            paradigm_epochs_df=fx['paradigm_epochs_df'],
            new_positions=fx['new_positions'],
            sigma=None,
        )
        self.assertEqual(obj.n_total_pos_bins, N_X * N_Y)
        obj.perform_compute_on_load()
        x_step = float(np.nanmean(np.diff(fx['xbin'])))
        y_step = float(np.nanmean(np.diff(fx['ybin'])))
        self.assertTrue(np.allclose(obj.sigma, np.nanmax([x_step, y_step]) * 5.0))
        df = obj.compute()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        for name in EPOCH_NAMES:
            self.assertTrue(any(c.endswith(f'_{name}') for c in df.columns))
        ## END for name in EPOCH_NAMES...
        df2 = obj.rebuild_locality_measures_df()
        self.assertEqual(set(df.columns), set(df2.columns))


if __name__ == '__main__':
    unittest.main()
