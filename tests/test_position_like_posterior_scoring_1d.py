"""Tests for PositionLikePosteriorScoring 1D PLI path (AST-loaded to avoid PendingNotebookCode import side effects)."""
import ast
import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))


def _load_position_like_posterior_scoring_1d():
    """Extract ``PositionLikePosteriorScoring`` from PendingNotebookCode without importing the module."""
    source_path = root_project_folder / 'src' / 'pyphoplacecellanalysis' / 'SpecificResults' / 'PendingNotebookCode.py'
    source_tree = ast.parse(source_path.read_text(encoding='utf-8'))
    class_node = None
    for node in source_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'PositionLikePosteriorScoring':
            class_node = node
            break
    ## END for node in source_tree.body...
    assert class_node is not None, 'PositionLikePosteriorScoring not found in PendingNotebookCode.py'

    # Strip project-only decorators; keep classmethod/staticmethod/property
    class_node.decorator_list = []
    keep_decorator_names = {'classmethod', 'staticmethod', 'property'}
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kept = []
            for dec in item.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in keep_decorator_names:
                    kept.append(dec)
                elif isinstance(dec, ast.Attribute) and dec.attr in keep_decorator_names:
                    kept.append(dec)
            ## END for dec in item.decorator_list...
            item.decorator_list = kept
        ## END if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))...
    ## END for item in class_node.body...

    namespace: Dict[str, Any] = {
        'np': np,
        'pd': pd,
        'plt': __import__('matplotlib.pyplot', fromlist=['pyplot']),
        'math': math,
        'Path': Path,
        'Optional': Optional,
        'List': List,
        'Dict': Dict,
        'Tuple': Tuple,
        'Any': Any,
        'NDArray': np.ndarray,
        'DecodedFilterEpochsResult': object,
    }
    # Preserve postponed annotations like the real module (`from __future__ import annotations`)
    future_import = ast.parse('from __future__ import annotations').body[0]
    exec(compile(ast.Module(body=[future_import, class_node], type_ignores=[]), filename=str(source_path), mode='exec'), namespace)
    return namespace['PositionLikePosteriorScoring']


class TestPositionLikePosteriorScoring1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.PLS = _load_position_like_posterior_scoring_1d()
        cls.n_pos = 40
        cls.x_edges = np.linspace(0.0, 200.0, cls.n_pos + 1)

    def test_uniform_score_near_zero(self):
        P = np.ones(self.n_pos, dtype=float)
        score, comps = self.PLS.calculate_pli_1d_score(P, x_edges=self.x_edges)
        self.assertLess(score, 0.25)
        self.assertLess(comps['sharpness'], 0.05)

    def test_delta_score_near_one(self):
        P = np.zeros(self.n_pos, dtype=float)
        P[self.n_pos // 2] = 1.0
        score, comps = self.PLS.calculate_pli_1d_score(P, x_edges=self.x_edges)
        self.assertGreater(score, 0.95)
        self.assertAlmostEqual(comps['sharpness'], 1.0, places=5)
        self.assertAlmostEqual(comps['locality'], 1.0, places=5)
        self.assertAlmostEqual(comps['rms_cm'], 0.0, places=5)

    def test_narrow_gaussian_between_uniform_and_delta(self):
        centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        map_x = centers[self.n_pos // 2]
        sigma_cm = 8.0
        P = np.exp(-0.5 * ((centers - map_x) / sigma_cm) ** 2)
        score_g, _ = self.PLS.calculate_pli_1d_score(P, x_edges=self.x_edges)
        score_u, _ = self.PLS.calculate_pli_1d_score(np.ones(self.n_pos), x_edges=self.x_edges)
        P_delta = np.zeros(self.n_pos); P_delta[self.n_pos // 2] = 1.0
        score_d, _ = self.PLS.calculate_pli_1d_score(P_delta, x_edges=self.x_edges)
        self.assertGreater(score_g, score_u)
        self.assertLess(score_g, score_d)

    def test_reject_2d_spatial_grid_in_1d_score(self):
        P_2d = np.ones((self.n_pos, 12), dtype=float)
        with self.assertRaises(ValueError) as ctx:
            self.PLS.calculate_pli_1d_score(P_2d, x_edges=self.x_edges)
        self.assertIn('calculate_pli_score', str(ctx.exception))

    def test_reject_3d_stack_in_normalize(self):
        stack_3d = np.ones((10, 8, 5), dtype=float)
        with self.assertRaises(ValueError) as ctx:
            self.PLS._normalize_posterior_1d_stack(stack_3d)
        self.assertIn('2D spatial stack', str(ctx.exception))

    def test_stack_row_count_equals_n_t(self):
        rng = np.random.default_rng(0)
        n_t = 17
        stack = np.abs(rng.standard_normal((self.n_pos, n_t)))
        stack = stack / np.sum(stack, axis=0, keepdims=True)
        df = self.PLS.compute_and_plot_posterior_stack_1d(stack, x_edges=self.x_edges, should_plot_results=False)
        self.assertEqual(len(df), n_t)
        self.assertIn('is_position_like_1d', df.columns)
        self.assertIn('score', df.columns)

    def test_x_edges_length_must_match_n_pos(self):
        P = np.ones(self.n_pos, dtype=float)
        with self.assertRaises(ValueError):
            self.PLS.calculate_pli_1d_score(P, x_edges=self.x_edges[:-2])

    def test_1d_quality_label_uses_1d_cutoffs(self):
        label_hq, _ = self.PLS.get_pli_1d_quality_label(self.PLS.high_quality_score_cutoff_1d)
        label_pl, _ = self.PLS.get_pli_1d_quality_label(self.PLS.position_like_score_cutoff_1d)
        label_amb, _ = self.PLS.get_pli_1d_quality_label(self.PLS.position_like_score_cutoff_1d - 0.05)
        self.assertEqual(label_hq, 'Position-Like HQ')
        self.assertEqual(label_pl, 'Position-Like')
        self.assertEqual(label_amb, 'Ambiguous')

    def test_score_h5_loaded_decoder_epochs_1d(self):
        rng = np.random.default_rng(1)
        epoch0 = np.abs(rng.standard_normal((self.n_pos, 5)))
        epoch1 = np.zeros((self.n_pos, 3)); epoch1[self.n_pos // 2, :] = 1.0
        h5_out_dict = {
            'long_LR': {
                'ripple': {
                    'p_x_given_n': [epoch0, epoch1],
                }
            }
        }
        concat_df, per_epoch = self.PLS.score_h5_loaded_decoder_epochs_1d(h5_out_dict, decoder_name='long_LR', epochs_name='ripple', x_edges=self.x_edges)
        self.assertEqual(len(per_epoch), 2)
        self.assertEqual(len(concat_df), 5 + 3)
        self.assertTrue(np.all(concat_df.loc[concat_df['epoch_idx'] == 1, 'score'] > 0.95))

    def test_optional_fixture_h5_skipped_if_missing(self):
        fixture = tests_folder / 'fixtures' / 'decoded_posteriors_mini.h5'
        if not fixture.exists():
            self.skipTest('optional fixtures/decoded_posteriors_mini.h5 not present')
        # If a fixture is added later, light-smoke via h5py shapes only (PosteriorExporting optional).
        import h5py
        with h5py.File(fixture, 'r') as h:
            self.assertGreater(len(h.keys()), 0)


if __name__ == '__main__':
    unittest.main()
