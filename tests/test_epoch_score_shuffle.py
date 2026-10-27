"""Pipeline-free tests for epoch-score shuffle stats, pickle load, and heuristic scoring."""
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import neuropy.core.position  # finish neuropy init before the computation-function package import

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent

try:
    import pyphoplacecellanalysis
except ModuleNotFoundError:
    sys.path.insert(0, str(root_project_folder.joinpath('src')))
finally:
    from neuropy.utils.mixins.indexing_helpers import get_dict_subset
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import DEFAULT_SHUFFLE_DECODER_NAMES, DecodedEpochScoreShuffle, WCorrShuffle
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData


def _epochs_df(n_epochs: int) -> pd.DataFrame:
    starts = np.arange(n_epochs, dtype=float) * 1.0
    return pd.DataFrame({
        'start': starts,
        'stop': starts + 0.2,
        'duration': np.full(n_epochs, 0.2),
        'long_best_dir_decoder_IDX': np.zeros(n_epochs, dtype=int),
        'short_best_dir_decoder_IDX': np.full(n_epochs, 2, dtype=int),
    })


def _wcorr_df_dict(wcorr_arr: np.ndarray):
    return DecodedEpochScoreShuffle._array_to_wcorr_df_dict(wcorr_arr, decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))


def _minimal_decoded_result_dict(n_epochs: int = 2) -> dict:
    return {
        'num_filter_epochs': n_epochs,
        'decoding_time_bin_size': 0.025,
        'nbins': np.full(n_epochs, 2, dtype=int),
        'filter_epochs': _epochs_df(n_epochs),
    }


def _decoded_result_mock(n_epochs: int = 2) -> "_DecodedResultMock":
    n_time = 2
    time_window_centers = np.array([0.025, 0.075])
    time_bin_edges = np.array([0.0, 0.05, 0.10])
    return _DecodedResultMock(
        num_filter_epochs=n_epochs,
        most_likely_positions_list=[np.array([0.0, 1.0]) for _epoch_idx in np.arange(n_epochs)],
        p_x_given_n_list=[np.ones((2, n_time)) for _epoch_idx in np.arange(n_epochs)],
        nbins=np.full(n_epochs, n_time, dtype=int),
        time_window_centers=[time_window_centers.copy() for _epoch_idx in np.arange(n_epochs)],
        time_bin_edges=[time_bin_edges.copy() for _epoch_idx in np.arange(n_epochs)],
        filter_epochs=_epochs_df(n_epochs),
    )


def _make_shuffle_object(n_epochs: int = 3, n_decoders: int = 4, n_shuffles: int = 5, seed: int = 0) -> DecodedEpochScoreShuffle:
    rng = np.random.default_rng(seed)
    real_wcorr = rng.uniform(-0.8, 0.8, size=(n_epochs, n_decoders))
    real_coverage = rng.uniform(0.1, 0.9, size=(n_epochs, n_decoders))
    real_travel = rng.uniform(0.05, 0.6, size=(n_epochs, n_decoders))
    output_extracted_score_arrays_list = []
    output_extracted_result_wcorrs_list = []
    for shuffle_idx in np.arange(n_shuffles):
        shuffle_wcorr = real_wcorr + rng.normal(0.0, 0.25, size=(n_epochs, n_decoders)) + (0.05 * shuffle_idx)
        shuffle_coverage = np.clip(real_coverage + rng.normal(0.0, 0.15, size=(n_epochs, n_decoders)), 0.0, 1.0)
        shuffle_travel = np.clip(real_travel + rng.normal(0.0, 0.12, size=(n_epochs, n_decoders)), 0.0, None)
        score_dict = {'wcorr': shuffle_wcorr, 'coverage': shuffle_coverage, 'travel': shuffle_travel}
        output_extracted_score_arrays_list.append(score_dict)
        output_extracted_result_wcorrs_list.append(_wcorr_df_dict(shuffle_wcorr))
    ## END for shuffle_idx in np.arange(n_shuffles)....

    return DecodedEpochScoreShuffle(
        filtered_epochs_df=_epochs_df(n_epochs),
        real_decoder_ripple_weighted_corr_arr=real_wcorr,
        real_score_arrays={'wcorr': real_wcorr, 'coverage': real_coverage, 'travel': real_travel},
        output_extracted_result_wcorrs_list=output_extracted_result_wcorrs_list,
        output_extracted_score_arrays_list=output_extracted_score_arrays_list,
        enable_heuristic_shuffle_scoring=True,
        enable_radon_shuffle_scoring=False,
        enable_saving_entire_decoded_shuffle_result=False,
    )


class _DecodedResultMock:
    """Duck-typed decoded-epoch result for HeuristicReplayScoring.compute_shuffle_heuristic_score_arrays."""
    def __init__(self, num_filter_epochs, most_likely_positions_list, p_x_given_n_list, nbins, time_window_centers, time_bin_edges, filter_epochs):
        self.num_filter_epochs = num_filter_epochs
        self.most_likely_positions_list = most_likely_positions_list
        self.p_x_given_n_list = p_x_given_n_list
        self.nbins = nbins
        self.time_window_centers = time_window_centers
        self.time_bin_edges = time_bin_edges
        self.filter_epochs = filter_epochs


class _DecoderXbinMock:
    def __init__(self, xbin):
        self.xbin = xbin


class _TemplatesMock:
    def __init__(self, xbin, track_length_dict):
        self.long_LR_decoder = _DecoderXbinMock(xbin)
        self._track_length_dict = track_length_dict


    def get_track_length_dict(self):
        return self._track_length_dict


class TestPostComputeMetric(unittest.TestCase):
    """DecodedEpochScoreShuffle.post_compute_metric extreme modes and NaN exclusion."""

    def test_abs_greater_counts_larger_absolute_shuffle_values(self):
        real_arr = np.array([[0.5, -0.8], [0.2, 0.9]], dtype=float)
        shuffle_stack = np.array([
            [[0.6, -0.9], [0.3, 0.4]],
            [[-0.7, 0.1], [-0.5, 0.95]],
            [[0.4, -0.2], [0.1, 0.2]],
            [[0.1, 0.85], [0.25, -0.91]],
        ], dtype=float)  # (n_shuffles=4, n_epochs=2, n_decoders=2)
        p, z, n_more_extreme = DecodedEpochScoreShuffle.post_compute_metric(real_arr, shuffle_stack, extreme_mode='abs_greater')

        expected_more_extreme = np.sum(np.abs(shuffle_stack) > np.abs(real_arr), axis=0)
        np.testing.assert_array_equal(n_more_extreme, expected_more_extreme)
        np.testing.assert_allclose(p, expected_more_extreme.astype(float) / 4.0)
        self.assertEqual(p.shape, (2, 2))
        self.assertEqual(z.shape, (2, 2))
        expected_z00 = DecodedEpochScoreShuffle.compute_z_score(shuffle_stack[:, 0, 0], real_arr[0, 0])
        self.assertAlmostEqual(float(z[0, 0]), float(expected_z00))


    def test_greater_counts_shuffles_above_real(self):
        real_arr = np.array([[0.3, 0.5]], dtype=float)
        shuffle_stack = np.array([[[0.4, 0.6]], [[0.2, 0.4]], [[0.5, 0.5]], [[0.3, 0.7]]], dtype=float)
        p, z, n_more_extreme = DecodedEpochScoreShuffle.post_compute_metric(real_arr, shuffle_stack, extreme_mode='greater')

        expected_more_extreme = np.sum(shuffle_stack > real_arr, axis=0)
        np.testing.assert_array_equal(n_more_extreme, expected_more_extreme)
        np.testing.assert_allclose(p, np.array([[2.0 / 4.0, 2.0 / 4.0]]))
        self.assertEqual(z.shape, (1, 2))


    def test_less_counts_shuffles_below_real(self):
        real_arr = np.array([[0.5, 0.2]], dtype=float)
        shuffle_stack = np.array([[[0.4, 0.1]], [[0.6, 0.3]], [[0.1, 0.2]], [[0.5, 0.05]]], dtype=float)
        p, z, n_more_extreme = DecodedEpochScoreShuffle.post_compute_metric(real_arr, shuffle_stack, extreme_mode='less')

        expected_more_extreme = np.sum(shuffle_stack < real_arr, axis=0)
        np.testing.assert_array_equal(n_more_extreme, expected_more_extreme)
        np.testing.assert_allclose(p, np.array([[2.0 / 4.0, 2.0 / 4.0]]))
        self.assertEqual(z.shape, (1, 2))


    def test_nans_in_shuffle_stack_are_excluded_from_n_valid(self):
        real_arr = np.array([[0.5, -0.5]], dtype=float)
        shuffle_stack = np.array([
            [[0.8, 0.8]],
            [[np.nan, np.nan]],
            [[0.2, -0.2]],
            [[0.6, -0.6]],
        ], dtype=float)
        p, _z, n_more_extreme = DecodedEpochScoreShuffle.post_compute_metric(real_arr, shuffle_stack, extreme_mode='abs_greater')

        n_valid = np.sum(np.logical_not(np.isnan(shuffle_stack)), axis=0)
        np.testing.assert_array_equal(n_valid, np.array([[3, 3]]))
        expected_more_extreme = np.sum(np.abs(shuffle_stack) > np.abs(real_arr), axis=0)
        np.testing.assert_array_equal(n_more_extreme, expected_more_extreme)
        np.testing.assert_allclose(p, expected_more_extreme.astype(float) / n_valid.astype(float))
        np.testing.assert_allclose(p, np.array([[2.0 / 3.0, 2.0 / 3.0]]))
        self.assertFalse(np.allclose(p, expected_more_extreme.astype(float) / 4.0))


class TestPostComputeAllMetricsWideDf(unittest.TestCase):
    """Construct DecodedEpochScoreShuffle from arrays; extra metric z/p columns in the wide DF."""

    def test_post_compute_all_metrics_and_wide_dataframe_columns(self):
        n_epochs, n_decoders, n_shuffles = 3, 4, 5
        obj = _make_shuffle_object(n_epochs=n_epochs, n_decoders=n_decoders, n_shuffles=n_shuffles)
        all_stats = obj.post_compute_all_metrics(decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))

        self.assertCountEqual(list(all_stats.keys()), ['wcorr', 'coverage', 'travel'])
        for metric_name, (p_arr, z_arr, n_more_extreme_arr) in all_stats.items():
            self.assertEqual(p_arr.shape, (n_epochs, n_decoders), msg=metric_name)
            self.assertEqual(z_arr.shape, (n_epochs, n_decoders), msg=metric_name)
            self.assertEqual(n_more_extreme_arr.shape, (n_epochs, n_decoders), msg=metric_name)
        ## END for metric_name, (p_arr, z_arr, n_more_extreme_arr) in all_stats.items()....

        post_compute_out = obj.post_compute(decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
        self.assertEqual(len(post_compute_out), 4)

        wide_df, _all_shuffles_wcorr_df = obj.build_all_shuffles_dataframes(decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
        self.assertIn('wcorr_long_LR', wide_df.columns)
        self.assertIn('wcorr_z_long', wide_df.columns)
        for metric_name in ('coverage', 'travel'):
            for decoder_name in DEFAULT_SHUFFLE_DECODER_NAMES:
                self.assertIn(f'{metric_name}_{decoder_name}', wide_df.columns)
                self.assertIn(f'{metric_name}_z_{decoder_name}', wide_df.columns)
                self.assertIn(f'{metric_name}_p_{decoder_name}', wide_df.columns)
                self.assertIn(f'{metric_name}_n_extreme_{decoder_name}', wide_df.columns)
            ## END for decoder_name in DEFAULT_SHUFFLE_DECODER_NAMES....

        ## END for metric_name in ('coverage', 'travel')....


    def test_post_compute_all_metrics_uses_score_extreme_mode(self):
        real = np.full((1, 4), 0.5, dtype=float)
        shuffle_vals = [0.1, 0.2, 0.3, 0.9]
        score_list = []
        wcorr_list = []
        for v in shuffle_vals:
            arr = np.full((1, 4), v, dtype=float)
            score_list.append({'wcorr': arr, 'coverage': arr, 'travel': arr})
            wcorr_list.append(_wcorr_df_dict(arr))
        ## END for v in shuffle_vals....

        obj = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(1),
            real_decoder_ripple_weighted_corr_arr=real,
            real_score_arrays={'wcorr': real, 'coverage': real, 'travel': real},
            output_extracted_result_wcorrs_list=wcorr_list,
            output_extracted_score_arrays_list=score_list,
        )
        all_stats = obj.post_compute_all_metrics(decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
        _p_travel, _z_travel, n_travel = all_stats['travel']
        _p_coverage, _z_coverage, n_coverage = all_stats['coverage']
        _p_wcorr, _z_wcorr, n_wcorr = all_stats['wcorr']
        np.testing.assert_array_equal(n_travel, np.full((1, 4), 3))  # less: 0.1, 0.2, 0.3
        np.testing.assert_array_equal(n_coverage, np.full((1, 4), 1))  # greater: 0.9
        np.testing.assert_array_equal(n_wcorr, np.full((1, 4), 1))  # abs_greater: 0.9


class TestPickleCompat(unittest.TestCase):
    """__setstate__ old-field fill, 3/4/5/7-tuple load_and_append_shuffle_data, batch reconstruct."""

    def test_setstate_fills_missing_score_arrays_and_flags(self):
        real_wcorr = np.array([[0.11, -0.22, 0.33, -0.44], [0.55, 0.66, -0.77, 0.88]], dtype=float)
        wcorr_list = [_wcorr_df_dict(real_wcorr + 0.1), _wcorr_df_dict(real_wcorr - 0.05)]
        old_state = {
            'filtered_epochs_df': _epochs_df(2),
            'output_extracted_result_wcorrs_list': wcorr_list,
            'real_decoder_ripple_weighted_corr_arr': real_wcorr,
        }
        obj = DecodedEpochScoreShuffle.__new__(DecodedEpochScoreShuffle)
        obj.__setstate__(old_state)

        self.assertIn('wcorr', obj.real_score_arrays)
        np.testing.assert_array_equal(obj.real_score_arrays['wcorr'], real_wcorr)
        self.assertEqual(len(obj.output_extracted_score_arrays_list), 2)
        for shuffle_idx, score_dict in enumerate(obj.output_extracted_score_arrays_list):
            self.assertCountEqual(list(score_dict.keys()), ['wcorr'])
            expected = DecodedEpochScoreShuffle._wcorr_df_dict_to_array(wcorr_list[shuffle_idx], decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
            np.testing.assert_allclose(score_dict['wcorr'], expected)
        ## END for shuffle_idx, score_dict in enumerate(obj.output_extracted_score_arrays_list)....

        self.assertTrue(obj.enable_heuristic_shuffle_scoring)
        self.assertFalse(obj.enable_radon_shuffle_scoring)
        self.assertFalse(obj.enable_saving_entire_decoded_shuffle_result)


    def test_load_and_append_shuffle_data_3_tuple_synthesizes_wcorr_score_arrays(self):
        real_wcorr = np.array([[0.2, -0.1, 0.4, 0.05], [0.3, 0.15, -0.25, 0.5]], dtype=float)
        loaded_wcorr = real_wcorr + np.array([[0.08, -0.03, 0.02, 0.01], [-0.04, 0.06, 0.07, -0.02]], dtype=float)
        payload = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [])
        obj = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(2),
            real_decoder_ripple_weighted_corr_arr=real_wcorr,
            real_score_arrays={},
            output_extracted_result_wcorrs_list=[],
            output_extracted_score_arrays_list=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir).joinpath('shuffle_3tuple.pkl')
            saveData(pkl_path, payload)
            obj.load_and_append_shuffle_data(filepath=pkl_path)

        self.assertEqual(len(obj.output_extracted_result_wcorrs_list), 1)
        self.assertEqual(len(obj.output_extracted_score_arrays_list), 1)
        self.assertCountEqual(list(obj.output_extracted_score_arrays_list[0].keys()), ['wcorr'])
        np.testing.assert_allclose(obj.output_extracted_score_arrays_list[0]['wcorr'], loaded_wcorr)


    def test_load_and_append_shuffle_data_4_tuple_synthesizes_wcorr_score_arrays(self):
        real_wcorr = np.array([[0.21, -0.11, 0.41, 0.06], [0.31, 0.16, -0.26, 0.51]], dtype=float)
        loaded_wcorr = real_wcorr + 0.04
        all_shuffles_wcorr_array = np.stack([loaded_wcorr], axis=0)
        payload = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [], all_shuffles_wcorr_array)
        obj = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(2),
            real_decoder_ripple_weighted_corr_arr=real_wcorr,
            real_score_arrays={'wcorr': real_wcorr},
            output_extracted_result_wcorrs_list=[],
            output_extracted_score_arrays_list=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir).joinpath('shuffle_4tuple.pkl')
            saveData(pkl_path, payload)
            obj.load_and_append_shuffle_data(filepath=pkl_path)

        self.assertEqual(len(obj.output_extracted_result_wcorrs_list), 1)
        self.assertEqual(len(obj.output_extracted_score_arrays_list), 1)
        self.assertCountEqual(list(obj.output_extracted_score_arrays_list[0].keys()), ['wcorr'])
        np.testing.assert_allclose(obj.output_extracted_score_arrays_list[0]['wcorr'], loaded_wcorr)


    def test_load_and_append_shuffle_data_5_tuple_sets_real_decode_dict(self):
        real_wcorr = np.array([[0.22, -0.12, 0.42, 0.07], [0.32, 0.17, -0.27, 0.52]], dtype=float)
        loaded_wcorr = real_wcorr - 0.03
        all_shuffles_wcorr_array = np.stack([loaded_wcorr], axis=0)
        real_decode_dict = {'long_LR': 'sentinel_decode'}
        payload = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [], all_shuffles_wcorr_array, real_decode_dict)
        obj = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(2),
            real_decoder_ripple_weighted_corr_arr=real_wcorr,
            real_decoder_ripple_filter_epochs_decoder_result_dict=None,
            real_score_arrays={'wcorr': real_wcorr},
            output_extracted_result_wcorrs_list=[],
            output_extracted_score_arrays_list=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir).joinpath('shuffle_5tuple.pkl')
            saveData(pkl_path, payload)
            obj.load_and_append_shuffle_data(filepath=pkl_path)

        self.assertEqual(obj.real_decoder_ripple_filter_epochs_decoder_result_dict, real_decode_dict)
        self.assertEqual(len(obj.output_extracted_score_arrays_list), 1)
        np.testing.assert_allclose(obj.output_extracted_score_arrays_list[0]['wcorr'], loaded_wcorr)


    def test_load_and_append_shuffle_data_7_tuple_copies_score_arrays(self):
        real_wcorr = np.array([[0.12, -0.34, 0.56, -0.78], [0.9, 0.1, -0.2, 0.3]], dtype=float)
        real_score_arrays = {'wcorr': real_wcorr, 'coverage': real_wcorr + 0.2, 'travel': np.abs(real_wcorr)}
        loaded_wcorr = real_wcorr + 0.07
        loaded_score_arrays_list = [{'wcorr': loaded_wcorr, 'coverage': loaded_wcorr + 0.1, 'travel': np.abs(loaded_wcorr)}]
        all_shuffles_wcorr_array = np.stack([loaded_wcorr], axis=0)
        payload = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [], all_shuffles_wcorr_array, None, loaded_score_arrays_list, real_score_arrays)
        obj = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(2),
            real_decoder_ripple_weighted_corr_arr=real_wcorr,
            real_score_arrays={},
            output_extracted_result_wcorrs_list=[],
            output_extracted_score_arrays_list=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir).joinpath('shuffle_7tuple.pkl')
            saveData(pkl_path, payload)
            obj.load_and_append_shuffle_data(filepath=pkl_path)

        self.assertEqual(len(obj.output_extracted_score_arrays_list), 1)
        self.assertCountEqual(list(obj.output_extracted_score_arrays_list[0].keys()), ['wcorr', 'coverage', 'travel'])
        np.testing.assert_allclose(obj.output_extracted_score_arrays_list[0]['wcorr'], loaded_wcorr)
        np.testing.assert_allclose(obj.output_extracted_score_arrays_list[0]['coverage'], loaded_wcorr + 0.1)
        self.assertCountEqual(list(obj.real_score_arrays.keys()), ['wcorr', 'coverage', 'travel'])
        np.testing.assert_allclose(obj.real_score_arrays['wcorr'], real_wcorr)
        np.testing.assert_allclose(obj.real_score_arrays['coverage'], real_wcorr + 0.2)


    def test_batch_reconstruct_from_to_dict_subset(self):
        obj = _make_shuffle_object(n_epochs=2, n_decoders=4, n_shuffles=3)
        mock_result = _decoded_result_mock(n_epochs=2)
        obj.real_decoder_ripple_filter_epochs_decoder_result_dict = {name: mock_result for name in DEFAULT_SHUFFLE_DECODER_NAMES}
        rebuilt = DecodedEpochScoreShuffle.modernize_from_instance(obj)

        self.assertCountEqual(list(rebuilt.real_score_arrays.keys()), list(obj.real_score_arrays.keys()))
        np.testing.assert_allclose(rebuilt.real_score_arrays['wcorr'], obj.real_score_arrays['wcorr'])
        np.testing.assert_allclose(rebuilt.real_score_arrays['coverage'], obj.real_score_arrays['coverage'])
        self.assertEqual(rebuilt.enable_heuristic_shuffle_scoring, obj.enable_heuristic_shuffle_scoring)
        self.assertTrue(rebuilt.enable_heuristic_shuffle_scoring)
        self.assertEqual(rebuilt.n_completed_shuffles, obj.n_completed_shuffles)
        self.assertEqual(rebuilt.n_completed_shuffles, 3)
        rebuilt_result = rebuilt.real_decoder_ripple_filter_epochs_decoder_result_dict['long_LR']
        self.assertFalse(isinstance(rebuilt_result, dict))
        self.assertTrue(hasattr(rebuilt_result, 'num_filter_epochs'))
        self.assertEqual(rebuilt_result.num_filter_epochs, 2)


    def test_coerce_decoded_epochs_result_dict_rebuilds_from_num_filter_epochs_dict(self):
        result_as_dict = _minimal_decoded_result_dict(n_epochs=2)
        decoder_result_dict = {'long_LR': result_as_dict, 'long_RL': None, 'short_LR': _decoded_result_mock(n_epochs=2)}
        coerced = DecodedEpochScoreShuffle._coerce_decoded_epochs_result_dict(decoder_result_dict)
        self.assertIsInstance(coerced['long_LR'], DecodedFilterEpochsResult)
        self.assertEqual(coerced['long_LR'].num_filter_epochs, 2)
        self.assertIsNone(coerced['long_RL'])
        self.assertTrue(hasattr(coerced['short_LR'], 'num_filter_epochs'))
        self.assertEqual(coerced['short_LR'].num_filter_epochs, 2)


    def test_ensure_real_score_arrays_coerces_dict_values_without_attribute_error(self):
        obj = _make_shuffle_object(n_epochs=2, n_decoders=4, n_shuffles=3)
        obj.real_score_arrays = {'wcorr': np.asarray(obj.real_decoder_ripple_weighted_corr_arr)}
        obj.real_decoder_ripple_filter_epochs_decoder_result_dict = {name: _minimal_decoded_result_dict(n_epochs=2) for name in DEFAULT_SHUFFLE_DECODER_NAMES}
        obj.enable_heuristic_shuffle_scoring = True
        obj.track_templates = _TemplatesMock(xbin=np.linspace(0.0, 200.0, 9), track_length_dict={'long_LR': 200.0, 'long_RL': 200.0, 'short_LR': 140.0, 'short_RL': 140.0})
        obj._ensure_real_score_arrays(decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
        coerced_result = obj.real_decoder_ripple_filter_epochs_decoder_result_dict['long_LR']
        self.assertTrue(hasattr(coerced_result, 'num_filter_epochs'))
        self.assertEqual(int(coerced_result.num_filter_epochs), 2)


    def test_ensure_real_score_arrays_skips_heuristic_scoring_when_travel_present(self):
        obj = _make_shuffle_object(n_epochs=2, n_decoders=4, n_shuffles=3)
        self.assertIn('travel', obj.real_score_arrays)
        obj.enable_heuristic_shuffle_scoring = True
        obj.track_templates = _TemplatesMock(xbin=np.linspace(0.0, 200.0, 9), track_length_dict={'long_LR': 200.0, 'long_RL': 200.0, 'short_LR': 140.0, 'short_RL': 140.0})
        obj.real_decoder_ripple_filter_epochs_decoder_result_dict = {name: _decoded_result_mock(n_epochs=2) for name in DEFAULT_SHUFFLE_DECODER_NAMES}
        with patch.object(HeuristicReplayScoring, 'compute_shuffle_heuristic_score_arrays') as mocked_heuristic:
            obj._ensure_real_score_arrays(decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
            mocked_heuristic.assert_not_called()


    def test_load_and_append_7_tuple_merges_missing_real_score_keys(self):
        real_wcorr = np.array([[0.13, -0.24, 0.35, -0.46], [0.57, 0.68, -0.79, 0.81]], dtype=float)
        loaded_wcorr = real_wcorr + 0.05
        loaded_score_arrays_list = [{'wcorr': loaded_wcorr, 'coverage': loaded_wcorr + 0.2}]
        real_score_arrays = {'wcorr': real_wcorr, 'coverage': real_wcorr + 0.3}
        payload = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [], np.stack([loaded_wcorr], axis=0), None, loaded_score_arrays_list, real_score_arrays)
        obj = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(2),
            real_decoder_ripple_weighted_corr_arr=real_wcorr,
            real_score_arrays={'wcorr': real_wcorr},
            output_extracted_result_wcorrs_list=[],
            output_extracted_score_arrays_list=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir).joinpath('shuffle_7tuple_merge.pkl')
            saveData(pkl_path, payload)
            obj.load_and_append_shuffle_data(filepath=pkl_path)

        np.testing.assert_allclose(obj.real_score_arrays['wcorr'], real_wcorr)
        np.testing.assert_allclose(obj.real_score_arrays['coverage'], real_wcorr + 0.3)


    def test_save_data_round_trip_7_tuple(self):
        src = _make_shuffle_object(n_epochs=2, n_decoders=4, n_shuffles=3)
        dst = DecodedEpochScoreShuffle(
            filtered_epochs_df=_epochs_df(2),
            real_decoder_ripple_weighted_corr_arr=src.real_decoder_ripple_weighted_corr_arr,
            real_score_arrays={},
            output_extracted_result_wcorrs_list=[],
            output_extracted_score_arrays_list=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir).joinpath('shuffle_save_roundtrip.pkl')
            src.save_data(pkl_path)
            dst.load_and_append_shuffle_data(filepath=pkl_path)

        self.assertEqual(len(dst.output_extracted_score_arrays_list), 3)
        self.assertCountEqual(list(dst.real_score_arrays.keys()), list(src.real_score_arrays.keys()))
        np.testing.assert_allclose(dst.real_score_arrays['coverage'], src.real_score_arrays['coverage'])
        np.testing.assert_allclose(dst.output_extracted_score_arrays_list[0]['travel'], src.output_extracted_score_arrays_list[0]['travel'])


    def test_init_from_data_only_file_3_5_and_7_tuple(self):
        real_wcorr = np.array([[0.14, -0.25, 0.36, -0.47], [0.58, 0.69, -0.71, 0.82]], dtype=float)
        loaded_wcorr = real_wcorr + 0.02
        real_score_arrays = {'wcorr': real_wcorr, 'coverage': real_wcorr + 0.1}
        loaded_score_arrays_list = [{'wcorr': loaded_wcorr, 'coverage': loaded_wcorr + 0.1}]
        payload_3 = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [])
        real_decode_dict = {'long_LR': 'sentinel_decode'}
        payload_5 = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [], np.stack([loaded_wcorr], axis=0), real_decode_dict)
        payload_7 = ([_wcorr_df_dict(loaded_wcorr)], real_wcorr, [], np.stack([loaded_wcorr], axis=0), None, loaded_score_arrays_list, real_score_arrays)

        def _stub_init_from_templates(*args, **kwargs):
            return DecodedEpochScoreShuffle(
                filtered_epochs_df=_epochs_df(2),
                real_decoder_ripple_weighted_corr_arr=None,
                real_score_arrays={},
                output_extracted_result_wcorrs_list=[],
                output_extracted_score_arrays_list=[],
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_3 = Path(tmp_dir).joinpath('init_3tuple.pkl')
            pkl_5 = Path(tmp_dir).joinpath('init_5tuple.pkl')
            pkl_7 = Path(tmp_dir).joinpath('init_7tuple.pkl')
            saveData(pkl_3, payload_3)
            saveData(pkl_5, payload_5)
            saveData(pkl_7, payload_7)
            with patch.object(DecodedEpochScoreShuffle, 'init_from_templates', side_effect=_stub_init_from_templates):
                obj_3 = DecodedEpochScoreShuffle.init_from_data_only_file(filepath=pkl_3, curr_active_pipeline=None)
                obj_5 = DecodedEpochScoreShuffle.init_from_data_only_file(filepath=pkl_5, curr_active_pipeline=None)
                obj_7 = DecodedEpochScoreShuffle.init_from_data_only_file(filepath=pkl_7, curr_active_pipeline=None)

        np.testing.assert_allclose(obj_3.real_decoder_ripple_weighted_corr_arr, real_wcorr)
        self.assertCountEqual(list(obj_3.real_score_arrays.keys()), ['wcorr'])
        self.assertEqual(len(obj_3.output_extracted_score_arrays_list), 1)
        np.testing.assert_allclose(obj_3.output_extracted_score_arrays_list[0]['wcorr'], loaded_wcorr)

        self.assertEqual(obj_5.real_decoder_ripple_filter_epochs_decoder_result_dict, real_decode_dict)
        self.assertCountEqual(list(obj_5.real_score_arrays.keys()), ['wcorr'])
        np.testing.assert_allclose(obj_5.output_extracted_score_arrays_list[0]['wcorr'], loaded_wcorr)

        np.testing.assert_allclose(obj_7.real_score_arrays['coverage'], real_wcorr + 0.1)
        self.assertCountEqual(list(obj_7.output_extracted_score_arrays_list[0].keys()), ['wcorr', 'coverage'])
        np.testing.assert_allclose(obj_7.output_extracted_score_arrays_list[0]['coverage'], loaded_wcorr + 0.1)


class TestHeuristicShuffleScoreArrays(unittest.TestCase):
    """Mocked decode + templates; compute_shuffle_heuristic_score_arrays does not mutate filter_epochs."""

    def test_compute_shuffle_heuristic_score_arrays_shapes_and_no_filter_epochs_mutation(self):
        from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring

        n_epochs, n_time, n_pos = 2, 4, 8
        xbin = np.linspace(0.0, 200.0, n_pos + 1)
        xbin_centers = 0.5 * (xbin[:-1] + xbin[1:])
        time_window_centers = np.array([0.025, 0.075, 0.125, 0.175])
        time_bin_edges = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
        filter_epochs = pd.DataFrame({'start': [0.0, 1.0], 'stop': [0.2, 1.2], 'duration': [0.2, 0.2]})
        filter_epochs_before = filter_epochs.copy(deep=True)

        peak_bins_by_epoch = [np.array([1, 2, 3, 4]), np.array([5, 4, 3, 2])]
        most_likely_positions_list = []
        p_x_given_n_list = []
        for peak_bins in peak_bins_by_epoch:
            posterior = np.full((n_pos, n_time), 0.01, dtype=float)
            for t_idx, bin_idx in enumerate(peak_bins):
                posterior[:, t_idx] = 0.01
                posterior[int(bin_idx), t_idx] = 1.0
                posterior[:, t_idx] = posterior[:, t_idx] / np.sum(posterior[:, t_idx])
            ## END for t_idx, bin_idx in enumerate(peak_bins)....

            p_x_given_n_list.append(posterior)
            most_likely_positions_list.append(xbin_centers[peak_bins])
        ## END for peak_bins in peak_bins_by_epoch....

        decoded_result = _DecodedResultMock(
            num_filter_epochs=n_epochs,
            most_likely_positions_list=most_likely_positions_list,
            p_x_given_n_list=p_x_given_n_list,
            nbins=np.full(n_epochs, n_time, dtype=int),
            time_window_centers=[time_window_centers.copy() for _epoch_idx in np.arange(n_epochs)],
            time_bin_edges=[time_bin_edges.copy() for _epoch_idx in np.arange(n_epochs)],
            filter_epochs=filter_epochs,
        )
        decoded_dict = {name: decoded_result for name in DEFAULT_SHUFFLE_DECODER_NAMES}
        templates = _TemplatesMock(xbin=xbin, track_length_dict={'long_LR': 200.0, 'long_RL': 200.0, 'short_LR': 140.0, 'short_RL': 140.0})

        with patch.object(HeuristicReplayScoring, 'compute_pho_heuristic_replay_scores') as mocked_old_path:
            score_arrays = HeuristicReplayScoring.compute_shuffle_heuristic_score_arrays(track_templates=templates, a_decoded_filter_epochs_decoder_result_dict=decoded_dict, decoder_names=deepcopy(DEFAULT_SHUFFLE_DECODER_NAMES))
            mocked_old_path.assert_not_called()

        n_decoders = len(DEFAULT_SHUFFLE_DECODER_NAMES)
        expected_keys = {
            'travel', 'coverage', 'avg_jump_cm',
            'mseq_len', 'mseq_len_ignoring_intrusions', 'mseq_len_ignoring_intrusions_and_repeats',
            'mseq_len_ratio_ignoring_intrusions_and_repeats', 'mseq_tcov', 'mseq_dtrav',
            'forwardicity', 'ratio_major_aligned_bins', 'ratio_decoder_aligned_bins',
        }
        self.assertTrue(expected_keys.issubset(set(score_arrays.keys())))
        for metric_name, metric_arr in score_arrays.items():
            self.assertEqual(metric_arr.shape, (n_epochs, n_decoders), msg=metric_name)
        ## END for metric_name, metric_arr in score_arrays.items()....

        self.assertIs(decoded_result.filter_epochs, filter_epochs)
        pd.testing.assert_frame_equal(decoded_result.filter_epochs, filter_epochs_before)
        pd.testing.assert_frame_equal(filter_epochs, filter_epochs_before)


try:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
except Exception:
    DecoderDecodedEpochsResult = None


@unittest.skipIf(DecoderDecodedEpochsResult is None, 'DecoderDecodedEpochsResult import requires optional GUI deps')
class TestAssignBooleanColumnFromLocIndicies(unittest.TestCase):
    """CSV export used `.iloc` with `.idxmin()` labels after dropna, raising IndexError (676 vs 669 rows)."""

    def test_gapped_index_labels_do_not_use_iloc(self):
        a_df = pd.DataFrame({'ripple_start_t': np.arange(4, dtype=float)}, index=[0, 1, 3, 676])
        n_set = DecoderDecodedEpochsResult._assign_boolean_column_from_loc_indicies(a_df, column_name='is_user_annotated_epoch', loc_indicies=np.array([1, 676, 999]))
        self.assertEqual(n_set, 2)
        self.assertEqual(len(a_df), 4)
        np.testing.assert_array_equal(a_df['is_user_annotated_epoch'].to_numpy(), np.array([False, True, False, True]))


    def test_try_add_is_user_annotated_epoch_column_with_gapped_index(self):
        starts = np.array([10.0, 20.0, 30.0, 40.0])
        a_df = pd.DataFrame({'start': starts, 'stop': starts + 0.2, 'label': np.arange(4), 'ripple_start_t': starts}, index=[0, 1, 3, 676])
        selected_times = np.array([[20.0, 20.2], [40.0, 40.2]])
        did_update = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(a_df, any_good_selected_epoch_times=selected_times, t_column_names=['ripple_start_t'])
        self.assertTrue(did_update)
        np.testing.assert_array_equal(a_df['is_user_annotated_epoch'].to_numpy(), np.array([False, True, False, True]))
