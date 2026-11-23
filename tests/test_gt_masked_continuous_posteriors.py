"""Smoke test for plot_gt_masked_continuous_1D_decoder_posteriors using mock pipeline objects (no real session data needed)."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
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
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_gt_masked_continuous_1D_decoder_posteriors

DECODER_NAMES = ('long_LR', 'long_RL', 'short_LR', 'short_RL')


class _TruthAccessorMock:
    """ Mimics `df.time_point_event.adding_true_decoder_identifier(...)` by assigning a known truth decoder per time bin. """
    def __init__(self, df: pd.DataFrame, truth_names_by_t: dict):
        self._df = df
        self._truth_names_by_t = truth_names_by_t


    def adding_true_decoder_identifier(self, t_start, t_delta, t_end):
        out_df = self._df.copy()
        out_df['maze_id'] = 0
        out_df['is_LR_dir'] = True
        out_df['truth_decoder_name'] = out_df['t'].map(self._truth_names_by_t) # NaN for non-lap bins
        return out_df


class _MockDF(pd.DataFrame):
    @property
    def time_point_event(self):
        return _TruthAccessorMock(self, self.attrs['_truth_names_by_t'])


class _LapsMock:
    def __init__(self, laps_df: pd.DataFrame):
        self._laps_df = laps_df


    def adding_true_decoder_identifier(self, t_start, t_delta, t_end):
        return self._laps_df.copy()


class _PositionMock:
    def __init__(self, pos_df: pd.DataFrame):
        self._pos_df = pos_df


    def to_dataframe(self):
        return self._pos_df.copy()


def _build_mock_pipeline(n_time: int = 40, n_pos: int = 6, time_bin_size: float = 0.1):
    time_window_centers = (np.arange(n_time) + 0.5) * time_bin_size
    xbin = np.linspace(0.0, 120.0, n_pos + 1)
    ## GT schedule: first 4 bins non-lap (no truth), then one 9-bin lap per decoder. Laps are placed to avoid straddling t_delta=2.0 (maze_id boundary):
    ## long_LR bins 4-12 (t 0.45-1.25), long_RL bins 13-19 (t 1.35-1.95), short_LR bins 22-30 (t 2.25-3.05), short_RL bins 31-39 (t 3.15-3.95). Bins 20-21 stay non-lap.
    ## lap_dir 0->LR, 1->RL (matches TimePointEventAccessor.adding_true_decoder_identifier)
    lap_specs = [('long_LR', 0, 0, 4, 12), ('long_RL', 0, 1, 13, 19), ('short_LR', 1, 0, 22, 30), ('short_RL', 1, 1, 31, 39)]
    laps_df = pd.DataFrame([{'lap_id': lap_id, 'start': float(time_window_centers[first_bin] - time_bin_size / 2.0), 'stop': float(time_window_centers[last_bin] + time_bin_size / 2.0), 'maze_id': maze_id, 'lap_dir': lap_dir, 'is_LR_dir': (lap_dir == 0), 'truth_decoder_name': name} for lap_id, (name, maze_id, lap_dir, first_bin, last_bin) in enumerate(lap_specs)])
    truth_sequence = [None] * n_time
    for name, maze_id, lap_dir, first_bin, last_bin in lap_specs:
        truth_sequence[first_bin:last_bin + 1] = [name] * (last_bin - first_bin + 1)
    ## END for name, maze_id, lap_dir, first_bin, last_bin in lap_specs...


    decoders_dict = {k: SimpleNamespace(xbin=xbin) for k in DECODER_NAMES}
    track_templates = SimpleNamespace(get_decoders_dict=lambda: decoders_dict)
    directional_laps_results = SimpleNamespace(get_templates=lambda minimum_inclusion_fr_Hz, included_qclu_values: track_templates)

    continuously_decoded_dict = {}
    for a_decoder_name in DECODER_NAMES:
        rng = np.random.default_rng(abs(hash(a_decoder_name)) % (2**32))
        posterior = rng.uniform(0.0, 1.0, size=(n_pos, n_time))
        posterior = posterior / posterior.sum(axis=0, keepdims=True)
        marginal_x = {'p_x_given_n': posterior, 'most_likely_positions_1D': np.interp(np.arange(n_time), [0, n_time - 1], [10.0, 110.0])}
        continuously_decoded_dict[a_decoder_name] = SimpleNamespace(p_x_given_n_list=[posterior.copy()], marginal_x_list=[marginal_x], time_bin_containers=[SimpleNamespace(centers=time_window_centers.copy(), edges=np.arange(n_time + 1) * time_bin_size)])
    ## END for a_decoder_name in DECODER_NAMES...

    directional_decoders_decode_result = SimpleNamespace(continuously_decoded_result_cache_dict={(time_bin_size, None): continuously_decoded_dict},
        most_recent_decoding_time_bin_size=time_bin_size, most_recent_continuously_decoded_dict=continuously_decoded_dict)

    pos_df = pd.DataFrame({'t': time_window_centers, 'x': np.interp(np.arange(n_time), [0, n_time - 1], [10.0, 110.0]), 'lin_pos': np.interp(np.arange(n_time), [0, n_time - 1], [0.0, 200.0])})
    sess = SimpleNamespace(laps=_LapsMock(laps_df), position=_PositionMock(pos_df))
    computation_config = SimpleNamespace(rank_order_shuffle_analysis=SimpleNamespace(minimum_inclusion_fr_Hz=2.0, included_qclu_values=[1, 2]))
    global_computation_results = SimpleNamespace(computation_config=computation_config, computed_data={'DirectionalLaps': directional_laps_results, 'DirectionalDecodersDecoded': directional_decoders_decode_result})
    pipeline = SimpleNamespace(global_computation_results=global_computation_results, sess=sess, find_LongShortDelta_times=lambda: (0.0, 2.0, 4.0))
    return pipeline, time_window_centers, truth_sequence, continuously_decoded_dict


class TestGTMaskedContinuousPosteriors(unittest.TestCase):
    def test_gt_masking_and_overlays(self):
        pipeline, time_window_centers, truth_sequence, continuously_decoded_dict = _build_mock_pipeline()
        fig, axs, out_artists = plot_gt_masked_continuous_1D_decoder_posteriors(pipeline)
        try:
            self.assertEqual(len(axs), len(DECODER_NAMES))
            self.assertEqual(set(out_artists.keys()), set(DECODER_NAMES))
            for i, a_decoder_name in enumerate(DECODER_NAMES):
                expected_active = np.array([name == a_decoder_name for name in truth_sequence])
                im = out_artists[a_decoder_name]['im_posterior_x']
                self.assertIsNotNone(im, f'{a_decoder_name}: posterior image missing')
                plotted = np.ma.filled(np.ma.asarray(im.get_array()).astype(float), np.nan) # handles both plain and masked arrays
                self.assertTrue(np.all(np.isnan(plotted[:, ~expected_active])), f'{a_decoder_name}: non-GT posterior columns not masked')
                self.assertTrue(np.all(np.isfinite(plotted[:, expected_active])), f'{a_decoder_name}: GT-active posterior columns unexpectedly masked')
                ## Most-likely line is NaN-padded outside GT-active bins:
                line = out_artists[a_decoder_name]['line_most_likely_position'][0]
                line_y = np.asarray(line.get_ydata(), dtype=float)
                self.assertEqual(len(line_y), len(time_window_centers))
                self.assertTrue(np.all(np.isnan(line_y[~expected_active])), f'{a_decoder_name}: non-GT most-likely positions not NaN')
                ## Measured position drawn on every row, using 'lin_pos' by default:
                measured_line = out_artists[a_decoder_name]['line_measured_position'][0]
                self.assertIsNotNone(measured_line, f'{a_decoder_name}: measured position line missing')
                self.assertEqual(measured_line.get_label(), 'measured lin_pos', f'{a_decoder_name}: measured overlay is not lin_pos')
            ## END for i, a_decoder_name in enumerate(DECODER_NAMES)...

            ## Original cached posterior must not be mutated by the masking:
            for a_decoder_name in DECODER_NAMES:
                cached_posterior = continuously_decoded_dict[a_decoder_name].marginal_x_list[0]['p_x_given_n']
                self.assertTrue(np.all(np.isfinite(cached_posterior)), f'{a_decoder_name}: cached posterior was mutated in-place')
            ## END for a_decoder_name in DECODER_NAMES...

        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


    def test_missing_decoder_fails_loudly(self):
        pipeline, _, _, continuously_decoded_dict = _build_mock_pipeline()
        del continuously_decoded_dict['short_RL'] ## simulate a missing continuous result
        with self.assertRaises(AssertionError):
            plot_gt_masked_continuous_1D_decoder_posteriors(pipeline)


if __name__ == '__main__':
    unittest.main()
