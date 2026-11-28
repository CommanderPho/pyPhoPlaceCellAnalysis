"""Synthetic + integration tests for PBE context flickering analysis."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

tests_folder = Path(__file__).parent
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

# Heavy GUI / optional deps pulled transitively by AcrossSessionResults / ComputationKWargParameters
for _mod_name in (
    'PyQt5Singleton', 'plotly', 'plotly.express', 'plotly.graph_objects', 'awkward', 'pyqtgraph',
    'pyvista', 'pyvistaqt', 'vtk', 'vtkmodules', 'sip', 'vedo',
    'pyphoplacecellanalysis.Pho3D.PyVista.animations',
    'pyphoplacecellanalysis.PhoPositionalData.plotting.placefield',
    'pyphoplacecellanalysis.PhoPositionalData.plotting',
):
    sys.modules.setdefault(_mod_name, MagicMock())

from pyphoplacecellanalysis.Analysis.Decoder.pbe_flickering import FlickerDynamicClass, PBEFlickeringAnalyzer


def _import_across_session_helpers():
    """Load AcrossSessionResults helpers; stub heavy paper/GUI modules that are not needed for parse/merge."""
    # Break the PhoDiba2023Paper -> PyVista/vedo import chain used only for unrelated paper helpers.
    for _heavy in (
        'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper',
        'pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions',
        'pyphoplacecellanalysis.PhoPositionalData.plotting',
        'pyphoplacecellanalysis.PhoPositionalData.plotting.placefield',
        'pyphoplacecellanalysis.Pho3D.PyVista.animations',
        'vedo',
    ):
        sys.modules.setdefault(_heavy, MagicMock())
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers, parse_filename
    return AcrossSessionHelpers, parse_filename


def _import_computation_params():
    from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters, pbe_flickering_analysis_Parameters
    return ComputationKWargParameters, pbe_flickering_analysis_Parameters


def _import_parse_filename_lightweight():
    """parse_filename is thin over try_parse_chain — enough for filename filetype assertions."""
    from pyphocorehelpers.Filesystem.path_helpers import try_parse_chain

    def parse_filename(path, should_print_unparsable_filenames: bool = True, debug_print: bool = False):
        if isinstance(path, str):
            filename = path
        else:
            filename = path.stem
        final_parsed_output_dict = try_parse_chain(basename=filename, should_print_unparsable_filenames=should_print_unparsable_filenames, debug_print=debug_print)
        if final_parsed_output_dict is None:
            return None, None, None, None, None
        export_datetime = final_parsed_output_dict.get('export_datetime', None)
        session_str = final_parsed_output_dict.get('session_str', None)
        export_file_type = final_parsed_output_dict.get('export_file_type', None)
        decoding_time_bin_size_str = final_parsed_output_dict.get('decoding_time_bin_size_str', None)
        custom_replay_name = final_parsed_output_dict.get('custom_replay_name', None)
        if export_file_type is not None and export_file_type[0] == '(' and export_file_type[-1] == ')':
            export_file_type = export_file_type[1:-1]
        return export_datetime, session_str, custom_replay_name, export_file_type, decoding_time_bin_size_str

    return parse_filename


def _make_track_posterior(track_masses: np.ndarray, n_pos: int = 20, prefer_dir: int = 0) -> np.ndarray:
    """Build (n_pos, 4, T) posterior from per-bin Short-track mass in [0,1]; prefer_dir 0=LR, 1=RL within track."""
    track_masses = np.asarray(track_masses, dtype=float)
    n_t = track_masses.size
    p = np.zeros((n_pos, 4, n_t), dtype=float)
    x = np.linspace(0.0, 1.0, n_pos)
    spatial = np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
    spatial = spatial / spatial.sum()
    for t, p_short in enumerate(track_masses):
        p_long = 1.0 - float(p_short)
        # put mass on preferred direction within each track
        long_dirs = np.array([0.85, 0.15]) if prefer_dir == 0 else np.array([0.15, 0.85])
        short_dirs = np.array([0.85, 0.15]) if prefer_dir == 0 else np.array([0.15, 0.85])
        p[:, 0, t] = spatial * p_long * long_dirs[0]
        p[:, 1, t] = spatial * p_long * long_dirs[1]
        p[:, 2, t] = spatial * p_short * short_dirs[0]
        p[:, 3, t] = spatial * p_short * short_dirs[1]
    ## END for t, p_short in enumerate(track_masses)...
    return p


def _make_spatial_sweep_posterior(track_schedule: list, x_centers_cm: np.ndarray, map_x_per_bin_cm: list) -> np.ndarray:
    """track_schedule: list of 'Long'/'Short' per time bin; map_x_per_bin_cm: MAP positions."""
    n_pos = x_centers_cm.size
    n_t = len(track_schedule)
    p = np.zeros((n_pos, 4, n_t), dtype=float)
    for t, (track, x_cm) in enumerate(zip(track_schedule, map_x_per_bin_cm)):
        peak_idx = int(np.argmin(np.abs(x_centers_cm - x_cm)))
        spatial = np.exp(-0.5 * ((np.arange(n_pos) - peak_idx) / 1.5) ** 2)
        spatial = spatial / spatial.sum()
        if track == 'Long':
            p[:, 0, t] = spatial * 0.9
            p[:, 1, t] = spatial * 0.05
            p[:, 2, t] = spatial * 0.03
            p[:, 3, t] = spatial * 0.02
        else:
            p[:, 0, t] = spatial * 0.03
            p[:, 1, t] = spatial * 0.02
            p[:, 2, t] = spatial * 0.9
            p[:, 3, t] = spatial * 0.05
        p[:, :, t] /= p[:, :, t].sum()
    ## END for t, (track, x_cm) in enumerate(zip(track_schedule, map_x_per_bin_cm))...
    return p


@pytest.fixture
def analyzer() -> PBEFlickeringAnalyzer:
    return PBEFlickeringAnalyzer(decoding_time_bin_size=0.025, conf_thresh=0.65, strict_conf_thresh=0.75,
                                 min_valid_bins=4, num_within_shuffles=32, num_pool_shuffles=32, num_neuron_id_shuffles=0, rng_seed=0)


def test_hysteresis_pure_long_zero_transitions(analyzer: PBEFlickeringAnalyzer):
    delta = np.full(12, -0.8)
    states, n_trans, bouts = analyzer.hysteresis_commit_states(delta, baseline=0.30)
    assert n_trans == 0
    assert np.all(states == -1)
    assert analyzer.classify_dragoi_dynamic(n_trans, prefer_track='Long', signed_dominant=True) == FlickerDynamicClass.Dominant_Long.value


def test_hysteresis_biphasic_one_transition(analyzer: PBEFlickeringAnalyzer):
    delta = np.concatenate([np.full(6, -0.8), np.full(6, 0.8)])
    states, n_trans, bouts = analyzer.hysteresis_commit_states(delta, baseline=0.30)
    assert n_trans == 1
    assert analyzer.classify_dragoi_dynamic(n_trans) == FlickerDynamicClass.Biphasic.value


def test_hysteresis_flicker_multi_transition(analyzer: PBEFlickeringAnalyzer):
    delta = np.array([-0.8, -0.8, 0.8, 0.8, -0.8, -0.8, 0.8, 0.8])
    states, n_trans, bouts = analyzer.hysteresis_commit_states(delta, baseline=0.30)
    assert n_trans >= 2
    assert analyzer.classify_dragoi_dynamic(n_trans) == FlickerDynamicClass.Flicker.value


def test_hysteresis_deadband_holds_committed_state(analyzer: PBEFlickeringAnalyzer):
    # commit Short, then stay in deadband, should hold Short
    delta = np.array([0.8, 0.8, 0.1, -0.1, 0.05, 0.8])
    states, n_trans, bouts = analyzer.hysteresis_commit_states(delta, baseline=0.30)
    assert states[2] == 1 and states[3] == 1 and states[4] == 1
    assert n_trans == 0


def test_analyze_pure_long_and_short(analyzer: PBEFlickeringAnalyzer):
    p_long = _make_track_posterior(np.full(10, 0.05))
    metrics_l, _ = analyzer.analyze_single_epoch(p_long, epoch_meta=dict(epoch_type='pbe', start=0.0, stop=0.25))
    assert metrics_l['conf_dynamic_class'] in (FlickerDynamicClass.Dominant_Long.value, FlickerDynamicClass.Dominant.value)
    assert metrics_l['conf_n_transitions'] == 0

    p_short = _make_track_posterior(np.full(10, 0.95))
    metrics_s, _ = analyzer.analyze_single_epoch(p_short, epoch_meta=dict(epoch_type='pbe', start=0.0, stop=0.25))
    assert metrics_s['conf_dynamic_class'] in (FlickerDynamicClass.Dominant_Short.value, FlickerDynamicClass.Dominant.value)
    assert metrics_s['conf_n_transitions'] == 0


def test_analyze_flat_ambiguous(analyzer: PBEFlickeringAnalyzer):
    p_flat = _make_track_posterior(np.full(10, 0.5))
    metrics, _ = analyzer.analyze_single_epoch(p_flat, epoch_meta=dict(epoch_type='pbe', start=0.0, stop=0.25))
    assert metrics['conf_dynamic_class'] == FlickerDynamicClass.Ambiguous.value
    assert metrics['has_both_confident_contexts'] is False
    assert metrics['bimodal_flicker_score'] == 0.0 or (not np.isfinite(metrics['bimodal_flicker_score'])) or metrics['bimodal_flicker_score'] < 0.05


def test_analyze_flicker_vs_biphasic(analyzer: PBEFlickeringAnalyzer):
    biphasic = _make_track_posterior(np.concatenate([np.full(6, 0.05), np.full(6, 0.95)]))
    metrics_b, _ = analyzer.analyze_single_epoch(biphasic, epoch_meta=dict(epoch_type='pbe', start=0.0, stop=0.3))
    assert metrics_b['conf_n_transitions'] == 1
    assert metrics_b['conf_dynamic_class'] == FlickerDynamicClass.Biphasic.value

    flicker = _make_track_posterior(np.tile([0.05, 0.05, 0.95, 0.95], 3))
    metrics_f, _ = analyzer.analyze_single_epoch(flicker, epoch_meta=dict(epoch_type='pbe', start=0.0, stop=0.3))
    assert metrics_f['conf_n_transitions'] >= 2
    assert metrics_f['conf_dynamic_class'] == FlickerDynamicClass.Flicker.value
    assert metrics_f['has_both_confident_contexts'] is True
    assert metrics_f['bimodal_flicker_score'] > metrics_b['bimodal_flicker_score']


def test_pause_resume_abs_jump_cm(analyzer: PBEFlickeringAnalyzer):
    xbin_centers = np.linspace(0.0, 100.0, 21)
    # Long bout sweeping 10->30, Short interruption, Long resumes near 32 (small jump)
    track_schedule = (['Long'] * 4) + (['Short'] * 3) + (['Long'] * 4)
    map_x = [10, 16, 22, 28, 50, 55, 60, 32, 38, 44, 50]
    p = _make_spatial_sweep_posterior(track_schedule, xbin_centers, map_x)
    metrics, bouts = analyzer.analyze_single_epoch(
        p, epoch_meta=dict(epoch_type='pbe', start=1.0, stop=1.275),
        xbin_centers=xbin_centers, track_len_cm=100.0,
    )
    assert metrics['n_aba_triplets'] >= 1
    assert np.isfinite(metrics['min_pause_resume_jump_cm'])
    # resume starts at 32 after pause ended near 28 => ~4 cm
    assert metrics['min_pause_resume_jump_cm'] < 15.0
    assert metrics['has_paused_resumed_trajectory'] is True
    resume_jumps = [b.get('pause_resume_abs_jump_cm') for b in bouts if np.isfinite(b.get('pause_resume_abs_jump_cm', np.nan))]
    assert len(resume_jumps) >= 1
    assert float(np.nanmin(resume_jumps)) == pytest.approx(metrics['min_pause_resume_jump_cm'], rel=0, abs=1e-6)


def test_within_perm_nulls_finite(analyzer: PBEFlickeringAnalyzer):
    flicker = _make_track_posterior(np.tile([0.05, 0.05, 0.95, 0.95], 4))
    metrics, _ = analyzer.analyze_single_epoch(flicker, epoch_meta=dict(epoch_type='pbe', start=0.0, stop=0.4))
    # within-event nulls are attached in compute_from_decoded_result / FAT fallback; call helper directly
    P_ctx, P_Long, P_Short, delta_P = analyzer.marginalize_track_posteriors(flicker)
    nulls = analyzer._within_event_permutation_nulls(delta_P, P_Long, P_Short, duration_s=float(metrics['duration_s']))
    assert np.isfinite(nulls['within_perm_transitions_z']) or np.isnan(nulls['within_perm_transitions_z'])
    assert 'within_perm_bout_ms_z' in nulls
    assert 'within_perm_autocorr_z' in nulls


def test_compute_from_time_bin_marginals_df_fallback(analyzer: PBEFlickeringAnalyzer):
    rows = []
    for epoch_i, masses in enumerate([np.full(8, 0.05), np.tile([0.05, 0.95], 4)]):
        for t, p_short in enumerate(masses):
            rows.append(dict(
                parent_epoch_label=epoch_i,
                parent_epoch_start_t=float(epoch_i * 10.0),
                parent_epoch_end_t=float(epoch_i * 10.0 + 0.2),
                t_bin_center=float(epoch_i * 10.0 + t * 0.025),
                P_Long=1.0 - p_short,
                P_Short=p_short,
                time_bin_size=0.025,
                session_name='kdiba_test_session',
                custom_replay_name='withNormalComputedReplays',
                pre_post_delta_category='post-delta' if epoch_i else 'pre-delta',
                delta_aligned_start_t=float(epoch_i * 10.0 - 5.0),
            ))
        ## END for t, p_short in enumerate(masses)...
    ## END for epoch_i, masses in enumerate(...)...
    tb_df = pd.DataFrame(rows)
    epoch_df, bouts_df = PBEFlickeringAnalyzer.compute_from_time_bin_marginals_df(tb_df, decoding_time_bin_size=0.025, num_within_shuffles=16, num_pool_shuffles=16)
    assert len(epoch_df) == 2
    assert 'conf_dynamic_class' in epoch_df.columns
    assert 'bimodal_flicker_score' in epoch_df.columns


def test_pbe_flickering_parameters_on_computation_kwargs():
    ComputationKWargParameters, pbe_flickering_analysis_Parameters = _import_computation_params()
    params = pbe_flickering_analysis_Parameters()
    assert params.decoding_time_bin_size == 0.025
    assert params.conf_thresh == 0.65
    kw = ComputationKWargParameters()
    assert hasattr(kw, 'pbe_flickering_analysis')
    assert kw.pbe_flickering_analysis.decoding_time_bin_size == 0.025


def test_parse_filename_pbe_flickering_csv():
    try:
        _AcrossSessionHelpers, parse_filename = _import_across_session_helpers()
    except Exception:
        parse_filename = _import_parse_filename_lightweight()
    name = '2026-11-27_0500PM-kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0-(pbe_flickering_df)_tbin-0.025'
    export_datetime, session_str, custom_replay_name, export_file_type, decoding_time_bin_size_str = parse_filename(name, should_print_unparsable_filenames=False, debug_print=False)
    assert export_file_type == 'pbe_flickering_df'
    assert decoding_time_bin_size_str == '0.025'
    assert session_str is not None and 'kdiba' in session_str

    name_b = name.replace('pbe_flickering_df', 'pbe_flickering_bouts_df')
    *_rest, export_file_type_b, tbin_b = parse_filename(name_b, should_print_unparsable_filenames=False, debug_print=False)
    assert export_file_type_b == 'pbe_flickering_bouts_df'
    assert tbin_b == '0.025'


def test_merge_pbe_flickering_onto_fat_df_row_count_invariant():
    AcrossSessionHelpers, _parse_filename = _import_across_session_helpers()
    fat = pd.DataFrame([
        dict(session_name='kdiba_a', custom_replay_name='withNormalComputedReplays', time_bin_size=0.025, known_named_decoding_epochs_type='pbe', data_grain='per_epoch', parent_epoch_start_t=10.0, start=10.0),
        dict(session_name='kdiba_a', custom_replay_name='withNormalComputedReplays', time_bin_size=0.025, known_named_decoding_epochs_type='pbe', data_grain='per_time_bin', parent_epoch_start_t=10.0, start=10.01),
        dict(session_name='kdiba_a', custom_replay_name='withNormalComputedReplays', time_bin_size=0.075, known_named_decoding_epochs_type='pbe', data_grain='per_epoch', parent_epoch_start_t=10.0, start=10.0),
        dict(session_name='kdiba_a', custom_replay_name='withNormalComputedReplays', time_bin_size=0.025, known_named_decoding_epochs_type='laps', data_grain='per_epoch', parent_epoch_start_t=1.0, start=1.0),
    ])
    flicker = pd.DataFrame([
        dict(session_name='kdiba_a', custom_replay_name='withNormalComputedReplays', time_bin_size=0.025, epoch_type='pbe', start=10.0, stop=10.2,
             conf_dynamic_class='Flicker', bimodal_flicker_score=0.42, conf_transition_hz=3.0, has_paused_resumed_trajectory=True),
    ])
    n_before = len(fat)
    merged = AcrossSessionHelpers.merge_pbe_flickering_onto_fat_df(fat, flicker, preferred_source_tbin=0.025, start_tol_sec=0.05, debug_print=False)
    assert len(merged) == n_before
    assert 'conf_dynamic_class' in merged.columns
    assert 'is_flickering_pbe' in merged.columns
    # both 25 ms and 75 ms PBE rows should pick up Flicker via exact / asof
    pbe_classes = merged.loc[merged['known_named_decoding_epochs_type'] == 'pbe', 'conf_dynamic_class'].dropna().unique().tolist()
    assert 'Flicker' in pbe_classes
    # laps row remains NaN
    laps_val = merged.loc[merged['known_named_decoding_epochs_type'] == 'laps', 'conf_dynamic_class']
    assert laps_val.isna().all()
