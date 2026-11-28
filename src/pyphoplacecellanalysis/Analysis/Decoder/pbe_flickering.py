"""PBE context flickering and pause-and-resume trajectory analysis.

Implements literature-standard within-event context dynamics (Dragoi 2024 hysteresis;
Jezek/Posani/Tirole log-odds & sticky Markov persistence) plus manuscript Fig. 4d
paused-and-resumed spatial trajectory scoring, with shuffle null controls.
"""
from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from attrs import define, field, Factory
from nptyping import NDArray

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes


CONTEXT_NAMES_4: List[str] = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
TRACK_NAMES_2: List[str] = ['Long', 'Short']


class FlickerDynamicClass(str, Enum):
    """Dragoi-style co-representation dynamics class (optionally track-signed)."""
    Dominant = 'Dominant'
    Dominant_Long = 'Dominant_Long'
    Dominant_Short = 'Dominant_Short'
    Biphasic = 'Biphasic'
    Flicker = 'Flicker'
    Ambiguous = 'Ambiguous'
    Insufficient = 'Insufficient'


@define(slots=False, repr=False)
class PBEFlickeringResult:
    """Container for session-level flickering outputs stored under `computed_data['PBEFlickering']`."""
    is_global: bool = True
    decoding_time_bin_size: float = 0.025
    conf_thresh: float = 0.65
    strict_conf_thresh: float = 0.75
    min_num_unique_active_neurons_per_time_bin: int = 2
    min_valid_bins: int = 4
    epoch_df: Optional[pd.DataFrame] = None
    bouts_df: Optional[pd.DataFrame] = None
    params: Dict[str, Any] = field(default=Factory(dict))


@metadata_attributes(short_name='PBEFlickeringAnalyzer', tags=['flickering', 'PBE', 'context', 'Dragoi2024', 'replay'], input_requires=[], output_provides=['PBEFlickering'], uses=['TransitionMatrixComputations'], used_by=[], creation_date='2026-11-27 17:00', related_items=[])
class PBEFlickeringAnalyzer:
    """Quantify within-PBE Long/Short context flickering and pause-and-resume trajectories.

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.pbe_flickering import PBEFlickeringAnalyzer

        analyzer = PBEFlickeringAnalyzer(conf_thresh=0.65, decoding_time_bin_size=0.025)
        epoch_df, bouts_df = analyzer.compute_from_decoded_result(a_result, epoch_type='pbe', t_delta=t_delta, xbin_centers=xbin_centers)
    """

    def __init__(self, decoding_time_bin_size: float = 0.025, conf_thresh: float = 0.65, strict_conf_thresh: float = 0.75,
                 min_num_unique_active_neurons_per_time_bin: int = 2, min_num_spikes_per_bin: int = 1, min_valid_bins: int = 4,
                 num_within_shuffles: int = 500, num_pool_shuffles: int = 500, num_neuron_id_shuffles: int = 0,
                 hmm_self_stay_prob: float = 0.80, log_odds_clip: float = 20.0, rng_seed: Optional[int] = 0):
        self.decoding_time_bin_size = float(decoding_time_bin_size)
        self.conf_thresh = float(conf_thresh)
        self.strict_conf_thresh = float(strict_conf_thresh)
        self.min_num_unique_active_neurons_per_time_bin = int(min_num_unique_active_neurons_per_time_bin)
        self.min_num_spikes_per_bin = int(min_num_spikes_per_bin)
        self.min_valid_bins = int(min_valid_bins)
        self.num_within_shuffles = int(num_within_shuffles)
        self.num_pool_shuffles = int(num_pool_shuffles)
        self.num_neuron_id_shuffles = int(num_neuron_id_shuffles)
        self.hmm_self_stay_prob = float(hmm_self_stay_prob)
        self.log_odds_clip = float(log_odds_clip)
        self.rng = np.random.default_rng(rng_seed)


    @classmethod
    def _safe_nanstd(cls, x: NDArray, ddof: int = 1) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < (ddof + 1):
            return np.nan
        return float(np.nanstd(x, ddof=ddof))


    @classmethod
    def _safe_se(cls, x: NDArray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.nan
        return float(np.nanstd(x, ddof=1) / np.sqrt(x.size))


    @classmethod
    def _lag1_autocorr(cls, x: NDArray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        x0 = x[:-1]
        x1 = x[1:]
        if (np.nanstd(x0) < 1e-12) or (np.nanstd(x1) < 1e-12):
            return np.nan
        return float(np.corrcoef(x0, x1)[0, 1])


    @classmethod
    def marginalize_track_posteriors(cls, p_x_given_n: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Return (P_contexts[4,T], P_Long[T], P_Short[T], delta_P[T]) from (n_pos, 4, T) posterior."""
        assert p_x_given_n.ndim == 3, f'expected (n_pos, 4, T), got {np.shape(p_x_given_n)}'
        n_pos, n_ctx, n_t = np.shape(p_x_given_n)
        assert n_ctx == 4, f'expected 4 contexts, got {n_ctx}'
        P_ctx = np.nansum(p_x_given_n, axis=0)  # (4, T)
        denom = np.nansum(P_ctx, axis=0, keepdims=True)
        denom = np.where(denom > 0.0, denom, np.nan)
        P_ctx = P_ctx / denom
        P_Long = np.nansum(P_ctx[0:2, :], axis=0)
        P_Short = np.nansum(P_ctx[2:4, :], axis=0)
        # renormalize Long/Short in case of NaNs
        track_sum = P_Long + P_Short
        good = np.isfinite(track_sum) & (track_sum > 0.0)
        P_Long = np.where(good, P_Long / track_sum, np.nan)
        P_Short = np.where(good, P_Short / track_sum, np.nan)
        delta_P = P_Short - P_Long
        return P_ctx, P_Long, P_Short, delta_P


    @classmethod
    def classify_dragoi_dynamic(cls, n_transitions: int, prefer_track: Optional[str] = None, signed_dominant: bool = False) -> str:
        if n_transitions <= 0:
            if signed_dominant and (prefer_track is not None):
                if prefer_track == 'Long':
                    return FlickerDynamicClass.Dominant_Long.value
                if prefer_track == 'Short':
                    return FlickerDynamicClass.Dominant_Short.value
            return FlickerDynamicClass.Dominant.value
        if n_transitions == 1:
            return FlickerDynamicClass.Biphasic.value
        return FlickerDynamicClass.Flicker.value


    @classmethod
    def hysteresis_commit_states(cls, delta_P: NDArray, baseline: float) -> Tuple[NDArray, int, NDArray]:
        """Commit Long(-1)/Short(+1) with deadband hysteresis around 0.

        Returns:
            states: int array in {-1, 0, +1} (0 = never committed / ambiguous throughout)
            n_transitions: number of committed-state flips
            bout_lengths: contiguous committed bout lengths (in bins)
        """
        delta_P = np.asarray(delta_P, dtype=float)
        n = delta_P.size
        states = np.zeros(n, dtype=int)
        if n == 0 or (not np.isfinite(baseline)) or (baseline < 0):
            return states, 0, np.array([], dtype=int)

        committed = 0
        for t in range(n):
            v = delta_P[t]
            if not np.isfinite(v):
                states[t] = committed
                continue
            if v > baseline:
                committed = 1
            elif v < -baseline:
                committed = -1
            states[t] = committed
        ## END for t in range(n)...

        committed_mask = states != 0
        if not np.any(committed_mask):
            return states, 0, np.array([], dtype=int)

        committed_states = states[committed_mask]
        n_transitions = int(np.sum(committed_states[1:] != committed_states[:-1])) if committed_states.size > 1 else 0

        bout_lengths: List[int] = []
        run_len = 1
        for i in range(1, states.size):
            if (states[i] != 0) and (states[i] == states[i - 1]):
                run_len += 1
            else:
                if states[i - 1] != 0:
                    bout_lengths.append(run_len)
                run_len = 1
        ## END for i in range(1, states.size)...
        if states[-1] != 0:
            bout_lengths.append(run_len)

        return states, n_transitions, np.asarray(bout_lengths, dtype=int)


    @classmethod
    def sticky_hmm_viterbi(cls, P_Long: NDArray, P_Short: NDArray, self_stay_prob: float = 0.80) -> Tuple[NDArray, int, NDArray]:
        """2-state sticky HMM Viterbi on track marginal emissions. States: 0=Long, 1=Short."""
        P_Long = np.asarray(P_Long, dtype=float)
        P_Short = np.asarray(P_Short, dtype=float)
        n = P_Long.size
        if n == 0:
            return np.array([], dtype=int), 0, np.array([], dtype=int)

        stay = float(np.clip(self_stay_prob, 1e-6, 1.0 - 1e-6))
        switch = 1.0 - stay
        logA = np.log(np.array([[stay, switch], [switch, stay]], dtype=float))
        eps = 1e-12
        emit = np.vstack([np.clip(P_Long, eps, 1.0), np.clip(P_Short, eps, 1.0)])  # (2, T)
        logE = np.log(emit)

        dp = np.full((2, n), -np.inf, dtype=float)
        ptr = np.zeros((2, n), dtype=int)
        dp[:, 0] = np.log(0.5) + logE[:, 0]
        for t in range(1, n):
            for s in (0, 1):
                scores = dp[:, t - 1] + logA[:, s]
                ptr[s, t] = int(np.argmax(scores))
                dp[s, t] = scores[ptr[s, t]] + logE[s, t]
            ## END for s in (0, 1)...
        ## END for t in range(1, n)...

        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(dp[:, -1]))
        for t in range(n - 2, -1, -1):
            path[t] = ptr[path[t + 1], t + 1]
        ## END for t in range(n - 2, -1, -1)...

        n_transitions = int(np.sum(path[1:] != path[:-1])) if n > 1 else 0
        bout_lengths: List[int] = []
        run_len = 1
        for i in range(1, n):
            if path[i] == path[i - 1]:
                run_len += 1
            else:
                bout_lengths.append(run_len)
                run_len = 1
        ## END for i in range(1, n)...
        bout_lengths.append(run_len)
        return path, n_transitions, np.asarray(bout_lengths, dtype=int)


    @classmethod
    def compute_bimodal_scores(cls, P_Long: NDArray, P_Short: NDArray, conf_thresh: float = 0.75) -> Dict[str, float]:
        P_Long = np.asarray(P_Long, dtype=float)
        P_Short = np.asarray(P_Short, dtype=float)
        valid = np.isfinite(P_Long) & np.isfinite(P_Short)
        if not np.any(valid):
            return dict(max_P_Long=np.nan, max_P_Short=np.nan, has_both_confident_contexts=False, bimodal_co_rep_min_conf=np.nan, bimodal_flicker_score=np.nan, std_P_Short=np.nan)

        max_P_Long = float(np.nanmax(P_Long[valid]))
        max_P_Short = float(np.nanmax(P_Short[valid]))
        std_P_Short = cls._safe_nanstd(P_Short[valid])
        has_both = bool((max_P_Long >= conf_thresh) and (max_P_Short >= conf_thresh))
        bimodal_co_rep_min_conf = float(min(max_P_Long, max_P_Short))
        bimodal_flicker_score = float(4.0 * max(0.0, min(max_P_Long, max_P_Short) - 0.5) * (0.0 if not np.isfinite(std_P_Short) else std_P_Short))
        return dict(max_P_Long=max_P_Long, max_P_Short=max_P_Short, has_both_confident_contexts=has_both, bimodal_co_rep_min_conf=bimodal_co_rep_min_conf, bimodal_flicker_score=bimodal_flicker_score, std_P_Short=std_P_Short)


    @classmethod
    def compute_log_odds_metrics(cls, P_Long: NDArray, P_Short: NDArray, clip: float = 20.0) -> Dict[str, float]:
        eps = 1e-12
        P_Long = np.clip(np.asarray(P_Long, dtype=float), eps, 1.0)
        P_Short = np.clip(np.asarray(P_Short, dtype=float), eps, 1.0)
        L = np.clip(np.log(P_Short / P_Long), -clip, clip)
        valid = np.isfinite(L)
        if not np.any(valid):
            return dict(log_odds_mean=np.nan, log_odds_std=np.nan, log_odds_range=np.nan, log_odds_mean_abs_step=np.nan)
        Lv = L[valid]
        steps = np.abs(np.diff(L))
        steps = steps[np.isfinite(steps)]
        return dict(
            log_odds_mean=float(np.nanmean(Lv)),
            log_odds_std=cls._safe_nanstd(Lv),
            log_odds_range=float(np.nanmax(Lv) - np.nanmin(Lv)),
            log_odds_mean_abs_step=float(np.nanmean(steps)) if steps.size else np.nan,
        )


    @classmethod
    def estimate_markov_metrics(cls, P_ctx: NDArray, P_Long: NDArray, P_Short: NDArray) -> Dict[str, float]:
        """Soft 2x2 / 4x4 Markov transition metrics from per-bin probabilities."""
        out = dict(markov_p_stay_long=np.nan, markov_p_stay_short=np.nan, markov_p_switch_track=np.nan, markov_p_flip_dir_same_track=np.nan)
        try:
            from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
            track_probs = np.vstack([P_Long, P_Short]).T  # (T, 2)
            good = np.all(np.isfinite(track_probs), axis=1)
            track_probs = track_probs[good]
            if track_probs.shape[0] >= 2:
                A2 = TransitionMatrixComputations.estimate_transition_matrix_weighted_avg(track_probs)
                out['markov_p_stay_long'] = float(A2[0, 0])
                out['markov_p_stay_short'] = float(A2[1, 1])
                out['markov_p_switch_track'] = float(0.5 * (A2[0, 1] + A2[1, 0]))
            ## END if track_probs.shape[0] >= 2...

            ctx_probs = np.asarray(P_ctx, dtype=float).T  # (T, 4)
            good4 = np.all(np.isfinite(ctx_probs), axis=1)
            ctx_probs = ctx_probs[good4]
            if ctx_probs.shape[0] >= 2:
                A4 = TransitionMatrixComputations.estimate_transition_matrix_weighted_avg(ctx_probs)
                # same-track direction flips: LR<->RL within Long (0,1) and Short (2,3)
                flip_long = 0.5 * (A4[0, 1] + A4[1, 0])
                flip_short = 0.5 * (A4[2, 3] + A4[3, 2])
                out['markov_p_flip_dir_same_track'] = float(0.5 * (flip_long + flip_short))
            ## END if ctx_probs.shape[0] >= 2...
        except Exception:
            pass
        return out


    @classmethod
    def _map_positions_cm(cls, p_x_given_n: NDArray, context_idx: int, xbin_centers: Optional[NDArray]) -> NDArray:
        """MAP position (cm) for one context strip; NaN if missing xbin_centers."""
        n_pos, n_ctx, n_t = np.shape(p_x_given_n)
        strip = p_x_given_n[:, context_idx, :]  # (n_pos, T)
        map_idx = np.nanargmax(strip, axis=0)
        if xbin_centers is None:
            return map_idx.astype(float)
        xbin_centers = np.asarray(xbin_centers, dtype=float)
        assert xbin_centers.size == n_pos
        return xbin_centers[map_idx]


    def _pause_resume_for_bouts(self, bouts: List[Dict[str, Any]], p_x_given_n: NDArray, xbin_centers: Optional[NDArray], track_len_cm: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Score A->B->A triplets for pause-and-resume continuity; enrich bout dicts."""
        summary = dict(n_aba_triplets=0, min_pause_resume_jump_cm=np.nan, max_pause_resume_continuity_score=np.nan, has_paused_resumed_trajectory=False)
        if len(bouts) < 3 or xbin_centers is None:
            return summary, bouts

        xbin_centers = np.asarray(xbin_centers, dtype=float)
        track_len_cm = float(track_len_cm) if np.isfinite(track_len_cm) and track_len_cm > 0 else float(np.nanmax(xbin_centers) - np.nanmin(xbin_centers))
        jumps: List[float] = []
        continuity_scores: List[float] = []

        for k in range(1, len(bouts) - 1):
            b_prev, b_mid, b_next = bouts[k - 1], bouts[k], bouts[k + 1]
            if (b_prev['track'] != b_next['track']) or (b_prev['track'] == b_mid['track']):
                continue
            ## A->B->A triplet
            summary['n_aba_triplets'] = int(summary['n_aba_triplets']) + 1
            ctx_idx = int(b_prev['dominant_context_idx'])
            x_pause_end = float(b_prev.get('map_x_end_cm', np.nan))
            x_resume_start = float(b_next.get('map_x_start_cm', np.nan))
            if (not np.isfinite(x_pause_end)) or (not np.isfinite(x_resume_start)):
                # fallback from MAP strip
                x_map = self._map_positions_cm(p_x_given_n, ctx_idx, xbin_centers)
                x_pause_end = float(x_map[int(b_prev['stop_idx'])])
                x_resume_start = float(x_map[int(b_next['start_idx'])])
            pause_resume_abs_jump_cm = float(np.abs(x_resume_start - x_pause_end))
            # constant-velocity extrapolation across gap
            gap_bins = max(int(b_mid['n_bins']), 1)
            vel = float(b_prev.get('bout_vel_cm_s', np.nan))
            dt = float(self.decoding_time_bin_size)
            if np.isfinite(vel):
                extrapolated = x_pause_end + vel * gap_bins * dt
                extrapolated_resume_err_cm = float(np.abs(x_resume_start - extrapolated))
            else:
                extrapolated_resume_err_cm = np.nan
            is_resume_dir_congruent = bool(np.sign(b_prev.get('bout_direction_sign', 0)) == np.sign(b_next.get('bout_direction_sign', 0))) if (b_prev.get('bout_direction_sign', 0) != 0 and b_next.get('bout_direction_sign', 0) != 0) else False
            continuity = float(np.clip(1.0 - (pause_resume_abs_jump_cm / track_len_cm), 0.0, 1.0)) if track_len_cm > 0 else np.nan
            jumps.append(pause_resume_abs_jump_cm)
            if np.isfinite(continuity):
                continuity_scores.append(continuity)
            b_mid['is_interrupting_bout'] = True
            b_next['pause_resume_abs_jump_cm'] = pause_resume_abs_jump_cm
            b_next['extrapolated_resume_err_cm'] = extrapolated_resume_err_cm
            b_next['is_resume_dir_congruent'] = is_resume_dir_congruent
            b_next['pause_resume_continuity_score'] = continuity
        ## END for k in range(1, len(bouts) - 1)...

        if jumps:
            summary['min_pause_resume_jump_cm'] = float(np.nanmin(jumps))
            summary['max_pause_resume_continuity_score'] = float(np.nanmax(continuity_scores)) if continuity_scores else np.nan
            summary['has_paused_resumed_trajectory'] = bool(summary['max_pause_resume_continuity_score'] >= 0.5) if continuity_scores else False
        return summary, bouts


    def _build_bouts_from_states(self, states: NDArray, P_ctx: NDArray, p_x_given_n: NDArray, xbin_centers: Optional[NDArray], epoch_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build contiguous committed-context bout records from hysteresis states (-1 Long / +1 Short)."""
        bouts: List[Dict[str, Any]] = []
        n = states.size
        if n == 0:
            return bouts
        i = 0
        while i < n:
            if states[i] == 0:
                i += 1
                continue
            j = i + 1
            while (j < n) and (states[j] == states[i]):
                j += 1
            track = 'Short' if states[i] > 0 else 'Long'
            ctx_slice = P_ctx[2:4, i:j] if track == 'Short' else P_ctx[0:2, i:j]
            # dominant directional context within track
            mean_ctx = np.nanmean(ctx_slice, axis=1)
            local_dom = int(np.nanargmax(mean_ctx)) if np.any(np.isfinite(mean_ctx)) else 0
            dominant_context_idx = (2 + local_dom) if track == 'Short' else local_dom
            dominant_context_name = CONTEXT_NAMES_4[dominant_context_idx]
            x_map = self._map_positions_cm(p_x_given_n, dominant_context_idx, xbin_centers) if (xbin_centers is not None) else np.full(n, np.nan)
            x_seg = x_map[i:j]
            finite_x = x_seg[np.isfinite(x_seg)]
            if finite_x.size >= 2:
                dx = np.diff(finite_x)
                bout_direction_sign = int(np.sign(np.nansum(dx)))
                bout_mean_jump_cm = float(np.nanmean(np.abs(dx)))
                bout_span_cm = float(np.nanmax(finite_x) - np.nanmin(finite_x))
                bout_vel_cm_s = float((finite_x[-1] - finite_x[0]) / max((j - i) * self.decoding_time_bin_size, 1e-9))
            else:
                bout_direction_sign = 0
                bout_mean_jump_cm = np.nan
                bout_span_cm = np.nan
                bout_vel_cm_s = np.nan
            bout = dict(
                **epoch_meta,
                bout_idx=len(bouts),
                start_idx=int(i),
                stop_idx=int(j - 1),
                n_bins=int(j - i),
                duration_s=float((j - i) * self.decoding_time_bin_size),
                track=track,
                dominant_context_idx=int(dominant_context_idx),
                dominant_context_name=dominant_context_name,
                map_x_start_cm=float(x_seg[0]) if np.isfinite(x_seg[0]) else np.nan,
                map_x_end_cm=float(x_seg[-1]) if np.isfinite(x_seg[-1]) else np.nan,
                bout_span_cm=bout_span_cm,
                bout_mean_jump_cm=bout_mean_jump_cm,
                bout_vel_cm_s=bout_vel_cm_s,
                bout_direction_sign=bout_direction_sign,
                is_interrupting_bout=False,
                pause_resume_abs_jump_cm=np.nan,
                extrapolated_resume_err_cm=np.nan,
                is_resume_dir_congruent=False,
                pause_resume_continuity_score=np.nan,
            )
            bouts.append(bout)
            i = j
        ## END while i < n...
        return bouts


    def analyze_single_epoch(self, p_x_given_n: NDArray, epoch_meta: Optional[Dict[str, Any]] = None,
                             valid_bin_mask: Optional[NDArray] = None, xbin_centers: Optional[NDArray] = None,
                             track_len_cm: Optional[float] = None, pre_post_delta_category: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Analyze one epoch posterior `(n_pos, 4, T)` -> (epoch_metrics_dict, bout_dicts)."""
        epoch_meta = dict(epoch_meta or {})
        P_ctx_full, P_Long_full, P_Short_full, delta_P_full = self.marginalize_track_posteriors(p_x_given_n)
        n_t = P_Long_full.size
        if valid_bin_mask is None:
            valid_bin_mask = np.isfinite(P_Long_full) & np.isfinite(P_Short_full)
        else:
            valid_bin_mask = np.asarray(valid_bin_mask, dtype=bool) & np.isfinite(P_Long_full) & np.isfinite(P_Short_full)
        n_valid = int(np.sum(valid_bin_mask))
        duration_s = float(n_t * self.decoding_time_bin_size)

        base = dict(
            **epoch_meta,
            n_total_bins=int(n_t),
            n_valid_bins=n_valid,
            duration_s=duration_s,
            time_bin_size=float(self.decoding_time_bin_size),
            mean_P_Long=float(np.nanmean(P_Long_full[valid_bin_mask])) if n_valid else np.nan,
            mean_P_Short=float(np.nanmean(P_Short_full[valid_bin_mask])) if n_valid else np.nan,
        )

        if n_valid < self.min_valid_bins:
            base.update(dict(
                dragoi_n_transitions=np.nan, dragoi_transition_hz=np.nan, dragoi_dynamic_class=FlickerDynamicClass.Insufficient.value,
                dragoi_mean_bout_ms=np.nan, prob_preference=np.nan, duration_preference=np.nan,
                conf_n_transitions=np.nan, conf_transition_hz=np.nan, conf_dynamic_class=FlickerDynamicClass.Insufficient.value,
                conf_mean_bout_ms=np.nan, conf_max_bout_ms=np.nan, n_long_bouts=0, n_short_bouts=0,
                hmm_n_transitions=np.nan, hmm_dynamic_class=FlickerDynamicClass.Insufficient.value, hmm_mean_bout_ms=np.nan,
                lag1_autocorr_delta_p=np.nan, n_aba_triplets=0, min_pause_resume_jump_cm=np.nan,
                max_pause_resume_continuity_score=np.nan, has_paused_resumed_trajectory=False,
            ))
            base.update(self.compute_bimodal_scores(P_Long_full, P_Short_full, conf_thresh=self.strict_conf_thresh))
            base.update(self.compute_log_odds_metrics(P_Long_full, P_Short_full, clip=self.log_odds_clip))
            base.update(self.estimate_markov_metrics(P_ctx_full, P_Long_full, P_Short_full))
            base['frac_discordant_bins'] = np.nan
            return base, []

        # Restrict analytics to valid bins while preserving original indices via compression map
        valid_idx = np.where(valid_bin_mask)[0]
        P_Long = P_Long_full[valid_idx]
        P_Short = P_Short_full[valid_idx]
        delta_P = delta_P_full[valid_idx]
        P_ctx = P_ctx_full[:, valid_idx]
        p_valid = p_x_given_n[:, :, valid_idx]

        # Dragoi SE baseline
        se = self._safe_se(delta_P)
        b_dragoi = 0.5 * se if np.isfinite(se) else 0.1
        dragoi_states, dragoi_n_transitions, dragoi_bouts = self.hysteresis_commit_states(delta_P, baseline=b_dragoi)
        prefer_track = 'Short' if (np.nansum(P_Short) >= np.nansum(P_Long)) else 'Long'
        dragoi_class = self.classify_dragoi_dynamic(dragoi_n_transitions, prefer_track=prefer_track, signed_dominant=False)
        dragoi_mean_bout_ms = float(np.nanmean(dragoi_bouts) * self.decoding_time_bin_size * 1000.0) if dragoi_bouts.size else np.nan
        mass_L = float(np.nansum(P_Long))
        mass_S = float(np.nansum(P_Short))
        mass_sum = mass_L + mass_S
        prob_preference = float(np.abs(mass_S - mass_L) / mass_sum) if mass_sum > 0 else np.nan
        # duration preference from committed dragoi bouts
        long_bins = int(np.sum(dragoi_states == -1))
        short_bins = int(np.sum(dragoi_states == 1))
        dur_sum = long_bins + short_bins
        duration_preference = float(np.abs(short_bins - long_bins) / dur_sum) if dur_sum > 0 else np.nan

        # Confidence deadband
        b_conf = float(2.0 * self.conf_thresh - 1.0)
        conf_states, conf_n_transitions, conf_bouts = self.hysteresis_commit_states(delta_P, baseline=b_conf)
        if not np.any(conf_states != 0):
            conf_class = FlickerDynamicClass.Ambiguous.value
        else:
            conf_class = self.classify_dragoi_dynamic(conf_n_transitions, prefer_track=prefer_track, signed_dominant=True)
        conf_mean_bout_ms = float(np.nanmean(conf_bouts) * self.decoding_time_bin_size * 1000.0) if conf_bouts.size else np.nan
        conf_max_bout_ms = float(np.nanmax(conf_bouts) * self.decoding_time_bin_size * 1000.0) if conf_bouts.size else np.nan
        n_long_bouts = int(np.sum([1 for b in conf_bouts]))  # temporary; refined below from states
        # recount long/short bouts from conf_states
        n_long_bouts = 0
        n_short_bouts = 0
        if conf_states.size:
            run_state = conf_states[0]
            for s in conf_states[1:]:
                if s != run_state:
                    if run_state == -1:
                        n_long_bouts += 1
                    elif run_state == 1:
                        n_short_bouts += 1
                    run_state = s
            ## END for s in conf_states[1:]...
            if run_state == -1:
                n_long_bouts += 1
            elif run_state == 1:
                n_short_bouts += 1

        # HMM
        hmm_path, hmm_n_transitions, hmm_bouts = self.sticky_hmm_viterbi(P_Long, P_Short, self_stay_prob=self.hmm_self_stay_prob)
        hmm_class = self.classify_dragoi_dynamic(hmm_n_transitions, prefer_track=prefer_track, signed_dominant=True)
        hmm_mean_bout_ms = float(np.nanmean(hmm_bouts) * self.decoding_time_bin_size * 1000.0) if hmm_bouts.size else np.nan

        # Discordant fraction (Jezek-like)
        if pre_post_delta_category == 'pre-delta':
            frac_discordant_bins = float(np.mean(P_Short >= self.conf_thresh)) if n_valid else np.nan
        elif pre_post_delta_category == 'post-delta':
            frac_discordant_bins = float(np.mean(P_Long >= self.conf_thresh)) if n_valid else np.nan
        else:
            frac_discordant_bins = np.nan

        # Bouts + pause/resume (use confidence states)
        bouts = self._build_bouts_from_states(conf_states, P_ctx, p_valid, xbin_centers=xbin_centers, epoch_meta=epoch_meta)
        # remap compressed indices to original time-bin indices for bout records
        for b in bouts:
            b['start_idx'] = int(valid_idx[int(b['start_idx'])])
            b['stop_idx'] = int(valid_idx[int(b['stop_idx'])])
        ## END for b in bouts...
        if track_len_cm is None:
            track_len_cm = float(np.nanmax(xbin_centers) - np.nanmin(xbin_centers)) if xbin_centers is not None else np.nan
        pause_summary, bouts = self._pause_resume_for_bouts(bouts, p_x_given_n, xbin_centers=xbin_centers, track_len_cm=float(track_len_cm) if track_len_cm is not None else np.nan)

        metrics = dict(base)
        metrics.update(dict(
            dragoi_n_transitions=int(dragoi_n_transitions),
            dragoi_transition_hz=float(dragoi_n_transitions / duration_s) if duration_s > 0 else np.nan,
            dragoi_dynamic_class=dragoi_class,
            dragoi_mean_bout_ms=dragoi_mean_bout_ms,
            prob_preference=prob_preference,
            duration_preference=duration_preference,
            conf_n_transitions=int(conf_n_transitions),
            conf_transition_hz=float(conf_n_transitions / duration_s) if duration_s > 0 else np.nan,
            conf_dynamic_class=conf_class,
            conf_mean_bout_ms=conf_mean_bout_ms,
            conf_max_bout_ms=conf_max_bout_ms,
            n_long_bouts=int(n_long_bouts),
            n_short_bouts=int(n_short_bouts),
            hmm_n_transitions=int(hmm_n_transitions),
            hmm_dynamic_class=hmm_class,
            hmm_mean_bout_ms=hmm_mean_bout_ms,
            lag1_autocorr_delta_p=self._lag1_autocorr(delta_P),
            frac_discordant_bins=frac_discordant_bins,
            **pause_summary,
        ))
        metrics.update(self.compute_bimodal_scores(P_Long, P_Short, conf_thresh=self.strict_conf_thresh))
        metrics.update(self.compute_log_odds_metrics(P_Long, P_Short, clip=self.log_odds_clip))
        metrics.update(self.estimate_markov_metrics(P_ctx, P_Long, P_Short))
        return metrics, bouts


    def _within_event_permutation_nulls(self, delta_P: NDArray, P_Long: NDArray, P_Short: NDArray, duration_s: float) -> Dict[str, float]:
        """Within-PBE time-bin permutation nulls for bout duration / transitions / autocorr."""
        out = dict(within_perm_bout_ms_z=np.nan, within_perm_bout_ms_p=np.nan, within_perm_transitions_z=np.nan, within_perm_transitions_p=np.nan, within_perm_autocorr_z=np.nan, within_perm_autocorr_p=np.nan)
        n = delta_P.size
        if (self.num_within_shuffles <= 0) or (n < self.min_valid_bins):
            return out
        b_conf = float(2.0 * self.conf_thresh - 1.0)
        _, real_n_trans, real_bouts = self.hysteresis_commit_states(delta_P, baseline=b_conf)
        real_bout_ms = float(np.nanmean(real_bouts) * self.decoding_time_bin_size * 1000.0) if real_bouts.size else np.nan
        real_ac = self._lag1_autocorr(delta_P)

        sh_bout = np.full(self.num_within_shuffles, np.nan)
        sh_trans = np.full(self.num_within_shuffles, np.nan)
        sh_ac = np.full(self.num_within_shuffles, np.nan)
        for s in range(self.num_within_shuffles):
            perm = self.rng.permutation(n)
            d = delta_P[perm]
            _, n_tr, bouts = self.hysteresis_commit_states(d, baseline=b_conf)
            sh_trans[s] = float(n_tr)
            sh_bout[s] = float(np.nanmean(bouts) * self.decoding_time_bin_size * 1000.0) if bouts.size else np.nan
            sh_ac[s] = self._lag1_autocorr(d)
        ## END for s in range(self.num_within_shuffles)...

        def _z_p(real_v, null_v, greater_is_extreme: bool = True):
            null_v = null_v[np.isfinite(null_v)]
            if (not np.isfinite(real_v)) or (null_v.size < 5):
                return np.nan, np.nan
            mu = float(np.nanmean(null_v))
            sd = float(np.nanstd(null_v, ddof=1))
            z = (real_v - mu) / sd if sd > 1e-12 else np.nan
            if greater_is_extreme:
                p = float((np.sum(null_v >= real_v) + 1) / (null_v.size + 1))
            else:
                p = float((np.sum(null_v <= real_v) + 1) / (null_v.size + 1))
            return z, p

        out['within_perm_bout_ms_z'], out['within_perm_bout_ms_p'] = _z_p(real_bout_ms, sh_bout, greater_is_extreme=True)
        out['within_perm_transitions_z'], out['within_perm_transitions_p'] = _z_p(float(real_n_trans), sh_trans, greater_is_extreme=True)
        out['within_perm_autocorr_z'], out['within_perm_autocorr_p'] = _z_p(real_ac, sh_ac, greater_is_extreme=True)
        return out


    def _pool_resampling_nulls(self, epoch_rows: List[Dict[str, Any]], pool_P_Short_by_split: Dict[str, NDArray]) -> None:
        """In-place add pool_* z/p columns using session-pooled P_Short bins (stratified when possible)."""
        if self.num_pool_shuffles <= 0:
            return
        b_conf = float(2.0 * self.conf_thresh - 1.0)
        for row in epoch_rows:
            n_valid = int(row.get('n_valid_bins', 0) or 0)
            split = str(row.get('pre_post_delta_category', 'all'))
            pool = pool_P_Short_by_split.get(split)
            if pool is None or pool.size < n_valid or n_valid < self.min_valid_bins:
                pool = pool_P_Short_by_split.get('all')
            if pool is None or pool.size < n_valid or n_valid < self.min_valid_bins:
                row.update(dict(pool_bimodal_z=np.nan, pool_bimodal_p=np.nan, pool_transitions_z=np.nan, pool_transitions_p=np.nan))
                continue
            real_bimodal = float(row.get('bimodal_flicker_score', np.nan))
            real_trans = float(row.get('conf_n_transitions', np.nan))
            sh_bimodal = np.full(self.num_pool_shuffles, np.nan)
            sh_trans = np.full(self.num_pool_shuffles, np.nan)
            for s in range(self.num_pool_shuffles):
                samp = self.rng.choice(pool, size=n_valid, replace=True)
                P_Short = np.asarray(samp, dtype=float)
                P_Long = 1.0 - P_Short
                delta_P = P_Short - P_Long
                _, n_tr, _ = self.hysteresis_commit_states(delta_P, baseline=b_conf)
                sh_trans[s] = float(n_tr)
                sh_bimodal[s] = float(self.compute_bimodal_scores(P_Long, P_Short, conf_thresh=self.strict_conf_thresh)['bimodal_flicker_score'])
            ## END for s in range(self.num_pool_shuffles)...

            def _z_p(real_v, null_v):
                null_v = null_v[np.isfinite(null_v)]
                if (not np.isfinite(real_v)) or (null_v.size < 5):
                    return np.nan, np.nan
                mu = float(np.nanmean(null_v)); sd = float(np.nanstd(null_v, ddof=1))
                z = (real_v - mu) / sd if sd > 1e-12 else np.nan
                p = float((np.sum(null_v >= real_v) + 1) / (null_v.size + 1))
                return z, p

            zb, pb = _z_p(real_bimodal, sh_bimodal)
            zt, pt = _z_p(real_trans, sh_trans)
            row.update(dict(pool_bimodal_z=zb, pool_bimodal_p=pb, pool_transitions_z=zt, pool_transitions_p=pt))
        ## END for row in epoch_rows...


    def compute_from_decoded_result(self, decoded_result, epoch_type: str = 'pbe', t_delta: Optional[float] = None,
                                    xbin_centers: Optional[NDArray] = None, track_len_cm: Optional[float] = None,
                                    session_name: Optional[str] = None, custom_replay_name: Optional[str] = None,
                                    active_bin_masks: Optional[List[Optional[NDArray]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze all epochs in a `DecodedFilterEpochsResult` (4-context pseudo2D)."""
        from neuropy.core.epoch import ensure_dataframe

        p_list = list(decoded_result.p_x_given_n_list)
        filter_epochs_df = ensure_dataframe(deepcopy(decoded_result.filter_epochs))
        n_epochs = len(p_list)
        assert len(filter_epochs_df) == n_epochs or ('start' in filter_epochs_df.columns)

        epoch_rows: List[Dict[str, Any]] = []
        bout_rows: List[Dict[str, Any]] = []
        pool_bins: Dict[str, List[float]] = {'pre-delta': [], 'post-delta': [], 'all': []}

        for epoch_i, p_x_given_n in enumerate(p_list):
            p_x_given_n = np.asarray(p_x_given_n)
            if p_x_given_n.ndim != 3:
                continue
            if p_x_given_n.shape[1] != 4:
                # try transpose variants
                if p_x_given_n.shape[0] == 4:
                    p_x_given_n = np.transpose(p_x_given_n, (1, 0, 2))
                else:
                    continue
            start_t = float(filter_epochs_df.iloc[epoch_i]['start']) if 'start' in filter_epochs_df.columns else np.nan
            stop_t = float(filter_epochs_df.iloc[epoch_i]['stop']) if 'stop' in filter_epochs_df.columns else np.nan
            label = filter_epochs_df.iloc[epoch_i]['label'] if 'label' in filter_epochs_df.columns else epoch_i
            if t_delta is not None and np.isfinite(start_t):
                delta_aligned_start_t = float(start_t - t_delta)
                pre_post = 'post-delta' if delta_aligned_start_t > 0.0 else 'pre-delta'
            else:
                delta_aligned_start_t = np.nan
                pre_post = None
            meta = dict(
                epoch_idx=int(epoch_i),
                epoch_type=str(epoch_type),
                label=label,
                start=start_t,
                stop=stop_t,
                delta_aligned_start_t=delta_aligned_start_t,
                pre_post_delta_category=pre_post,
                session_name=session_name,
                custom_replay_name=custom_replay_name,
            )
            valid_mask = active_bin_masks[epoch_i] if (active_bin_masks is not None and epoch_i < len(active_bin_masks)) else None
            metrics, bouts = self.analyze_single_epoch(p_x_given_n, epoch_meta=meta, valid_bin_mask=valid_mask, xbin_centers=xbin_centers, track_len_cm=track_len_cm, pre_post_delta_category=pre_post)

            # within-event permutation nulls on valid compressed series
            P_ctx, P_Long, P_Short, delta_P = self.marginalize_track_posteriors(p_x_given_n)
            if valid_mask is None:
                valid_mask = np.isfinite(P_Long) & np.isfinite(P_Short)
            if int(np.sum(valid_mask)) >= self.min_valid_bins:
                metrics.update(self._within_event_permutation_nulls(delta_P[valid_mask], P_Long[valid_mask], P_Short[valid_mask], duration_s=float(metrics['duration_s'])))
                for v in P_Short[valid_mask]:
                    if np.isfinite(v):
                        pool_bins['all'].append(float(v))
                        if pre_post in pool_bins:
                            pool_bins[pre_post].append(float(v))
                ## END for v in P_Short[valid_mask]...
            else:
                metrics.update(dict(within_perm_bout_ms_z=np.nan, within_perm_bout_ms_p=np.nan, within_perm_transitions_z=np.nan, within_perm_transitions_p=np.nan, within_perm_autocorr_z=np.nan, within_perm_autocorr_p=np.nan))

            # placeholder cell-shuffle columns (filled only if templates provided elsewhere)
            metrics.setdefault('cell_shuffle_bimodal_z', np.nan)
            metrics.setdefault('cell_shuffle_bimodal_p', np.nan)
            metrics.setdefault('cell_shuffle_transitions_z', np.nan)
            metrics.setdefault('cell_shuffle_transitions_p', np.nan)

            epoch_rows.append(metrics)
            bout_rows.extend(bouts)
        ## END for epoch_i, p_x_given_n in enumerate(p_list)...

        pool_arr = {k: np.asarray(v, dtype=float) for k, v in pool_bins.items()}
        self._pool_resampling_nulls(epoch_rows, pool_arr)

        epoch_df = pd.DataFrame(epoch_rows)
        bouts_df = pd.DataFrame(bout_rows)
        if len(epoch_df) and 'start' in epoch_df.columns:
            epoch_df = epoch_df.sort_values('start', ascending=True).reset_index(drop=True)
        return epoch_df, bouts_df


    @classmethod
    def compute_from_time_bin_marginals_df(cls, time_bin_df: pd.DataFrame, decoding_time_bin_size: Optional[float] = None,
                                           conf_thresh: float = 0.65, strict_conf_thresh: float = 0.75, min_valid_bins: int = 4,
                                           num_within_shuffles: int = 500, num_pool_shuffles: int = 500,
                                           epoch_type: str = 'pbe', parent_epoch_col: str = 'parent_epoch_label') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fallback: compute context-marginal flickering metrics from FAT/per-time-bin marginals DF (no spatial pause/resume)."""
        df = deepcopy(time_bin_df)
        assert 'P_Long' in df.columns and 'P_Short' in df.columns
        if decoding_time_bin_size is None:
            if 'time_bin_size' in df.columns and df['time_bin_size'].notna().any():
                decoding_time_bin_size = float(df['time_bin_size'].dropna().iloc[0])
            else:
                decoding_time_bin_size = 0.025
        analyzer = cls(decoding_time_bin_size=float(decoding_time_bin_size), conf_thresh=conf_thresh, strict_conf_thresh=strict_conf_thresh,
                       min_valid_bins=min_valid_bins, num_within_shuffles=num_within_shuffles, num_pool_shuffles=num_pool_shuffles, num_neuron_id_shuffles=0)

        if parent_epoch_col not in df.columns:
            # treat whole DF as one epoch stream partitioned by contiguous blocks of parent_epoch_id if present
            if 'parent_epoch_id' in df.columns:
                parent_epoch_col = 'parent_epoch_id'
            else:
                raise KeyError(f'need {parent_epoch_col} or parent_epoch_id on time_bin_df')

        epoch_rows: List[Dict[str, Any]] = []
        bout_rows: List[Dict[str, Any]] = []
        pool_bins: Dict[str, List[float]] = {'pre-delta': [], 'post-delta': [], 'all': []}

        for epoch_label, an_epoch_df in df.groupby(parent_epoch_col, sort=False):
            an_epoch_df = an_epoch_df.sort_values('t_bin_center' if 't_bin_center' in an_epoch_df.columns else an_epoch_df.columns[0])
            P_Long = an_epoch_df['P_Long'].to_numpy(dtype=float)
            P_Short = an_epoch_df['P_Short'].to_numpy(dtype=float)
            n_t = P_Long.size
            # fabricate (n_pos=1, 4, T) with LR/RL split half of track mass when directional cols absent
            if set(['long_LR', 'long_RL', 'short_LR', 'short_RL']).issubset(an_epoch_df.columns):
                P_ctx = an_epoch_df[['long_LR', 'long_RL', 'short_LR', 'short_RL']].to_numpy(dtype=float).T
            else:
                P_ctx = np.vstack([0.5 * P_Long, 0.5 * P_Long, 0.5 * P_Short, 0.5 * P_Short])
            p_x_given_n = P_ctx[None, :, :]  # (1, 4, T)

            start_t = float(an_epoch_df['parent_epoch_start_t'].iloc[0]) if 'parent_epoch_start_t' in an_epoch_df.columns else (float(an_epoch_df['start'].iloc[0]) if 'start' in an_epoch_df.columns else np.nan)
            stop_t = float(an_epoch_df['parent_epoch_end_t'].iloc[0]) if 'parent_epoch_end_t' in an_epoch_df.columns else (float(an_epoch_df['stop'].iloc[-1]) if 'stop' in an_epoch_df.columns else np.nan)
            pre_post = an_epoch_df['pre_post_delta_category'].iloc[0] if 'pre_post_delta_category' in an_epoch_df.columns else None
            delta_aligned = float(an_epoch_df['delta_aligned_start_t'].iloc[0]) if 'delta_aligned_start_t' in an_epoch_df.columns else np.nan
            meta = dict(
                epoch_idx=epoch_label,
                epoch_type=str(epoch_type),
                label=epoch_label,
                start=start_t,
                stop=stop_t,
                delta_aligned_start_t=delta_aligned,
                pre_post_delta_category=pre_post,
                session_name=an_epoch_df['session_name'].iloc[0] if 'session_name' in an_epoch_df.columns else None,
                custom_replay_name=an_epoch_df['custom_replay_name'].iloc[0] if 'custom_replay_name' in an_epoch_df.columns else None,
            )
            metrics, bouts = analyzer.analyze_single_epoch(p_x_given_n, epoch_meta=meta, xbin_centers=None, track_len_cm=None, pre_post_delta_category=pre_post)
            valid = np.isfinite(P_Long) & np.isfinite(P_Short)
            if int(np.sum(valid)) >= analyzer.min_valid_bins:
                metrics.update(analyzer._within_event_permutation_nulls((P_Short - P_Long)[valid], P_Long[valid], P_Short[valid], duration_s=float(metrics['duration_s'])))
                for v in P_Short[valid]:
                    pool_bins['all'].append(float(v))
                    if pre_post in pool_bins:
                        pool_bins[pre_post].append(float(v))
                ## END for v in P_Short[valid]...
            metrics.setdefault('cell_shuffle_bimodal_z', np.nan)
            metrics.setdefault('cell_shuffle_bimodal_p', np.nan)
            metrics.setdefault('cell_shuffle_transitions_z', np.nan)
            metrics.setdefault('cell_shuffle_transitions_p', np.nan)
            epoch_rows.append(metrics)
            bout_rows.extend(bouts)
        ## END for epoch_label, an_epoch_df in df.groupby(parent_epoch_col, sort=False)...

        pool_arr = {k: np.asarray(v, dtype=float) for k, v in pool_bins.items()}
        analyzer._pool_resampling_nulls(epoch_rows, pool_arr)
        epoch_df = pd.DataFrame(epoch_rows)
        bouts_df = pd.DataFrame(bout_rows)
        return epoch_df, bouts_df


@metadata_attributes(short_name='PBEFlickeringAcrossSessionPlotter', tags=['plotting', 'publication', 'flickering'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-11-27 17:30', related_items=[])
class PBEFlickeringAcrossSessionPlotter:
    """Publication helpers for across-session flickering figures (Panels A–D)."""

    @staticmethod
    def summarize_class_proportions(epoch_df: pd.DataFrame, class_col: str = 'conf_dynamic_class', split_col: str = 'pre_post_delta_category') -> pd.DataFrame:
        if epoch_df is None or len(epoch_df) == 0:
            return pd.DataFrame()
        rows = []
        for split_name, sub in epoch_df.groupby(split_col):
            counts = sub[class_col].value_counts(dropna=False)
            total = float(counts.sum()) if counts.sum() else np.nan
            for cls_name, cnt in counts.items():
                rows.append(dict(pre_post_delta_category=split_name, dynamic_class=cls_name, n=int(cnt), proportion=(float(cnt) / total) if np.isfinite(total) and total > 0 else np.nan))
            ## END for cls_name, cnt in counts.items()...
        ## END for split_name, sub in epoch_df.groupby(split_col)...
        return pd.DataFrame(rows)


    @staticmethod
    def plot_class_proportions_bars(epoch_df: pd.DataFrame, class_col: str = 'conf_dynamic_class', ax=None, title: str = 'Flickering class proportions'):
        import matplotlib.pyplot as plt
        summary = PBEFlickeringAcrossSessionPlotter.summarize_class_proportions(epoch_df, class_col=class_col)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=160)
        else:
            fig = ax.figure
        if len(summary) == 0:
            ax.set_title(title + ' (empty)')
            return fig, ax
        pivot = summary.pivot_table(index='pre_post_delta_category', columns='dynamic_class', values='proportion', aggfunc='mean').fillna(0.0)
        pivot.plot(kind='bar', stacked=True, ax=ax)
        ax.set_ylabel('Proportion')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7)
        return fig, ax


    @staticmethod
    def plot_timecourse_scatter(epoch_df: pd.DataFrame, y_col: str = 'bimodal_flicker_score', ax=None, title: Optional[str] = None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3.0), dpi=160)
        else:
            fig = ax.figure
        if epoch_df is None or len(epoch_df) == 0 or ('delta_aligned_start_t' not in epoch_df.columns):
            ax.set_title((title or y_col) + ' (empty)')
            return fig, ax
        ax.scatter(epoch_df['delta_aligned_start_t'], epoch_df[y_col], s=8, alpha=0.45)
        ax.axvline(0.0, color='k', lw=1.0, ls='--')
        ax.set_xlabel('delta_aligned_start_t (s)')
        ax.set_ylabel(y_col)
        ax.set_title(title or y_col)
        return fig, ax


    @staticmethod
    def plot_p_short_vs_bimodal(epoch_df: pd.DataFrame, ax=None, title: str = 'Whole-epoch P_Short vs bimodal flicker'):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
        else:
            fig = ax.figure
        if epoch_df is None or len(epoch_df) == 0:
            ax.set_title(title + ' (empty)')
            return fig, ax
        x = epoch_df['mean_P_Short'] if 'mean_P_Short' in epoch_df.columns else epoch_df.get('P_Short')
        y = epoch_df['bimodal_flicker_score']
        ax.scatter(x, y, s=10, alpha=0.4)
        ax.set_xlabel('mean_P_Short')
        ax.set_ylabel('bimodal_flicker_score')
        ax.set_title(title)
        return fig, ax
