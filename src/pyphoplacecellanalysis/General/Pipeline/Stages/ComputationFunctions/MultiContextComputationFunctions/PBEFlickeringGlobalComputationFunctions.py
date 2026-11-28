"""Global computation functions for within-PBE context flickering analysis."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


def validate_has_pbe_flickering_results(curr_active_pipeline, computation_filter_name: str = 'maze') -> bool:
    """Top-level validator for `pbe_flickering_analysis` (must be picklable; avoid lambdas)."""
    flickering_result = curr_active_pipeline.global_computation_results.computed_data.get('PBEFlickering', None)
    if flickering_result is None:
        return False
    epoch_df = getattr(flickering_result, 'epoch_df', None)
    if epoch_df is None or (not isinstance(epoch_df, pd.DataFrame)) or (len(epoch_df) == 0):
        return False
    required_cols = ['conf_dynamic_class', 'dragoi_dynamic_class', 'bimodal_flicker_score', 'conf_transition_hz']
    missing = [c for c in required_cols if c not in epoch_df.columns]
    return len(missing) == 0


class PBEFlickeringGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """Within-PBE Long/Short context flickering + pause-and-resume trajectory scoring."""
    _computationGroupName = 'pbe_flickering'
    _computationPrecidence = 1003.5
    _is_global = True


    @function_attributes(short_name='pbe_flickering_analysis', tags=['flickering', 'PBE', 'context', 'Dragoi2024', 'replay', 'main-computation-function'],
                        input_requires=[], output_provides=['PBEFlickering'],
                        uses=['PBEFlickeringAnalyzer', 'DirectionalPseudo2DDecodersResult'], used_by=['compute_and_export_pbe_flickering_completion_function'], creation_date='2026-11-27 17:10', related_items=[],
                        requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders'], provides_global_keys=['PBEFlickering'],
                        validate_computation_test=validate_has_pbe_flickering_results, is_global=True, computation_precidence=1003.5)
    def perform_pbe_flickering_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False,
                                        decoding_time_bin_size: float = 0.025, conf_thresh: float = 0.65, strict_conf_thresh: float = 0.75,
                                        min_num_unique_active_neurons_per_time_bin: int = 2, min_num_spikes_per_bin: int = 1, min_valid_bins: int = 4,
                                        num_within_shuffles: int = 500, num_pool_shuffles: int = 500, num_neuron_id_shuffles: int = 0,
                                        include_laps_control: bool = True, drop_previous_result_and_compute_fresh: bool = False):
        """Compute within-PBE (and optional laps-control) context flickering metrics.

        Provides:
            global_computation_results.computed_data['PBEFlickering']
        """
        from neuropy.core.epoch import ensure_dataframe
        from pyphoplacecellanalysis.Analysis.Decoder.pbe_flickering import PBEFlickeringAnalyzer, PBEFlickeringResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult, get_proper_global_spikes_df

        if include_includelist is not None:
            print(f'WARN: perform_pbe_flickering_analysis(...): include_includelist={include_includelist} is currently ignored.')

        if drop_previous_result_and_compute_fresh:
            removed = global_computation_results.computed_data.pop('PBEFlickering', None)
            if removed is not None:
                print('removed previous "PBEFlickering" result and computing fresh since drop_previous_result_and_compute_fresh == True')

        directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data['DirectionalMergedDecoders']
        all_directional_pf1D_Decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
        ripple_result = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result
        laps_result = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result

        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        session_name: str = owning_pipeline_reference.session_name
        try:
            complete_session_context, (_session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()
            custom_replay_name = additional_session_context.get_description() if additional_session_context is not None else None
        except Exception:
            custom_replay_name = None

        xbin_centers = None
        track_len_cm = None
        try:
            xbin_centers = np.asarray(all_directional_pf1D_Decoder.xbin_centers, dtype=float)
            track_len_cm = float(np.nanmax(xbin_centers) - np.nanmin(xbin_centers))
        except Exception:
            pass

        # Optionally re-decode PBEs at requested tbin without mutating cached DirectionalMergedDecoders
        active_ripple_result = ripple_result
        cached_ripple_tbin = float(getattr(ripple_result, 'decoding_time_bin_size', np.nan)) if ripple_result is not None else np.nan
        if (ripple_result is None) or (not np.isfinite(cached_ripple_tbin)) or (abs(float(decoding_time_bin_size) - cached_ripple_tbin) > 1e-9):
            print(f'perform_pbe_flickering_analysis: re-decoding PBEs at decoding_time_bin_size={decoding_time_bin_size} (cached ripple tbin={cached_ripple_tbin})')
            spikes_df = deepcopy(get_proper_global_spikes_df(owning_pipeline_reference))
            filter_epochs = ensure_dataframe(deepcopy(ripple_result.filter_epochs)) if ripple_result is not None else ensure_dataframe(deepcopy(owning_pipeline_reference.sess.replay))
            active_ripple_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=float(decoding_time_bin_size), debug_print=False)

        analyzer = PBEFlickeringAnalyzer(
            decoding_time_bin_size=float(decoding_time_bin_size),
            conf_thresh=float(conf_thresh),
            strict_conf_thresh=float(strict_conf_thresh),
            min_num_unique_active_neurons_per_time_bin=int(min_num_unique_active_neurons_per_time_bin),
            min_num_spikes_per_bin=int(min_num_spikes_per_bin),
            min_valid_bins=int(min_valid_bins),
            num_within_shuffles=int(num_within_shuffles),
            num_pool_shuffles=int(num_pool_shuffles),
            num_neuron_id_shuffles=int(num_neuron_id_shuffles),
        )

        pbe_epoch_df, pbe_bouts_df = analyzer.compute_from_decoded_result(
            active_ripple_result, epoch_type='pbe', t_delta=t_delta, xbin_centers=xbin_centers, track_len_cm=track_len_cm,
            session_name=session_name, custom_replay_name=custom_replay_name,
        )

        epoch_dfs = [pbe_epoch_df]
        bout_dfs = [pbe_bouts_df]

        if include_laps_control and (laps_result is not None):
            active_laps_result = laps_result
            cached_laps_tbin = float(getattr(laps_result, 'decoding_time_bin_size', np.nan))
            # For laps control, reuse cached result unless wildly different; optional re-decode if needed
            if (not np.isfinite(cached_laps_tbin)) or (abs(float(decoding_time_bin_size) - cached_laps_tbin) > 1e-9):
                print(f'perform_pbe_flickering_analysis: re-decoding Laps control at decoding_time_bin_size={decoding_time_bin_size} (cached laps tbin={cached_laps_tbin})')
                spikes_df = deepcopy(get_proper_global_spikes_df(owning_pipeline_reference))
                filter_epochs = ensure_dataframe(deepcopy(laps_result.filter_epochs))
                active_laps_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=float(decoding_time_bin_size), debug_print=False)
            laps_epoch_df, laps_bouts_df = analyzer.compute_from_decoded_result(
                active_laps_result, epoch_type='laps', t_delta=t_delta, xbin_centers=xbin_centers, track_len_cm=track_len_cm,
                session_name=session_name, custom_replay_name=custom_replay_name,
            )
            epoch_dfs.append(laps_epoch_df)
            bout_dfs.append(laps_bouts_df)
        ## END if include_laps_control...

        epoch_df = pd.concat([d for d in epoch_dfs if d is not None and len(d)], ignore_index=True) if any((d is not None and len(d)) for d in epoch_dfs) else pd.DataFrame()
        bouts_df = pd.concat([d for d in bout_dfs if d is not None and len(d)], ignore_index=True) if any((d is not None and len(d)) for d in bout_dfs) else pd.DataFrame()

        result = PBEFlickeringResult(
            is_global=True,
            decoding_time_bin_size=float(decoding_time_bin_size),
            conf_thresh=float(conf_thresh),
            strict_conf_thresh=float(strict_conf_thresh),
            min_num_unique_active_neurons_per_time_bin=int(min_num_unique_active_neurons_per_time_bin),
            min_valid_bins=int(min_valid_bins),
            epoch_df=epoch_df,
            bouts_df=bouts_df,
            params=dict(
                decoding_time_bin_size=float(decoding_time_bin_size),
                conf_thresh=float(conf_thresh),
                strict_conf_thresh=float(strict_conf_thresh),
                min_num_unique_active_neurons_per_time_bin=int(min_num_unique_active_neurons_per_time_bin),
                min_num_spikes_per_bin=int(min_num_spikes_per_bin),
                min_valid_bins=int(min_valid_bins),
                num_within_shuffles=int(num_within_shuffles),
                num_pool_shuffles=int(num_pool_shuffles),
                num_neuron_id_shuffles=int(num_neuron_id_shuffles),
                include_laps_control=bool(include_laps_control),
            ),
        )
        global_computation_results.computed_data['PBEFlickering'] = result
        if debug_print:
            print(f'PBEFlickering: epoch_df={len(epoch_df)} rows, bouts_df={len(bouts_df)} rows, tbin={decoding_time_bin_size}')
        return global_computation_results
