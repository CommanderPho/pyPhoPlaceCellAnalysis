from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Callable

from pathlib import Path
import inspect
from jinja2 import Template
from neuropy.utils.result_context import IdentifyingContext
import numpy as np
import pandas as pd





# %% [markdown]
# ## Build Processing Scripts:

# %%


# get like: generated_header_code = curr_runtime_context_header_template.render(BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, collected_outputs_path_str=str(collected_outputs_path))
curr_runtime_context_header_template: str = Template("""
BATCH_DATE_TO_USE = '{{ BATCH_DATE_TO_USE }}'
collected_outputs_path = Path('{{ collected_outputs_path_str }}').resolve()
""")



""" In general, results of your callback can be added to the output dict like:

    across_session_results_extended_dict['compute_and_export_marginals_dfs_completion_function'] = _out

and can be extracted from batch output by:

    # Extracts the callback results 'determine_computation_datetimes_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_computation_datetimes_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}



    All capture:
        BATCH_DATE_TO_USE, collected_outputs_path
"""

## These are temporary captured references that should never appear in generated code or runtime, they're just here to prevent typechecking errors in the editor
BATCH_DATE_TO_USE: str = '0000-00-00_Fake' # TODO: Change this as needed, templating isn't actually doing anything rn.
# collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve() # Linux
collected_outputs_path: Path = Path('/home/halechr/cloud/turbo/Data/Output/collected_outputs').resolve() # GreatLakes
# collected_outputs_path = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\output\collected_outputs').resolve() # Apogee


def export_rank_order_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'export_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    assert collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import save_rank_order_results, SaveStringGenerator
    save_rank_order_results(curr_active_pipeline, day_date=f"{CURR_BATCH_OUTPUT_PREFIX}", override_output_parent_path=collected_outputs_path) # "2024-01-02_301pm" "2024-01-02_734pm""

    ## 2023-12-21 - Export to CSV:
    spikes_df = curr_active_pipeline.sess.spikes_df
    rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
    minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
    included_qclu_values: List[int] = rank_order_results.included_qclu_values
    ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
    print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
    print(f'included_qclu_values: {included_qclu_values}')

    print(f'\t try saving to CSV...')
    # active_csv_parent_output_path = curr_active_pipeline.get_output_path().resolve()
    active_csv_parent_output_path = collected_outputs_path.resolve()
    merged_complete_epoch_stats_df = rank_order_results.ripple_merged_complete_epoch_stats_df ## New method
    merged_complete_ripple_epoch_stats_df_output_path = active_csv_parent_output_path.joinpath(f'{CURR_BATCH_OUTPUT_PREFIX}_merged_complete_epoch_stats_df.csv').resolve()
    merged_complete_epoch_stats_df.to_csv(merged_complete_ripple_epoch_stats_df_output_path)
    print(f'\t saving to CSV: {merged_complete_ripple_epoch_stats_df_output_path} done.')
    
    # across_session_results_extended_dict['merged_complete_epoch_stats_df'] = merged_complete_ripple_epoch_stats_df_output_path
    
    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # return True
    return across_session_results_extended_dict


def figures_rank_order_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderGlobalDisplayFunctions
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'figures_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    assert collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    curr_active_pipeline.reload_default_display_functions()
    _display_rank_order_z_stats_results_out = curr_active_pipeline.display('_display_rank_order_z_stats_results', defer_render=True, save_figure=True)
    # across_session_results_extended_dict['merged_complete_epoch_stats_df'] = merged_complete_ripple_epoch_stats_df_output_path
    
    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # return True
    return across_session_results_extended_dict


def compute_and_export_marginals_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_marginals_dfs_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    
    assert collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations
    curr_active_pipeline.reload_default_computation_functions()
    batch_extended_computations(curr_active_pipeline, include_includelist=['merged_directional_placefields'], include_global_functions=True, fail_on_exception=True, force_recompute=True)
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']

    active_context = curr_active_pipeline.get_session_context()
    _out = directional_merged_decoders_result.compute_and_export_marginals_df_csvs(parent_output_path=collected_outputs_path, active_context=active_context)
    print(f'successfully exported marginals_df_csvs to {collected_outputs_path}!')
    # (laps_marginals_df, laps_out_path), (ripple_marginals_df, ripple_out_path) = _out
    (laps_marginals_df, laps_out_path, laps_time_bin_marginals_df, laps_time_bin_marginals_out_path), (ripple_marginals_df, ripple_out_path, ripple_time_bin_marginals_df, ripple_time_bin_marginals_out_path) = _out
    print(f'\tlaps_out_path: {laps_out_path}\n\tripple_out_path: {ripple_out_path}\n\tdone.')

    # add to output dict
    # across_session_results_extended_dict['compute_and_export_marginals_dfs_completion_function'] = _out

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict

def determine_computation_datetimes_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """ 
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import determine_computation_datetimes_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_computation_datetimes_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_computation_datetimes_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'determine_computation_datetimes_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    
    global_pickle_path = curr_active_pipeline.global_computation_results_pickle_path.resolve()
    assert global_pickle_path.exists()
    global_pickle_path

    callback_outputs = {
     'global_pickle_path': global_pickle_path   
    }

    across_session_results_extended_dict['determine_computation_datetimes_completion_function'] = callback_outputs
    

    # print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict

def determine_session_t_delta_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """ 
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import determine_computation_datetimes_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'determine_session_t_delta_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    print(f'\t{curr_session_basedir}:\tt_start: {t_start}, t_delta: {t_delta}, t_end: {t_end}')
    
    callback_outputs = {
     't_start': t_start, 't_delta':t_delta, 't_end': t_end   
    }
    across_session_results_extended_dict['determine_session_t_delta_completion_function'] = callback_outputs
    
    # print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


def perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    from neuropy.utils.debug_helpers import parameter_sweeps
    from neuropy.core.laps import Laps
    from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalMergedDecodersResult
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

    ## Single decode:
    def _try_single_decode(owning_pipeline_reference, directional_merged_decoders_result, use_single_time_bin_per_epoch: bool, desired_laps_decoding_time_bin_size: Optional[float], desired_ripple_decoding_time_bin_size: Optional[float]):

        # ripple_epochs_df = deepcopy(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
        
        ## Decode Laps:
        laps_epochs_df = deepcopy(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.filter_epochs)
        if not isinstance(laps_epochs_df, pd.DataFrame):
            laps_epochs_df = laps_epochs_df.to_dataframe()
        # global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
        min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(laps_epochs_df['duration'].to_numpy())
        min_bounded_laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002
        
        if desired_laps_decoding_time_bin_size < min_bounded_laps_decoding_time_bin_size:
            print(f'WARN: desired_laps_decoding_time_bin_size: {desired_laps_decoding_time_bin_size} < min_bounded_laps_decoding_time_bin_size: {min_bounded_laps_decoding_time_bin_size}... hopefully it works.')
        laps_decoding_time_bin_size: float = desired_laps_decoding_time_bin_size # allow direct use
        if use_single_time_bin_per_epoch:
            laps_decoding_time_bin_size = None
        directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=laps_epochs_df,
                                                                                                                                                        decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
        directional_merged_decoders_result.perform_compute_marginals()        
        return directional_merged_decoders_result
        

    def _update_result_laps(a_result: DecodedFilterEpochsResult, laps_df: pd.DataFrame) -> pd.DataFrame:
        """ captures nothing. Can reusing the same laps_df as it makes no modifications to it. 
        
        e.g. a_result=output_alt_directional_merged_decoders_result[a_sweep_tuple]
        """
        result_laps_epochs_df: pd.DataFrame = a_result.laps_epochs_df
        ## 2024-01-17 - Updates the `a_directional_merged_decoders_result.laps_epochs_df` with both the ground-truth values and the decoded predictions
        result_laps_epochs_df['maze_id'] = laps_df['maze_id'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching
        ## add the 'is_LR_dir' groud-truth column in:
        result_laps_epochs_df['is_LR_dir'] = laps_df['is_LR_dir'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching
        
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = a_result.laps_directional_marginals_tuple
        laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = a_result.laps_track_identity_marginals_tuple
        ## Add the decoded results to the laps df:
        result_laps_epochs_df['is_most_likely_track_identity_Long'] = laps_is_most_likely_track_identity_Long
        result_laps_epochs_df['is_most_likely_direction_LR'] = laps_is_most_likely_direction_LR_dir
        return result_laps_epochs_df

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    assert collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    active_context = curr_active_pipeline.get_session_context()
    session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
    
    all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_laps_decoding_time_bin_size=np.linspace(0.125, 1.0, num=20), use_single_time_bin_per_epoch=[False], desired_ripple_decoding_time_bin_size=[None])
    # len(all_param_sweep_options)
    
    ## Perfrom the computations:

    # DirectionalMergedDecoders: Get the result after computation:
    ## Copy the default result:
    directional_merged_decoders_result: DirectionalMergedDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result: DirectionalMergedDecodersResult = deepcopy(directional_merged_decoders_result)

    # laps_decoding_time_bin_size = alt_directional_merged_decoders_result.laps_decoding_time_bin_size
    # now_day_str: str = BATCH_DATE_TO_USE
    # active_context: IdentifyingContext = curr_active_pipeline.get_session_context()
    # data_identifier_str=f'(laps_time_bin_marginals_df)'

    # out_path_basename_str: str = f"{now_day_str}_{active_context}_time_bin_size-{laps_decoding_time_bin_size}_{data_identifier_str}"
    # out_path_basename_str: str = f"{now_day_str}_{active_context}_time_bin_size_sweep_results"
    out_path_basename_str: str = f"{CURR_BATCH_OUTPUT_PREFIX}_time_bin_size_sweep_results"
    # out_path_filenname_str: str = f"{out_path_basename_str}.csv"

    out_path_filenname_str: str = f"{out_path_basename_str}.h5"
    out_path: Path = collected_outputs_path.resolve().joinpath(out_path_filenname_str).resolve()
    print(f'\out_path_str: "{out_path_filenname_str}"')
    print(f'\tout_path: "{out_path}"')

    # time_bin_size-{laps_decoding_time_bin_size}
    # curr_time_bin_size_str: str = f"time_bin_size-{laps_decoding_time_bin_size}"
    # laps_time_bin_marginals_df: pd.DataFrame = alt_directional_merged_decoders_result.laps_time_bin_marginals_df.copy()
    # laps_all_epoch_bins_marginals_df: pd.DataFrame = alt_directional_merged_decoders_result.laps_all_epoch_bins_marginals_df.copy()
    
    # Ensure it has the 'lap_track' column
    ## Compute the ground-truth information using the position information:
    # adds columns: ['maze_id', 'is_LR_dir']
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_obj.update_lap_dir_from_smoothed_velocity(pos_input=curr_active_pipeline.sess.position)
    laps_obj.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
    laps_df = laps_obj.to_dataframe()
    # laps_df: pd.DataFrame = Laps._update_dataframe_computed_vars(laps_df=laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=curr_active_pipeline.sess, replace_existing=True) # NOTE: .sess is used because global_session is missing the last two laps   
    # # computes 'track_id' from t_start, t_delta, and t_end where t_delta corresponds to the transition point (track change).
    # laps_df = Laps._update_dataframe_maze_id_if_needed(laps_df=laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end)
    # ## computes proper 'is_LR_dir' and 'lap_dir' columns:
    # laps_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df=laps_df, global_session=curr_active_pipeline.sess) # adds 'is_LR_dir'

    # Uses: session_ctxt_key, all_param_sweep_options
    output_alt_directional_merged_decoders_result = {} # empty dict
    output_laps_decoding_accuracy_results_dict = {} # empty dict

    for a_sweep_dict in all_param_sweep_options:
        a_sweep_tuple = frozenset(a_sweep_dict.items())
        print(f'a_sweep_dict: {a_sweep_dict}')
        # Convert parameters to string because Parquet supports metadata as string
        a_sweep_str_params = {key: str(value) for key, value in a_sweep_dict.items() if value is not None}
        
        output_alt_directional_merged_decoders_result[a_sweep_tuple] = _try_single_decode(curr_active_pipeline, alt_directional_merged_decoders_result, **a_sweep_dict)

        laps_time_bin_marginals_df: pd.DataFrame = output_alt_directional_merged_decoders_result[a_sweep_tuple].laps_time_bin_marginals_df.copy()
        laps_all_epoch_bins_marginals_df: pd.DataFrame = output_alt_directional_merged_decoders_result[a_sweep_tuple].laps_all_epoch_bins_marginals_df.copy()

        desired_laps_decoding_time_bin_size_str: str = a_sweep_str_params.get('desired_laps_decoding_time_bin_size', None)
        laps_decoding_time_bin_size: float = output_alt_directional_merged_decoders_result[a_sweep_tuple].laps_decoding_time_bin_size
        # ripple_decoding_time_bin_size: float = output_alt_directional_merged_decoders_result[a_sweep_tuple].ripple_decoding_time_bin_size
        actual_laps_decoding_time_bin_size_str: str = str(laps_decoding_time_bin_size)
        if actual_laps_decoding_time_bin_size_str is not None:
            laps_time_bin_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_laps_decoding_time_bin_size_str}/laps_time_bin_marginals_df', format='table', data_columns=True)
            laps_all_epoch_bins_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_laps_decoding_time_bin_size_str}/laps_all_epoch_bins_marginals_df', format='table', data_columns=True)

        # get the current lap object and determine the percentage correct:
        result_laps_epochs_df: pd.DataFrame = _update_result_laps(a_result=output_alt_directional_merged_decoders_result[a_sweep_tuple], laps_df=laps_df)
        (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = _check_result_laps_epochs_df_performance(result_laps_epochs_df)
        output_laps_decoding_accuracy_results_dict[laps_decoding_time_bin_size] = (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly)
        

    ## Output the performance:
    output_laps_decoding_accuracy_results_df: pd.DataFrame = pd.DataFrame(output_laps_decoding_accuracy_results_dict.values(), index=output_laps_decoding_accuracy_results_dict.keys(), 
                    columns=['percent_laps_track_identity_estimated_correctly',
                            'percent_laps_direction_estimated_correctly',
                            'percent_laps_estimated_correctly'])
    output_laps_decoding_accuracy_results_df.index.name = 'laps_decoding_time_bin_size'
    # output_laps_decoding_accuracy_results_df
    ## Save out the laps peformance result
    output_laps_decoding_accuracy_results_df.to_hdf(out_path, key=f'{session_ctxt_key}/laps_decoding_accuracy_results', format='table', data_columns=True)

    # add to output dict
    # across_session_results_extended_dict['compute_and_export_marginals_dfs_completion_function'] = _out
    across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'] = (out_path, output_laps_decoding_accuracy_results_df)

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


_pre_user_completion_functions_header_template_str: str = f"""
# ==================================================================================================================== #
# BEGIN USER COMPLETION FUNCTIONS                                                                                      #
# ==================================================================================================================== #
from copy import deepcopy


custom_user_completion_functions = []
"""

_post_user_completion_functions_footer_template_str: str = f"""
# ==================================================================================================================== #
# END USER COMPLETION FUNCTIONS                                                                                        #
# ==================================================================================================================== #
"""

def write_test_script(custom_user_completion_function_template_code: str, script_file_override: Optional[Path]=None):
    """Save out the generated `custom_user_completion_function_template_code` string to file for testing"""
    test_python_script_path = script_file_override or Path('output/test_script.py').resolve()
    with open(test_python_script_path, 'w') as script_file:
        script_content = custom_user_completion_function_template_code
        script_file.write(script_content)
    print(f'wrote: {test_python_script_path}')
    return test_python_script_path


def MAIN_get_template_string(BATCH_DATE_TO_USE: str, collected_outputs_path:Path, override_custom_user_completion_functions_dict: Optional[Dict]=None):
    """ 
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import MAIN_get_template_string
    

    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import MAIN_get_template_string
    
    custom_user_completion_function_template_code, custom_user_completion_functions_dict = MAIN_get_template_string()

    """
    custom_user_completion_functions_dict = override_custom_user_completion_functions_dict or {
                                    "export_rank_order_results_completion_function": export_rank_order_results_completion_function,
                                    "figures_rank_order_results_completion_function": figures_rank_order_results_completion_function,
                                    "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function,
                                    'determine_session_t_delta_completion_function': determine_session_t_delta_completion_function,
                                    'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function,
                                    }
    
    generated_header_code: str = curr_runtime_context_header_template.render(BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, collected_outputs_path_str=str(collected_outputs_path))

    ## Build the template string:
    template_str: str = f"{generated_header_code}\n{_pre_user_completion_functions_header_template_str}"

    for a_name, a_fn in custom_user_completion_functions_dict.items():
        fcn_defn_str: str = inspect.getsource(a_fn)
        template_str = f"{template_str}\n{fcn_defn_str}\ncustom_user_completion_functions.append({a_name})\n# END `{a_name}` USER COMPLETION FUNCTION  _______________________________________________________________________________________ #\n\n"
        
    template_str += _post_user_completion_functions_footer_template_str 

    custom_user_completion_function_template_code = template_str

    # print(custom_user_completion_function_template_code)
    return custom_user_completion_function_template_code, custom_user_completion_functions_dict




