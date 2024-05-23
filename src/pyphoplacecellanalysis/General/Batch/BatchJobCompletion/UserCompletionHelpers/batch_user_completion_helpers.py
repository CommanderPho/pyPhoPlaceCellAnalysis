from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types

from pathlib import Path
import inspect
from jinja2 import Template
from neuropy.utils.result_context import IdentifyingContext
from nptyping import NDArray
import numpy as np
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes



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
# self.BATCH_DATE_TO_USE: str = '0000-00-00_Fake' # TODO: Change this as needed, templating isn't actually doing anything rn.
# collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve() # Linux
# self.collected_outputs_path: Path = Path('/home/halechr/cloud/turbo/Data/Output/collected_outputs').resolve() # GreatLakes
# collected_outputs_path = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\output\collected_outputs').resolve() # Apogee

@function_attributes(short_name=None, tags=['batch', 'rank-order'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-27 21:21', related_items=[])
def export_rank_order_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'export_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import save_rank_order_results, SaveStringGenerator
    save_rank_order_results(curr_active_pipeline, day_date=f"{CURR_BATCH_OUTPUT_PREFIX}", override_output_parent_path=self.collected_outputs_path) # "2024-01-02_301pm" "2024-01-02_734pm""

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
    active_csv_parent_output_path = self.collected_outputs_path.resolve()
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

@function_attributes(short_name=None, tags=['figures'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-27 21:21', related_items=[])
def figures_rank_order_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderGlobalDisplayFunctions
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'figures_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    curr_active_pipeline.reload_default_display_functions()
    _display_rank_order_z_stats_results_out = curr_active_pipeline.display('_display_rank_order_z_stats_results', defer_render=True, save_figure=True)
    # across_session_results_extended_dict['merged_complete_epoch_stats_df'] = merged_complete_ripple_epoch_stats_df_output_path
    
    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # return True
    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['marginal', 'across-sessions', 'CSV'], input_requires=[], output_provides=[], uses=['directional_merged_decoders_result.compute_and_export_marginals_df_csvs'], used_by=[], creation_date='2024-04-27 21:22', related_items=[])
def compute_and_export_marginals_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_marginals_dfs_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations
    curr_active_pipeline.reload_default_computation_functions()
    batch_extended_computations(curr_active_pipeline, include_includelist=['merged_directional_placefields'], include_global_functions=True, fail_on_exception=True, force_recompute=False)
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']

    active_context = curr_active_pipeline.get_session_context()
    _out = directional_merged_decoders_result.compute_and_export_marginals_df_csvs(parent_output_path=self.collected_outputs_path, active_context=active_context)
    print(f'successfully exported marginals_df_csvs to {self.collected_outputs_path}!')
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
    print(f'determine_computation_datetimes_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
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
    print(f'determine_session_t_delta_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
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


@function_attributes(short_name=None, tags=['CSV', 'time_bin_sizes', 'marginals'], input_requires=['DirectionalMergedDecoders'], output_provides=[], uses=[], used_by=[], creation_date='2024-04-27 21:24', related_items=[])
def perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                             save_hdf=True, save_csvs=True, return_full_decoding_results:bool=False, 
                                                                             custom_all_param_sweep_options=None,
                                                                             desired_shared_decoding_time_bin_sizes:Optional[NDArray]=None) -> dict:
    """
    if `return_full_decoding_results` == True, returns the full decoding results for debugging purposes. `output_alt_directional_merged_decoders_result`

    custom_all_param_sweep_options: if provided, these parameters will be used as the parameter sweeps instead of building new ones.

    custom_all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_laps_decoding_time_bin_size=np.linspace(start=0.030, stop=0.10, num=6),
                                                                            use_single_time_bin_per_epoch=[False],
                                                                            minimum_event_duration=[desired_shared_decoding_time_bin_sizes[-1]])

    """
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    from copy import deepcopy
    import numpy as np
    import pandas as pd
    from typing_extensions import TypeAlias
    from typing import NewType
    from nptyping import NDArray

    import neuropy.utils.type_aliases as types
    from neuropy.utils.indexing_helpers import PandasHelpers
    from neuropy.utils.debug_helpers import parameter_sweeps
    from neuropy.core.laps import Laps
    from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
    from pyphocorehelpers.print_helpers import get_now_day_str
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult


    DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names

    # Export CSVs:
    def export_marginals_df_csv(marginals_df: pd.DataFrame, data_identifier_str: str, parent_output_path: Path, active_context):
        """ captures nothing
        """
        # output_date_str: str = get_now_rounded_time_str()
        output_date_str: str = get_now_day_str()
        # parent_output_path: Path = Path('output').resolve()
        # active_context = curr_active_pipeline.get_session_context()
        session_identifier_str: str = active_context.get_description()
        assert output_date_str is not None
        out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-01-04|kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
        out_filename = f"{out_basename}.csv"
        out_path = parent_output_path.joinpath(out_filename).resolve()
        marginals_df.to_csv(out_path)
        return out_path 

    def _subfn_process_time_bin_swept_results(output_extracted_result_tuples, active_context):
        """ After the sweeps are complete and multiple (one for each time_bin_size swept) indepdnent dfs are had with the four results types this function concatenates each of the four into a single dataframe for all time_bin_size values with a column 'time_bin_size'. 
        It also saves them out to CSVs in a manner similar to what `compute_and_export_marginals_dfs_completion_function` did to be compatible with `2024-01-23 - Across Session Point and YellowBlue Marginal CSV Exports.ipynb`
        Captures: save_csvs
        GLOBAL Captures: collected_outputs_path
        
        
        """
        several_time_bin_sizes_laps_time_bin_marginals_df_list = []
        several_time_bin_sizes_laps_per_epoch_marginals_df_list = []

        several_time_bin_sizes_ripple_time_bin_marginals_df_list = []
        several_time_bin_sizes_ripple_per_epoch_marginals_df_list = []


        # for a_sweep_tuple, (a_laps_time_bin_marginals_df, a_laps_all_epoch_bins_marginals_df) in output_extracted_result_tuples.items():
        for a_sweep_tuple, (a_laps_time_bin_marginals_df, a_laps_all_epoch_bins_marginals_df, a_ripple_time_bin_marginals_df, a_ripple_all_epoch_bins_marginals_df) in output_extracted_result_tuples.items():
            a_sweep_dict = dict(a_sweep_tuple)
            

            if 'desired_shared_decoding_time_bin_size' in a_sweep_dict:
                # Shared
                desired_laps_decoding_time_bin_size = float(a_sweep_dict['desired_shared_decoding_time_bin_size'])
                desired_ripple_decoding_time_bin_size = float(a_sweep_dict['desired_shared_decoding_time_bin_size'])
            else:
                # Separate:
                desired_laps_decoding_time_bin_size = float(a_sweep_dict.get('desired_laps_decoding_time_bin_size', None))
                if desired_laps_decoding_time_bin_size is not None:
                    desired_laps_decoding_time_bin_size = float(desired_laps_decoding_time_bin_size)
                
                desired_ripple_decoding_time_bin_size = a_sweep_dict.get('desired_ripple_decoding_time_bin_size', None)
                if desired_ripple_decoding_time_bin_size is not None:
                    desired_ripple_decoding_time_bin_size = float(desired_ripple_decoding_time_bin_size)
            

            if desired_laps_decoding_time_bin_size is not None:
                df = a_laps_time_bin_marginals_df
                df['time_bin_size'] = desired_laps_decoding_time_bin_size # desired_laps_decoding_time_bin_size
                # df['session_name'] = session_name
                df = a_laps_all_epoch_bins_marginals_df
                df['time_bin_size'] = desired_laps_decoding_time_bin_size

                several_time_bin_sizes_laps_time_bin_marginals_df_list.append(a_laps_time_bin_marginals_df)
                several_time_bin_sizes_laps_per_epoch_marginals_df_list.append(a_laps_all_epoch_bins_marginals_df)
                

            if desired_ripple_decoding_time_bin_size is not None:
                df = a_ripple_time_bin_marginals_df
                df['time_bin_size'] = desired_ripple_decoding_time_bin_size
                df = a_ripple_all_epoch_bins_marginals_df
                df['time_bin_size'] = desired_ripple_decoding_time_bin_size

                several_time_bin_sizes_ripple_time_bin_marginals_df_list.append(a_ripple_time_bin_marginals_df)
                several_time_bin_sizes_ripple_per_epoch_marginals_df_list.append(a_ripple_all_epoch_bins_marginals_df)
            

        ## Build across_sessions join dataframes:
        several_time_bin_sizes_time_bin_laps_df: Optional[pd.DataFrame] = PandasHelpers.safe_concat(several_time_bin_sizes_laps_time_bin_marginals_df_list, axis='index', ignore_index=True)
        several_time_bin_sizes_laps_df: Optional[pd.DataFrame] = PandasHelpers.safe_concat(several_time_bin_sizes_laps_per_epoch_marginals_df_list, axis='index', ignore_index=True) # per epoch
        several_time_bin_sizes_time_bin_ripple_df: Optional[pd.DataFrame] = PandasHelpers.safe_concat(several_time_bin_sizes_ripple_time_bin_marginals_df_list, axis='index', ignore_index=True)
        several_time_bin_sizes_ripple_df: Optional[pd.DataFrame] = PandasHelpers.safe_concat(several_time_bin_sizes_ripple_per_epoch_marginals_df_list, axis='index', ignore_index=True) # per epoch

        # Export time_bin_swept results to CSVs:
        laps_time_bin_marginals_out_path, laps_out_path, ripple_time_bin_marginals_out_path, ripple_out_path = None, None, None, None
        if save_csvs:
            assert self.collected_outputs_path.exists()
            assert active_context is not None
            if several_time_bin_sizes_time_bin_laps_df is not None:
                laps_time_bin_marginals_out_path = export_marginals_df_csv(several_time_bin_sizes_time_bin_laps_df, data_identifier_str=f'(laps_time_bin_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)
            if several_time_bin_sizes_laps_df is not None:
                laps_out_path = export_marginals_df_csv(several_time_bin_sizes_laps_df, data_identifier_str=f'(laps_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)
            if several_time_bin_sizes_time_bin_ripple_df is not None:
                ripple_time_bin_marginals_out_path = export_marginals_df_csv(several_time_bin_sizes_time_bin_ripple_df, data_identifier_str=f'(ripple_time_bin_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)
            if several_time_bin_sizes_ripple_df is not None:
                ripple_out_path = export_marginals_df_csv(several_time_bin_sizes_ripple_df, data_identifier_str=f'(ripple_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)

        return (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path)
        # (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path)
        
    def add_session_df_columns(df: pd.DataFrame, session_name: str, curr_session_t_delta: Optional[float], time_col: str) -> pd.DataFrame:
        """ adds session-specific information to the marginal dataframes """
        df['session_name'] = session_name 
        if curr_session_t_delta is not None:
            df['delta_aligned_start_t'] = df[time_col] - curr_session_t_delta
        return df

    ## Merged-only decode:
    def _try_single_decode(owning_pipeline_reference, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, use_single_time_bin_per_epoch: bool,
                            desired_laps_decoding_time_bin_size: Optional[float]=None, desired_ripple_decoding_time_bin_size: Optional[float]=None, desired_shared_decoding_time_bin_size: Optional[float]=None, minimum_event_duration: Optional[float]=None) -> DirectionalPseudo2DDecodersResult:
        """ decodes laps and ripples for a single bin size. 
        
        desired_laps_decoding_time_bin_size
        desired_ripple_decoding_time_bin_size
        minimum_event_duration: if provided, excludes all events shorter than minimum_event_duration

        Looks like it updates:
            .all_directional_laps_filter_epochs_decoder_result, .all_directional_ripple_filter_epochs_decoder_result, and whatever .perform_compute_marginals() updates

        
        Compared to `_compute_lap_and_ripple_epochs_decoding_for_decoder`, it looks like this only computes for the `*all*_directional_pf1D_Decoder` while `_compute_lap_and_ripple_epochs_decoding_for_decoder` is called for each separate directional pf1D decoder
        """
        if desired_shared_decoding_time_bin_size is not None:
            assert desired_laps_decoding_time_bin_size is None
            assert desired_ripple_decoding_time_bin_size is None
            desired_laps_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            desired_ripple_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            
        # Separate the decoder first so they're all independent:
        directional_merged_decoders_result = deepcopy(directional_merged_decoders_result)

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

        ## Decode Ripples: ripples are kinda optional (if `desired_ripple_decoding_time_bin_size is None` they are not computed.
        if desired_ripple_decoding_time_bin_size is not None:
            # global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
            replay_epochs_df = deepcopy(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
            if not isinstance(replay_epochs_df, pd.DataFrame):
                replay_epochs_df = replay_epochs_df.to_dataframe()
            # min_possible_ripple_time_bin_size: float = find_minimum_time_bin_duration(replay_epochs_df['duration'].to_numpy())
            # min_bounded_ripple_decoding_time_bin_size: float = min(desired_ripple_decoding_time_bin_size, min_possible_ripple_time_bin_size) # 10ms # 0.002
            # if desired_ripple_decoding_time_bin_size < min_bounded_ripple_decoding_time_bin_size:
            #     print(f'WARN: desired_ripple_decoding_time_bin_size: {desired_ripple_decoding_time_bin_size} < min_bounded_ripple_decoding_time_bin_size: {min_bounded_ripple_decoding_time_bin_size}... hopefully it works.')
            ripple_decoding_time_bin_size: float = desired_ripple_decoding_time_bin_size # allow direct use            
            ## Drop those less than the time bin duration
            print(f'DropShorterMode:')
            pre_drop_n_epochs = len(replay_epochs_df)
            if minimum_event_duration is not None:                
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > minimum_event_duration]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
            else:
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > desired_ripple_decoding_time_bin_size]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                print(f'\tdropping {n_dropped_epochs} that are shorter than our ripple decoding time bin size of {desired_ripple_decoding_time_bin_size}', end='\t') 

            print(f'{post_drop_n_epochs} remain.')
            # returns a `DecodedFilterEpochsResult`
            directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=replay_epochs_df,
                                                                                                                                                                                            decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

        directional_merged_decoders_result.perform_compute_marginals()
        return directional_merged_decoders_result
        
    

    ## All templates AND merged decode:
    def _try_all_templates_decode(owning_pipeline_reference, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, use_single_time_bin_per_epoch: bool,
                            desired_laps_decoding_time_bin_size: Optional[float]=None, desired_ripple_decoding_time_bin_size: Optional[float]=None, desired_shared_decoding_time_bin_size: Optional[float]=None, minimum_event_duration: Optional[float]=None) -> Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]]: #-> Dict[str, DirectionalPseudo2DDecodersResult]:
        """ decodes laps and ripples for a single bin size but for each of the four track templates. 
        
        Added 2024-05-23 04:23 

        desired_laps_decoding_time_bin_size
        desired_ripple_decoding_time_bin_size
        minimum_event_duration: if provided, excludes all events shorter than minimum_event_duration

        Looks like it updates:
            .all_directional_laps_filter_epochs_decoder_result, .all_directional_ripple_filter_epochs_decoder_result, and whatever .perform_compute_marginals() updates

        
        Compared to `_compute_lap_and_ripple_epochs_decoding_for_decoder`, it looks like this only computes for the `*all*_directional_pf1D_Decoder` while `_compute_lap_and_ripple_epochs_decoding_for_decoder` is called for each separate directional pf1D decoder
        """
        ripple_decoding_time_bin_size = None
        if desired_shared_decoding_time_bin_size is not None:
            assert desired_laps_decoding_time_bin_size is None
            assert desired_ripple_decoding_time_bin_size is None
            desired_laps_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            desired_ripple_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            
        # Separate the decoder first so they're all independent:
        directional_merged_decoders_result = deepcopy(directional_merged_decoders_result)

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

        ## Decode Ripples: ripples are kinda optional (if `desired_ripple_decoding_time_bin_size is None` they are not computed.
        if desired_ripple_decoding_time_bin_size is not None:
            # global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
            replay_epochs_df = deepcopy(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
            if not isinstance(replay_epochs_df, pd.DataFrame):
                replay_epochs_df = replay_epochs_df.to_dataframe()
            # min_possible_ripple_time_bin_size: float = find_minimum_time_bin_duration(replay_epochs_df['duration'].to_numpy())
            # min_bounded_ripple_decoding_time_bin_size: float = min(desired_ripple_decoding_time_bin_size, min_possible_ripple_time_bin_size) # 10ms # 0.002
            # if desired_ripple_decoding_time_bin_size < min_bounded_ripple_decoding_time_bin_size:
            #     print(f'WARN: desired_ripple_decoding_time_bin_size: {desired_ripple_decoding_time_bin_size} < min_bounded_ripple_decoding_time_bin_size: {min_bounded_ripple_decoding_time_bin_size}... hopefully it works.')
            ripple_decoding_time_bin_size: float = desired_ripple_decoding_time_bin_size # allow direct use            
            ## Drop those less than the time bin duration
            print(f'DropShorterMode:')
            pre_drop_n_epochs = len(replay_epochs_df)
            if minimum_event_duration is not None:                
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > minimum_event_duration]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
            else:
                replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > desired_ripple_decoding_time_bin_size]
                post_drop_n_epochs = len(replay_epochs_df)
                n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
                print(f'\tdropping {n_dropped_epochs} that are shorter than our ripple decoding time bin size of {desired_ripple_decoding_time_bin_size}', end='\t') 

            print(f'{post_drop_n_epochs} remain.')

            # returns a `DecodedFilterEpochsResult`
            directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=replay_epochs_df,
                                                                                                                                                                                            decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)


        directional_merged_decoders_result.perform_compute_marginals() # this only works for the pseudo2D decoder, not the individual 1D ones


        # directional_merged_decoders_result_dict: Dict[types.DecoderName, DirectionalPseudo2DDecodersResult] = {}

        decoder_laps_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        decoder_ripple_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        
        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            # external-function way:
            decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _compute_lap_and_ripple_epochs_decoding_for_decoder(a_decoder, curr_active_pipeline, desired_laps_decoding_time_bin_size=laps_decoding_time_bin_size, desired_ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)

            # # this function's way:
            # directional_merged_decoders_result_dict[a_name] = deepcopy(directional_merged_decoders_result)
            
            # directional_merged_decoders_result_dict[a_name].all_directional_laps_filter_epochs_decoder_result = a_decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=deepcopy(laps_epochs_df),
            #                                                                                                                                     decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
            # if ripple_decoding_time_bin_size is not None:
            #     directional_merged_decoders_result_dict[a_name].all_directional_ripple_filter_epochs_decoder_result = a_decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=deepcopy(replay_epochs_df),
            #                                                                                                                                                                                 decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

            # directional_merged_decoders_result_dict[a_name].perform_compute_marginals() # this only works for the pseudo2D decoder, not the individual 1D ones


        return directional_merged_decoders_result, (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
        




        decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict = _perform_compute_custom_epoch_decoding(curr_active_pipeline, directional_merged_decoders_result, track_templates) # Dict[str, Optional[DecodedFilterEpochsResult]]

        ## Recompute the epoch scores/metrics such as radon transform and wcorr:
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(directional_merged_decoders_result, track_templates,
                                                                                                                                                                                            decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                            spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df),
                                                                                                                                                                                            should_skip_radon_transform=True)
        
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df, laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df = merged_df_outputs_tuple
        decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_extras_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict = raw_dict_outputs_tuple


        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _compute_lap_and_ripple_epochs_decoding_for_decoder(a_decoder, curr_active_pipeline, desired_laps_decoding_time_bin_size=laps_decoding_time_bin_size, desired_ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)


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

        ## re-apply the laps changes:
        a_result.laps_epochs_df = result_laps_epochs_df # 2024-04-05 - think this is good so the result has the updated columns, but not exactly certain.

        return result_laps_epochs_df

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    should_output_lap_decoding_performance_info: bool = False

    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    active_context = curr_active_pipeline.get_session_context()
    session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
    
    ## INPUT PARAMETER: time_bin_size sweep paraemters    
    if custom_all_param_sweep_options is None:
        if desired_shared_decoding_time_bin_sizes is None:
            desired_shared_decoding_time_bin_sizes = np.linspace(start=0.030, stop=0.10, num=6)
        # Shared time bin sizes
        custom_all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_shared_decoding_time_bin_size=desired_shared_decoding_time_bin_sizes, use_single_time_bin_per_epoch=[False], minimum_event_duration=[desired_shared_decoding_time_bin_sizes[-1]]) # with Ripples

        # ## Laps Only:
        # custom_all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_laps_decoding_time_bin_size=desired_shared_decoding_time_bin_sizes,
        #                                                                         use_single_time_bin_per_epoch=[False],
        #                                                                         minimum_event_duration=[desired_shared_decoding_time_bin_sizes[-1]])

    else:
        assert desired_shared_decoding_time_bin_sizes is None, f"when providing `custom_all_param_sweep_options`, desired_shared_decoding_time_bin_sizes must be None (not specified)."


    all_param_sweep_options = custom_all_param_sweep_options


    ## Perfrom the computations:

    # DirectionalMergedDecoders: Get the result after computation:
    rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder'] # : "RankOrderComputationsContainer"
    minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
    # included_qclu_values: List[int] = rank_order_results.included_qclu_values
    directional_laps_results: "DirectionalLapsResult" = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    track_templates: "TrackTemplates" = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?

    ## Copy the default result:
    directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)

    # out_path_basename_str: str = f"{now_day_str}_{active_context}_time_bin_size-{laps_decoding_time_bin_size}_{data_identifier_str}"
    # out_path_basename_str: str = f"{now_day_str}_{active_context}_time_bin_size_sweep_results"
    out_path_basename_str: str = f"{CURR_BATCH_OUTPUT_PREFIX}_time_bin_size_sweep_results"
    # out_path_filenname_str: str = f"{out_path_basename_str}.csv"

    out_path_filenname_str: str = f"{out_path_basename_str}.h5"
    out_path: Path = self.collected_outputs_path.resolve().joinpath(out_path_filenname_str).resolve()
    print(f'\out_path_str: "{out_path_filenname_str}"')
    print(f'\tout_path: "{out_path}"')
    
    # Ensure it has the 'lap_track' column
    ## Compute the ground-truth information using the position information:
    # adds columns: ['maze_id', 'is_LR_dir']
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_obj.update_lap_dir_from_smoothed_velocity(pos_input=curr_active_pipeline.sess.position)
    laps_obj.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
    laps_df = laps_obj.to_dataframe()
    assert 'maze_id' in laps_df.columns, f"laps_df is still missing the 'maze_id' column after calling `laps_obj.update_maze_id_if_needed(...)`. laps_df.columns: {print(list(laps_df.columns))}"

    # # BEGIN BLOCK ________________________________________________________________________________________________________ #

    # # Uses: session_ctxt_key, all_param_sweep_options
    # output_alt_directional_merged_decoders_result: Dict[Tuple, DirectionalPseudo2DDecodersResult] = {} # empty dict
    # output_laps_decoding_accuracy_results_dict = {} # empty dict
    # output_extracted_result_tuples = {}

    # for a_sweep_dict in all_param_sweep_options:
    #     a_sweep_tuple = frozenset(a_sweep_dict.items())
    #     print(f'a_sweep_dict: {a_sweep_dict}')
    #     # Convert parameters to string because Parquet supports metadata as string
    #     a_sweep_str_params = {key: str(value) for key, value in a_sweep_dict.items() if value is not None}
        
    #     output_alt_directional_merged_decoders_result[a_sweep_tuple] = _try_single_decode(curr_active_pipeline, alt_directional_merged_decoders_result, **a_sweep_dict)

    #     laps_time_bin_marginals_df: pd.DataFrame = output_alt_directional_merged_decoders_result[a_sweep_tuple].laps_time_bin_marginals_df.copy()
    #     laps_all_epoch_bins_marginals_df: pd.DataFrame = output_alt_directional_merged_decoders_result[a_sweep_tuple].laps_all_epoch_bins_marginals_df.copy()
        
    #     ## Ripples:
    #     ripple_time_bin_marginals_df: pd.DataFrame = output_alt_directional_merged_decoders_result[a_sweep_tuple].ripple_time_bin_marginals_df.copy()
    #     ripple_all_epoch_bins_marginals_df: pd.DataFrame = output_alt_directional_merged_decoders_result[a_sweep_tuple].ripple_all_epoch_bins_marginals_df.copy()

    #     session_name = curr_session_name
    #     curr_session_t_delta = t_delta
        
    #     for a_df, a_time_bin_column_name in zip((laps_time_bin_marginals_df, laps_all_epoch_bins_marginals_df, ripple_time_bin_marginals_df, ripple_all_epoch_bins_marginals_df), ('t_bin_center', 'lap_start_t', 't_bin_center', 'ripple_start_t')):
    #         ## Add the session-specific columns:
    #         a_df = add_session_df_columns(a_df, session_name, curr_session_t_delta, a_time_bin_column_name)

    #     ## Build the output tuple:
    #     output_extracted_result_tuples[a_sweep_tuple] = (laps_time_bin_marginals_df, laps_all_epoch_bins_marginals_df, ripple_time_bin_marginals_df, ripple_all_epoch_bins_marginals_df)
        
    #     # desired_laps_decoding_time_bin_size_str: str = a_sweep_str_params.get('desired_laps_decoding_time_bin_size', None)
    #     laps_decoding_time_bin_size: float = output_alt_directional_merged_decoders_result[a_sweep_tuple].laps_decoding_time_bin_size
    #     # ripple_decoding_time_bin_size: float = output_alt_directional_merged_decoders_result[a_sweep_tuple].ripple_decoding_time_bin_size
    #     actual_laps_decoding_time_bin_size_str: str = str(laps_decoding_time_bin_size)
    #     if save_hdf and (actual_laps_decoding_time_bin_size_str is not None):
    #         laps_time_bin_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_laps_decoding_time_bin_size_str}/laps_time_bin_marginals_df', format='table', data_columns=True)
    #         laps_all_epoch_bins_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_laps_decoding_time_bin_size_str}/laps_all_epoch_bins_marginals_df', format='table', data_columns=True)

    #     ## TODO: output ripple .h5 here if desired.
            
    #     # get the current lap object and determine the percentage correct:
    #     result_laps_epochs_df: pd.DataFrame = _update_result_laps(a_result=output_alt_directional_merged_decoders_result[a_sweep_tuple], laps_df=laps_df)
    #     (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = _check_result_laps_epochs_df_performance(result_laps_epochs_df)
    #     output_laps_decoding_accuracy_results_dict[laps_decoding_time_bin_size] = (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly)
        

    # BEGIN BLOCK 2 - modernizing from `_perform_compute_custom_epoch_decoding`  ________________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _compute_lap_and_ripple_epochs_decoding_for_decoder, _perform_compute_custom_epoch_decoding, _compute_all_df_score_metrics

    # Uses: session_ctxt_key, all_param_sweep_options
    # output_alt_directional_merged_decoders_result: Dict[Tuple, Dict[types.DecoderName, DirectionalPseudo2DDecodersResult]] = {} # empty dict

    output_alt_directional_merged_decoders_result: Dict[Tuple, DirectionalPseudo2DDecodersResult] = {} # empty dict
    # output_alt_directional_merged_decoders_result: Dict[Tuple, Dict[types.DecoderName, DirectionalPseudo2DDecodersResult]] = {} # empty dict

    # Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]]
    output_directional_decoders_epochs_decode_results_dict: Dict[Tuple, DecoderDecodedEpochsResult] = {} # `_decode_and_evaluate_epochs_using_directional_decoders`-style output

    output_laps_decoding_accuracy_results_dict = {} # empty dict
    output_extracted_result_tuples = {}


    for a_sweep_dict in all_param_sweep_options:
        a_sweep_tuple = frozenset(a_sweep_dict.items())
        print(f'a_sweep_dict: {a_sweep_dict}')
        # Convert parameters to string because Parquet supports metadata as string
        # a_sweep_str_params = {key: str(value) for key, value in a_sweep_dict.items() if value is not None}
        


        # for a_name, a_decoder in track_templates.get_decoders_dict().items():
        #     decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _compute_lap_and_ripple_epochs_decoding_for_decoder(a_decoder, curr_active_pipeline, desired_laps_decoding_time_bin_size=laps_decoding_time_bin_size, desired_ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)


        # output_alt_directional_merged_decoders_result[a_sweep_tuple] = _try_all_templates_decode(curr_active_pipeline, alt_directional_merged_decoders_result, **a_sweep_dict) # type: ignore


        output_alt_directional_merged_decoders_result[a_sweep_tuple], (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = _try_all_templates_decode(curr_active_pipeline, alt_directional_merged_decoders_result, **a_sweep_dict)
        an_alt_dir_Pseudo2D_decoders_result = output_alt_directional_merged_decoders_result[a_sweep_tuple]


        ## Decode epochs for all four decoders:


        ## This function post-compute:

        # for k, an_alt_dir_Pseudo2D_decoders_result in output_alt_directional_merged_decoders_result[a_sweep_tuple].items():

        laps_time_bin_marginals_df: pd.DataFrame = an_alt_dir_Pseudo2D_decoders_result.laps_time_bin_marginals_df.copy()
        laps_all_epoch_bins_marginals_df: pd.DataFrame = an_alt_dir_Pseudo2D_decoders_result.laps_all_epoch_bins_marginals_df.copy()
        
        ## Ripples:
        ripple_time_bin_marginals_df: pd.DataFrame = an_alt_dir_Pseudo2D_decoders_result.ripple_time_bin_marginals_df.copy()
        ripple_all_epoch_bins_marginals_df: pd.DataFrame = an_alt_dir_Pseudo2D_decoders_result.ripple_all_epoch_bins_marginals_df.copy()

        session_name = curr_session_name
        curr_session_t_delta = t_delta
        
        for a_df, a_time_bin_column_name in zip((laps_time_bin_marginals_df, laps_all_epoch_bins_marginals_df, ripple_time_bin_marginals_df, ripple_all_epoch_bins_marginals_df), ('t_bin_center', 'lap_start_t', 't_bin_center', 'ripple_start_t')):
            ## Add the session-specific columns:
            a_df = add_session_df_columns(a_df, session_name, curr_session_t_delta, a_time_bin_column_name)

        ## Build the output tuple:
        output_extracted_result_tuples[a_sweep_tuple] = (laps_time_bin_marginals_df, laps_all_epoch_bins_marginals_df, ripple_time_bin_marginals_df, ripple_all_epoch_bins_marginals_df) # output tuples are extracted here, where changes are needed I think
        
        ## Laps:
        # desired_laps_decoding_time_bin_size_str: str = a_sweep_str_params.get('desired_laps_decoding_time_bin_size', None)
        laps_decoding_time_bin_size: float = an_alt_dir_Pseudo2D_decoders_result.laps_decoding_time_bin_size
        # ripple_decoding_time_bin_size: float = v.ripple_decoding_time_bin_size
        actual_laps_decoding_time_bin_size_str: str = str(laps_decoding_time_bin_size)
        if save_hdf and (actual_laps_decoding_time_bin_size_str is not None):
            laps_time_bin_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_laps_decoding_time_bin_size_str}/laps_time_bin_marginals_df', format='table', data_columns=True)
            laps_all_epoch_bins_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_laps_decoding_time_bin_size_str}/laps_all_epoch_bins_marginals_df', format='table', data_columns=True)

        ## TODO: output ripple .h5 here if desired.
        
        # get the current lap object and determine the percentage correct:
        if should_output_lap_decoding_performance_info:
            result_laps_epochs_df: pd.DataFrame = _update_result_laps(a_result=an_alt_dir_Pseudo2D_decoders_result, laps_df=laps_df)
            (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = _check_result_laps_epochs_df_performance(result_laps_epochs_df)
            output_laps_decoding_accuracy_results_dict[laps_decoding_time_bin_size] = (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly)



        # `_decode_and_evaluate_epochs_using_directional_decoders` post compute ______________________________________________ #

        # decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict = _perform_compute_custom_epoch_decoding(curr_active_pipeline, directional_merged_decoders_result, track_templates) # Dict[str, Optional[DecodedFilterEpochsResult]]

        ## Recompute the epoch scores/metrics such as radon transform and wcorr:
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(an_alt_dir_Pseudo2D_decoders_result, track_templates,
                                                                                                                                                                                            decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                            spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df),
                                                                                                                                                                                            should_skip_radon_transform=True)
        
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df, laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df = merged_df_outputs_tuple
        decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_extras_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict = raw_dict_outputs_tuple


        #TODO 2024-02-16 13:46: - [ ] Currently always replace
        ## Create or update the global directional_merged_decoders_result:
        # directional_decoders_epochs_decode_result: DirectionalPseudo2DDecodersResult = global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
        # if directional_decoders_epochs_decode_result is not None:
        # directional_decoders_epochs_decode_result.__dict__.update(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
        #                                               long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
        #                                               short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder)
        ripple_decoding_time_bin_size = an_alt_dir_Pseudo2D_decoders_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size = an_alt_dir_Pseudo2D_decoders_result.laps_decoding_time_bin_size
        pos_bin_size = an_alt_dir_Pseudo2D_decoders_result.all_directional_pf1D_Decoder.pos_bin_size

        curr_sweep_directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = DecoderDecodedEpochsResult(is_global=True, **{'pos_bin_size': pos_bin_size, 'ripple_decoding_time_bin_size':ripple_decoding_time_bin_size, 'laps_decoding_time_bin_size':laps_decoding_time_bin_size,
                                                                                                'decoder_laps_filter_epochs_decoder_result_dict':decoder_laps_filter_epochs_decoder_result_dict,
            'decoder_ripple_filter_epochs_decoder_result_dict':decoder_ripple_filter_epochs_decoder_result_dict, 'decoder_laps_radon_transform_df_dict':decoder_laps_radon_transform_df_dict, 'decoder_ripple_radon_transform_df_dict':decoder_ripple_radon_transform_df_dict,
            'decoder_laps_radon_transform_extras_dict': decoder_laps_radon_transform_extras_dict, 'decoder_ripple_radon_transform_extras_dict': decoder_ripple_radon_transform_extras_dict,
            'laps_weighted_corr_merged_df': laps_weighted_corr_merged_df, 'ripple_weighted_corr_merged_df': ripple_weighted_corr_merged_df, 'decoder_laps_weighted_corr_df_dict': decoder_laps_weighted_corr_df_dict, 'decoder_ripple_weighted_corr_df_dict': decoder_ripple_weighted_corr_df_dict,
            'laps_simple_pf_pearson_merged_df': laps_simple_pf_pearson_merged_df, 'ripple_simple_pf_pearson_merged_df': ripple_simple_pf_pearson_merged_df,
            })
        output_directional_decoders_epochs_decode_results_dict[a_sweep_tuple] = curr_sweep_directional_decoders_epochs_decode_result
    
    # END FOR a_sweep_dict in all_param_sweep_options



    # END BLOCK __________________________________________________________________________________________________________ #

    if should_output_lap_decoding_performance_info:
        ## Output the performance:
        output_laps_decoding_accuracy_results_df: pd.DataFrame = pd.DataFrame(output_laps_decoding_accuracy_results_dict.values(), index=output_laps_decoding_accuracy_results_dict.keys(), 
                        columns=['percent_laps_track_identity_estimated_correctly',
                                'percent_laps_direction_estimated_correctly',
                                'percent_laps_estimated_correctly'])
        output_laps_decoding_accuracy_results_df.index.name = 'laps_decoding_time_bin_size'
        ## Save out the laps peformance result
        if save_hdf:
            output_laps_decoding_accuracy_results_df.to_hdf(out_path, key=f'{session_ctxt_key}/laps_decoding_accuracy_results', format='table', data_columns=True)
    else:
        output_laps_decoding_accuracy_results_df = None

    
    ## Call the subfunction to process the time_bin_size swept result and produce combined output dataframes:
    combined_multi_timebin_outputs_tuple = _subfn_process_time_bin_swept_results(output_extracted_result_tuples, active_context=curr_active_pipeline.get_session_context())
    # Unpacking:    
    # (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path) = combined_multi_timebin_outputs_tuple

    # add to output dict
    # across_session_results_extended_dict['compute_and_export_marginals_dfs_completion_function'] = _out
    across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'] = [out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple]

    if return_full_decoding_results:
        # output_alt_directional_merged_decoders_result: Dict[Tuple, DirectionalPseudo2DDecodersResult]
        across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'].append(output_alt_directional_merged_decoders_result) # append the real full results
        across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'].append(output_directional_decoders_epochs_decode_results_dict) # append the real full results

        ## Save out the laps peformance result
        if save_hdf:
            ## figure out how to save the actual dict out to HDF
            print(f'`return_full_decoding_results` is True and `save_hdf` is True, but I do not yet know how to propperly output the `output_alt_directional_merged_decoders_result`')
            # for a_sweep_dict in all_param_sweep_options:
            #     a_sweep_tuple = frozenset(a_sweep_dict.items())
            #     print(f'a_sweep_dict: {a_sweep_dict}')
            #     # Convert parameters to string because Parquet supports metadata as string
            #     a_sweep_str_params = {key: str(value) for key, value in a_sweep_dict.items() if value is not None}
            #     a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = output_alt_directional_merged_decoders_result[a_sweep_tuple]

            #     # 2024-04-03 `DirectionalPseudo2DDecodersResult` is actually missing a `to_hdf` implementation, so no dice.

        #     output_alt_directional_merged_decoders_result.to_hdf(out_path, key=f'{session_ctxt_key}/alt_directional_merged_decoders_result')
        across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'] = tuple(across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'])


    ## UNPACKING
    # # with `return_full_decoding_results == False`
    # out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple = across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']

    # # with `return_full_decoding_results == True`
    # out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple, output_full_directional_merged_decoders_result = across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']


    # can unpack like:
    (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path) = combined_multi_timebin_outputs_tuple

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['CSVs', 'export', 'across-sessions', 'batch', 'ripple_all_scores_merged_df'], input_requires=['DirectionalDecodersEpochsEvaluations'], output_provides=[], uses=[], used_by=[], creation_date='2024-04-27 21:20', related_items=[])
def compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """
    Aims to export the results of the global 'directional_decoders_evaluate_epochs' calculation

    Exports: 'ripple_all_scores_merged_df'

    Uses result computed by `_decode_and_evaluate_epochs_using_directional_decoders`


    """
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function(global_data_root_parent_path: "{global_data_root_parent_path}", curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)') # ,across_session_results_extended_dict: {across_session_results_extended_dict}
    from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _workaround_validate_has_directional_decoded_epochs_heuristic_scoring

    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring

    assert self.collected_outputs_path.exists()
    active_context = curr_active_pipeline.get_session_context()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    ## Doesn't force recompute! Assumes that the DirectionalDecodersEpochsEvaluations result is new
    # curr_active_pipeline.reload_default_computation_functions()
    # batch_extended_computations(curr_active_pipeline, include_includelist=['directional_decoders_evaluate_epochs'], include_global_functions=True, fail_on_exception=True, force_recompute=True)
    
    ## Extract Data:
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # DirectionalLapsResult
    rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
    minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
    included_qclu_values: float = rank_order_results.included_qclu_values
    track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only # TrackTemplates


    directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
    pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
    ripple_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
    laps_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
    # decoder_laps_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict
    # decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict

    # Radon Transforms:
    decoder_laps_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_df_dict
    decoder_ripple_radon_transform_df_dict = directional_decoders_epochs_decode_result.decoder_ripple_radon_transform_df_dict
    decoder_laps_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_laps_radon_transform_extras_dict
    decoder_ripple_radon_transform_extras_dict = directional_decoders_epochs_decode_result.decoder_ripple_radon_transform_extras_dict

    # Weighted correlations:
    laps_weighted_corr_merged_df: pd.DataFrame = directional_decoders_epochs_decode_result.laps_weighted_corr_merged_df
    ripple_weighted_corr_merged_df: pd.DataFrame = directional_decoders_epochs_decode_result.ripple_weighted_corr_merged_df
    decoder_laps_weighted_corr_df_dict: Dict[str, pd.DataFrame] = directional_decoders_epochs_decode_result.decoder_laps_weighted_corr_df_dict
    decoder_ripple_weighted_corr_df_dict: Dict[str, pd.DataFrame] = directional_decoders_epochs_decode_result.decoder_ripple_weighted_corr_df_dict

    # Pearson's correlations:
    laps_simple_pf_pearson_merged_df: pd.DataFrame = directional_decoders_epochs_decode_result.laps_simple_pf_pearson_merged_df
    ripple_simple_pf_pearson_merged_df: pd.DataFrame = directional_decoders_epochs_decode_result.ripple_simple_pf_pearson_merged_df

    ## FILTERING FOR GOOD ROWS:
    

    ## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict

    # 2024-03-04 - Filter out the epochs based on the criteria:
    _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
    filtered_valid_epoch_times = filtered_epochs_df[['start', 'stop']].to_numpy()

    ## filter the epochs by something and only show those:
    # INPUTS: filtered_epochs_df
    # filtered_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(active_epochs_df[['start', 'stop']].to_numpy())

    ## 2024-03-08 - Also constrain the user-selected ones (just to try it):
    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates)

    ## run 'directional_decoders_epoch_heuristic_scoring',
    directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=True)

    # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores`
    if (not _workaround_validate_has_directional_decoded_epochs_heuristic_scoring(curr_active_pipeline)):
        print(f'\tmissing heuristic columns. Recomputing:')
        directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict)
        print(f'\tdone recomputing heuristics.')

    print(f'\tComputation complete. Exporting .CSVs...')

    ## Export CSVs:
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    _output_csv_paths = directional_decoders_epochs_decode_result.export_csvs(parent_output_path=self.collected_outputs_path.resolve(), active_context=active_context, session_name=curr_session_name, curr_session_t_delta=t_delta,
                                                                              user_annotation_selections={'ripple': any_good_selected_epoch_times},
                                                                              valid_epochs_selections={'ripple': filtered_valid_epoch_times})
    

    print(f'\t\tsuccessfully exported directional_decoders_epochs_decode_result to {self.collected_outputs_path}!')
    _output_csv_paths_info_str: str = '\n'.join([f'{a_name}: "{file_uri_from_path(a_path)}"' for a_name, a_path in _output_csv_paths.items()])
    # print(f'\t\t\tCSV Paths: {_output_csv_paths}\n')
    print(f'\t\t\tCSV Paths: {_output_csv_paths_info_str}\n')

    # add to output dict
    # across_session_results_extended_dict['compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function'] = _out

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict



def reload_exported_kdiba_session_position_info_mat_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """ 
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import reload_exported_kdiba_session_position_info_mat_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
    from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
    from neuropy.core.session.Formats.SessionSpecifications import SessionConfig

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'reload_exported_kdiba_session_position_info_mat_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    active_data_mode_name: str = curr_active_pipeline.session_data_type

    # known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    # active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    a_session = deepcopy(curr_active_pipeline.sess)
    # sess_config: SessionConfig = SessionConfig(**deepcopy(session.config.to_dict()))
    sess_config: SessionConfig = SessionConfig(**deepcopy(a_session.config.__getstate__()))
    a_session.config = sess_config

    # a_session = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=deepcopy(curr_active_pipeline.sess))
    a_session = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=a_session)
    # a_session

    curr_active_pipeline.stage.sess = a_session ## apply the session
    # curr_active_pipeline.sess.config = a_session.config # apply the config only...

    loaded_track_limits = a_session.config.loaded_track_limits
    
    a_config_dict = a_session.config.to_dict()

    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    print(f'\t{curr_session_basedir}:\tloaded_track_limits: {loaded_track_limits}, a_config_dict: {a_config_dict}')  # , t_end: {t_end}
    
    callback_outputs = {
     'loaded_track_limits': loaded_track_limits, 'a_config_dict':a_config_dict, #'t_end': t_end   
    }
    across_session_results_extended_dict['position_info_mat_reload_completion_function'] = callback_outputs
    
    # print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict



def export_session_h5_file_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """  Export the pipeline's HDF5 as 'pipeline_results.h5'
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import reload_exported_kdiba_session_position_info_mat_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    import sys
    from datetime import timedelta, datetime
    from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'export_session_h5_file_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    

    hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('pipeline_results.h5').resolve()
    print(f'pipeline hdf5_output_path: {hdf5_output_path}')
    err = None
    # Only get files newer than date
    skip_overwriting_files_newer_than_specified:bool = False

    was_write_good: bool = False
    newest_file_to_overwrite_date = datetime.now() - timedelta(days=1) # don't overwrite any files more recent than 1 day ago
    can_skip_if_allowed: bool = (hdf5_output_path.exists() and (FilesystemMetadata.get_last_modified_time(hdf5_output_path)<=newest_file_to_overwrite_date))
    if (not skip_overwriting_files_newer_than_specified) or (not can_skip_if_allowed):
        # if skipping is disabled OR skipping is enabled but it's not valid to skip, overwrite.
        # file is folder than the date to overwrite, so overwrite it
        print(f'OVERWRITING (or writing) the file {hdf5_output_path}!')
        # with ExceptionPrintingContext(suppress=False):
        try:
            curr_active_pipeline.export_pipeline_to_h5()
            was_write_good = True
        except BaseException as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {err} while trying to build the session HDF output for {curr_session_context}")
            hdf5_output_path = None # set to None because it failed.
            if self.fail_on_exception:
                raise err.exc
    else:
        print(f'WARNING: file {hdf5_output_path} is newer than the allowed overwrite date, so it will be skipped.')
        print(f'\t\tnewest_file_to_overwrite_date: {newest_file_to_overwrite_date}\t can_skip_if_allowed: {can_skip_if_allowed}\n')
        # return (hdf5_output_path, None)


    callback_outputs = {
     'hdf5_output_path': hdf5_output_path, 'e':err, #'t_end': t_end   
    }
    across_session_results_extended_dict['export_session_h5_file_completion_function'] = callback_outputs
    
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
    """ Gets the python code template string that will be inserted into the produced python scripts.
    
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
                                    'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function,
                                    'reload_exported_kdiba_session_position_info_mat_completion_function': reload_exported_kdiba_session_position_info_mat_completion_function,
                                    'export_session_h5_file_completion_function': export_session_h5_file_completion_function,
                                    }
    
    
    generated_header_code: str = curr_runtime_context_header_template.render(BATCH_DATE_TO_USE=BATCH_DATE_TO_USE, collected_outputs_path_str=str(collected_outputs_path.as_posix()))

    ## Build the template string:
    template_str: str = f"{generated_header_code}\n{_pre_user_completion_functions_header_template_str}"

    for a_name, a_fn in custom_user_completion_functions_dict.items():
        fcn_defn_str: str = inspect.getsource(a_fn)
        template_str = f"{template_str}\n{fcn_defn_str}\ncustom_user_completion_functions.append({a_name})\n# END `{a_name}` USER COMPLETION FUNCTION  _______________________________________________________________________________________ #\n\n"
        
    template_str += _post_user_completion_functions_footer_template_str 

    custom_user_completion_function_template_code = template_str

    # print(custom_user_completion_function_template_code)
    return custom_user_completion_function_template_code, custom_user_completion_functions_dict




