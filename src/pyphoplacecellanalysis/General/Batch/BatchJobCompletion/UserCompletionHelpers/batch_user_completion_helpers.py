from copy import deepcopy
import shutil
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from neuropy.core.epoch import ensure_dataframe
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


from attrs import make_class


SimpleBatchComputationDummy = make_class('SimpleBatchComputationDummy', attrs=['BATCH_DATE_TO_USE', 'collected_outputs_path', 'fail_on_exception'])


# ==================================================================================================================== #
# batch_user_completion_helpers                                                                                        #
# ==================================================================================================================== #
""" This file contains custom user-defined functions to be executed at the end of a batch pipeline run (after the rest of the computations are done). Can be used to define post-hoc corrections, perform exports of results to file, etc.



from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import 

a_dummy = SimpleBatchComputationDummy(BATCH_DATE_TO_USE, collected_outputs_path, True)


"""

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
    print(f'successfully exported marginals_df_csvs to "{self.collected_outputs_path}"!')
    # (laps_marginals_df, laps_out_path), (ripple_marginals_df, ripple_out_path) = _out
    (laps_marginals_df, laps_out_path, laps_time_bin_marginals_df, laps_time_bin_marginals_out_path), (ripple_marginals_df, ripple_out_path, ripple_time_bin_marginals_df, ripple_time_bin_marginals_out_path) = _out
    print(f'\tlaps_out_path: {laps_out_path}\n\tripple_out_path: {ripple_out_path}\n\tdone.')

    # add to output dict
    # across_session_results_extended_dict['compute_and_export_marginals_dfs_completion_function'] = _out

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict

@function_attributes(short_name=None, tags=['export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
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

@function_attributes(short_name=None, tags=['t_delta', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
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


@function_attributes(short_name=None, tags=['CSV', 'time_bin_sizes', 'marginals', 'multi_timebin'], input_requires=['DirectionalMergedDecoders'], output_provides=[], uses=['_compute_all_df_score_metrics'], used_by=[], creation_date='2024-04-27 21:24', related_items=[])
def perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                             save_hdf=True, save_csvs=True, return_full_decoding_results:bool=False, 
                                                                             custom_all_param_sweep_options=None,
                                                                             desired_shared_decoding_time_bin_sizes:Optional[NDArray]=None,
                                                                             additional_session_context: Optional[IdentifyingContext]=None) -> dict:
    """
    if `return_full_decoding_results` == True, returns the full decoding results for debugging purposes. `output_alt_directional_merged_decoders_result`

    custom_all_param_sweep_options: if provided, these parameters will be used as the parameter sweeps instead of building new ones.

    custom_all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_laps_decoding_time_bin_size=np.linspace(start=0.030, stop=0.10, num=6),
                                                                            use_single_time_bin_per_epoch=[False],
                                                                            minimum_event_duration=[desired_shared_decoding_time_bin_sizes[-1]])



    additional_session_context if provided, this is combined with the session context.
    ## CSVs are saved out in `_subfn_process_time_bin_swept_results`

    
    'K:/scratch/collected_outputs/2024-09-25_Apogee-2006-4-28_12-38-13-None_time_bin_size_sweep_results.h5'
    'K:/scratch/collected_outputs/2024-09-25-kdiba_vvp01_two_2006-4-28_12-38-13_None-(ripple_time_bin_marginals_df).csv'
    
    
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

    suppress_exceptions: bool = (not self.fail_on_exception)

    if additional_session_context is None:
        print(f'\t!!!! 2024-07-10 WARNING: additional_session_context is None!')

    # BEGIN _SUBFNS_ _____________________________________________________________________________________________________ #
    # Export CSVs:
    def export_marginals_df_csv(marginals_df: pd.DataFrame, data_identifier_str: str, parent_output_path: Path, active_context: IdentifyingContext):
        """ captures nothing
        
        Outputs: '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
        
        """
        # output_date_str: str = get_now_rounded_time_str()
        output_date_str: str = get_now_day_str()
        # parent_output_path: Path = Path('output').resolve()
        # active_context = curr_active_pipeline.get_session_context()
        session_identifier_str: str = active_context.get_description()
        assert output_date_str is not None
        out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
        out_filename = f"{out_basename}.csv"
        out_path = parent_output_path.joinpath(out_filename).resolve()
        marginals_df.to_csv(out_path)
        return out_path 

    def _subfn_process_time_bin_swept_results(output_extracted_result_tuples, active_context: IdentifyingContext):
        """ After the sweeps are complete and multiple (one for each time_bin_size swept) indepdnent dfs are had with the four results types this function concatenates each of the four into a single dataframe for all time_bin_size values with a column 'time_bin_size'. 
        It also saves them out to CSVs in a manner similar to what `compute_and_export_marginals_dfs_completion_function` did to be compatible with `2024-01-23 - Across Session Point and YellowBlue Marginal CSV Exports.ipynb`
        Captures: save_csvs
        GLOBAL Captures: collected_outputs_path
        
        Produces: a single output df flattened acrossed all time bin sizes
        
        Outputs:
        [laps_time_bin_marginals_out_path, laps_out_path, ripple_time_bin_marginals_out_path, ripple_out_path]:
            '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_time_bin_marginals_df).csv'
            '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
            '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(ripple_time_bin_marginals_df).csv'
            '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(ripple_marginals_df).csv'
        
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

    ## All templates AND merged decode:
    def _try_all_templates_decode(owning_pipeline_reference, directional_merged_decoders_result: DirectionalPseudo2DDecodersResult, use_single_time_bin_per_epoch: bool,
                            desired_laps_decoding_time_bin_size: Optional[float]=None, desired_ripple_decoding_time_bin_size: Optional[float]=None, desired_shared_decoding_time_bin_size: Optional[float]=None, minimum_event_duration: Optional[float]=None, suppress_exceptions: bool=True) -> Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]]: #-> Dict[str, DirectionalPseudo2DDecodersResult]:
        """ decodes laps and ripples for a single bin size but for each of the four track templates. 
        
        Added 2024-05-23 04:23 

        desired_laps_decoding_time_bin_size
        desired_ripple_decoding_time_bin_size
        minimum_event_duration: if provided, excludes all events shorter than minimum_event_duration

        Looks like it updates:
            .all_directional_laps_filter_epochs_decoder_result, .all_directional_ripple_filter_epochs_decoder_result, and whatever .perform_compute_marginals() updates

        
        Compared to `_compute_lap_and_ripple_epochs_decoding_for_decoder`, it looks like this only computes for the `*all*_directional_pf1D_Decoder` while `_compute_lap_and_ripple_epochs_decoding_for_decoder` is called for each separate directional pf1D decoder

        Usage:

            from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function, _try_all_templates_decode

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode
        
        ripple_decoding_time_bin_size = None
        if desired_shared_decoding_time_bin_size is not None:
            assert desired_laps_decoding_time_bin_size is None
            assert desired_ripple_decoding_time_bin_size is None
            desired_laps_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            desired_ripple_decoding_time_bin_size = desired_shared_decoding_time_bin_size
            
        # Separate the decoder first so they're all independent:
        directional_merged_decoders_result = deepcopy(directional_merged_decoders_result)

        ## Decode Laps: laps are also optional (if `desired_laps_decoding_time_bin_size is None` they are not computed.
        if desired_laps_decoding_time_bin_size is not None:
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

        else:
            laps_decoding_time_bin_size = None

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
        else:
            ripple_decoding_time_bin_size = None

        directional_merged_decoders_result.perform_compute_marginals() # this only works for the pseudo2D decoder, not the individual 1D ones

        # directional_merged_decoders_result_dict: Dict[types.DecoderName, DirectionalPseudo2DDecodersResult] = {}

        decoder_laps_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        decoder_ripple_filter_epochs_decoder_result_dict: DecodedEpochsResultsDict = {}
        
        for a_name, a_decoder in track_templates.get_decoders_dict().items():
            # external-function way:
            decoder_laps_filter_epochs_decoder_result_dict[a_name], decoder_ripple_filter_epochs_decoder_result_dict[a_name] = _compute_lap_and_ripple_epochs_decoding_for_decoder(a_decoder, curr_active_pipeline, desired_laps_decoding_time_bin_size=laps_decoding_time_bin_size, desired_ripple_decoding_time_bin_size=ripple_decoding_time_bin_size, epochs_filtering_mode=EpochFilteringMode.DropShorter)

        return directional_merged_decoders_result, (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict)
        
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

    # active_context = curr_active_pipeline.get_session_context()
    if additional_session_context is not None:
        if isinstance(additional_session_context, dict):
            additional_session_context = IdentifyingContext(**additional_session_context)
        active_context = (curr_active_pipeline.get_session_context() | additional_session_context)
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=(IdentifyingContext._get_session_context_keys() + list(additional_session_context.keys())))
        CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{additional_session_context.get_description()}"
    else:
        active_context = curr_active_pipeline.get_session_context()
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
        CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"

    print(f'\tactive_context: {active_context}')    
    print(f'\tsession_ctxt_key: {session_ctxt_key}')
    print(f'\tCURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')    
    
    ## INPUT PARAMETER: time_bin_size sweep paraemters    
    if custom_all_param_sweep_options is None:
        if desired_shared_decoding_time_bin_sizes is None:
            # desired_shared_decoding_time_bin_sizes = np.linspace(start=0.030, stop=0.10, num=6) ####### <<<------ Default sweep is defined here
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058, 0.072, 0.086, 0.100, 0.250, 1.5]) ####### <<<------ Default sweep is defined here
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058, 0.072, 0.086, 0.100]) ####### <<<------ Default sweep is defined here
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058, 0.072, 0.086, 0.100])
            desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058,])

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
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # : "DirectionalLapsResult"
    track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference? : "TrackTemplates"

    ## Copy the default result:
    directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)

    # out_path_basename_str: str = f"{now_day_str}_{active_context}_time_bin_size-{laps_decoding_time_bin_size}_{data_identifier_str}"
    # out_path_basename_str: str = f"{now_day_str}_{active_context}_time_bin_size_sweep_results"
    out_path_basename_str: str = f"{CURR_BATCH_OUTPUT_PREFIX}_time_bin_size_sweep_results"
    # out_path_filenname_str: str = f"{out_path_basename_str}.csv"

    out_path_filenname_str: str = f"{out_path_basename_str}.h5"
    out_path: Path = self.collected_outputs_path.resolve().joinpath(out_path_filenname_str).resolve()
    print(f'\tout_path_str: "{out_path_filenname_str}"')
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

    # BEGIN BLOCK 2 - modernizing from `_perform_compute_custom_epoch_decoding`  ________________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _compute_lap_and_ripple_epochs_decoding_for_decoder, _perform_compute_custom_epoch_decoding, _compute_all_df_score_metrics
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

    # Uses: session_ctxt_key, all_param_sweep_options
    # output_alt_directional_merged_decoders_result: Dict[Tuple, Dict[types.DecoderName, DirectionalPseudo2DDecodersResult]] = {} # empty dict

    output_alt_directional_merged_decoders_result: Dict[Tuple, DirectionalPseudo2DDecodersResult] = {} # empty dict
    # output_alt_directional_merged_decoders_result: Dict[Tuple, Dict[types.DecoderName, DirectionalPseudo2DDecodersResult]] = {} # empty dict

    # Tuple[DirectionalPseudo2DDecodersResult, Tuple[DecodedEpochsResultsDict, DecodedEpochsResultsDict]]
    output_directional_decoders_epochs_decode_results_dict: Dict[Tuple, DecoderDecodedEpochsResult] = {} # `_decode_and_evaluate_epochs_using_directional_decoders`-style output

    output_laps_decoding_accuracy_results_dict = {} # empty dict
    output_extracted_result_tuples = {}


    for a_sweep_dict in all_param_sweep_options:
        ## Looks like each iteration of the loop serves to update: `output_alt_directional_merged_decoders_result`, `output_extracted_result_tuples`, `output_directional_decoders_epochs_decode_results_dict`, 
        a_sweep_tuple = frozenset(a_sweep_dict.items())
        print(f'\ta_sweep_dict: {a_sweep_dict}')

        output_alt_directional_merged_decoders_result[a_sweep_tuple], (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict) = _try_all_templates_decode(curr_active_pipeline, alt_directional_merged_decoders_result, **a_sweep_dict)
        an_alt_dir_Pseudo2D_decoders_result = output_alt_directional_merged_decoders_result[a_sweep_tuple]

        ## Decode epochs for all four decoders:
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

            
            #TODO 2024-07-05 20:53: - [ ] Note that these new columns aren't added back to the source `an_alt_dir_Pseudo2D_decoders_result`
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

        ## Ripple .h5 export:
        ripple_decoding_time_bin_size: float = an_alt_dir_Pseudo2D_decoders_result.ripple_decoding_time_bin_size
        actual_ripple_decoding_time_bin_size_str: str = str(ripple_decoding_time_bin_size)
        if save_hdf and (actual_ripple_decoding_time_bin_size_str is not None):
            ripple_time_bin_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_ripple_decoding_time_bin_size_str}/ripple_time_bin_marginals_df', format='table', data_columns=True)
            ripple_all_epoch_bins_marginals_df.to_hdf(out_path, key=f'{session_ctxt_key}/{actual_ripple_decoding_time_bin_size_str}/ripple_all_epoch_bins_marginals_df', format='table', data_columns=True)

        
        # get the current lap object and determine the percentage correct:
        if should_output_lap_decoding_performance_info:
            result_laps_epochs_df: pd.DataFrame = _update_result_laps(a_result=an_alt_dir_Pseudo2D_decoders_result, laps_df=laps_df)
            (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = _check_result_laps_epochs_df_performance(result_laps_epochs_df)
            output_laps_decoding_accuracy_results_dict[laps_decoding_time_bin_size] = (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly)



        # `_decode_and_evaluate_epochs_using_directional_decoders` post compute ______________________________________________ #

        ## Recompute the epoch scores/metrics such as radon transform and wcorr:
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(an_alt_dir_Pseudo2D_decoders_result, track_templates,
                                                                                                                                                                                            decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                            spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df),
                                                                                                                                                                                            should_skip_radon_transform=True, suppress_exceptions=suppress_exceptions)
        
        laps_radon_transform_merged_df, ripple_radon_transform_merged_df, laps_weighted_corr_merged_df, ripple_weighted_corr_merged_df, laps_simple_pf_pearson_merged_df, ripple_simple_pf_pearson_merged_df = merged_df_outputs_tuple ## Here is where `ripple_weighted_corr_merged_df` is being returned (badly)
        decoder_laps_radon_transform_df_dict, decoder_ripple_radon_transform_df_dict, decoder_laps_radon_transform_extras_dict, decoder_ripple_radon_transform_extras_dict, decoder_laps_weighted_corr_df_dict, decoder_ripple_weighted_corr_df_dict = raw_dict_outputs_tuple

        ripple_decoding_time_bin_size = an_alt_dir_Pseudo2D_decoders_result.ripple_decoding_time_bin_size
        laps_decoding_time_bin_size = an_alt_dir_Pseudo2D_decoders_result.laps_decoding_time_bin_size
        pos_bin_size = an_alt_dir_Pseudo2D_decoders_result.all_directional_pf1D_Decoder.pos_bin_size

        ## This is where the result is being built, so this must be where the wrong merged _df is being made!
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
    combined_multi_timebin_outputs_tuple = _subfn_process_time_bin_swept_results(output_extracted_result_tuples, active_context=active_context)
    # Unpacking:    
    # (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path) = combined_multi_timebin_outputs_tuple
    
    # combined_multi_timebin_outputs_tuple
    

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
    _temp_saved_files_dict = {'laps_out_path': laps_out_path, 'laps_time_bin_marginals_out_path': laps_time_bin_marginals_out_path, 'ripple_out_path': ripple_out_path, 'ripple_time_bin_marginals_out_path': ripple_time_bin_marginals_out_path}
    print(f'>>>>>>>>>> exported files: {_temp_saved_files_dict}\n\n')
    
    # 2024-07-12 - Export computed CSVs in here?!?! ______________________________________________________________________ #
    ## INPUTS: active_context (from before), should be correct
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

    for a_sweep_tuple, a_directional_decoders_epochs_decode_result in output_directional_decoders_epochs_decode_results_dict.items():
        # active_context = curr_active_pipeline.get_session_context()
        # a_sweep_tuple
        ## add the additional contexts:
        # a_sweep_active_context = deepcopy(active_context).adding_context_if_missing(custom_replay_name='TESTNEW', time_bin_size=directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size)
        # additional_session_context = None
        # try:
        # 	if custom_suffix is not None:
        # 		additional_session_context = IdentifyingContext(custom_suffix=custom_suffix)
        # 		print(f'Using custom suffix: "{custom_suffix}" - additional_session_context: "{additional_session_context}"')
        # except NameError as err:
        # 	additional_session_context = None
        # 	print(f'NO CUSTOM SUFFIX.')    

        decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates)
        print(f'\tComputation complete. Exporting .CSVs...')

        # 2024-03-04 - Filter out the epochs based on the criteria: -- #TODO 2024-07-12 08:30: - [ ] This is nearly certainly going to ruin it
        _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
        filtered_valid_epoch_times = filtered_epochs_df[['start', 'stop']].to_numpy()

        ## Export CSVs:
        _output_csv_paths = a_directional_decoders_epochs_decode_result.export_csvs(parent_output_path=self.collected_outputs_path.resolve(), active_context=active_context, session_name=curr_session_name, curr_session_t_delta=t_delta,
                                                                                        user_annotation_selections={'ripple': any_good_selected_epoch_times},
                                                                                        valid_epochs_selections={'ripple': filtered_valid_epoch_times},
                                                                                    )
        print(f'\t>>>>>>>>>> exported files: {_output_csv_paths}\n\n')



    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['CSVs', 'export', 'across-sessions', 'batch', 'single-time-bin-size', 'ripple_all_scores_merged_df'], input_requires=['DirectionalDecodersEpochsEvaluations'], output_provides=[], uses=['filter_and_update_epochs_and_spikes', 'DecoderDecodedEpochsResult', 'DecoderDecodedEpochsResult.export_csvs'], used_by=[], creation_date='2024-04-27 21:20', related_items=[])
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


@function_attributes(short_name=None, tags=['UNFINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
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
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    # active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]


    def _update_loaded_track_limits(a_session):
        """ captures: curr_active_pipeline
        """
        sess_config: SessionConfig = SessionConfig(**deepcopy(a_session.config.__getstate__()))
        a_session.config = sess_config
        _bak_loaded_track_limits = deepcopy(a_session.config.loaded_track_limits)
        ## Apply fn
        a_session = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=a_session)
        _new_loaded_track_limits = deepcopy(a_session.config.loaded_track_limits)
        # did_change: bool = ((_bak_loaded_track_limits is None) or (_new_loaded_track_limits != _bak_loaded_track_limits))
        did_change: bool = ((_bak_loaded_track_limits is None) or np.any((np.array(_new_loaded_track_limits) != np.array(_bak_loaded_track_limits))))
        return did_change, a_session

    ## Check if they changed
    did_change: bool = False

    ## Do main session:
    a_session = deepcopy(curr_active_pipeline.sess)
    new_did_change, a_session = _update_loaded_track_limits(a_session=a_session)
    curr_active_pipeline.stage.sess = a_session ## apply the session
    did_change = did_change | new_did_change
    print(f'curr_active_pipeline.sess changed its track limits!')
    # curr_active_pipeline.sess.config = a_session.config # apply the config only...

    # --------------------- Do for filtered sessions as well --------------------- #

    # did_change = did_change or np.any(ensure_dataframe(_backup_session_configs['sess']).to_numpy() != ensure_dataframe(new_replay_epochs).to_numpy())

    _new_sessions = {}

    # curr_active_pipeline.sess.replay = deepcopy(new_replay_epochs)
    for k, a_filtered_session in curr_active_pipeline.filtered_sessions.items():
        ## backup original values:
        # _backup_session_replay_epochs[k] = deepcopy(a_filtered_session.config.preprocessing_parameters.epoch_estimation_parameters.replays)
        # _backup_session_configs[k] = deepcopy(a_filtered_session.replay)


        a_filtered_session = deepcopy(a_filtered_session)
        # sess_config: SessionConfig = SessionConfig(**deepcopy(a_filtered_session.config.__getstate__()))
        # a_filtered_session.config = sess_config
        # _new_sessions[k] = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=a_filtered_session)
        new_did_change, a_filtered_session = _update_loaded_track_limits(a_session=a_filtered_session)
        if new_did_change:
            print(f'\tfiltered_session[{k}] changed!')
        did_change = did_change | new_did_change
        

    for k, a_filtered_session in _new_sessions.items():
        curr_active_pipeline.filtered_sessions[k] = a_filtered_session



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


@function_attributes(short_name=None, tags=['hdf5', 'h5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
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


@function_attributes(short_name=None, tags=['backup', 'versioning', 'copy'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-25 06:22', related_items=[])
def backup_previous_session_files_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """  Makes a backup copy of the pipeline's files (pkl, HDF5) with desired suffix
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import backup_previous_session_files_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    import sys
    from datetime import timedelta, datetime
    from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    import shutil

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'backup_previous_session_files_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    desired_suffix: str = 'Pre2024-07-16'
    was_write_good: bool = False

    _out = {'pipeline_pkl': curr_active_pipeline.pickle_path,
            'global_computation_pkl': curr_active_pipeline.global_computation_results_pickle_path,
            'pipeline_h5': curr_active_pipeline.h5_export_path,
    }
    _existing_session_files_dict = {a_name:a_path for a_name, a_path in _out.items() if a_path.exists()}

    _successful_copies_dict = {}
    # for a_name, a_path in _out.items():
    for a_name, a_path in _existing_session_files_dict.items():
        if a_path.exists():
            # file really exists, do something with it
            desired_path: Path = a_path.with_stem(f"{a_path.stem}_{desired_suffix}").resolve() # # WindowsPath('W:/Data/KDIBA/gor01/one/2006-6-08_14-26-15/output/pipeline_results_Pre2024-07-09.h5')
            ## hopefully it doesn't already exist!
            desired_path.exists()

            ## copy
            try:
                print(f"'{a_path}' backing up -> to desired_path: '{desired_path}'")
                shutil.copy(a_path, desired_path)
                print('done.')
                was_write_good = True
                _successful_copies_dict[a_name] = desired_path

            except BaseException as e:
                exception_info = sys.exc_info()
                err = CapturedException(e, exception_info)
                print(f"ERROR: encountered exception {err} while trying to backup the {a_name} file at {a_path} for {curr_session_context}")
                if self.fail_on_exception:
                    raise err.exc
            
    callback_outputs = {
     'desired_suffix': desired_suffix,
     'session_files_dict': _existing_session_files_dict, 'successfully_copied_files_dict':_successful_copies_dict,
    }
    across_session_results_extended_dict['backup_previous_session_files_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict




@function_attributes(short_name=None, tags=['wcorr', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def compute_and_export_session_wcorr_shuffles_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """  Computes the shuffled wcorrs and export them to several formats: .mat, .pkl, and .csv
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import reload_exported_kdiba_session_position_info_mat_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'compute_and_export_session_wcorr_shuffles_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('compute_and_export_session_wcorr_shuffles_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    import sys
    import numpy as np
    from datetime import timedelta, datetime
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import SequenceBasedComputationsContainer, WCorrShuffle
    from neuropy.utils.mixins.indexing_helpers import get_dict_subset


    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_wcorr_shuffles_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    # desired_total_num_shuffles: int = 4096
    desired_total_num_shuffles: int = 700
    minimum_required_unique_num_shuffles: int = 700

    allow_update_global_result: bool = False
    callback_outputs = {
        'wcorr_shuffles_data_output_filepath': None, #'t_end': t_end   
        'standalone_MAT_filepath': None,
        'ripple_WCorrShuffle_df_export_CSV_path': None,
    }


    if ('SequenceBased' not in curr_active_pipeline.global_computation_results.computed_data) or (not hasattr(curr_active_pipeline.global_computation_results.computed_data, 'SequenceBased')):
            # initialize
            a_sequence_computation_container: SequenceBasedComputationsContainer = SequenceBasedComputationsContainer(wcorr_ripple_shuffle=None, is_global=True)
    else:
        a_sequence_computation_container: SequenceBasedComputationsContainer = deepcopy(curr_active_pipeline.global_computation_results.computed_data['SequenceBased'])


    # global_computation_results.computed_data['SequenceBased'].included_qclu_values = included_qclu_values
    if (not hasattr(a_sequence_computation_container, 'wcorr_ripple_shuffle') or (a_sequence_computation_container.wcorr_ripple_shuffle is None)):
        # initialize a new wcorr result            
        wcorr_shuffles: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline, enable_saving_entire_decoded_shuffle_result=False)
        a_sequence_computation_container.wcorr_ripple_shuffle = wcorr_shuffles
    else:
        ## get the existing one:
        wcorr_shuffles = a_sequence_computation_container.wcorr_ripple_shuffle
        wcorr_shuffles: WCorrShuffle = WCorrShuffle(**get_dict_subset(wcorr_shuffles.to_dict(), subset_excludelist=['_VersionedResultMixin_version'])) # modernize the object


    wcorr_shuffles.compute_shuffles(num_shuffles=2, curr_active_pipeline=curr_active_pipeline) # do one more shuffle
    
    # try load previous compatible shuffles: _____________________________________________________________________________ #
    wcorr_shuffles.discover_load_and_append_shuffle_data_from_directory(save_directory=curr_active_pipeline.get_output_path().resolve())

    n_completed_shuffles: int = wcorr_shuffles.n_completed_shuffles

    if (minimum_required_unique_num_shuffles is not None) and (n_completed_shuffles < minimum_required_unique_num_shuffles):
        ## skipping
        print(f'\tskipping session {curr_active_pipeline.session_name} because n_completed_shuffles: {n_completed_shuffles} < minimum_required_unique_num_shuffles: {minimum_required_unique_num_shuffles}')

        across_session_results_extended_dict['compute_and_export_session_wcorr_shuffles_completion_function'] = callback_outputs
        ## EXITS HERE:
        return across_session_results_extended_dict

    if n_completed_shuffles < desired_total_num_shuffles:   
        print(f'n_prev_completed_shuffles: {n_completed_shuffles}.')
        print(f'needed desired_total_num_shuffles: {desired_total_num_shuffles}.')
        desired_new_num_shuffles: int = max((desired_total_num_shuffles - wcorr_shuffles.n_completed_shuffles), 0)
        print(f'need desired_new_num_shuffles: {desired_new_num_shuffles} more shuffles.')
        ## add some more shuffles to it:
        wcorr_shuffles.compute_shuffles(num_shuffles=desired_new_num_shuffles, curr_active_pipeline=curr_active_pipeline)


    # (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict) = wcorr_tool.post_compute(debug_print=False)
    # wcorr_tool.save_data(filepath='temp100.pkl')

    a_sequence_computation_container.wcorr_ripple_shuffle = wcorr_shuffles

    n_completed_shuffles: int = wcorr_shuffles.n_completed_shuffles
    print(f'loaded and computed shuffles with {n_completed_shuffles} unique shuffles')
    
    if allow_update_global_result:
        print(f'updating global result because allow_update_global_result is True ')
        curr_active_pipeline.global_computation_results.computed_data['SequenceBased'] = a_sequence_computation_container
        # need to mark it as dirty?

    ## standalone saving:
    # datetime.today().strftime('%Y-%m-%d')

    err = None

    try:
        active_context = curr_active_pipeline.get_session_context()
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
        session_name: str = curr_active_pipeline.session_name
        export_files_dict = wcorr_shuffles.export_csvs(parent_output_path=self.collected_outputs_path.resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=curr_active_pipeline)
        ripple_WCorrShuffle_df_export_CSV_path = export_files_dict['ripple_WCorrShuffle_df']
        print(f'Successfully exported ripple_WCorrShuffle_df_export_CSV_path: "{ripple_WCorrShuffle_df_export_CSV_path}" with wcorr_shuffles.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles} unique shuffles.')
        callback_outputs['ripple_WCorrShuffle_df_export_CSV_path'] = ripple_WCorrShuffle_df_export_CSV_path
    except BaseException as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"ERROR: encountered exception {err} while trying to perform wcorr_ripple_shuffle.export_csvs(parent_output_path='{self.collected_outputs_path.resolve()}', ...) for {curr_session_context}")
        ripple_WCorrShuffle_df_export_CSV_path = None # set to None because it failed.
        if self.fail_on_exception:
            raise err.exc
        

    ## Pickle Saving:
    standalone_filename: str = f'{get_now_day_str()}_standalone_wcorr_ripple_shuffle_data_only_{a_sequence_computation_container.wcorr_ripple_shuffle.n_completed_shuffles}.pkl'
    wcorr_shuffles_data_standalone_filepath = curr_active_pipeline.get_output_path().joinpath(standalone_filename).resolve()
    print(f'wcorr_shuffles_data_standalone_filepath: "{wcorr_shuffles_data_standalone_filepath}"')

    try:
        wcorr_shuffles.save_data(wcorr_shuffles_data_standalone_filepath)
        was_write_good = True
        callback_outputs['wcorr_shuffles_data_output_filepath'] = wcorr_shuffles_data_standalone_filepath

    except BaseException as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"ERROR: encountered exception {err} while trying to perform wcorr_shuffles.save_data('{wcorr_shuffles_data_standalone_filepath}') for {curr_session_context}")
        wcorr_shuffles_data_standalone_filepath = None # set to None because it failed.
        if self.fail_on_exception:
            raise err.exc

    # (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict), _out_shuffle_wcorr_arr = wcorr_shuffles.post_compute()

    ## MATLAB .mat format output
    standalone_mat_filename: str = f'{get_now_rounded_time_str()}_standalone_all_shuffles_wcorr_array.mat' 
    standalone_MAT_filepath = curr_active_pipeline.get_output_path().joinpath(standalone_mat_filename).resolve() # Path("W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\2024-05-30_0925AM_standalone_wcorr_ripple_shuffle_data_only_1100.pkl")
    print(f'\tsaving .mat format to "{standalone_MAT_filepath}"...')
    
    try:
        wcorr_shuffles.save_data_mat(filepath=standalone_MAT_filepath, additional_mat_elements={'session': curr_active_pipeline.get_session_context().to_dict()})
        callback_outputs['standalone_MAT_filepath'] = standalone_MAT_filepath
    except BaseException as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"ERROR: encountered exception {err} while trying to perform savemat('{standalone_MAT_filepath}') for {curr_session_context}")
        standalone_MAT_filepath = None # set to None because it failed.
        if self.fail_on_exception:
            raise err.exc
        
    # callback_outputs = {
    #  'wcorr_shuffles_data_output_filepath': wcorr_shuffles_data_standalone_filepath, 'e':err, #'t_end': t_end   
    #  'standalone_MAT_filepath': standalone_MAT_filepath,
    #  'ripple_WCorrShuffle_df_export_CSV_path': ripple_WCorrShuffle_df_export_CSV_path,
    # }
    across_session_results_extended_dict['compute_and_export_session_wcorr_shuffles_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict








@function_attributes(short_name=None, tags=['wcorr', 'shuffle', 'replay', 'epochs', 'alternative_replays'], input_requires=[], output_provides=[], uses=['compute_all_replay_epoch_variations'], used_by=[], creation_date='2024-06-28 01:50', related_items=[])
def compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """  Computes several different alternative replay-detection variants and computes and exports the shuffled wcorrs for each of them
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    import sys
    import numpy as np
    from datetime import timedelta, datetime
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import SequenceBasedComputationsContainer, WCorrShuffle
    from neuropy.utils.mixins.indexing_helpers import get_dict_subset
    from neuropy.core.epoch import Epoch, ensure_Epoch, ensure_dataframe
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
    from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_diba_quiescent_style_replay_events, overwrite_replay_epochs_and_recompute, try_load_neuroscope_EVT_file_epochs, replace_replay_epochs
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_replay_wcorr_histogram
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import finalize_output_shuffled_wcorr, _get_custom_suffix_for_replay_filename
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_replay_epoch_variations

    # SimpleBatchComputationDummy = make_class('SimpleBatchComputationDummy', attrs=['BATCH_DATE_TO_USE', 'collected_outputs_path'])
    # a_dummy = SimpleBatchComputationDummy(BATCH_DATE_TO_USE, collected_outputs_path)
    
    # "self" already is a dummy
    
    base_BATCH_DATE_TO_USE: str = f"{self.BATCH_DATE_TO_USE}" ## backup original string
    should_suppress_errors: bool = (not self.fail_on_exception) # get('fail_on_exception', False)    

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    callback_outputs = {
        'replay_epoch_variations': None,
        'replay_epoch_outputs': None,
        # 'wcorr_shuffles_data_output_filepath': None, #'t_end': t_end   
        # 'standalone_MAT_filepath': None,
        # 'ripple_WCorrShuffle_df_export_CSV_path': None,
    }


    # ==================================================================================================================== #
    # Compute Alternative Replays: `replay_epoch_variations`                                                               #
    # ==================================================================================================================== #

    replay_epoch_outputs = {} # replay_epochs_key

    ## Compute new epochs: 
    replay_epoch_variations = {}

    replay_epoch_variations = compute_all_replay_epoch_variations(curr_active_pipeline)

    print(f'completed replay extraction, have: {list(replay_epoch_variations.keys())}')
    
    ## OUTPUT: replay_epoch_variations
    callback_outputs['replay_epoch_variations'] = replay_epoch_variations

    ## Save the replay epochs out to event files:
    for replay_epochs_key, a_replay_epochs in replay_epoch_variations.items():
        # ## Use `diba_evt_file_replay_epochs` as `new_replay_epochs`
        # replay_epochs_key = 'diba_quiescent_method_replay_epochs'
        # a_replay_epochs = replay_epoch_variations[replay_epochs_key]
        print(f'performing comp for "{replay_epochs_key}"...')
        replay_epoch_outputs[replay_epochs_key] = {} # init to empty

        custom_suffix: str = _get_custom_suffix_for_replay_filename(new_replay_epochs=a_replay_epochs)
        print(f'\treplay_epochs_key: {replay_epochs_key}: custom_suffix: "{custom_suffix}"')

        ## Export to .evt file

        ## Save computed epochs out to a neuroscope .evt file:
        filename = f"{curr_active_pipeline.session_name}{custom_suffix}"
        good_filename: str = sanitize_filename_for_Windows(filename)
        print(f'\tgood_filename: {good_filename}')
        filepath = curr_active_pipeline.get_output_path().joinpath(good_filename).resolve()
        
        curr_replay_epoch = deepcopy(a_replay_epochs)

        ## set the filename of the Epoch:
        curr_replay_epoch.filename = filepath
        filepath = curr_replay_epoch.to_neuroscope(ext='PHONEW')
        assert filepath.exists()
        print(F'saved out newly computed epochs of type "{replay_epochs_key} to "{filepath}".')
        replay_epoch_outputs[replay_epochs_key].update(dict(exported_evt_file_path=str(filepath.as_posix())))

    # ==================================================================================================================== #
    # For each replay, duplicate entire pipeline to perform desired computations to ensure no downstream effects           #
    # ==================================================================================================================== #
    print(f'=====================================>> Starting computations for the 4 new epochs...')

    ## Duplicate Copy of pipeline to perform desired computations:
    for replay_epochs_key, a_replay_epochs in replay_epoch_variations.items():
        # ## Use `diba_evt_file_replay_epochs` as `new_replay_epochs`
        # replay_epochs_key = 'diba_quiescent_method_replay_epochs'
        # a_replay_epochs = replay_epoch_variations[replay_epochs_key]
        print(f'\t=====================================>> performing comp for "{replay_epochs_key}"...')
        # replay_epoch_outputs[replay_epochs_key] = {} # init to empty

        custom_suffix: str = _get_custom_suffix_for_replay_filename(new_replay_epochs=a_replay_epochs)
        print(f'\treplay_epochs_key: {replay_epochs_key}: custom_suffix: "{custom_suffix}"')

        ## Modify .BATCH_DATE_TO_USE to include the custom suffix
        curr_BATCH_DATE_TO_USE: str = f"{base_BATCH_DATE_TO_USE}{custom_suffix}"
        print(f'\tcurr_BATCH_DATE_TO_USE: "{curr_BATCH_DATE_TO_USE}"')
        self.BATCH_DATE_TO_USE = curr_BATCH_DATE_TO_USE # set the internal BATCH_DATE_TO_USE which is used to determine the .csv and .h5 export names

        print(f'\tWARNING: should_suppress_errors: {should_suppress_errors}')
        with ExceptionPrintingContext(suppress=should_suppress_errors, exception_print_fn=(lambda formatted_exception_str: print(f'\tfailed epoch computations for replay_epochs_key: "{replay_epochs_key}". Failed with error: {formatted_exception_str}. Skipping.'))):
            # for replay_epochs_key, a_replay_epochs in replay_epoch_variations.items():
            a_curr_active_pipeline = deepcopy(curr_active_pipeline)
            did_change, custom_save_filenames, custom_save_filepaths = overwrite_replay_epochs_and_recompute(curr_active_pipeline=a_curr_active_pipeline, new_replay_epochs=a_replay_epochs,
                                                                                                              enable_save_pipeline_pkl=True, enable_save_global_computations_pkl=False, enable_save_h5=False,
                                                                                                              user_completion_dummy=self)

            replay_epoch_outputs[replay_epochs_key].update(dict(did_change=did_change, custom_save_filenames=custom_save_filenames, custom_save_filepaths=custom_save_filepaths))

            print(f'<<<<<<<<<<<<<<< Done with `overwrite_replay_epochs_and_recompute(...)` for replay_epochs_key: {replay_epochs_key}')
            ## modifies `_temp_curr_active_pipeline`

            # ==================================================================================================================== #
            # OUTPUT OF WCORR                                                                                                      #
            # ==================================================================================================================== #

            ## Call on `a_replay_epochs`, _temp_curr_active_pipeline:
            replay_epoch_outputs[replay_epochs_key].update(dict(custom_suffix=custom_suffix))

            ## INPUTS: a_curr_active_pipeline, custom_suffix
            decoder_names = TrackTemplates.get_decoder_names()

            wcorr_shuffle_results: SequenceBasedComputationsContainer = a_curr_active_pipeline.global_computation_results.computed_data.get('SequenceBased', None)
            if wcorr_shuffle_results is not None:    
                wcorr_shuffles: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
                wcorr_shuffles: WCorrShuffle = WCorrShuffle(**get_dict_subset(wcorr_shuffles.to_dict(), subset_excludelist=['_VersionedResultMixin_version']))
                a_curr_active_pipeline.global_computation_results.computed_data.SequenceBased.wcorr_ripple_shuffle = wcorr_shuffles
                filtered_epochs_df: pd.DataFrame = deepcopy(wcorr_shuffles.filtered_epochs_df)
                print(f'\t\twcorr_ripple_shuffle.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles}')
            else:
                print(f'\t\tSequenceBased is not computed.')
                wcorr_shuffles = None
                raise ValueError(f'SequenceBased is not computed.')

            # wcorr_ripple_shuffle: WCorrShuffle = WCorrShuffle.init_from_templates(curr_active_pipeline=curr_active_pipeline, enable_saving_entire_decoded_shuffle_result=True)

            n_epochs: int = wcorr_shuffles.n_epochs
            print(f'n_epochs: {n_epochs}')
            n_completed_shuffles: int = wcorr_shuffles.n_completed_shuffles
            print(f'n_completed_shuffles: {n_completed_shuffles}')
            wcorr_shuffles.compute_shuffles(num_shuffles=2, curr_active_pipeline=a_curr_active_pipeline)
            n_completed_shuffles: int = wcorr_shuffles.n_completed_shuffles
            print(f'n_completed_shuffles: {n_completed_shuffles}')
            desired_ripple_decoding_time_bin_size: float = wcorr_shuffle_results.wcorr_ripple_shuffle.all_templates_decode_kwargs['desired_ripple_decoding_time_bin_size']
            print(f'{desired_ripple_decoding_time_bin_size = }')
            # filtered_epochs_df

            # 7m - 200 shuffles
            # (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict), _out_shuffle_wcorr_arr = wcorr_shuffles.post_compute(decoder_names=deepcopy(decoder_names))
            wcorr_ripple_shuffle_all_df, all_shuffles_wcorr_df = wcorr_shuffles.build_all_shuffles_dataframes(decoder_names=deepcopy(decoder_names))
            ## Prepare for plotting in histogram:
            wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.dropna(subset=['start', 'stop'], how='any', inplace=False)
            wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.dropna(subset=['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL'], how='all', inplace=False)
            wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.convert_dtypes()
            # {'long_best_dir_decoder_IDX': int, 'short_best_dir_decoder_IDX': int}
            # wcorr_ripple_shuffle_all_df
            ## Gets the absolutely most extreme value from any of the four decoders and uses that:
            # Replace pandas.NA with np.nan before doing Nanargmax
            _wcorr_results_arr = wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].to_numpy(dtype=float, na_value=np.nan) 
            best_wcorr_max_indices = np.nanargmax(np.abs(_wcorr_results_arr), axis=1) # Compute argmax ignoring NaNs

            wcorr_ripple_shuffle_all_df[f'abs_best_wcorr'] = [wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].values[i, best_idx] for i, best_idx in enumerate(best_wcorr_max_indices)] #  np.where(direction_max_indices, wcorr_ripple_shuffle_all_df['long_LR'].filter_epochs[a_column_name].to_numpy(), wcorr_ripple_shuffle_all_df['long_RL'].filter_epochs[a_column_name].to_numpy())
            # wcorr_ripple_shuffle_all_df

            all_shuffles_only_best_decoder_wcorr_df = pd.concat([all_shuffles_wcorr_df[np.logical_and((all_shuffles_wcorr_df['epoch_idx'] == epoch_idx), (all_shuffles_wcorr_df['decoder_idx'] == best_idx))] for epoch_idx, best_idx in enumerate(best_wcorr_max_indices)])
            
            ## OUTPUTS: wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df
            replay_epoch_outputs[replay_epochs_key].update(dict(wcorr_ripple_shuffle_all_df=wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df=all_shuffles_only_best_decoder_wcorr_df))


            ## INPUTS: wcorr_ripple_shuffle, a_curr_active_pipeline, wcorr_shuffles, custom_suffix

            # standalone save
            standalone_pkl_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_wcorr_ripple_shuffle_data_only_{wcorr_shuffles.n_completed_shuffles}.pkl' 
            standalone_pkl_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_pkl_filename).resolve() # Path("W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\2024-05-30_0925AM_standalone_wcorr_ripple_shuffle_data_only_1100.pkl")
            print(f'saving to "{standalone_pkl_filepath}"...')
            wcorr_shuffles.save_data(standalone_pkl_filepath)
            ## INPUTS: wcorr_ripple_shuffle
            standalone_mat_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_all_shuffles_wcorr_array.mat' 
            standalone_mat_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_mat_filename).resolve() # r"W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43\output\2024-06-03_0400PM_standalone_all_shuffles_wcorr_array.mat"
            wcorr_shuffles.save_data_mat(filepath=standalone_mat_filepath, **{'session': a_curr_active_pipeline.get_session_context().to_dict()})

            replay_epoch_outputs[replay_epochs_key].update(dict(standalone_pkl_filepath=standalone_pkl_filepath, standalone_mat_filepath=standalone_mat_filepath))

            try:
                active_context = a_curr_active_pipeline.get_session_context()
                session_name: str = f"{a_curr_active_pipeline.session_name}{custom_suffix}" ## appending this here is a hack, but it makes the correct filename
                active_context = active_context.adding_context_if_missing(suffix=custom_suffix)

                export_files_dict = wcorr_shuffles.export_csvs(parent_output_path=a_curr_active_pipeline.get_output_path().resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=a_curr_active_pipeline,
                                                            #    source='diba_evt_file',
                                                                source='compute_diba_quiescent_style_replay_events',
                                                                )
                ripple_WCorrShuffle_df_export_CSV_path = export_files_dict['ripple_WCorrShuffle_df']
                print(f'Successfully exported ripple_WCorrShuffle_df_export_CSV_path: "{ripple_WCorrShuffle_df_export_CSV_path}" with wcorr_shuffles.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles} unique shuffles.')
                replay_epoch_outputs[replay_epochs_key].update(dict(active_context=active_context, export_files_dict=export_files_dict))
                # callback_outputs['ripple_WCorrShuffle_df_export_CSV_path'] = ripple_WCorrShuffle_df_export_CSV_path
            except BaseException as e:
                raise e
                # exception_info = sys.exc_info()
                # err = CapturedException(e, exception_info)
                # print(f"ERROR: encountered exception {err} while trying to perform wcorr_ripple_shuffle.export_csvs(parent_output_path='{self.collected_outputs_path.resolve()}', ...) for {curr_session_context}")
                # ripple_WCorrShuffle_df_export_CSV_path = None # set to None because it failed.
                # if self.fail_on_exception:
                #     raise err.exc
                
            # wcorr_ripple_shuffle.discover_load_and_append_shuffle_data_from_directory(save_directory=curr_active_pipeline.get_output_path().resolve())
            # active_context = curr_active_pipeline.get_session_context()
            # session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
            # session_name: str = curr_active_pipeline.session_name
            # export_files_dict = wcorr_ripple_shuffle.export_csvs(parent_output_path=collected_outputs_path.resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=curr_active_pipeline)
            # export_files_dict

            ## FINAL STAGE: generate histogram:
            
            ## INPUTS: wcorr_ripple_shuffle_all_df, wcorr_ripple_shuffle_all_df, custom_suffix
            plot_var_name: str = 'abs_best_wcorr'
            a_fig_context = a_curr_active_pipeline.build_display_context_for_session(display_fn_name='replay_wcorr', custom_suffix=custom_suffix)
            params_description_str: str = " | ".join([f"{str(k)}:{str(v)}" for k, v in get_dict_subset(a_replay_epochs.metadata, subset_excludelist=['qclu_included_aclus']).items()])
            footer_annotation_text = f'{a_curr_active_pipeline.get_session_context()}<br>{params_description_str}'

            fig = plot_replay_wcorr_histogram(df=wcorr_ripple_shuffle_all_df, plot_var_name=plot_var_name,
                    all_shuffles_only_best_decoder_wcorr_df=all_shuffles_only_best_decoder_wcorr_df, footer_annotation_text=footer_annotation_text)

            # Save figure to disk:
            out_hist_fig_result = a_curr_active_pipeline.output_figure(a_fig_context, fig=fig)

            replay_epoch_outputs[replay_epochs_key].update(dict(params_description_str=params_description_str, footer_annotation_text=footer_annotation_text, out_hist_fig_result=out_hist_fig_result))

            # Show the figure
            # fig.show()
        ## end error handler

    # END FOR
    ## restore original base_BATCH_DATE_TO_USE
    self.BATCH_DATE_TO_USE = base_BATCH_DATE_TO_USE

    callback_outputs = {
        'custom_suffix': custom_suffix,
        'replay_epoch_variations': deepcopy(replay_epoch_variations),
        'replay_epoch_outputs': deepcopy(replay_epoch_outputs),
        #  'wcorr_shuffles_data_output_filepath': wcorr_shuffles_data_standalone_filepath, 'e':err, #'t_end': t_end   
        #  'standalone_pkl_filepath': standalone_pkl_filepath,
        #  'standalone_MAT_filepath': standalone_MAT_filepath,
        #  'ripple_WCorrShuffle_df_export_CSV_path': ripple_WCorrShuffle_df_export_CSV_path,
    }

    across_session_results_extended_dict['compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function'] = callback_outputs

    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict





@function_attributes(short_name=None, tags=['recomputed_inst_firing_rate', 'inst_fr', 'independent'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def compute_and_export_session_instantaneous_spike_rates_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                            #  instantaneous_time_bin_size_seconds_list:List[float]=[0.0005, 0.0009, 0.0015, 0.0025, 0.025], epoch_handling_mode:str='DropShorterMode',
                                                                            instantaneous_time_bin_size_seconds_list:List[float]=[1000.0], epoch_handling_mode:str='UseAllEpochsMode', # single-bin per epoch
                                                                            save_hdf:bool=True, save_pickle:bool=True, save_across_session_hdf:bool=False, 
                                                                ) -> dict:
    """  Computes the `InstantaneousSpikeRateGroupsComputation` for the pipleine (completely independent of the internal implementations), and exports it as several output files:

    Output Files:
        if save_pickle:
            PKL Output:     f'{get_now_day_str()}_recomputed_inst_fr_comps_{_out_recomputed_inst_fr_comps.instantaneous_time_bin_size_seconds}.pkl'
        
        if save_hdf:
            HDF5 Output:    f'{get_now_day_str()}_recomputed_inst_fr_comps_{_out_recomputed_inst_fr_comps.instantaneous_time_bin_size_seconds}.h5'
        
        if save_across_session_hdf:
            Across Session HDF5 Output: f'{get_now_day_str()}_across_session_recomputed_inst_fr_comps.h5'
        

    ## previous `instantaneous_time_bin_size_seconds` values:
    0.0005    
    
    
    """
    import sys
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import InstantaneousFiringRatesDataframeAccessor

    # Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]
    callback_outputs = {
        'recomputed_inst_fr_time_bin_dict': None,
    }
    
        
    def _subfn_single_time_bin_size_compute_and_export_session_instantaneous_spike_rates_completion_function(instantaneous_time_bin_size_seconds:float):
        print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f'compute_and_export_session_instantaneous_spike_rates_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, instantaneous_time_bin_size_seconds: {instantaneous_time_bin_size_seconds}, ...)')
        
        subfn_callback_outputs = {
            'recomputed_inst_fr_comps_filepath': None, #'t_end': t_end   
            'recomputed_inst_fr_comps_h5_filepath': None,
            'common_across_session_h5': None,
        }
        err = None

        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_recomputed_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds) # 3ms, 10ms
            _out_recomputed_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context(), epoch_handling_mode=epoch_handling_mode)

            # LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            # LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
            print(f'\t\t done (success).')

        except BaseException as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"WARN: on_complete_success_execution_session: encountered exception {err} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            # if self.fail_on_exception:
            #     raise e.exc
            # _out_inst_fr_comps = None
            _out_recomputed_inst_fr_comps = None
            pass


        ## standalone saving:
        if (_out_recomputed_inst_fr_comps is not None) and save_pickle:
            ## Pickle Saving:
            standalone_filename: str = f'{get_now_day_str()}_recomputed_inst_fr_comps_{_out_recomputed_inst_fr_comps.instantaneous_time_bin_size_seconds}.pkl'
            recomputed_inst_fr_comps_filepath = curr_active_pipeline.get_output_path().joinpath(standalone_filename).resolve()
            print(f'recomputed_inst_fr_comps_filepath: "{recomputed_inst_fr_comps_filepath}"')

            try:
                saveData(recomputed_inst_fr_comps_filepath, (curr_session_context, _out_recomputed_inst_fr_comps, _out_recomputed_inst_fr_comps.instantaneous_time_bin_size_seconds))
                was_write_good = True
                subfn_callback_outputs['recomputed_inst_fr_comps_filepath'] = recomputed_inst_fr_comps_filepath

            except BaseException as e:
                exception_info = sys.exc_info()
                err = CapturedException(e, exception_info)
                print(f"ERROR: encountered exception {err} while trying to perform _out_recomputed_inst_fr_comps.save_data('{recomputed_inst_fr_comps_filepath}') for {curr_session_context}")
                recomputed_inst_fr_comps_filepath = None # set to None because it failed.
                if self.fail_on_exception:
                    raise err.exc
        else:
            recomputed_inst_fr_comps_filepath = None

        subfn_callback_outputs['recomputed_inst_fr_comps_filepath'] = recomputed_inst_fr_comps_filepath


        ## HDF5 output:
        if (_out_recomputed_inst_fr_comps is not None) and save_hdf:
            ## Pickle Saving:
            standalone_h5_filename: str = f'{get_now_day_str()}_recomputed_inst_fr_comps_{_out_recomputed_inst_fr_comps.instantaneous_time_bin_size_seconds}.h5'
            recomputed_inst_fr_comps_h5_filepath = curr_active_pipeline.get_output_path().joinpath(standalone_h5_filename).resolve()
            print(f'recomputed_inst_fr_comps_h5_filepath: "{recomputed_inst_fr_comps_h5_filepath}"')
            try:
                _out_recomputed_inst_fr_comps.to_hdf(recomputed_inst_fr_comps_h5_filepath, key='recomputed_inst_fr_comps', debug_print=False, enable_hdf_testing_mode=False)
                was_write_good = True
                subfn_callback_outputs['recomputed_inst_fr_comps_h5_filepath'] = recomputed_inst_fr_comps_h5_filepath

            except BaseException as e:
                exception_info = sys.exc_info()
                err = CapturedException(e, exception_info)
                print(f"ERROR: encountered exception {err} while trying to perform _out_recomputed_inst_fr_comps.to_hdf('/recomputed_inst_fr_comps', '{recomputed_inst_fr_comps_h5_filepath}') for {curr_session_context}")
                recomputed_inst_fr_comps_h5_filepath = None # set to None because it failed.
                if self.fail_on_exception:
                    raise err.exc
        else:
            recomputed_inst_fr_comps_h5_filepath = None


        ## common_across_session_h5
        if (_out_recomputed_inst_fr_comps is not None) and save_across_session_hdf:
            common_across_session_h5_filename: str = f'{get_now_day_str()}_across_session_recomputed_inst_fr_comps.h5'
            common_file_path = self.collected_outputs_path.resolve().joinpath(common_across_session_h5_filename).resolve()
            print(f'common_file_path: "{common_file_path}"')

            try:
                InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(inst_fr_comps=_out_recomputed_inst_fr_comps, curr_active_pipeline=curr_active_pipeline, common_file_path=common_file_path)
                subfn_callback_outputs['common_across_session_h5'] = common_file_path

            except BaseException as e:
                exception_info = sys.exc_info()
                err = CapturedException(e, exception_info)
                print(f"ERROR: encountered exception {err} while trying to perform InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(..., common_file_path='{common_file_path}') for {curr_session_context}")
                common_file_path = None # set to None because it failed.
                if self.fail_on_exception:
                    raise err.exc
        else:
            common_file_path = None
            
        return subfn_callback_outputs
        # END _subfn_single_time_bin_size_compute_and_export_session_instantaneous_spike_rates_completion_function

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    callback_outputs['recomputed_inst_fr_time_bin_dict'] = {}
    for an_instantaneous_time_bin_size_seconds in instantaneous_time_bin_size_seconds_list:
        subfn_callback_outputs = _subfn_single_time_bin_size_compute_and_export_session_instantaneous_spike_rates_completion_function(instantaneous_time_bin_size_seconds=an_instantaneous_time_bin_size_seconds)
        callback_outputs['recomputed_inst_fr_time_bin_dict'][an_instantaneous_time_bin_size_seconds] = subfn_callback_outputs


    across_session_results_extended_dict['compute_and_export_session_instantaneous_spike_rates_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['JSON', 'CSV', 'peak', 'pf', 'peak_promenance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def compute_and_export_session_extended_placefield_peak_information_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                             save_hdf:bool=True, save_across_session_hdf:bool=False) -> dict:
    """  Extracts peak information for the placefields for each neuron
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import reload_exported_kdiba_session_position_info_mat_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'compute_and_export_session_extended_placefield_peak_information_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('compute_and_export_session_extended_placefield_peak_information_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    """
    import sys
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult

    # Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_extended_placefield_peak_information_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    callback_outputs = {
        'json_output_path': None, #'t_end': t_end   
        'csv_output_path': None,
    }
    err = None

    # active_csv_parent_output_path = curr_active_pipeline.get_output_path().resolve()
    active_export_parent_output_path = self.collected_outputs_path.resolve()

    try:
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = rank_order_results.included_qclu_values
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        print(f'included_qclu_values: {included_qclu_values}')
    
        print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
        jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
        neuron_replay_stats_df: pd.DataFrame = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)
        neuron_replay_stats_df, all_pf2D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        neuron_replay_stats_df, all_pf1D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_directional_pf_maximum_peaks(track_templates=track_templates)
        # both_included_neuron_stats_df = deepcopy(neuron_replay_stats_df[neuron_replay_stats_df['LS_pf_peak_x_diff'].notnull()]).drop(columns=['track_membership', 'neuron_type'])
        
        print(f'\t\t done (success).')

    except BaseException as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"WARN: on_complete_success_execution_session: encountered exception {err} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
        # if self.fail_on_exception:
        #     raise e.exc
        # _out_inst_fr_comps = None
        neuron_replay_stats_df = None
        pass

    if (neuron_replay_stats_df is not None):
        print(f'\t try saving to CSV...')
        # Save DataFrame to CSV
        csv_output_path = active_export_parent_output_path.joinpath(f'{CURR_BATCH_OUTPUT_PREFIX}_neuron_replay_stats_df.csv').resolve()
        try:
            
            neuron_replay_stats_df.to_csv(csv_output_path)
            print(f'\t saving to CSV: "{csv_output_path}" done.')
            callback_outputs['csv_output_path'] = csv_output_path

        except BaseException as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {err} while trying to save to CSV for {curr_session_context}")
            csv_output_path = None # set to None because it failed.
            if self.fail_on_exception:
                raise err.exc
    else:
        csv_output_path = None


    ## standalone saving:
    if (neuron_replay_stats_df is not None):
        print(f'\t try saving to JSON...')
        # Save DataFrame to JSON
        json_output_path = active_export_parent_output_path.joinpath(f'{CURR_BATCH_OUTPUT_PREFIX}_neuron_replay_stats_df.json').resolve()
        try:
            neuron_replay_stats_df.to_json(json_output_path, orient='records', lines=True) ## This actually looks pretty good!
            print(f'\t saving to JSON: "{json_output_path}" done.')
            callback_outputs['json_output_path'] = json_output_path

        except BaseException as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {err} while trying to save to json for {curr_session_context}")
            json_output_path = None # set to None because it failed.
            if self.fail_on_exception:
                raise err.exc
    else:
        json_output_path = None

    across_session_results_extended_dict['compute_and_export_session_extended_placefield_peak_information_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict



# ==================================================================================================================== #
# END COMPLETION FUNCTIONS                                                                                             #
# ==================================================================================================================== #

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
    if override_custom_user_completion_functions_dict is None:
        # If a literall None is provided, provide ALL
        custom_user_completion_functions_dict = {
                                    "export_rank_order_results_completion_function": export_rank_order_results_completion_function,
                                    "figures_rank_order_results_completion_function": figures_rank_order_results_completion_function,
                                    "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function,
                                    'determine_session_t_delta_completion_function': determine_session_t_delta_completion_function,
                                    'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function,
                                    'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function,
                                    'reload_exported_kdiba_session_position_info_mat_completion_function': reload_exported_kdiba_session_position_info_mat_completion_function,
                                    'export_session_h5_file_completion_function': export_session_h5_file_completion_function,
                                    'compute_and_export_session_wcorr_shuffles_completion_function': compute_and_export_session_wcorr_shuffles_completion_function,
                                    'compute_and_export_session_instantaneous_spike_rates_completion_function': compute_and_export_session_instantaneous_spike_rates_completion_function,
                                    'compute_and_export_session_extended_placefield_peak_information_completion_function': compute_and_export_session_extended_placefield_peak_information_completion_function,
                                    'compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function': compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function,
                                    'backup_previous_session_files_completion_function': backup_previous_session_files_completion_function,
                                    }
    else:
        # use the user one:
        custom_user_completion_functions_dict = override_custom_user_completion_functions_dict
    
    
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




