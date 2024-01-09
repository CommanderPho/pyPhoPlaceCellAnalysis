from typing import List, Dict, Tuple, Optional, Callable

from pathlib import Path

# %% [markdown]
# ## Build Processing Scripts:

# %%
import inspect

curr_context_template: str = """ 
    BATCH_DATE_TO_USE = '2024-01-04_Lab' # TODO: Change this as needed, templating isn't actually doing anything rn.
    collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve()
    assert collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')
"""

def _helper_get_curr_batch_context(curr_active_pipeline):
    BATCH_DATE_TO_USE = '2024-01-04_Lab' # TODO: Change this as needed, templating isn't actually doing anything rn.
    collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve()
    assert collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')
    return CURR_BATCH_OUTPUT_PREFIX, curr_session_name, collected_outputs_path, BATCH_DATE_TO_USE


def export_rank_order_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'export_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    BATCH_DATE_TO_USE = '2024-01-04_Lab' # TODO: Change this as needed, templating isn't actually doing anything rn.
    collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve()
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
    # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'figures_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    BATCH_DATE_TO_USE = '2024-01-04_Lab' # TODO: Change this as needed, templating isn't actually doing anything rn.
    collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve()
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
    # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_marginals_dfs_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

    BATCH_DATE_TO_USE = '2024-01-09_GL' # TODO: Change this as needed, templating isn't actually doing anything rn.
    # collected_outputs_path = Path('/nfs/turbo/umms-kdiba/Data/Output/collected_outputs').resolve() # Linux
    collected_outputs_path = Path('/home/halechr/cloud/turbo/Data/Outputcollected_outputs').resolve() # GreatLakes
    # collected_outputs_path = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\output\collected_outputs').resolve() # Apogee
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

    # add to output dict    
    # across_session_results_extended_dict['compute_and_export_marginals_dfs_completion_function'] = _out

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict



def MAIN_get_template_string():
    """ 
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import MAIN_get_template_string
    


    """
    # custom_user_completion_functions = [a_test_completion_function]

    custom_user_completion_functions_dict = {
                                        # "export_rank_order_results_completion_function": export_rank_order_results_completion_function,
                                        # "figures_rank_order_results_completion_function": figures_rank_order_results_completion_function,
                                        # "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function,
                                        }


    # _template_defn_string: str = '\n\n'.join([inspect.getsource(a_fn) for a_name, a_fn in custom_user_completion_functions_dict.items()])
        
    ## Build the template string:
    template_str: str = f"""
    custom_user_completion_functions = []
    """

    for a_name, a_fn in custom_user_completion_functions_dict.items():
        fcn_defn_str: str = inspect.getsource(a_fn)
        template_str = f"{template_str}\n{fcn_defn_str}\ncustom_user_completion_functions.append({a_name})\n\n"
        
    template_str

    # template_str: str = f"""
    # {inspect.getsource(export_rank_order_results_completion_function)}
    # custom_user_completion_functions = [export_rank_order_results_completion_function]
    # """

    custom_user_completion_function_template_code = template_str
    print(custom_user_completion_function_template_code)
    return custom_user_completion_function_template_code, custom_user_completion_functions_dict




