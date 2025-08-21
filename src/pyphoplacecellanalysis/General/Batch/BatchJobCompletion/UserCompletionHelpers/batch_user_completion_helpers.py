from copy import deepcopy
import shutil
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from neuropy.analyses import Epoch
from neuropy.core.epoch import TimeColumnAliasesProtocol, ensure_dataframe
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types

from pathlib import Path
import inspect
from jinja2 import Template
from neuropy.utils.result_context import IdentifyingContext
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

# ==================================================================================================================== #
# Specific Decoding Parameter Sweeps                                                                                   #
# ==================================================================================================================== #

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
    


    Outputs:

        After the sweeps are complete and multiple (one for each time_bin_size swept) indepdnent dfs are had with the four results types this function concatenates each of the four into a single dataframe for all time_bin_size values with a column 'time_bin_size'. 
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
    from neuropy.core.epoch import TimeColumnAliasesProtocol, ensure_dataframe
    from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _compute_lap_and_ripple_epochs_decoding_for_decoder, _perform_compute_custom_epoch_decoding, _compute_all_df_score_metrics
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df, co_filter_epochs_and_spikes
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes
    
    DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names

    suppress_exceptions: bool = (not self.fail_on_exception)

    if additional_session_context is None:
        print(f'\t!!!! 2024-07-10 WARNING: additional_session_context is None!')

    # BEGIN _SUBFNS_ _____________________________________________________________________________________________________ #

    # Export CSVs:
    def export_marginals_df_csv(marginals_df: pd.DataFrame, data_identifier_str: str, parent_output_path: Path, active_context: IdentifyingContext):
        """ 
        captures: curr_active_pipeline,
        
        Outputs: '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
        
        """
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_day_str(), data_identifier_str=data_identifier_str, parent_output_path=parent_output_path, out_extension='.csv')
        marginals_df.to_csv(out_path)
        return out_path 


    def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
        """ captures `curr_active_pipeline`
        """
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(rounded_minutes=10), data_identifier_str=data_identifier_str,
                                                                                                                 parent_output_path=parent_output_path, out_extension='.csv')
        export_df.to_csv(out_path)
        return out_path 
    
    custom_export_df_to_csv_fn = _subfn_custom_export_df_to_csv
    

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
            # assert active_context is not None
            if several_time_bin_sizes_time_bin_laps_df is not None:
                laps_time_bin_marginals_out_path = export_marginals_df_csv(several_time_bin_sizes_time_bin_laps_df, data_identifier_str=f'(laps_time_bin_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)
            if several_time_bin_sizes_laps_df is not None:
                laps_out_path = export_marginals_df_csv(several_time_bin_sizes_laps_df, data_identifier_str=f'(laps_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)
            if several_time_bin_sizes_time_bin_ripple_df is not None:
                ripple_time_bin_marginals_out_path = export_marginals_df_csv(several_time_bin_sizes_time_bin_ripple_df, data_identifier_str=f'(ripple_time_bin_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)
            if several_time_bin_sizes_ripple_df is not None:
                ripple_out_path = export_marginals_df_csv(several_time_bin_sizes_ripple_df, data_identifier_str=f'(ripple_marginals_df)', parent_output_path=self.collected_outputs_path, active_context=active_context)

        return (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path)


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

            directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), filter_epochs=laps_epochs_df,
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
            directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), filter_epochs=replay_epochs_df,
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
    #TODO 2024-11-19 02:05: - [ ] Depricated in favor of using `curr_active_pipeline.build_complete_session_identifier_filename_string(...)`
    _, _, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters() # this `custom_suffix` is correct and not duplicated
    # curr_active_pipeline.get
    #TODO 2024-11-19 00:11: - [ ] Get proper names
    if len(custom_suffix) > 0:
        if additional_session_context is not None:
            if isinstance(additional_session_context, dict):
                additional_session_context = IdentifyingContext(**additional_session_context)
            ## easiest to update as dict:	
            additional_session_context = additional_session_context.to_dict()
            ## do not duplicate context:

            existing_custom_suffix: str = (additional_session_context.get('custom_suffix', '') or '')
            print(f'final_custom_suffix: "{existing_custom_suffix}"')	
            final_custom_suffix: str = existing_custom_suffix # 
            found_existing_same_custom_suffix_idx: int = existing_custom_suffix.find(custom_suffix)
            if found_existing_same_custom_suffix_idx > -1:
                # prevent duplication of custom suffix:
                print(f'\tdropping "{custom_suffix}" to prevent duplication...')	
                final_custom_suffix = final_custom_suffix.replace(custom_suffix, '') # drop the custom_suffix so it isn't duplicated

            final_custom_suffix = final_custom_suffix + custom_suffix ## add it back on to the end, so there's only one repetition.
            print(f'final_custom_suffix: "{final_custom_suffix}"')	
            # additional_session_context['custom_suffix'] = (additional_session_context.get('custom_suffix', '') or '') + custom_suffix # this is where the duplication happens
            additional_session_context['custom_suffix'] =  final_custom_suffix
            additional_session_context = IdentifyingContext(**additional_session_context)
            
        else:
            additional_session_context = IdentifyingContext(custom_suffix=custom_suffix)
    
    assert (additional_session_context is not None), f"perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function: additional_session_context is None even after trying to add the computation params as additional_session_context"
    # active_context = curr_active_pipeline.get_session_context()
    if additional_session_context is not None:
        if isinstance(additional_session_context, dict):
            additional_session_context = IdentifyingContext(**additional_session_context)
        active_context = (curr_active_pipeline.get_session_context() | additional_session_context)
        # if len(custom_suffix) == 0:
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=(IdentifyingContext._get_session_context_keys() + list(additional_session_context.keys())))
        # CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{additional_session_context.get_description()}"
        # else:
        # 	session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=(IdentifyingContext._get_session_context_keys() + list(additional_session_context.keys()))) + f'|{custom_suffix}'
        # 	CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{additional_session_context.get_description()}-{custom_suffix}"
    else:
        active_context = curr_active_pipeline.get_session_context()
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
        # if len(custom_suffix) == 0:
        # CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
        # else:
        # 	session_ctxt_key:str = session_ctxt_key + custom_suffix
        # 	CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{custom_suffix}"

    active_context = None
    
    print(f'\tactive_context: {active_context}')
    print(f'\tsession_ctxt_key: {session_ctxt_key}')
    # print(f'\tCURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')
    
    ## INPUT PARAMETER: time_bin_size sweep paraemters    
    if custom_all_param_sweep_options is None:
        if desired_shared_decoding_time_bin_sizes is None:
            # desired_shared_decoding_time_bin_sizes = np.linspace(start=0.030, stop=0.10, num=6) ####### <<<------ Default sweep is defined here
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058, 0.072, 0.086, 0.100, 0.250, 1.5]) ####### <<<------ Default sweep is defined here
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058, 0.072, 0.086, 0.100]) ####### <<<------ Default sweep is defined here
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058, 0.072, 0.086, 0.100])
            # desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.030, 0.044, 0.050, 0.058,])
            desired_shared_decoding_time_bin_sizes = np.array([0.025, 0.058,])

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
    rank_order_results = curr_active_pipeline.global_computation_results.computed_data.get('RankOrder', None)
    if rank_order_results is not None:
        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = rank_order_results.included_qclu_values
    else:        
        ## get from parameters:
        minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
        
    # DirectionalMergedDecoders: Get the result after computation:
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # : "DirectionalLapsResult"
    track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference? : "TrackTemplates"

    ## Copy the default result:
    directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)
    out_path, out_path_filenname_str, out_path_basename_str = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=self.BATCH_DATE_TO_USE, data_identifier_str="(time_bin_size_sweep_results)", parent_output_path=self.collected_outputs_path.resolve(), out_extension='.h5')
    print(f'\tout_path_str: "{out_path_filenname_str}"')
    print(f'\tout_path: "{out_path}"')
    
    # Ensure it has the 'lap_track' column
    ## Compute the ground-truth information using the position information:
    # adds columns: ['maze_id', 'is_LR_dir']
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_obj.update_lap_dir_from_net_displacement(pos_input=curr_active_pipeline.sess.position)
    laps_obj.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
    laps_df = laps_obj.to_dataframe()
    assert 'maze_id' in laps_df.columns, f"laps_df is still missing the 'maze_id' column after calling `laps_obj.update_maze_id_if_needed(...)`. laps_df.columns: {print(list(laps_df.columns))}"

    # # BEGIN BLOCK ________________________________________________________________________________________________________ #

    # BEGIN BLOCK 2 - modernizing from `_perform_compute_custom_epoch_decoding`  ________________________________________________________________________________________________________ #
    
    # Uses: session_ctxt_key, all_param_sweep_options
    output_alt_directional_merged_decoders_result: Dict[Tuple, DirectionalPseudo2DDecodersResult] = {} # empty dict
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
        ripple_time_bin_marginals_df: pd.DataFrame = an_alt_dir_Pseudo2D_decoders_result.ripple_time_bin_marginals_df.copy() ## calling .copy() is triggering an issue with the @property `ripple_time_bin_marginals_df`
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
        (decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict), merged_df_outputs_tuple, raw_dict_outputs_tuple = _compute_all_df_score_metrics(directional_merged_decoders_result=an_alt_dir_Pseudo2D_decoders_result, track_templates=track_templates,
                                                                                                                                                                                            decoder_laps_filter_epochs_decoder_result_dict=decoder_laps_filter_epochs_decoder_result_dict,
                                                                                                                                                                                            decoder_ripple_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                                                                                            # spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df),
                                                                                                                                                                                            spikes_df=get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=curr_active_pipeline.global_computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values),
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
        # across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'] = tuple(across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'])


    # across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'] ## add the output files?
    
    ## UNPACKING
    # # with `return_full_decoding_results == False`
    # out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple, output_saved_individual_sweep_files_dict = across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']

    # # with `return_full_decoding_results == True`
    # out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple, output_full_directional_merged_decoders_result, output_saved_individual_sweep_files_dict = across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']


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
                                                                                        custom_export_df_to_csv_fn=custom_export_df_to_csv_fn, 
                                                                                    )
        print(f'\t>>>>>>>>>> exported files: {_output_csv_paths}\n\n')
        for k, v in _output_csv_paths.items():
            ## current iteration only
            if k not in _temp_saved_files_dict:
                _temp_saved_files_dict[k] = [] # initialize an empty array for this key
            _temp_saved_files_dict[k].append(v) ## append the path to the output files dict				

    ## update output files:
    across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'].append(_temp_saved_files_dict)
    
    across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'] = tuple(across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'])
    

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['CSVs', 'export', 'across-sessions', 'batch', 'single-time-bin-size', 'ripple_all_scores_merged_df', 'DIRECTIONAL-ONLY'], input_requires=['DirectionalLaps', 'RankOrder', 'DirectionalDecodersEpochsEvaluations'], output_provides=[], uses=['filter_and_update_epochs_and_spikes', 'DecoderDecodedEpochsResult', 'DecoderDecodedEpochsResult.export_csvs'], used_by=[], creation_date='2024-04-27 21:20', related_items=[])
def compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                               ripple_decoding_time_bin_size_override: Optional[float]=None, laps_decoding_time_bin_size_override: Optional[float]=None,
                                                needs_recompute_heuristics: bool = False, force_recompute_all_decoding: bool = False,
                                                save_hdf:bool=True, allow_append_to_session_h5_file:bool=True, max_ignore_bins: float = 2, same_thresh_cm: float = 10.7, max_jump_distance_cm: float = 60.0) -> dict:
    """
    Aims to export the results of the global 'directional_decoders_evaluate_epochs' calculation

    Exports: 'ripple_all_scores_merged_df'

    Uses result computed by `_decode_and_evaluate_epochs_using_directional_decoders`

    
    Updates: ['DirectionalDecodersEpochsEvaluations'


    Exports:

    "K:/scratch/collected_outputs/2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(laps_simple_pf_pearson_merged_df)_tbin-0.25.csv"
    "K:/scratch/collected_outputs/2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(laps_weighted_corr_merged_df)_tbin-0.25.csv"
    
    ## why using the wrong size (0.016)?
    "K:/scratch/collected_outputs/2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_all_scores_merged_df)_tbin-0.016.csv"
    "K:/scratch/collected_outputs/2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_simple_pf_pearson_merged_df)_tbin-0.016.csv"
    "K:/scratch/collected_outputs/2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_weighted_corr_merged_df)_tbin-0.016.csv"
    
    ## why missing the custom_replay suffix?
    "K:/scratch/collected_outputs/2024-11-26_0240AM-kdiba_gor01_one_2006-6-09_1-22-43-(ripple_all_scores_merged_df)_tbin-0.025.csv"
    "K:/scratch/collected_outputs/2024-11-26_0240AM-kdiba_gor01_one_2006-6-09_1-22-43-(laps_simple_pf_pearson_merged_df)_tbin-0.25.csv"
    "K:/scratch/collected_outputs/2024-11-26_0240AM-kdiba_gor01_one_2006-6-09_1-22-43-(laps_weighted_corr_merged_df)_tbin-0.25.csv"
    "K:/scratch/collected_outputs/2024-11-26_0240AM-kdiba_gor01_one_2006-6-09_1-22-43-(ripple_simple_pf_pearson_merged_df)_tbin-0.025.csv"
    "K:/scratch/collected_outputs/2024-11-26_0240AM-kdiba_gor01_one_2006-6-09_1-22-43-(ripple_weighted_corr_merged_df)_tbin-0.025.csv"
    
    if save_hdf:
        "2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(decoded_posteriors).h5"

    """
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function(global_data_root_parent_path: "{global_data_root_parent_path}", curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)') # ,across_session_results_extended_dict: {across_session_results_extended_dict}
    from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _workaround_validate_has_directional_decoded_epochs_heuristic_scoring

    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring
    from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
    from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
    

    assert self.collected_outputs_path.exists()
    active_context = curr_active_pipeline.get_session_context()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')
    CURR_BATCH_DATE_TO_USE: str = self.BATCH_DATE_TO_USE

    across_session_results_extended_dict['compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function'] = {'ripple_decoding_time_bin_size_override': ripple_decoding_time_bin_size_override,
        'laps_decoding_time_bin_size_override': laps_decoding_time_bin_size_override,
        'needs_recompute_heuristics': needs_recompute_heuristics, 'save_hdf': save_hdf, 'allow_append_to_session_h5_file': allow_append_to_session_h5_file,
        'output_csv_paths': None, 'output_hdf_paths': None, # 'allow_append_to_session_h5_file': allow_append_to_session_h5_file,
        'max_ignore_bins':max_ignore_bins, 'same_thresh_cm':same_thresh_cm, 'max_jump_distance_cm':max_jump_distance_cm,		
    }
    
    # across_session_results_extended_dict['compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function'].update({'output_csv_paths': []})
    # across_session_results_extended_dict['compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function'].update({'output_hdf_paths': []})
    

    def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
        """ captures CURR_BATCH_DATE_TO_USE, `curr_active_pipeline`
        """
        output_date_str: str = deepcopy(CURR_BATCH_DATE_TO_USE)
        if (output_date_str is None) or (len(output_date_str) < 1):
            output_date_str = get_now_rounded_time_str(rounded_minutes=10)
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=data_identifier_str, parent_output_path=parent_output_path, out_extension='.csv')
        export_df.to_csv(out_path)
        return out_path 
    

    def _subfn_build_custom_export_to_h5_path(data_identifier_str: str = f'(decoded_posteriors)', a_tbin_size: float=None, parent_output_path: Path=None):
        """ captures CURR_BATCH_DATE_TO_USE, `curr_active_pipeline`
        """
        output_date_str: str = deepcopy(CURR_BATCH_DATE_TO_USE)
        if (output_date_str is None) or (len(output_date_str) < 1):
            output_date_str = get_now_rounded_time_str(rounded_minutes=10)
            
        if (a_tbin_size is not None):
            ## add optional time bin suffix:
            a_tbin_size_str: str = f"{round(a_tbin_size, ndigits=5)}"
            a_data_identifier_str: str = f'{data_identifier_str}_tbin-{a_tbin_size_str}' ## build the identifier '(decoded_posteriors)_tbin-1.5'
            
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=output_date_str, data_identifier_str=a_data_identifier_str, parent_output_path=parent_output_path, out_extension='.h5')
        return out_path 
    

    
    custom_export_df_to_csv_fn = _subfn_custom_export_df_to_csv
    
    ## Doesn't force recompute! Assumes that the DirectionalDecodersEpochsEvaluations result is new
    # curr_active_pipeline.reload_default_computation_functions()
    # batch_extended_computations(curr_active_pipeline, include_includelist=['directional_decoders_evaluate_epochs'], include_global_functions=True, fail_on_exception=True, force_recompute=True)
    
    ## Extract Data:
    directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # DirectionalLapsResult
    rank_order_results = curr_active_pipeline.global_computation_results.computed_data.get('RankOrder', None)
    if rank_order_results is not None:
        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = rank_order_results.included_qclu_values
    else:        
        ## get from parameters:
        minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
    track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only # TrackTemplates

    if force_recompute_all_decoding:
        ## force recompute all
        print(f'\tforce_recompute_all_decoding is True. dropping previous results.')
        needs_recompute: bool = True # need to recompute if we're lacking result
    else:
        ## try to use previous decoding result (but check it for matching time bin sizes, etc first):
        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
        if (directional_decoders_epochs_decode_result is None):
            needs_recompute: bool = True # need to recompute if we're lacking result
        else:
            needs_recompute: bool = False
            pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
            prev_computed_ripple_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
            prev_computed_laps_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
            if ripple_decoding_time_bin_size_override is not None:
                if ripple_decoding_time_bin_size_override != prev_computed_ripple_decoding_time_bin_size:
                    print(f'ripple_decoding_time_bin_size_override is specfied ({ripple_decoding_time_bin_size_override}) and is not equal to the computed value ({prev_computed_ripple_decoding_time_bin_size}). Will recompmute!')
                    needs_recompute = True
                else: 
                    print(f'ripple_decoding_time_bin_size_override is the same size as computed ({ripple_decoding_time_bin_size_override})')

            if laps_decoding_time_bin_size_override is not None:
                if laps_decoding_time_bin_size_override != prev_computed_laps_decoding_time_bin_size:
                    print(f'laps_decoding_time_bin_size_override is specfied ({laps_decoding_time_bin_size_override}) and is not equal to the computed value ({prev_computed_laps_decoding_time_bin_size}). Will recompmute!')
                    needs_recompute = True
                else:
                    print(f'laps_decoding_time_bin_size_override is the same size as computed ({laps_decoding_time_bin_size_override})')
                

    if needs_recompute:
        ## Drop 'DirectionalDecodersEpochsEvaluations', and recompute
        global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['DirectionalDecodersEpochsEvaluations'], debug_print=True) # don't need to drop 'DirectionalMergedDecoders', , just recompute it
        #TODO 2024-11-27 13:24: - [ ] Can't I just use the independent version, `_perform_compute_custom_epoch_decoding(...)` or whatnot?

    if needs_recompute:
        print(f'recompute is needed!')
        curr_active_pipeline.reload_default_computation_functions()
        ## This recreates the Pseudo2D placefields from the 4 directional ones, which is NOT needed and seems a bit excessive.
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields'], # 'merged_directional_placefields': `_build_merged_directional_placefields`
                                                        computation_kwargs_list=[{'laps_decoding_time_bin_size': laps_decoding_time_bin_size_override, 'ripple_decoding_time_bin_size': ripple_decoding_time_bin_size_override},],
                                                        enabled_filter_names=None, fail_on_exception=True, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
        
        # 'DirectionalDecodersEpochsEvaluations': `directional_decoders_evaluate_epochs`
        global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['DirectionalDecodersEpochsEvaluations'], debug_print=True)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_evaluate_epochs'], # ,  'directional_decoders_epoch_heuristic_scoring'
                        computation_kwargs_list=[{'should_skip_radon_transform': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
        needs_recompute_heuristics = True
        
        ## gets the newly computed value
        directional_decoders_epochs_decode_result = curr_active_pipeline.global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None) # 'DirectionalDecodersEpochsEvaluations': `directional_decoders_evaluate_epochs`
        assert directional_decoders_epochs_decode_result is not None, f"directional_decoders_epochs_decode_result is None even after recompute!"
        pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
        newly_computed_ripple_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
        newly_computed_laps_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
        print(f'\tnewly_computed_ripple_decoding_time_bin_size: {newly_computed_ripple_decoding_time_bin_size}')
        print(f'\tnewly_computed_laps_decoding_time_bin_size: {newly_computed_laps_decoding_time_bin_size}')
        if ripple_decoding_time_bin_size_override is not None:
            assert ripple_decoding_time_bin_size_override == newly_computed_ripple_decoding_time_bin_size, f'ripple_decoding_time_bin_size_override is specfied ({ripple_decoding_time_bin_size_override}) and is not equal to the computed value ({newly_computed_ripple_decoding_time_bin_size}). ERROR: Should match after computation!'
        if laps_decoding_time_bin_size_override is not None:
            assert laps_decoding_time_bin_size_override == newly_computed_laps_decoding_time_bin_size, f'laps_decoding_time_bin_size_override is specfied ({laps_decoding_time_bin_size_override}) and is not equal to the computed value ({newly_computed_laps_decoding_time_bin_size}). ERROR: Should match after computation!'
    # end if needs_recompute
    
    
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

    
    # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores` - updates `directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict`
    if (needs_recompute_heuristics or (not _workaround_validate_has_directional_decoded_epochs_heuristic_scoring(curr_active_pipeline))):
        print(f'\tmissing heuristic columns. Recomputing:')
        directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict, _out_new_scores, _out_new_partition_result_dict = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates,
                                                                                     a_decoded_filter_epochs_decoder_result_dict=directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict, max_ignore_bins=max_ignore_bins, same_thresh_cm=same_thresh_cm, max_jump_distance_cm=max_jump_distance_cm)
        print(f'\tdone recomputing heuristics.')


    print(f'\tComputation complete. Exporting .CSVs...')
    print(f"\t\t ripple_decoding_time_bin_size_override: {ripple_decoding_time_bin_size_override}")
    print(f"\t\t directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size: {directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size}")
    # print(f"\t\t directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size: {directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size}")
    ## Export CSVs:
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    _output_csv_paths = directional_decoders_epochs_decode_result.export_csvs(parent_output_path=self.collected_outputs_path.resolve(), active_context=active_context, session_name=curr_session_name, curr_session_t_delta=t_delta,
                                                                              user_annotation_selections={'ripple': any_good_selected_epoch_times},
                                                                              valid_epochs_selections={'ripple': filtered_valid_epoch_times},
                                                                              custom_export_df_to_csv_fn=custom_export_df_to_csv_fn,
                                                                              should_export_complete_all_scores_df=True, export_df_variable_names=[], # `export_df_variable_names=[]` means export no non-complete dfs
                                                                              )
    _output_csv_paths_info_str: str = '\n'.join([f'{a_name}: "{a_path}"' for a_name, a_path in _output_csv_paths.items()])
    # print(f'\t\t\tCSV Paths: {_output_csv_paths}\n')
    print(f'\t\t\tCSV Paths: {_output_csv_paths_info_str}\n')
    across_session_results_extended_dict['compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function'].update({'output_csv_paths': _output_csv_paths})
    
    
    
    # Export HDF5 ________________________________________________________________________________________________________ #
    if save_hdf:
        ## Exports: "2024-11-26_Lab-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(decoded_posteriors).h5"
        print(f'save_hdf == True, so exporting posteriors to HDF file...')
        # parent_output_path = self.collected_outputs_path.resolve()
        save_path: Path = _subfn_build_custom_export_to_h5_path(data_identifier_str='(decoded_posteriors)', a_tbin_size=directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size, parent_output_path=self.collected_outputs_path.resolve())
        # save_path = Path(f'output/{BATCH_DATE_TO_USE}_newest_all_decoded_epoch_posteriors.h5').resolve()
        complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
        _, _, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters()
        custom_params_hdf_key: str = custom_suffix.strip('_') # strip leading/trailing underscores
        # _parent_save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_decoded_posteriors_to_HDF5', custom_suffix=custom_suffix)
        _parent_save_context: DisplaySpecifyingIdentifyingContext = deepcopy(session_context).overwriting_context(custom_suffix=custom_params_hdf_key, display_fn_name='save_decoded_posteriors_to_HDF5')
        # _parent_save_context: DisplaySpecifyingIdentifyingContext = complete_session_context.overwriting_context(display_fn_name='save_decoded_posteriors_to_HDF5')
        _parent_save_context.display_dict = {
            'custom_suffix': lambda k, v: f"{v}", # just include the name
            'display_fn_name': lambda k, v: f"{v}", # just include the name
        }
        out_contexts, _flat_all_HDF5_out_paths = PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5(decoder_laps_filter_epochs_decoder_result_dict=None,
                                                                                    decoder_ripple_filter_epochs_decoder_result_dict=deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict),
                                                                                    _save_context=_parent_save_context.get_raw_identifying_context(), save_path=save_path, should_overwrite_extant_file=(not allow_append_to_session_h5_file))

        _flat_all_HDF5_out_paths = list(dict.fromkeys([v.as_posix() for v in _flat_all_HDF5_out_paths]).keys())
        # _output_HDF5_paths_info_str: str = '\n'.join([f'"{file_uri_from_path(a_path)}"' for a_path in _flat_all_HDF5_out_paths])
        _output_HDF5_paths_info_str: str = '\n'.join([f'"{a_path}"' for a_path in _flat_all_HDF5_out_paths])
        # print(f'\t\t\tHDF5 Paths: {_flat_all_HDF5_out_paths}\n')
        print(f'\t\t\tHDF5 Paths: {_output_HDF5_paths_info_str}\n')
        across_session_results_extended_dict['compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function'].update({'output_hdf_paths': _flat_all_HDF5_out_paths})
        

    print(f'\t\tsuccessfully exported directional_decoders_epochs_decode_result to {self.collected_outputs_path}!')


    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['TrialByTrialActivityResult'], input_requires=[], output_provides=[], uses=['TrialByTrialActivity.directional_compute_trial_by_trial_correlation_matrix', '_perform_run_rigorous_decoder_performance_assessment'], used_by=[], creation_date='2024-10-08 16:07', 
                     requires_global_keys=['DirectionalLaps'], provides_global_keys=['DirectionalMergedDecoders'], related_items=[])
def compute_and_export_session_trial_by_trial_performance_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                              active_laps_decoding_time_bin_size: float = 0.25, minimum_one_point_stability: float = 0.6, zero_point_stability: float = 0.1, save_hdf:bool=True, save_across_session_hdf:bool=False,
                                                                               additional_session_context: Optional[IdentifyingContext]=None) -> dict:
    """  Computes the trial-by-trial deoding performance
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import compute_and_export_session_trial_by_trial_performance_completion_function
    
    Results can be extracted from batch output by 
   
    Unpacking:
    
        callback_outputs = _across_session_results_extended_dict['compute_and_export_session_trial_by_trial_performance_completion_function']
        a_trial_by_trial_result: TrialByTrialActivityResult = callback_outputs['a_trial_by_trial_result']
        stability_df: pd.DataFrame = callback_outputs['stability_df']
        subset_neuron_IDs_dict = callback_outputs['subset_neuron_IDs_dict']
        subset_decode_results_dict = callback_outputs['subset_decode_results_dict']
        subset_decode_results_track_id_correct_performance_dict = callback_outputs['subset_decode_results_track_id_correct_performance_dict']
        directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = a_trial_by_trial_result.directional_active_lap_pf_results_dicts
        _out_subset_decode_results_track_id_correct_performance_dict = callback_outputs['subset_decode_results_track_id_correct_performance_dict']
        _out_subset_decode_results_dict = callback_outputs['subset_decode_results_dict']
        (complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results)  = _out_subset_decode_results_dict['any_decoder'] ## get the result for all cells
        filtered_laps_time_bin_marginals_df: pd.DataFrame = callback_outputs['subset_decode_results_time_bin_marginals_df_dict']['filtered_laps_time_bin_marginals_df']
        active_results: Dict[types.DecoderName, DecodedFilterEpochsResult] = deepcopy({k:v.decoder_result for k, v in _out_separate_decoder_results[0].items()})

        neuron_group_split_stability_dfs_tuple = callback_outputs['neuron_group_split_stability_dfs_tuple']
        neuron_group_split_stability_aclus_tuple = callback_outputs['neuron_group_split_stability_aclus_tuple']

        appearing_stability_df, disappearing_stability_df, appearing_or_disappearing_stability_df, stable_both_stability_df, stable_neither_stability_df, stable_long_stability_df, stable_short_stability_df = neuron_group_split_stability_dfs_tuple
        appearing_aclus, disappearing_aclus, appearing_or_disappearing_aclus, stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus = neuron_group_split_stability_aclus_tuple
        
    """
    import sys
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

    from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
    from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_run_rigorous_decoder_performance_assessment
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df, co_filter_epochs_and_spikes

    # from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

    # Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_trial_by_trial_performance_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
    # filtered_valid_epoch_times = filtered_epochs_df[['start', 'stop']].to_numpy()

    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    _, _, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters()
    if len(custom_suffix) > 0:
        if additional_session_context is not None:
            if isinstance(additional_session_context, dict):
                additional_session_context = IdentifyingContext(**additional_session_context)

            ## easiest to update as dict:	
            additional_session_context = additional_session_context.to_dict()
            additional_session_context['custom_suffix'] = (additional_session_context.get('custom_suffix', '') or '') + custom_suffix
            additional_session_context = IdentifyingContext(**additional_session_context)
            
        else:
            additional_session_context = IdentifyingContext(custom_suffix=custom_suffix)
    
        assert (additional_session_context is not None), f"compute_and_export_session_trial_by_trial_performance_completion_function: additional_session_context is None even after trying to add the computation params as additional_session_context"
        # active_context = curr_active_pipeline.get_session_context()
        if isinstance(additional_session_context, dict):
            additional_session_context = IdentifyingContext(**additional_session_context)
        active_context = (curr_active_pipeline.get_session_context() | additional_session_context)
        CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{additional_session_context.get_description()}"
    else:
        CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}" ## OLD:
    
    print(f'\tCURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    callback_outputs = {
        'active_laps_decoding_time_bin_size': None,
        'subset_decode_results_track_id_correct_performance_dict': None, #'t_end': t_end   
        'subset_decode_results_dict': None,
        'a_trial_by_trial_result': None, 'subset_neuron_IDs_dict': None,
        'neuron_group_split_stability_dfs_tuple': None, 'neuron_group_split_stability_aclus_tuple': None,
        'subset_decode_results_time_bin_marginals_df_dict': None
    }
    err = None

    callback_outputs['active_laps_decoding_time_bin_size'] = active_laps_decoding_time_bin_size

    try:
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data.get('RankOrder', None)
        if rank_order_results is not None:
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = rank_order_results.included_qclu_values
        else:
            ## get from parameters:
            minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
            included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        print(f'\tminimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        print(f'\tincluded_qclu_values: {included_qclu_values}')
    
        ## INPUTS: curr_active_pipeline, track_templates, global_epoch_name, (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)
        # any_decoder_neuron_IDs: NDArray = deepcopy(track_templates.any_decoder_neuron_IDs)
        any_decoder_neuron_IDs = None ## set to None to auto-build them
        # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

        # ## Directional Trial-by-Trial Activity:
        if 'pf1D_dt' not in curr_active_pipeline.computation_results[global_epoch_name].computed_data:
            # if `KeyError: 'pf1D_dt'` recompute
            curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)


        ## Copy previously existing one:
        # active_pf_1D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'])
        # active_pf_2D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt'])
                
        ## REBUILD NEW pf1D_dt
        computation_result = curr_active_pipeline.computation_results[global_epoch_name]
        active_session, pf_computation_config = computation_result.sess, computation_result.computation_config.pf_params
        active_session_spikes_df, active_pos, computation_config, active_epoch_placefields1D, active_epoch_placefields2D = active_session.spikes_df, active_session.position, pf_computation_config, None, None
        included_epochs = deepcopy(pf_computation_config.computation_epochs)
        should_force_recompute_placefields: bool = True

        if any_decoder_neuron_IDs is None:
            any_decoder_neuron_IDs = deepcopy(active_session_spikes_df.spikes.neuron_ids)
            print(f'any_decoder_neuron_IDs is None, using all aclus in spikes_df:\n\tany_decoder_neuron_IDs: {any_decoder_neuron_IDs}')
        # NOTE: even in TimeDependentPlacefieldSurpriseMode.STATIC_METHOD_ONLY a PfND_TimeDependent object is used to access its properties for the Static Method (although it isn't modified)
        active_pf_1D_dt = PfND_TimeDependent(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=deepcopy(included_epochs),
                                             config=deepcopy(pf_computation_config),
                                            )

        active_pf_dt: PfND_TimeDependent = active_pf_1D_dt

        # Limit only to the placefield aclus:
        # active_pf_dt = active_pf_dt.get_by_id(ids=any_decoder_neuron_IDs) ###TODO 2025-07-31 07:01: - [ ] DISABLE LIMITING TO JUST PLACEFIELDS 

        # long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = track_templates.get_decoders()

        # Unpack all directional variables:
        ## {"even": "RL", "odd": "LR"}
        ## Ancient Names:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # note has global also
        long_LR_name, long_RL_name, short_LR_name, short_RL_name = track_templates.get_decoder_names() ## Modern Names

        directional_lap_epochs_dict = dict(zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)))
        directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = TrialByTrialActivity.directional_compute_trial_by_trial_correlation_matrix(active_pf_dt=active_pf_dt, directional_lap_epochs_dict=directional_lap_epochs_dict, included_neuron_IDs=any_decoder_neuron_IDs)

        ## OUTPUTS: directional_active_lap_pf_results_dicts
        a_trial_by_trial_result: TrialByTrialActivityResult = TrialByTrialActivityResult(any_decoder_neuron_IDs=any_decoder_neuron_IDs,
                                                                                        active_pf_dt=active_pf_dt,
                                                                                        directional_lap_epochs_dict=directional_lap_epochs_dict,
                                                                                        directional_active_lap_pf_results_dicts=directional_active_lap_pf_results_dicts,
                                                                                        is_global=True)  # type: Tuple[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, BasePositionDecoder], Any]

        ## UNPACKING:
        directional_lap_epochs_dict: Dict[str, Epoch] = a_trial_by_trial_result.directional_lap_epochs_dict
        # stability_df = a_trial_by_trial_result.get_stability_df()
        # appearing_or_disappearing_aclus, appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus, (stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus) = a_trial_by_trial_result.get_cell_stability_info(minimum_one_point_stability=0.6, zero_point_stability=0.1)
        stability_df, _neuron_group_split_stability_dfs_tuple, _neuron_group_split_stability_aclus_tuple = a_trial_by_trial_result.get_cell_stability_info(minimum_one_point_stability=minimum_one_point_stability, zero_point_stability=zero_point_stability)
        # appearing_stability_df, disappearing_stability_df, appearing_or_disappearing_stability_df, stable_both_stability_df, stable_neither_stability_df, stable_long_stability_df, stable_short_stability_df = _neuron_group_split_stability_dfs_tuple
        appearing_aclus, disappearing_aclus, appearing_or_disappearing_aclus, stable_both_aclus, stable_neither_aclus, stable_long_aclus, stable_short_aclus = _neuron_group_split_stability_aclus_tuple
        ## Compute the track_ID deoding performance for the merged_decoder with some cells left out:
        subset_neuron_IDs_dict: Dict[str, NDArray] = dict(any_decoder=any_decoder_neuron_IDs,
            stable_both=stable_both_aclus, stable_neither=stable_neither_aclus,
            stable_long=stable_long_aclus, stable_short=stable_short_aclus,
            appearing_or_disappearing=appearing_or_disappearing_aclus,
            appearing=appearing_aclus, disappearing=disappearing_aclus,
        )
        
        ## OUTPUTS: a_trial_by_trial_result
        callback_outputs.update(dict(a_trial_by_trial_result=a_trial_by_trial_result,
                                     stability_df=deepcopy(stability_df),
                                     subset_neuron_IDs_dict=subset_neuron_IDs_dict,
                    neuron_group_split_stability_dfs_tuple=_neuron_group_split_stability_dfs_tuple, neuron_group_split_stability_aclus_tuple=_neuron_group_split_stability_aclus_tuple,
        ))
        
    except Exception as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"WARN: encountered exception {err} while performing .compute_and_export_session_trial_by_trial_performance_completion_function(...) - PHASE I\n\tfor curr_session_context: {curr_session_context}")
        if self.fail_on_exception:
            raise
            # raise e.exc
        # _out_inst_fr_comps = None
        neuron_replay_stats_df = None
        pass
    

    # ==================================================================================================================================================================================================================================================================================== #
    # PHASE II                                                                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    callback_outputs['subset_decode_results_track_id_correct_performance_dict'] = None
    callback_outputs['subset_decode_results_dict'] = None
    callback_outputs['subset_decode_results_time_bin_marginals_df_dict'] = None
        
    # try:        
    #     # ==================================================================================================================== #
    #     # Performs for each subset of cells                                                                                    #
    #     # ==================================================================================================================== #
    #     _out_subset_decode_results_dict: Dict[str, Tuple] = {}
    #     _out_subset_decode_results_track_id_correct_performance_dict: Dict[str, float] = {}
    #     for a_subset_name, a_neuron_IDs_subset in subset_neuron_IDs_dict.items():
    #         has_valid_result: bool = False
    #         if len(a_neuron_IDs_subset) > 0:
    #             try:
    #                 _out_subset_decode_results_dict[a_subset_name] = _perform_run_rigorous_decoder_performance_assessment(curr_active_pipeline=curr_active_pipeline, included_neuron_IDs=a_neuron_IDs_subset, active_laps_decoding_time_bin_size=active_laps_decoding_time_bin_size)
    #                 ## extract results:
    #                 complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results = _out_subset_decode_results_dict[a_subset_name]
    #                 (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
    #                 _out_subset_decode_results_track_id_correct_performance_dict[a_subset_name] = float(percent_laps_track_identity_estimated_correctly)
    #                 has_valid_result = True
    #             except ValueError as err:
    #                 # empty pfs: ValueError: need at least one array to concatenate
    #                 has_valid_result = False
    #             except Exception as err:
    #                 raise

    #         if (not has_valid_result):
    #             ## no result, initialize the key to empty/bad values:
    #             _out_subset_decode_results_dict[a_subset_name] = None
    #             _out_subset_decode_results_track_id_correct_performance_dict[a_subset_name] = np.nan

    #     _out_subset_decode_results_track_id_correct_performance_dict

    #     # ## OUTPUTS: `_out_subset_decode_results_track_id_correct_performance_dict`
    #     # {'any_decoder': 0.8351648351648352,
    #     #  'stable_both': 0.7692307692307693,
    #     #  'stable_neither': nan,
    #     #  'stable_long': 0.8131868131868132,
    #     #  'stable_short': 0.8241758241758241,
    #     #  'appearing_or_disappearing': 0.6593406593406593,
    #     #  'appearing': 0.7142857142857143,
    #     #  'disappearing': 0.6043956043956044}

    #     callback_outputs['subset_decode_results_track_id_correct_performance_dict'] = _out_subset_decode_results_track_id_correct_performance_dict
    #     callback_outputs['subset_decode_results_dict'] = _out_subset_decode_results_dict
        
    #     for a_subset_name, a_neuron_IDs_subset in subset_neuron_IDs_dict.items():
    #         percent_laps_track_identity_estimated_correctly: float = (round(_out_subset_decode_results_track_id_correct_performance_dict[a_subset_name], ndigits=5) * 100.0)
    #         print(f'aclu subset: "{a_subset_name}"\n\ta_neuron_IDs_subset: {a_neuron_IDs_subset}\n\tpercent_laps_track_identity_estimated_correctly: {percent_laps_track_identity_estimated_correctly} %')
            


    #     # ==================================================================================================================== #
    #     # Process Outputs to get marginals                                                                                     #
    #     # ==================================================================================================================== #
    #     print(f'\t computing time_bin marginal for context: {curr_session_context}...')

    #     #TODO 2024-10-09 09:08: - [ ] Could easily do for each set of cells by looping through `_out_subset_decode_results_dict` dict
    #     callback_outputs['subset_decode_results_time_bin_marginals_df_dict'] = None
        
    #     (complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results)  = _out_subset_decode_results_dict['any_decoder'] ## get the result for all cells
        

    #     ## INPUTS: all_directional_laps_filter_epochs_decoder_result
    #     transfer_column_names_list: List[str] = ['maze_id', 'lap_dir', 'lap_id']
    #     TIME_OVERLAP_PREVENTION_EPSILON: float = 1e-12
    #     (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
    #                                                                                                                                                         additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])
    #     laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = laps_directional_marginals_tuple
    #     laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = laps_track_identity_marginals_tuple
    #     non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = laps_non_marginalized_decoder_marginals_tuple
    #     laps_time_bin_marginals_df: pd.DataFrame = all_directional_laps_filter_epochs_decoder_result.build_per_time_bin_marginals_df(active_marginals_tuple=(laps_directional_marginals, laps_track_identity_marginals, non_marginalized_decoder_marginals),
    #                                                                                                                                 columns_tuple=(['P_LR', 'P_RL'], ['P_Long', 'P_Short'], ['long_LR', 'long_RL', 'short_LR', 'short_RL']), transfer_column_names_list=transfer_column_names_list)
    #     laps_time_bin_marginals_df['start'] = laps_time_bin_marginals_df['start'] + TIME_OVERLAP_PREVENTION_EPSILON ## ENSURE NON-OVERLAPPING

    #     ## INPUTS: laps_time_bin_marginals_df
    #     # active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.33333333333333)
    #     active_min_num_unique_aclu_inclusions_requirement = None # must be none for individual `time_bin` periods
    #     filtered_laps_time_bin_marginals_df, active_spikes_df = co_filter_epochs_and_spikes(active_spikes_df=get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=curr_active_pipeline.global_computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values),
    #                                                                     active_epochs_df=laps_time_bin_marginals_df, included_aclus=track_templates.any_decoder_neuron_IDs, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement,
    #                                                                     epoch_id_key_name='lap_individual_time_bin_id', no_interval_fill_value=-1, add_unique_aclus_list_column=True, drop_non_epoch_spikes=True)
    #     callback_outputs['subset_decode_results_time_bin_marginals_df_dict'] = {'filtered_laps_time_bin_marginals_df': filtered_laps_time_bin_marginals_df,
    #                                                                             # 'laps_marginals_df': laps_marginals_df,
    #     }

        
    #     # aclu subset: "any_decoder"
    #     # 	a_neuron_IDs_subset: [  3   5   7   9  10  11  14  15  16  17  19  21  24  25  26  31  32  33  34  35  36  37  41  45  48  49  50  51  53  54  55  56  57  58  59  60  61  62  63  64  66  67  68  69  70  71  73  74  75  76  78  81  82  83  84  85  86  87  88  89  90  92  93  96  98 100 102 107 108]
    #     # 	percent_laps_track_identity_estimated_correctly: 86.02199999999999 %
    #     # aclu subset: "stable_both"
    #     # 	a_neuron_IDs_subset: [  5   7   9  10  17  25  26  31  33  36  41  45  48  49  50  54  55  56  59  61  62  64  66  69  71  75  76  78  83  84  86  88  89  90  92  93  96 107 108]
    #     # 	percent_laps_track_identity_estimated_correctly: 82.796 %
    #     # aclu subset: "stable_neither"
    #     # 	a_neuron_IDs_subset: [16 19 37 60 73 87]
    #     # 	percent_laps_track_identity_estimated_correctly: 58.065 %
    #     # aclu subset: "stable_long"
    #     # 	a_neuron_IDs_subset: [  5   7   9  10  17  25  26  31  32  33  35  36  41  45  48  49  50  53  54  55  56  59  61  62  64  66  68  69  71  74  75  76  78  82  83  84  86  88  89  90  92  93  96 107 108]
    #     # 	percent_laps_track_identity_estimated_correctly: 80.645 %
    #     # aclu subset: "stable_short"
    #     # 	a_neuron_IDs_subset: [  3   5   7   9  10  11  14  15  17  24  25  26  31  33  34  36  41  45  48  49  50  51  54  55  56  57  58  59  61  62  64  66  67  69  71  75  76  78  83  84  85  86  88  89  90  92  93  96 100 102 107 108]
    #     # 	percent_laps_track_identity_estimated_correctly: 82.796 %
    #     # aclu subset: "appearing_or_disappearing"
    #     # 	a_neuron_IDs_subset: [ 3 11 14 15 24 34 35 51 58 67 74 82]
    #     # 	percent_laps_track_identity_estimated_correctly: 75.26899999999999 %
    #     # aclu subset: "appearing"
    #     # 	a_neuron_IDs_subset: [ 3 11 14 15 24 34 51 58 67]
    #     # 	percent_laps_track_identity_estimated_correctly: 76.344 %
    #     # aclu subset: "disappearing"
    #     # 	a_neuron_IDs_subset: [35 74 82]
    #     # 	percent_laps_track_identity_estimated_correctly: 61.29 %

    #     # stability_df

    #     # a_trial_by_trial_result

    #     # # Time-dependent
    #     # long_pf1D_dt, short_pf1D_dt, global_pf1D_dt = long_results.pf1D_dt, short_results.pf1D_dt, global_results.pf1D_dt
    #     # # long_pf2D_dt, short_pf2D_dt, global_pf2D_dt = long_results.pf2D_dt, short_results.pf2D_dt, global_results.pf2D_dt
    #     # global_pf1D_dt: PfND_TimeDependent = global_results.pf1D_dt
    #     # # global_pf2D_dt: PfND_TimeDependent = global_results.pf2D_dt
    #     # _flat_z_scored_tuning_map_matrix, _flat_decoder_identity_arr = a_trial_by_trial_result.build_combined_decoded_epoch_z_scored_tuning_map_matrix() # .shape: (n_epochs, n_neurons, n_pos_bins) 
    #     # modified_directional_active_lap_pf_results_dicts: Dict[types.DecoderName, TrialByTrialActivity] = a_trial_by_trial_result.build_separated_nan_filled_decoded_epoch_z_scored_tuning_map_matrix()
    #     # # _flat_z_scored_tuning_map_matrix

    #     ## OUTPUTS: override_active_neuron_IDs


    #     print(f'\t\t done (success).')

    # except (Exception, AssertionError) as e:
    #     exception_info = sys.exc_info()
    #     err = CapturedException(e, exception_info)
    #     print(f"WARN: encountered exception {err} while performing .compute_and_export_session_trial_by_trial_performance_completion_function(...) - PHASE II\n\tfor curr_session_context: {curr_session_context}")
    #     if self.fail_on_exception:
    #         raise
    #         # raise e.exc
    #     # _out_inst_fr_comps = None
    #     neuron_replay_stats_df = None
    #     pass


    across_session_results_extended_dict['compute_and_export_session_trial_by_trial_performance_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


# ==================================================================================================================== #
# Rank-Order and WCorr                                                                                                 #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['batch', 'rank-order'], input_requires=['rank_order_results.ripple_merged_complete_epoch_stats_df'], output_provides=[], uses=[], used_by=[], creation_date='2024-04-27 21:21', related_items=[])
def export_rank_order_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict, should_save_pkl:bool=False, should_save_CSV:bool=True) -> dict:
    """
    provides_files=['']
    
    Unpacking:	
        callback_outputs = _across_session_results_extended_dict['export_rank_order_results_completion_function']
        merged_complete_ripple_epoch_stats_df_output_path = callback_outputs['merged_complete_ripple_epoch_stats_df_output_path']
        minimum_inclusion_fr_Hz = callback_outputs['minimum_inclusion_fr_Hz']
        included_qclu_values = callback_outputs['included_qclu_values']
        print(f'merged_complete_ripple_epoch_stats_df_output_path: {merged_complete_ripple_epoch_stats_df_output_path}')

    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import save_rank_order_results
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor

    # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'export_rank_order_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
    session_name: str = curr_active_pipeline.session_name
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    
    assert self.collected_outputs_path.exists()
    # curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    complete_session_identifier_string: str = curr_active_pipeline.get_complete_session_identifier_string() # 'kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]'
    custom_replay_source: str = curr_active_pipeline.get_session_additional_parameters_context().to_dict()['epochs_source']
    
    # CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}" # self.BATCH_DATE_TO_USE: '2024-11-15_Lab'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{complete_session_identifier_string}" # self.BATCH_DATE_TO_USE: '2024-11-15_Lab'
    print(f'CURR_BATCH_OUTPUT_PREFIX: "{CURR_BATCH_OUTPUT_PREFIX}"') # CURR_BATCH_OUTPUT_PREFIX: "2024-11-15_Lab-kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]"

    callback_outputs = {
        'pkl_rank_order_output_path': None,'pkl_directional_laps_output_path': None,'pkl_directional_merged_decoders_output_path': None, 'out_filename_str': None,
        'merged_complete_ripple_epoch_stats_df_output_path': None, 
        'minimum_inclusion_fr_Hz': None,
        'included_qclu_values': None,
    }
    

    if should_save_pkl:
        ## Save out pickled results:
        rank_order_output_path, directional_laps_output_path, directional_merged_decoders_output_path, out_filename_str = save_rank_order_results(curr_active_pipeline, day_date=f"{self.BATCH_DATE_TO_USE}", override_output_parent_path=self.collected_outputs_path) # don't pass `CURR_BATCH_OUTPUT_PREFIX` or it will double up the descriptors
        callback_outputs['pkl_rank_order_output_path'] = rank_order_output_path
        callback_outputs['pkl_directional_laps_output_path'] = directional_laps_output_path
        callback_outputs['pkl_directional_merged_decoders_output_path'] = directional_merged_decoders_output_path
        callback_outputs['out_filename_str'] = out_filename_str
    
    # save_rank_order_results(curr_active_pipeline, day_date=f"{CURR_BATCH_OUTPUT_PREFIX}", override_output_parent_path=self.collected_outputs_path) # "2024-01-02_301pm" "2024-01-02_734pm""

    ## 2023-12-21 - Export to CSV:
    # spikes_df = curr_active_pipeline.sess.spikes_df
    rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
    minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
    included_qclu_values: List[int] = rank_order_results.included_qclu_values
    callback_outputs['minimum_inclusion_fr_Hz'] = minimum_inclusion_fr_Hz
    callback_outputs['included_qclu_values'] = included_qclu_values

    # ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
    # directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    # track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
    # print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
    # print(f'included_qclu_values: {included_qclu_values}')
    if should_save_CSV:
        print(f'\t try saving to CSV...')
        # active_csv_parent_output_path = curr_active_pipeline.get_output_path().resolve()
        active_csv_parent_output_path = self.collected_outputs_path.resolve()
        merged_complete_epoch_stats_df: pd.DataFrame = rank_order_results.ripple_merged_complete_epoch_stats_df ## New method
        ## add the missing context columns, like session, time_bin_size, etc.
        merged_complete_epoch_stats_df = merged_complete_epoch_stats_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=None, custom_replay_source=custom_replay_source, curr_session_t_delta=t_delta, time_col='start')

        merged_complete_ripple_epoch_stats_df_output_path = active_csv_parent_output_path.joinpath(f'{CURR_BATCH_OUTPUT_PREFIX}-(merged_complete_epoch_stats_df).csv').resolve()
        merged_complete_epoch_stats_df.to_csv(merged_complete_ripple_epoch_stats_df_output_path)
        print(f'\t saving to CSV: "{merged_complete_ripple_epoch_stats_df_output_path}" done.')
        callback_outputs['merged_complete_ripple_epoch_stats_df_output_path'] = str(merged_complete_ripple_epoch_stats_df_output_path.as_posix())
        # across_session_results_extended_dict['merged_complete_epoch_stats_df'] = merged_complete_ripple_epoch_stats_df_output_path

    across_session_results_extended_dict['export_rank_order_results_completion_function'] = callback_outputs
    

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


@function_attributes(short_name=None, tags=['wcorr', 'shuffle', 'export', 'mat', 'pkl', 'csv'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def compute_and_export_session_wcorr_shuffles_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                   should_skip_previous_saved_shuffles:bool=False, with_data_name: Optional[str]=None) -> dict:
    """  Computes the shuffled wcorrs and export them to several formats: .mat, .pkl, and .csv
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import compute_and_export_session_wcorr_shuffles_completion_function
    
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

    def _subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
        """ captures `curr_active_pipeline`
        """
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(rounded_minutes=10), data_identifier_str=data_identifier_str,
                                                                                                                 parent_output_path=parent_output_path, out_extension='.csv')
        export_df.to_csv(out_path)
        return out_path 
    
    custom_export_df_to_csv_fn = _subfn_custom_export_df_to_csv
    
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_wcorr_shuffles_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    # desired_total_num_shuffles: int = 4096
    desired_total_num_shuffles: int = 700
    minimum_required_unique_num_shuffles: int = 700

    newer_than: Optional[datetime] = datetime(2024, 11, 11)

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
    
    if not should_skip_previous_saved_shuffles:
        # try load previous compatible shuffles: _____________________________________________________________________________ #
        wcorr_shuffles.discover_load_and_append_shuffle_data_from_directory(save_directory=curr_active_pipeline.get_output_path().resolve(), newer_than=newer_than, with_data_name=with_data_name)

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
        # active_context = curr_active_pipeline.get_session_context()
        active_context, (curr_session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context() ## #TODO 2024-11-08 10:29: - [ ] complete instead of basic session context
        # session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
        session_name: str = curr_active_pipeline.session_name
        export_files_dict = wcorr_shuffles.export_csvs(parent_output_path=self.collected_outputs_path.resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=curr_active_pipeline, custom_export_df_to_csv_fn=custom_export_df_to_csv_fn) # "2024-11-15_0200PM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays_qclu_[1, 2, 4, 6, 7, 9]_frateThresh_5.0-(ripple_WCorrShuffle_df)_tbin-0.025.csv"
        ripple_WCorrShuffle_df_export_CSV_path = export_files_dict['ripple_WCorrShuffle_df']
        print(f'Successfully exported ripple_WCorrShuffle_df_export_CSV_path: "{ripple_WCorrShuffle_df_export_CSV_path}" with wcorr_shuffles.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles} unique shuffles.')
        callback_outputs['ripple_WCorrShuffle_df_export_CSV_path'] = ripple_WCorrShuffle_df_export_CSV_path
    except Exception as e:
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

    except Exception as e:
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
    except Exception as e:
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


@function_attributes(short_name=None, tags=['wcorr', 'shuffle', 'replay', 'epochs', 'alternative_replays'], input_requires=[], output_provides=[], uses=['compute_all_replay_epoch_variations', 'BatchCompletionHelpers.overwrite_replay_epochs_and_recompute'], used_by=[], creation_date='2024-06-28 01:50', related_items=[])
def compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                                      included_qclu_values = [1,2,4,6,7,9], minimum_inclusion_fr_Hz=5.0, ripple_decoding_time_bin_size: float = 0.025, num_wcorr_shuffles: int = 2048, drop_previous_result_and_compute_fresh:bool=True, enable_plot_wcorr_hist_figure:bool=False) -> dict:
    """  Computes several different alternative replay-detection variants and computes and exports the shuffled wcorrs for each of them
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

    2024-11-08 12:25 - output filenames seem correct and don't have duplicated parameter strings

    
    
    """
    import sys
    import numpy as np
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import SequenceBasedComputationsContainer, WCorrShuffle
    from neuropy.utils.mixins.indexing_helpers import get_dict_subset
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
    from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows

    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_completion_helpers import BatchCompletionHelpers
    
    base_BATCH_DATE_TO_USE: str = f"{self.BATCH_DATE_TO_USE}" ## backup original string
    should_suppress_errors: bool = (not self.fail_on_exception) # get('fail_on_exception', False)    
        
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    callback_outputs = {
        'custom_suffix': None,
        'replay_epoch_variations': None,
        'replay_epoch_outputs': None,
        'included_qclu_values': included_qclu_values,
        'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
        'ripple_decoding_time_bin_size': ripple_decoding_time_bin_size,
        'num_wcorr_shuffles': num_wcorr_shuffles,
        'drop_previous_result_and_compute_fresh': drop_previous_result_and_compute_fresh,
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

    replay_epoch_variations = BatchCompletionHelpers.compute_all_replay_epoch_variations(curr_active_pipeline, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

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

        custom_suffix: str = BatchCompletionHelpers._get_custom_suffix_for_replay_filename(new_replay_epochs=a_replay_epochs)
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

        custom_suffix: str = BatchCompletionHelpers._get_custom_suffix_for_replay_filename(new_replay_epochs=a_replay_epochs) # looks right
        print(f'\treplay_epochs_key: {replay_epochs_key}: custom_suffix: "{custom_suffix}"')

        ## Modify .BATCH_DATE_TO_USE to include the custom suffix
        # curr_BATCH_DATE_TO_USE: str = f"{base_BATCH_DATE_TO_USE}{custom_suffix}"

        curr_BATCH_DATE_TO_USE: str = f"{base_BATCH_DATE_TO_USE}"
        print(f'\tcurr_BATCH_DATE_TO_USE: "{curr_BATCH_DATE_TO_USE}"')
        self.BATCH_DATE_TO_USE = curr_BATCH_DATE_TO_USE # set the internal BATCH_DATE_TO_USE which is used to determine the .csv and .h5 export names
        # self.BATCH_DATE_TO_USE = '2024-11-01_Apogee'
        print(f'\tWARNING: should_suppress_errors: {should_suppress_errors}')
        with ExceptionPrintingContext(suppress=should_suppress_errors, exception_print_fn=(lambda formatted_exception_str: print(f'\tfailed epoch computations for replay_epochs_key: "{replay_epochs_key}". Failed with error: {formatted_exception_str}. Skipping.'))):
            # for replay_epochs_key, a_replay_epochs in replay_epoch_variations.items():
            a_curr_active_pipeline = deepcopy(curr_active_pipeline)
            did_change, custom_save_filenames, custom_save_filepaths = BatchCompletionHelpers.overwrite_replay_epochs_and_recompute(curr_active_pipeline=a_curr_active_pipeline, new_replay_epochs=a_replay_epochs,
                                                                                                              enable_save_pipeline_pkl=True, enable_save_global_computations_pkl=False, enable_save_h5=False,
                                                                                                              num_wcorr_shuffles=num_wcorr_shuffles,
                                                                                                              user_completion_dummy=self, drop_previous_result_and_compute_fresh=drop_previous_result_and_compute_fresh, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)
            # 'ripple_h5_out_path': WindowsPath('K:/scratch/collected_outputs/2024-11-18_Lab-2006-6-09_1-22-43-_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_withNormalComputedReplays-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]_time_bin_size_sweep_results.h5'),
            # 'ripple_csv_out_path': WindowsPath('K:/scratch/collected_outputs/2024-11-18-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_withNormalComputedReplays-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(ripple_marginals_df).csv'),
            # 'ripple_csv_time_bin_marginals': WindowsPath('K:/scratch/collected_outputs/2024-11-18-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_withNormalComputedReplays-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(ripple_time_bin_marginals_df).csv'),
            # 'laps_csv_out_path': WindowsPath('K:/scratch/collected_outputs/2024-11-18-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_withNormalComputedReplays-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(laps_marginals_df).csv'),
            # 'laps_csv_time_bin_marginals_out_path': WindowsPath('K:/scratch/collected_outputs/2024-11-18-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_1.0_withNormalComputedReplays-frateThresh_1.0-qclu_[1, 2, 4, 6, 7, 9]-(laps_time_bin_marginals_df).csv'),

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
            def a_subfn_custom_export_df_to_csv(export_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)', parent_output_path: Path=None):
                """ captures `a_curr_active_pipeline`
                """
                out_path, out_filename, out_basename = a_curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(rounded_minutes=10), data_identifier_str=data_identifier_str,
                                                                                                                        parent_output_path=parent_output_path, out_extension='.csv')
                export_df.to_csv(out_path)
                return out_path 
            

            #TODO 2024-11-19 03:33: - [ ] `custom_export_df_to_csv_fn`
            custom_export_df_to_csv_fn = a_subfn_custom_export_df_to_csv
            
            #TODO 2024-11-20 06:16: - [ ] I feel like these need the time_bin_size as a suffix
            # standalone save (2024-11-19 03:40 Identical to `vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_completion_helpers.py:806`
            out_path, standalone_pkl_filename, standalone_pkl_filepath = a_curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(), data_identifier_str='standalone_wcorr_ripple_shuffle_data_only', parent_output_path=a_curr_active_pipeline.get_output_path(), out_extension='.pkl', suffix_string=f'_{wcorr_shuffles.n_completed_shuffles}')		
            _prev_standalone_pkl_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_wcorr_ripple_shuffle_data_only_{wcorr_shuffles.n_completed_shuffles}.pkl' 
            if _prev_standalone_pkl_filename != standalone_pkl_filename:
                print(f'standalone_pkl_filename:\n\t"{standalone_pkl_filename}"')
                print(f'_prev_standalone_pkl_filename:\n\t"{_prev_standalone_pkl_filename}"')
            # standalone_pkl_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_pkl_filename).resolve() # Path("W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\2024-05-30_0925AM_standalone_wcorr_ripple_shuffle_data_only_1100.pkl")
            print(f'saving to "{standalone_pkl_filepath}"...')
            wcorr_shuffles.save_data(standalone_pkl_filepath)
            ## INPUTS: wcorr_ripple_shuffle
            _prev_standalone_mat_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_all_shuffles_wcorr_array.mat' 
            # standalone_mat_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_mat_filename).resolve() # r"W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43\output\2024-06-03_0400PM_standalone_all_shuffles_wcorr_array.mat"
            out_path, standalone_mat_filename, standalone_mat_filepath = a_curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_rounded_time_str(), data_identifier_str='standalone_all_shuffles_wcorr_array', parent_output_path=a_curr_active_pipeline.get_output_path(), out_extension='.mat')
            if _prev_standalone_mat_filename != standalone_mat_filename:
                print(f'standalone_mat_filepath:\n\t"{standalone_mat_filepath}"')
                print(f'_prev_standalone_mat_filename:\n\t"{_prev_standalone_mat_filename}"')
            #TODO 2024-11-19 03:25: - [ ] Previously used `custom_suffix`
            wcorr_shuffles.save_data_mat(filepath=standalone_mat_filepath, **{'session': a_curr_active_pipeline.get_session_context().to_dict()})
            
            replay_epoch_outputs[replay_epochs_key].update(dict(standalone_pkl_filepath=standalone_pkl_filepath, standalone_mat_filepath=standalone_mat_filepath))

            try:
                active_context = a_curr_active_pipeline.get_session_context()
                session_name: str = f"{a_curr_active_pipeline.session_name}{custom_suffix}" ## appending this here is a hack, but it makes the correct filename
                # complete_session_context, (curr_session_context,  additional_session_context) = a_curr_active_pipeline.get_complete_session_context()
                # active_context = complete_session_context
                active_context = active_context.adding_context_if_missing(suffix=custom_suffix)

                export_files_dict = wcorr_shuffles.export_csvs(parent_output_path=a_curr_active_pipeline.get_output_path().resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=a_curr_active_pipeline,
                                                            #    source='diba_evt_file',
                                                                source='compute_diba_quiescent_style_replay_events',
                                                                custom_export_df_to_csv_fn=custom_export_df_to_csv_fn,
                                                                )
                ripple_WCorrShuffle_df_export_CSV_path = export_files_dict['ripple_WCorrShuffle_df']
                print(f'Successfully exported ripple_WCorrShuffle_df_export_CSV_path: "{ripple_WCorrShuffle_df_export_CSV_path}" with wcorr_shuffles.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles} unique shuffles.')
                replay_epoch_outputs[replay_epochs_key].update(dict(active_context=active_context, export_files_dict=export_files_dict))
                # callback_outputs['ripple_WCorrShuffle_df_export_CSV_path'] = ripple_WCorrShuffle_df_export_CSV_path
            except Exception as e:
                raise
                
            # wcorr_ripple_shuffle.discover_load_and_append_shuffle_data_from_directory(save_directory=curr_active_pipeline.get_output_path().resolve())
            # active_context = curr_active_pipeline.get_session_context()
            # session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
            # session_name: str = curr_active_pipeline.session_name
            # export_files_dict = wcorr_ripple_shuffle.export_csvs(parent_output_path=collected_outputs_path.resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=curr_active_pipeline)
            # export_files_dict

            ## FINAL STAGE: generate histogram:
            if enable_plot_wcorr_hist_figure:
                ## INPUTS: wcorr_ripple_shuffle_all_df, wcorr_ripple_shuffle_all_df, custom_suffix
                plot_var_name: str = 'abs_best_wcorr'
                a_fig_context = a_curr_active_pipeline.build_display_context_for_session(display_fn_name='replay_wcorr', custom_suffix=custom_suffix)
                params_description_str: str = " | ".join([f"{str(k)}:{str(v)}" for k, v in get_dict_subset(a_replay_epochs.metadata, subset_excludelist=['qclu_included_aclus']).items()])
                footer_annotation_text = f'{a_curr_active_pipeline.get_session_context()}<br>{params_description_str}'

                fig = BatchCompletionHelpers.plot_replay_wcorr_histogram(df=wcorr_ripple_shuffle_all_df, plot_var_name=plot_var_name,
                        all_shuffles_only_best_decoder_wcorr_df=all_shuffles_only_best_decoder_wcorr_df, footer_annotation_text=footer_annotation_text)

                # Save figure to disk:
                out_hist_fig_result = a_curr_active_pipeline.output_figure(a_fig_context, fig=fig)
                
                # Show the figure
                # fig.show()
                
            else:
                ## disable plotting the histogram:
                params_description_str = None
                footer_annotation_text = None
                out_hist_fig_result = None
            
            replay_epoch_outputs[replay_epochs_key].update(dict(params_description_str=params_description_str, footer_annotation_text=footer_annotation_text, out_hist_fig_result=out_hist_fig_result))


        ## end error handler

    # END FOR
    ## restore original base_BATCH_DATE_TO_USE
    self.BATCH_DATE_TO_USE = base_BATCH_DATE_TO_USE

    callback_outputs = {
        'custom_suffix': custom_suffix,
        'replay_epoch_variations': deepcopy(replay_epoch_variations),
        'replay_epoch_outputs': deepcopy(replay_epoch_outputs),
        'included_qclu_values': included_qclu_values,
        'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
        'ripple_decoding_time_bin_size': ripple_decoding_time_bin_size,
        'num_wcorr_shuffles': num_wcorr_shuffles,
        'drop_previous_result_and_compute_fresh': drop_previous_result_and_compute_fresh,
        #  'wcorr_shuffles_data_output_filepath': wcorr_shuffles_data_standalone_filepath, 'e':err, #'t_end': t_end   
        #  'standalone_pkl_filepath': standalone_pkl_filepath,
        #  'standalone_MAT_filepath': standalone_MAT_filepath,
        #  'ripple_WCorrShuffle_df_export_CSV_path': ripple_WCorrShuffle_df_export_CSV_path,
    }

    across_session_results_extended_dict['compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function'] = callback_outputs

    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


# ==================================================================================================================== #
# cell_first_spikes_characteristics                                                                                    #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['first-spikes', 'neurons', 'HDF5', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-01 18:30', related_items=[])
def compute_and_export_cell_first_spikes_characteristics_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """ Exports this session's cell first-firing information (HDF5) with custom suffix derived from parameters
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import compute_and_export_cell_first_spikes_characteristics_completion_function
    
    
    #TODO 2024-11-01 21:17: - [ ] need to export those globally unique identifiers for each aclu within a session, look at other user fcns for inspiration.
    #TODO 2024-11-01 21:17: - [ ] Need to stanardize the output filename so that it can be parsed by `pyphoplacecellanalysis.SpecificResults.AcrossSessionResults.find_most_recent_files`. I think it might just need parens
        "K:/scratch/collected_outputs/kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]_first_spike_activity_data.h5",
        "K:/scratch/collected_outputs/kdiba-gor01-one-2006-6-12_15-55-31__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]_first_spike_activity_data.h5",
        "K:/scratch/collected_outputs/kdiba-vvp01-one-2006-4-10_12-25-50__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]_first_spike_activity_data.h5",
        "K:/scratch/collected_outputs/kdiba-vvp01-two-2006-4-09_16-40-54__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]_first_spike_activity_data.h5",
        "K:/scratch/collected_outputs/kdiba-pin01-one-11-03_12-3-25__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]_first_spike_activity_data.h5",
        "K:/scratch/collected_outputs/kdiba-vvp01-two-2006-4-10_12-58-3__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]_first_spike_activity_data.h5",
        "K:/scratch/collected_outputs/kdiba-gor01-two-2006-6-12_16-53-46__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]_first_spike_activity_data.h5",

    """
    import sys
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphocorehelpers.print_helpers import get_now_day_str
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes


    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_cell_first_spikes_characteristics_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    assert self.collected_outputs_path.exists()
    collected_outputs = self.collected_outputs_path.resolve()	
    print(f'collected_outputs: {collected_outputs}')

     # 'kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]'
    # complete_session_identifier_string: str = curr_active_pipeline.get_complete_session_identifier_string()
    # session_identifier_str: str = active_context.get_description()
    # hdf5_out_path, out_filename, out_basename = get_export_name(data_identifier_str="(first_spike_activity_data)", parent_output_path=collected_outputs, session_identifier_str=complete_session_identifier_string, out_extension='.h5')
    hdf5_out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=get_now_day_str(), data_identifier_str="(first_spike_activity_data)", parent_output_path=collected_outputs, out_extension='.h5')

    was_write_good: bool = False
    try:
        # all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict), hdf5_out_path = CellsFirstSpikeTimes.compute_cell_first_firings(curr_active_pipeline, hdf_save_parent_path=collected_outputs)	
        _obj: CellsFirstSpikeTimes = CellsFirstSpikeTimes.init_from_pipeline(curr_active_pipeline=curr_active_pipeline, hdf_save_parent_path=None, should_include_only_spikes_after_initial_laps=False)
        _obj.save_to_hdf5(hdf_save_path=hdf5_out_path)
        was_write_good = True

    except Exception as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"ERROR: encountered exception {err} while trying to export the first_firings for {curr_session_context}")
        if self.fail_on_exception:
            raise

    callback_outputs = {
        'hdf5_out_path': hdf5_out_path,
        #  'all_cells_first_spike_time_df': all_cells_first_spike_time_df,
        'was_write_good': was_write_good,
    }
    across_session_results_extended_dict['compute_and_export_cell_first_spikes_characteristics_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['first-spikes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-07 21:31', related_items=[])
def figures_plot_cell_first_spikes_characteristics_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict, later_appearing_cell_lap_start_id: int=4) -> dict:
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode	
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes

    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')
    
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'figures_plot_cell_first_spikes_characteristics_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    custom_figure_output_path = self.collected_outputs_path
    custom_fig_man: FileOutputManager = FileOutputManager(figure_output_location=FigureOutputLocation.CUSTOM, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE, override_output_parent_path=custom_figure_output_path)
    # test_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='display_long_short_laps')
    # custom_fig_man.get_figure_save_file_path(test_context, make_folder_if_needed=False)
    cells_first_spike_times: CellsFirstSpikeTimes = CellsFirstSpikeTimes.init_from_pipeline(curr_active_pipeline, hdf_save_parent_path=self.collected_outputs_path, should_include_only_spikes_after_initial_laps=False)
    later_lap_appearing_aclus = cells_first_spike_times.all_cells_first_spike_time_df[cells_first_spike_times.all_cells_first_spike_time_df['lap_spike_lap'] > later_appearing_cell_lap_start_id]['aclu'].unique() ## only get
    if later_lap_appearing_aclus is not None and (len(later_lap_appearing_aclus) > 0):
        # later_lap_appearing_aclus = [32, 33,34, 35, 62, 67]
        # later_lap_appearing_aclus = [62]
        filtered_cells_first_spike_times: CellsFirstSpikeTimes = cells_first_spike_times.sliced_by_neuron_id(later_lap_appearing_aclus)
        later_lap_appearing_aclus_df = filtered_cells_first_spike_times.all_cells_first_spike_time_df ## find ones that appear only on later laps
        later_lap_appearing_aclus = np.unique(later_lap_appearing_aclus_df['aclu'].to_numpy()) ## get the aclus that only appear on later laps
        print(f'later_lap_appearing_aclus: {later_lap_appearing_aclus}')
        later_lap_appearing_figures_dict = filtered_cells_first_spike_times.plot_PhoJonathan_plots_with_time_indicator_lines(curr_active_pipeline, included_neuron_ids=later_lap_appearing_aclus, n_max_page_rows=16,
                                                                                                                        write_vector_format=False, write_png=True, override_fig_man=custom_fig_man)

    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # return True
    return across_session_results_extended_dict

# ==================================================================================================================== #
# Unsorted                                                                                                             #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['recomputed_inst_firing_rate', 'inst_fr', 'independent'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def compute_and_export_session_instantaneous_spike_rates_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                            #  instantaneous_time_bin_size_seconds_list:List[float]=[0.0005, 0.0009, 0.0015, 0.0025, 0.025], epoch_handling_mode:str='DropShorterMode',
                                                                            instantaneous_time_bin_size_seconds_list:List[float]=[1000.0], minimum_inclusion_fr_Hz: Optional[float]=0.0, epoch_handling_mode:str='UseAllEpochsMode', # single-bin per epoch
                                                                            save_hdf:bool=True, save_pickle:bool=True, save_across_session_hdf:bool=False, save_FAT_csv:bool=False, 
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
    
    inst_fr_comps: InstantaneousSpikeRateGroupsComputation = callback_outputs['recomputed_inst_fr_time_bin_dict'][1000.0]['inst_fr_comps']
    
    
    subfn_callback_outputs['inst_fr_comps'] 
    
    """
    import sys
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphocorehelpers.assertion_helpers import Assert
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import InstantaneousFiringRatesDataframeAccessor

    # Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]
    callback_outputs = {
        'recomputed_inst_fr_time_bin_dict': None,
    }
    
    active_export_parent_output_path: Path = self.collected_outputs_path.resolve()
    Assert.path_exists(active_export_parent_output_path)
    ## OUTPUTS: active_export_parent_output_path
        
    def _subfn_single_time_bin_size_compute_and_export_session_instantaneous_spike_rates_completion_function(instantaneous_time_bin_size_seconds:float):
        """ Captures: active_export_parent_output_path, curr_session_context, ...
        """
        print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f'compute_and_export_session_instantaneous_spike_rates_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, instantaneous_time_bin_size_seconds: {instantaneous_time_bin_size_seconds}, ...)')
        
        subfn_callback_outputs = {
            'inst_fr_comps': None, # the actual InstantaneousSpikeRateGroupsComputation
            'recomputed_inst_fr_comps_filepath': None, #'t_end': t_end   
            'recomputed_inst_fr_comps_h5_filepath': None,
            'recomputed_inst_fr_comps_FAT_CSV_filepath': None,
            'common_across_session_h5': None,
            
        }
        err = None

        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_recomputed_inst_fr_comps: InstantaneousSpikeRateGroupsComputation = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=instantaneous_time_bin_size_seconds) # 3ms, 10ms
            _out_recomputed_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context(), epoch_handling_mode=epoch_handling_mode, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)

            # LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            # LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus

            if _out_recomputed_inst_fr_comps is not None:
                subfn_callback_outputs['inst_fr_comps'] = _out_recomputed_inst_fr_comps            

            print(f'\t\t done (success).')

        except Exception as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"WARN: compute_and_export_session_instantaneous_spike_rates_completion_function: encountered exception {err} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            # if self.fail_on_exception:
            #     raise e.exc
            # _out_inst_fr_comps = None
            _out_recomputed_inst_fr_comps = None
            pass


        ## 2025-07-17 - FAT_df CSV saving:
        if (_out_recomputed_inst_fr_comps is not None) and save_FAT_csv:
            ## FAT_csv Saving:
            ## Export to CSVs:
            # FAT_df: pd.DataFrame = _out_recomputed_inst_fr_comps.get_comprehensive_dataframe()

            recomputed_inst_fr_comps_FAT_CSV_filepath = None
            try:
                _csv_save_paths_dict = _out_recomputed_inst_fr_comps.export_as_FAT_df_CSV(active_export_parent_output_path=active_export_parent_output_path, owning_pipeline_reference=curr_active_pipeline, decoding_time_bin_size=_out_recomputed_inst_fr_comps.instantaneous_time_bin_size_seconds)
                recomputed_inst_fr_comps_FAT_CSV_filepath = list(_csv_save_paths_dict.values())[0]
                print(f'recomputed_inst_fr_comps_FAT_CSV_filepath: "{recomputed_inst_fr_comps_FAT_CSV_filepath}"\n')
                was_write_good = True
                subfn_callback_outputs['recomputed_inst_fr_comps_FAT_CSV_filepath'] = deepcopy(recomputed_inst_fr_comps_FAT_CSV_filepath)

            except Exception as e:
                exception_info = sys.exc_info()
                err = CapturedException(e, exception_info)
                print(f"ERROR: encountered exception {err} while trying to perform _out_recomputed_inst_fr_comps.export_as_FAT_df_CSV(...) for {curr_session_context}")
                recomputed_inst_fr_comps_FAT_CSV_filepath = None # set to None because it failed.
                if self.fail_on_exception:
                    raise err.exc
        else:
            recomputed_inst_fr_comps_FAT_CSV_filepath = None

        subfn_callback_outputs['recomputed_inst_fr_comps_FAT_CSV_filepath'] = recomputed_inst_fr_comps_FAT_CSV_filepath




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

            except Exception as e:
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

            except Exception as e:
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
            ## #TODO 2025-06-11 06:39: - [ ] NOT THREAD/PARALLEL SAFE because they access the same output file
            common_across_session_h5_filename: str = f'{get_now_day_str()}_across_session_recomputed_inst_fr_comps.h5'
            common_file_path = self.collected_outputs_path.resolve().joinpath(common_across_session_h5_filename).resolve()
            print(f'common_file_path: "{common_file_path}"')

            try:
                InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(inst_fr_comps=_out_recomputed_inst_fr_comps, curr_active_pipeline=curr_active_pipeline, common_file_path=common_file_path)
                subfn_callback_outputs['common_across_session_h5'] = common_file_path

            except Exception as e:
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



# ==================================================================================================================== #
# Utility/Helpers                                                                                                      #
# ==================================================================================================================== #

from neuropy.utils.mixins.binning_helpers import safe_limit_num_grid_bin_values
from neuropy.core.session.Formats.SessionSpecifications import SessionConfig
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
from neuropy.core.user_annotations import UserAnnotationsManager

# ==================================================================================================================== #
# NOTE: LOCAL CLASS DEFNITION                                                                                          #
# ==================================================================================================================== #
@metadata_attributes(short_name=None, tags=['grid_bin_bounds', 'grid_bin', 'FIXUP', 'post-hoc'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-13 12:33', related_items=['kdiba_session_post_fixup_completion_function'])
class PostHocPipelineFixup:
    """ Fixes the grid_bin_bounds, grid_bin, track_limits, and some other properties and recomputes if needed.
            
    #TODO 2025-02-13 12:35: - [ ] Format `FINAL_FIX_GRID_BIN_BOUNDS` as a user_ function, replace `kdiba_session_post_fixup_completion_function` with it.
    Inspired by `kdiba_session_post_fixup_completion_function`, but does additional things, and performs needed recomputes.
            
    Usage:
        from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import PostHocPipelineFixup
        
        (did_any_change, change_dict), correct_grid_bin_bounds = PostHocPipelineFixup.FINAL_FIX_GRID_BIN_BOUNDS(curr_active_pipeline=curr_active_pipeline, is_dry_run=True)
    
    """
    across_session_results_extended_dict_data_name: str = 'PostHocPipelineFixup' # like 'position_info_mat_reload_completion_function'

    # ==================================================================================================================== #
    # `grid_bin_bounds` and `grid_bin`                                                                                     #
    # ==================================================================================================================== #
    @classmethod
    def find_percent_pos_samples_within_grid_bin_bounds(cls, pos_df: pd.DataFrame, grid_bin_bounds):
        """ sanity-checks the grid_bin_bounds against the pos_df to see what percent of positions fall within the bounds
        
        percentage_within_ranges, filtered_df = find_percent_pos_samples_within_grid_bin_bounds(pos_df=pos_df, grid_bin_bounds=correct_grid_bin_bounds)
        
        percentage_within_ranges, filtered_df = find_percent_pos_samples_within_grid_bin_bounds(pos_df=pos_df, grid_bin_bounds=((0.0, 287.7697841726619), (115.10791366906477, 172.66187050359713)))

        percentage_within_ranges, filtered_df = find_percent_pos_samples_within_grid_bin_bounds(pos_df=pos_df, grid_bin_bounds=((37.0773897438341, 250.69004399129707), (107.8177789584226, 113.7570079192343)))

        """
        (xmin, xmax), (ymin, ymax) = grid_bin_bounds
        pos_df = pos_df
        # Filter the DataFrame for rows where 'x' and 'y' are within their respective ranges
        filtered_df = pos_df[(pos_df['x'] >= xmin) & (pos_df['x'] <= xmax) & 
                        (pos_df['y'] >= ymin) & (pos_df['y'] <= ymax)]

        # Calculate the percentage of rows within both ranges
        percentage_within_ranges = (len(filtered_df) / len(pos_df)) * 100
        print(f'percentage_within_ranges: {percentage_within_ranges}')
        return percentage_within_ranges, filtered_df

    @function_attributes(short_name=None, tags=['active', 'fix', 'grid_bin_bounds'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-12 03:54', related_items=[])
    @classmethod
    def get_hardcoded_known_good_grid_bin_bounds(cls, curr_active_pipeline):
        """ gets the actually correct grid_bin_bounds, fixing months worth of problems
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import get_hardcoded_known_good_grid_bin_bounds
            correct_grid_bin_bounds = get_hardcoded_known_good_grid_bin_bounds(curr_active_pipeline)
            correct_grid_bin_bounds

        """
        a_session_context = curr_active_pipeline.get_session_context() # IdentifyingContext.try_init_from_session_key(session_str=a_session_uid, separator='|')
        correct_grid_bin_bounds = UserAnnotationsManager.get_hardcoded_specific_session_override_dict().get(a_session_context, {}).get('grid_bin_bounds', None)
        assert correct_grid_bin_bounds is not None, f"session: {a_session_context} was not found in overrides!"
        return deepcopy(correct_grid_bin_bounds) ## returns the correct grid_bin_bounds for the pipeline


    @function_attributes(short_name=None, tags=['IMPORTANT', 'hardcoded', 'override', 'grid_bin_bounds'], input_requires=[], output_provides=[], uses=['safe_limit_num_grid_bin_values'], used_by=[], creation_date='2025-02-12 08:17', related_items=[])
    @classmethod
    def HARD_OVERRIDE_grid_bin_bounds(cls, curr_active_pipeline, hard_manual_override_grid_bin_bounds = ((0.0, 287.7697841726619), (80.0, 200.0)), desired_grid_bin = (2.0, 2.0), max_allowed_num_bins=(60, 9), is_dry_run: bool=False):
        """ manually overrides the `grid_bin_bounds` and `grid_bin` in all places needed to ensure they are correct. 
            
        did_any_change, change_dict = HARD_OVERRIDE_grid_bin_bounds(curr_active_pipeline, hard_manual_override_grid_bin_bounds = ((0.0, 287.7697841726619), (80.0, 200.0)))
        change_dict
        """            
        active_data_mode_name: str = curr_active_pipeline.session_data_type
        active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
        active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
        # active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

        if is_dry_run:
            print(f'NOTE: HARD_OVERRIDE_grid_bin_bounds(...): is_dry_run == True, so changes will be determined but not applied!')

        change_dict = {}


        def _sub_sub_fn_did_potentially_arr_or_None_variable_change(_old_val, new_val) -> bool:
            """ compares two variables, accounting for Nones, list-likes, and more """
            # For array/tuple type config values:
            if (_old_val is None) and (new_val is None):
                return False
            elif (_old_val is None) and (new_val is not None):
                return True
            elif (_old_val is not None) and (new_val is None):
                return True
            else:
                ## both are not None
                assert (_old_val is not None) and (new_val is not None)
                if (isinstance(_old_val, (np.ndarray, tuple, list)) or isinstance(new_val, (np.ndarray, tuple, list))):
                    # Convert to numpy arrays if needed
                    _old_val_array = np.array(deepcopy(_old_val))
                    new_val_array = np.array(new_val)
                    return (not np.all(np.isclose(_old_val_array, new_val_array)))
                else:
                    # For non-array types, use direct comparison
                    will_change: bool = (_old_val != new_val) # Python's != operator returns True when comparing None with any non-None value
                    ## warn for direct comparison
                    if will_change:
                        print(f'DEBUGWARN - direct comparison used for variables \n\tnew_val: {new_val} and \n\t_old_val: {_old_val}. and found (will_change == True)')
                    return will_change
        


        def _subfn_update_session_config(a_session, is_dry_run: bool=False):
            """ captures: curr_active_pipeline, hard_manual_override_grid_bin_bounds, change_dict
            """
            did_any_change: bool = False
            
            # sess_config: SessionConfig = SessionConfig(**deepcopy(a_session.config.__getstate__()))
            a_session_context = a_session.get_context() # IdentifyingContext.try_init_from_session_key(session_str=a_session_uid, separator='|')
            a_session_override_dict = UserAnnotationsManager.get_hardcoded_specific_session_override_dict().get(a_session_context, {})
            
            allowed_sess_Config_override_keys = ['pix2cm', 'real_unit_grid_bin_bounds', 'real_cm_grid_bin_bounds', 'grid_bin_bounds', 'grid_bin', 'track_start_t', 'track_end_t']

            # loaded_track_limits ________________________________________________________________________________________________ #
            if not is_dry_run:
                sess_config: SessionConfig = deepcopy(a_session.config)
                # 'first_valid_pos_time'
                a_session.config = sess_config
                _bak_loaded_track_limits = deepcopy(a_session.config.loaded_track_limits)
                ## Apply fn
                a_session = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=a_session) ## does this reset the overrides?
                ## do we want the UserAnnotations here?
                
                _new_loaded_track_limits = deepcopy(a_session.config.loaded_track_limits)
                # did_change: bool = ((_bak_loaded_track_limits is None) or (_new_loaded_track_limits != _bak_loaded_track_limits))
                # change_dict[f'filtered_sessions["{a_decoder_name}"]'] = {}
                # did_loaded_track_limits_change: bool = ((_bak_loaded_track_limits is None) or np.any([np.array(_bak_loaded_track_limits.get(k, [])) != np.array(v) for k, v in _new_loaded_track_limits.items()]))
                try:
                    did_loaded_track_limits_change: bool = (((_bak_loaded_track_limits is None) and (_new_loaded_track_limits is not None)) or np.any([_sub_sub_fn_did_potentially_arr_or_None_variable_change(np.array(_bak_loaded_track_limits.get(k, [])), np.array(v)) for k, v in _new_loaded_track_limits.items()]))
                                
                except ValueError as e:
                    ## something went wrong with the values! Consider them changed.
                    print(f'\tsomething went wrong with the values, consider them changed!')
                    did_loaded_track_limits_change: bool = True
                    pass
                except Exception as e:
                    raise

                
                change_dict[f'filtered_sessions["{a_decoder_name}"].loaded_track_limits'] = did_loaded_track_limits_change
                if did_loaded_track_limits_change:
                    did_any_change = True
            else:
                print(f'WARN: loaded_track_limits are not correctly checked in is_dry_run==True mode.')

            # all UserAnnotations overrides ______________________________________________________________________________________ #
            for k, new_val in a_session_override_dict.items():
                if k in allowed_sess_Config_override_keys:
                    _old_val = getattr(a_session.config, k, None)
                    will_change: bool = _sub_sub_fn_did_potentially_arr_or_None_variable_change(_old_val, new_val)
                    _curr_key: str = f'filtered_sessions["{a_decoder_name}"].config.{k}'
                    if will_change:
                        print(f'\t {_curr_key} changing! {_old_val} -> {new_val}.')
                    change_dict[_curr_key] = will_change
                    if will_change and (not is_dry_run):
                        setattr(a_session.config, k, deepcopy(new_val))
                    if will_change:
                        did_any_change = True


            # grid_bin_bounds ____________________________________________________________________________________________________ #
            _old_val = getattr(a_session.config, 'grid_bin_bounds', None)
            if _old_val is not None:
                _old_val = deepcopy(_old_val)
            # will_change: bool = (_old_val != hard_manual_override_grid_bin_bounds)
            will_change: bool = _sub_sub_fn_did_potentially_arr_or_None_variable_change(_old_val, hard_manual_override_grid_bin_bounds)
            change_dict[f'filtered_sessions["{a_decoder_name}"].config.grid_bin_bounds'] = (change_dict.get(f'filtered_sessions["{a_decoder_name}"].config.grid_bin_bounds', False) | will_change)
            if not is_dry_run:
                a_session.config.grid_bin_bounds = deepcopy(hard_manual_override_grid_bin_bounds) ## FORCEIPLY UPDATE
            if will_change:
                did_any_change = True
            
            # grid_bin ___________________________________________________________________________________________________________ #
            _old_val = getattr(a_session.config, 'grid_bin', None)
            if _old_val is not None:
                _old_val = deepcopy(_old_val)
            (constrained_grid_bin_sizes, constrained_num_grid_bins) = safe_limit_num_grid_bin_values(hard_manual_override_grid_bin_bounds, desired_grid_bin_sizes=deepcopy(desired_grid_bin), max_allowed_num_bins=max_allowed_num_bins, debug_print=False)
            will_change: bool = _sub_sub_fn_did_potentially_arr_or_None_variable_change(_old_val, constrained_grid_bin_sizes)
            change_dict[f'filtered_sessions["{a_decoder_name}"].config.grid_bin'] = (change_dict.get(f'filtered_sessions["{a_decoder_name}"].config.grid_bin', False) | will_change)
            if not is_dry_run:
                a_session.config.grid_bin = constrained_grid_bin_sizes
            if will_change:
                did_any_change = True
                
            return did_any_change, a_session
        # END def _subfn_update_session_config(....

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        did_any_change: bool = False

        for a_decoder_name, a_config in curr_active_pipeline.active_configs.items():
            # a_config: InteractivePlaceCellConfig
            _old_val = deepcopy(a_config.computation_config.pf_params.grid_bin_bounds)
            # will_change: bool = (_old_val != hard_manual_override_grid_bin_bounds)
            will_change: bool = _sub_sub_fn_did_potentially_arr_or_None_variable_change(_old_val, hard_manual_override_grid_bin_bounds)
            change_dict[f'active_configs["{a_decoder_name}"]'] = will_change
            grid_bin_bounds = deepcopy(hard_manual_override_grid_bin_bounds) ## FORCEIPLY UPDATE
            if not is_dry_run:
                a_config.computation_config.pf_params.grid_bin_bounds = grid_bin_bounds
            (constrained_grid_bin_sizes, constrained_num_grid_bins) = safe_limit_num_grid_bin_values(grid_bin_bounds, desired_grid_bin_sizes=deepcopy(desired_grid_bin), max_allowed_num_bins=max_allowed_num_bins, debug_print=False)
            if not is_dry_run:
                a_config.computation_config.pf_params.grid_bin = constrained_grid_bin_sizes
            if will_change:
                did_any_change = True
                
        ## THE ONES THAT START WRONG
        for a_decoder_name, a_config in curr_active_pipeline.computation_results.items():
            # a_config: InteractivePlaceCellConfig
            _old_val = deepcopy(a_config.computation_config.pf_params.grid_bin_bounds)
            # will_change: bool = (_old_val != hard_manual_override_grid_bin_bounds)
            will_change: bool = _sub_sub_fn_did_potentially_arr_or_None_variable_change(_old_val, hard_manual_override_grid_bin_bounds)
            change_dict[f'computation_results["{a_decoder_name}"]'] = will_change
            grid_bin_bounds = deepcopy(hard_manual_override_grid_bin_bounds) ## FORCEIPLY UPDATE
            if not is_dry_run:
                a_config.computation_config.pf_params.grid_bin_bounds = grid_bin_bounds
            (constrained_grid_bin_sizes, constrained_num_grid_bins) = safe_limit_num_grid_bin_values(grid_bin_bounds, desired_grid_bin_sizes=deepcopy(desired_grid_bin), max_allowed_num_bins=max_allowed_num_bins, debug_print=False)
            if not is_dry_run:
                a_config.computation_config.pf_params.grid_bin = constrained_grid_bin_sizes
            if will_change:
                did_any_change = True
        
        # sessions ___________________________________________________________________________________________________________ #
        for a_decoder_name, a_filtered_session in curr_active_pipeline.filtered_sessions.items():
            # a_filtered_session = deepcopy(a_filtered_session)
            # ## update the config
            # a_filtered_session.config.grid_bin_bounds = deepcopy(hard_manual_override_grid_bin_bounds) ## FORCEIPLY UPDATE ## needs it
            # (constrained_grid_bin_sizes, constrained_num_grid_bins) = safe_limit_num_grid_bin_values(a_filtered_session.config.grid_bin_bounds, desired_grid_bin_sizes=deepcopy(desired_grid_bin), max_allowed_num_bins=max_allowed_num_bins, debug_print=False)
            # a_filtered_session.config.grid_bin = constrained_grid_bin_sizes

            # loaded_track_limits, other overrides
            new_did_change, a_filtered_session = _subfn_update_session_config(a_session=a_filtered_session, is_dry_run=is_dry_run)
            if new_did_change:
                print(f'\tfiltered_session[{a_decoder_name}] changed!')
            did_any_change = did_any_change | new_did_change



        ### root/unfiltered session:
        new_did_change, curr_active_pipeline.stage.sess = _subfn_update_session_config(a_session=curr_active_pipeline.sess, is_dry_run=is_dry_run)
        if new_did_change:
            print(f'\tcurr_active_pipeline.sess[{a_decoder_name}] changed!')
        did_any_change = did_any_change | new_did_change
        
        # ## just to be safe:
        # curr_active_pipeline.sess.config.grid_bin_bounds = deepcopy(hard_manual_override_grid_bin_bounds) ## FORCEIPLY UPDATE ## needs it
        # (constrained_grid_bin_sizes, constrained_num_grid_bins) = safe_limit_num_grid_bin_values(curr_active_pipeline.sess.config.grid_bin_bounds, desired_grid_bin_sizes=deepcopy(desired_grid_bin), max_allowed_num_bins=max_allowed_num_bins, debug_print=False)
        # curr_active_pipeline.sess.config.grid_bin = constrained_grid_bin_sizes
        
        return did_any_change, change_dict
    

    @classmethod
    def _perform_required_recompute_on_change(cls, curr_active_pipeline):
            """ called to actually perform a recomputation when changes are detected/required and it's not dry_run
            """
            ## if not dry_run, do the recomputations:
            ## All invalidated ones:
            computation_functions_name_includelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                                '_perform_position_decoding_computation', 
                                                '_perform_firing_rate_trends_computation',
                                                '_perform_pf_find_ratemap_peaks_computation',
                                                '_perform_time_dependent_pf_sequential_surprise_computation'
                                                '_perform_two_step_position_decoding_computation',
                                                # '_perform_recursive_latent_placefield_decoding'
                                            ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'

            # ## Only Essentials:
            # computation_functions_name_includelist=['_perform_baseline_placefield_computation',
            #                                         '_perform_time_dependent_placefield_computation',
            #                                         '_perform_extended_statistics_computation',
            #                                     '_perform_position_decoding_computation', 
            #                                     '_perform_firing_rate_trends_computation',
            #                                     # '_perform_pf_find_ratemap_peaks_computation',
            #                                     # '_perform_time_dependent_pf_sequential_surprise_computation'
            #                                     '_perform_two_step_position_decoding_computation',
            #                                     # '_perform_recursive_latent_placefield_decoding'
            #                                 ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'

            # computation_functions_name_includelist=['_perform_baseline_placefield_computation']
            # curr_active_pipeline.perform_computations(computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=None, fail_on_exception=True, debug_print=FalTruese, overwrite_extant_results=True) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)
            # curr_active_pipeline.perform_computations(computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=None, enabled_filter_names=[global_epoch_name], fail_on_exception=True, debug_print=True) # , overwrite_extant_results=False #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)

            # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

            # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pf_computation', 'pfdt_computation'], enabled_filter_names=[global_epoch_name], fail_on_exception=True, debug_print=False)
            # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pf_computation'], enabled_filter_names=[global_epoch_name], fail_on_exception=True, debug_print=True)


            # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=computation_functions_name_includelist, enabled_filter_names=[global_epoch_name], fail_on_exception=True, debug_print=True)
            curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=computation_functions_name_includelist, fail_on_exception=True, debug_print=True)
            print(f'\trecomputation complete!')
                        

    @function_attributes(short_name=None, tags=['MAIN', 'ESSENTIAL', 'UNUSED', 'grid_bin_bounds', 'grid_bin'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-12 19:50', related_items=[])
    @classmethod
    def FINAL_FIX_GRID_BIN_BOUNDS(cls, curr_active_pipeline, force_recompute:bool=False, is_dry_run: bool=False, debug_skip_computations_only:bool=False, defer_required_compute: bool=False):
        """ perform all fixes regarding the grid_bin_bounds and grid_bin """
        print(f'\t !!!||||||||||||||||||> RUNNING `PostHocPipelineFixup.FINAL_FIX_GRID_BIN_BOUNDS(...)`:')
        correct_grid_bin_bounds = cls.get_hardcoded_known_good_grid_bin_bounds(curr_active_pipeline)
        did_any_change, change_dict = cls.HARD_OVERRIDE_grid_bin_bounds(curr_active_pipeline, hard_manual_override_grid_bin_bounds=deepcopy(correct_grid_bin_bounds), is_dry_run=is_dry_run)
        only_changing_change_keys = [k for k, v in change_dict.items() if v is True]
        if (did_any_change or force_recompute):
            if force_recompute:
                print(f'only_changing_change_keys: {only_changing_change_keys}\n\t(force_recompute==True), (did_any_change: {did_any_change}) recomputing...')
            else:
                print(f'only_changing_change_keys: {only_changing_change_keys}\n\tat least one grid_bin_bound was changed, recomputing...')

            if debug_skip_computations_only:
                print(f'2025-02-19 17:52 WARNING!!!! debug_skip_computations_only == True should only be for testing, not doing comps.')
            else:
                if (not is_dry_run):
                    ## if not dry_run, do the recomputations:
                    if not defer_required_compute:
                        cls._perform_required_recompute_on_change(curr_active_pipeline=curr_active_pipeline)
                        print(f'\trecomputation complete!')
                    else:
                        print(f'\trecomputation required but defer_required_compute == True, so skipping until later phase.')
                else:
                    print(f'\tWARNING: is_dry_run is true so no recompute will be done.')
                ## END if debug_skip_computations_only....
                
        else:
            ## no changes and not force_recompute
            print(f'No grid bin bounds were changed. Everything should be up-to-date!')

        return (did_any_change, change_dict), correct_grid_bin_bounds


    @function_attributes(short_name=None, tags=['laps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-15 12:45', related_items=[])
    @classmethod
    def FINAL_FIX_LAPS_FROM_OVERRIDES(cls, curr_active_pipeline, force_recompute:bool=False, is_dry_run: bool=False, debug_skip_computations_only:bool=False) -> bool:
        """ perform all fixes regarding any needed replacements for laps indicated in UserAnnotations """
        print(f'\t !!!||||||||||||||||||> RUNNING `PostHocPipelineFixup.FINAL_FIX_LAPS_FROM_OVERRIDES(...)`:')
        from neuropy.core.user_annotations import UserAnnotationsManager
        # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import override_laps
        from neuropy.core.laps import Laps

        override_laps_df: Optional[pd.DataFrame] = UserAnnotationsManager.get_hardcoded_laps_override_dict().get(curr_active_pipeline.get_session_context(), None)
        if override_laps_df is None:
            return False ## no changes
        else:
            ## non-None, override laps
            print(f'\toverriding laps....')
            override_laps_df['lap_id'] = override_laps_df.index + 1
            override_laps_df['label'] = override_laps_df.index
            ## OUTPUTS: override_laps_df
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            override_laps_obj: Laps = Laps(laps=override_laps_df)
            override_laps_obj.update_lap_dir_from_net_displacement(pos_input=curr_active_pipeline.sess.position)
            override_laps_obj.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
            override_laps_df = override_laps_obj.to_dataframe()

            return curr_active_pipeline.override_laps(override_laps_df=override_laps_df, debug_print=True)


    # ==================================================================================================================== #
    # Filepaths                                                                                                            #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['non_pbe', 'epochs', 'sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-19 00:00', related_items=[])
    @classmethod
    def FINAL_UPDATE_FILEPATHS(cls, curr_active_pipeline, force_update:bool=True) -> bool:
        """ perform all fixes regarding the pipeline's loaded session's .basepath and any paths in its configs """
        print(f'\t =================> RUNNING `PostHocPipelineFixup.FINAL_UPDATE_FILEPATHS(...)`:')
        did_fixup_any_missing_basepath = curr_active_pipeline.post_load_fixup_sess_basedirs(updated_session_basepath=deepcopy(curr_active_pipeline.sess.basepath), force_update=force_update)
        return did_fixup_any_missing_basepath
        
    # ==================================================================================================================== #
    # Non-PBE Epochs                                                                                                       #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['non_pbe', 'epochs', 'sessions'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-19 00:00', related_items=[])
    @classmethod
    def FINAL_UPDATE_NON_PBE_EPOCHS(cls, curr_active_pipeline) -> bool:
        """ perform all fixes regarding computation of the non-PBE epochs """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs
        print(f'\t =================> RUNNING `PostHocPipelineFixup.FINAL_UPDATE_NON_PBE_EPOCHS(...)`:')
        did_any_non_pbe_epochs_change, curr_active_pipeline.stage.sess, curr_active_pipeline.stage.filtered_sessions = Compute_NonPBE_Epochs.update_session_non_pbe_epochs(curr_active_pipeline.sess, filtered_sessions=curr_active_pipeline.filtered_sessions)

        return did_any_non_pbe_epochs_change


    # ==================================================================================================================== #
    # ALL/MAIN                                                                                                             #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['MAIN', 'epochs', 'sessions'], input_requires=[], output_provides=[], uses=[], used_by=['run_as_batch_user_completion_function'], creation_date='2025-02-19 15:00', related_items=[])
    @classmethod
    def FINAL_UPDATE_ALL(cls, curr_active_pipeline, force_recompute:bool=True, is_dry_run: bool=False) -> bool:
        """ perform all known fixes to the pipeline and return whether any fixes were needed/performed """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers, DirectionalLapsResult
        print(f'\t !!!||||||||||||||||||> RUNNING `PostHocPipelineFixup.FINAL_UPDATE_ALL(...)`:')
        print(f'starting `PostHocPipelineFixup.FINAL_UPDATE_ALL(...)`...')

        did_any_change: bool = False

        did_fixup_any_missing_basepath = PostHocPipelineFixup.FINAL_UPDATE_FILEPATHS(curr_active_pipeline=curr_active_pipeline)

        did_any_non_pbe_epochs_change = PostHocPipelineFixup.FINAL_UPDATE_NON_PBE_EPOCHS(curr_active_pipeline=curr_active_pipeline)

        did_override_any_laps = PostHocPipelineFixup.FINAL_FIX_LAPS_FROM_OVERRIDES(curr_active_pipeline=curr_active_pipeline)

        (did_any_grid_bin_change, change_dict), correct_grid_bin_bounds = PostHocPipelineFixup.FINAL_FIX_GRID_BIN_BOUNDS(curr_active_pipeline=curr_active_pipeline, force_recompute=force_recompute, is_dry_run=is_dry_run, defer_required_compute=True)

        did_any_change = (did_any_grid_bin_change or did_fixup_any_missing_basepath or did_any_non_pbe_epochs_change or did_override_any_laps)
        
        if (did_any_change or force_recompute):
            if force_recompute:
                print(f'(force_recompute==True), (did_any_change: {did_any_change}) recomputing...')
            else:
                print(f'at least one grid_bin_bound was changed, recomputing...')

            if (not is_dry_run):
                ## if not dry_run, do the recomputations:
                cls._perform_required_recompute_on_change(curr_active_pipeline=curr_active_pipeline)
                print(f'\trecomputation complete!')

            else:
                print(f'\tWARNING: is_dry_run is true so no recompute will be done.')
                
        else:
            ## no changes and not force_recompute
            print(f'No grid bin bounds were changed. Everything should be up-to-date!')
            
        # Fix the computation epochs to be constrained to the proper long/short intervals:
        # was_directional_pipeline_modified = DirectionalLapsResult.fix_computation_epochs_if_needed(curr_active_pipeline=curr_active_pipeline)
        was_directional_pipeline_modified = DirectionalLapsHelpers.fixup_directional_pipeline_if_needed(curr_active_pipeline)
        print(f'\tDirectionalLapsResult.init_from_pipeline_natural_epochs(...): was_modified: {was_directional_pipeline_modified}')
        
        # curr_active_pipeline, directional_lap_specific_configs = DirectionalLapsHelpers.split_to_directional_laps(curr_active_pipeline=curr_active_pipeline, add_created_configs_to_pipeline=True)

        did_any_change = (did_any_grid_bin_change or did_fixup_any_missing_basepath or did_any_non_pbe_epochs_change or was_directional_pipeline_modified or did_override_any_laps)
        print(f'\tPostHocPipelineFixup.FINAL_UPDATE_ALL(...): did_any_change: {did_any_change}')
        return did_any_change


    @metadata_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=[], used_by=['kdiba_session_post_fixup_completion_function'], creation_date='2025-02-19 07:24', related_items=[])
    @staticmethod
    def run_as_batch_user_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict, force_recompute: bool=False, is_dry_run: bool=False) -> dict:
        """ meant to be executed as a _batch_user_completion_function, called by `kdiba_session_post_fixup_completion_function` 
        """
        # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers, DirectionalLapsResult

        print(f'\t !!!||||||||||||||||||> RUNNING `PostHocPipelineFixup.run_as_batch_user_completion_function(...)`:')
        print(f'starting `PostHocPipelineFixup.run_as_batch_user_completion_function(...)`...')

        did_any_change: bool = PostHocPipelineFixup.FINAL_UPDATE_ALL(curr_active_pipeline, force_recompute=force_recompute, is_dry_run=is_dry_run)
        
        print(f'\tPostHocPipelineFixup.run_as_batch_user_completion_function(...): did_any_change: {did_any_change}')

        loaded_track_limits = curr_active_pipeline.sess.config.loaded_track_limits
        a_config_dict = curr_active_pipeline.sess.config.to_dict()

        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        print(f'\t{curr_session_basedir}:\tloaded_track_limits: {loaded_track_limits}, a_config_dict: {a_config_dict}')  # , t_end: {t_end}
        
        callback_outputs = {
        # 'correct_grid_bin_bounds': correct_grid_bin_bounds, 'loaded_track_limits': loaded_track_limits, 'change_dict': change_dict,
        'config_dict': a_config_dict, #'t_end': t_end 
        # 'did_any_grid_bin_change': did_any_grid_bin_change, 'did_fixup_any_missing_basepath': did_fixup_any_missing_basepath, 'did_any_non_pbe_epochs_change': did_any_non_pbe_epochs_change, 'was_directional_pipeline_modified': was_directional_pipeline_modified,
        'did_any_change': did_any_change,
        }
        print(f'\t\tcallback will be assigned to `across_session_results_extended_dict[{PostHocPipelineFixup.across_session_results_extended_dict_data_name}]`:')
        print(f'\t\t\tcallback_outputs: {callback_outputs}')
        across_session_results_extended_dict[PostHocPipelineFixup.across_session_results_extended_dict_data_name] = callback_outputs
        print(f'_____________________________________________________________________________________________________\n')
        if did_any_change:
            print(f'================= CHANGES WERE MADE BY FIXUP! VALUES WILL NEED RECOMPUTE!\n')
        else:
            print(f'================= NO changes needed!\n')            

        print(f'_____________________________________________________________________________________________________\n')        
        print('\tdone.')
        return across_session_results_extended_dict
    

@function_attributes(short_name=None, tags=['IMPORTANT', 'PostHocPipelineFixup', 'non_PBE'], input_requires=[], output_provides=[], uses=['PostHocPipelineFixup'], used_by=[], creation_date='2025-02-19 00:00', related_items=['PostHocPipelineFixup'])
def kdiba_session_post_fixup_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict, force_recompute:bool=True, is_dry_run: bool=False) -> dict:
    """ Called to update the pipeline's important position info parameters (such as the grid_bin_bounds, positions, etc) from a loaded .mat file
    
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import kdiba_session_post_fixup_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

    ['basepath', 'session_spec', 'session_name', 'session_context', 'format_name', 'preprocessing_parameters', 'absolute_start_timestamp', 'position_sampling_rate_Hz', 'microseconds_to_seconds_conversion_factor', 'pix2cm', 'x_midpoint', 'loaded_track_limits', 'is_resolved', 'resolved_required_filespecs_dict', 'resolved_optional_filespecs_dict', 'x_unit_midpoint', 'first_valid_pos_time', 'last_valid_pos_time']

    """
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import PostHocPipelineFixup

    # ==================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                  #
    # ==================================================================================================================== #


    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'kdiba_session_post_fixup_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ..., is_dry_run: {is_dry_run})')
    if is_dry_run:
        print(f'WARN: is_dry_run == True')
    
    across_session_results_extended_dict = PostHocPipelineFixup.run_as_batch_user_completion_function(self=self, global_data_root_parent_path=global_data_root_parent_path, curr_session_context=curr_session_context, curr_session_basedir=curr_session_basedir, curr_active_pipeline=curr_active_pipeline, across_session_results_extended_dict=across_session_results_extended_dict,
                                                                                                       force_recompute=force_recompute, is_dry_run=is_dry_run)

    # print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['hdf5', 'h5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def export_session_h5_file_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                should_write_pipeline_h5: bool=False, should_write_posterior_h5: bool=True) -> dict:
    """  Export the pipeline's HDF5 as 'pipeline_results.h5'
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import kdiba_session_post_fixup_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('export_session_h5_file_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    Exports:
        '/nfs/turbo/umms-kdiba/KDIBA/pin01/one/fet11-01_12-58-54/output/pipeline_results.h5'

    """
    import sys
    from datetime import timedelta, datetime
    from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
    from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import ContextToPathMode

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'export_session_h5_file_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    # custom_save_filepaths, custom_save_filenames, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters()
    # print(f'\tcustom_save_filenames: {custom_save_filenames}')
    # print(f'\tcustom_suffix: "{custom_suffix}"')
    err = None
    hdf5_output_path: Path = None
    was_write_good: bool = False
    

    active_export_parent_output_path: Path = self.collected_outputs_path.resolve()
    # Assert.path_exists(active_export_parent_output_path)
    callback_outputs = {}


    if should_write_pipeline_h5:
        try:
            custom_save_filepaths_dict, custom_suffix = curr_active_pipeline.custom_save_pipeline_as(enable_save_pipeline_pkl=False, enable_save_global_computations_pkl=False, enable_save_h5=True)
            was_write_good = True
            hdf5_output_path = custom_save_filepaths_dict['pipeline_h5']
            print(f'\tpipeline hdf5_output_path: "{hdf5_output_path}"')

        except Exception as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"\tERROR: encountered exception {err} while trying to build the session HDF output for {curr_session_context}")
            if self.fail_on_exception:
                raise err.exc
            
        callback_outputs.update(**{
            'hdf5_output_path': hdf5_output_path, #'t_end': t_end   
        })
    # END if should_write_pipeline_h5

    if should_write_posterior_h5:
        try:
            ## Get needed results
            rank_order_results = curr_active_pipeline.global_computation_results.computed_data.get('RankOrder', None)
            if rank_order_results is not None:
                minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = rank_order_results.included_qclu_values
            else:        
                ## get from parameters:
                minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
                
            # DirectionalMergedDecoders: Get the result after computation:
            directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] # : "DirectionalLapsResult"
            track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference? : "TrackTemplates"
            directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']
            directional_decoders_epochs_decode_result.add_all_extra_epoch_columns(curr_active_pipeline, track_templates=track_templates, required_min_percentage_of_active_cells=0.33333333, debug_print=False)

            ## Get save paths
            # posteriors_save_path = Path('output/newest_all_decoded_epoch_posteriors.h5').resolve()
            _parent_save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_decoded_posteriors_to_HDF5')
            # output_man = curr_active_pipeline.get_output_manager(context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE, override_output_parent_path=active_export_parent_output_path)
            # print(f'_parent_save_context: {_parent_save_context}')
            # posteriors_save_path: Path = output_man.get_figure_save_file_path(final_context=_parent_save_context).with_suffix('.h5')
            posteriors_save_path: Path = PosteriorExporting.build_custom_export_to_h5_path(curr_active_pipeline, output_date_str=None, data_identifier_str='(decoded_posteriors)', a_tbin_size=directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size, parent_output_path=active_export_parent_output_path)
            print(f'posteriors_save_path: "{posteriors_save_path}')

            # pos_bin_size: float = directional_decoders_epochs_decode_result.pos_bin_size
            # ripple_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.ripple_decoding_time_bin_size
            # laps_decoding_time_bin_size: float = directional_decoders_epochs_decode_result.laps_decoding_time_bin_size
            decoder_laps_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict
            decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict

            out_contexts = PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5(decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict, _save_context=_parent_save_context, save_path=posteriors_save_path)
            callback_outputs['posteriors_h5'] = posteriors_save_path
            print(f'\tposteriors_save_path: "{posteriors_save_path}"')

        except Exception as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"\tERROR: encountered exception {err} while trying to build the session POSTERIORS HDF output for {curr_session_context}")
            if self.fail_on_exception:
                raise err.exc
    # END if should_write_posterior_h5

    across_session_results_extended_dict['export_session_h5_file_completion_function'] = callback_outputs

    if was_write_good:
        print(f'\tHDF5 file "{hdf5_output_path}" successfully written out! done.')    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['save_custom', 'versioning', 'backup', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-28 14:25', related_items=[])
def save_custom_session_files_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
    """ Saves a copy of this pipeline's files (pkl, HDF5) with custom suffix derived from parameters
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import backup_previous_session_files_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

    """
    import sys
    from datetime import timedelta, datetime
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException

    custom_save_filepaths_dict, custom_save_filenames, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters()
    
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'save_custom_session_files_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    print(f'custom_save_filenames: {custom_save_filenames}')
    print(f'custom_suffix: "{custom_suffix}"')
    
    was_write_good: bool = False
    try:
        custom_save_filepaths_dict, custom_suffix = curr_active_pipeline.custom_save_pipeline_as(enable_save_pipeline_pkl=True, enable_save_global_computations_pkl=True, enable_save_h5=True)
        was_write_good = True

    except Exception as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"ERROR: encountered exception {err} while trying to backup the pipeline for {curr_session_context}")
        if self.fail_on_exception:
            raise err.exc

    callback_outputs = {
     'desired_suffix': custom_suffix,
    #  'session_files_dict': _existing_session_files_dict, 'successfully_copied_files_dict':_successful_copies_dict,
     'session_files_dict': custom_save_filepaths_dict, 'successfully_copied_files_dict': custom_save_filepaths_dict,
     'was_write_good': was_write_good,
    }
    across_session_results_extended_dict['save_custom_session_files_completion_function'] = callback_outputs
    
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict


@function_attributes(short_name=None, tags=['backup', 'versioning', 'copy'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-25 06:22', related_items=[])
def backup_previous_session_files_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict, desired_suffix: str = 'Pre2024-07-16') -> dict:
    """ Makes a backup copy of the pipeline's files (pkl, HDF5) with desired suffix
    
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

    ## Recover qclu, fr_Hz filters:
    all_params_dict = curr_active_pipeline.get_all_parameters()
    preprocessing_parameters = all_params_dict['preprocessing']
    rank_order_shuffle_analysis_parameters = all_params_dict['rank_order_shuffle_analysis']
    minimum_inclusion_fr_Hz = deepcopy(rank_order_shuffle_analysis_parameters['minimum_inclusion_fr_Hz'])
    included_qclu_values = deepcopy(rank_order_shuffle_analysis_parameters['included_qclu_values'])

    ## what computes track_templates? I think that is where the qclu and fr_Hz limits are actually applied
    minimum_inclusion_fr_Hz
    included_qclu_values

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


@function_attributes(short_name=None, tags=['all_neuron_stats_table', 'neuron_replay_stats_df_CSV', 'final-publication', 'JSON', 'CSV', 'peak', 'pf', 'peak_promenance'], input_requires=[], output_provides=[], uses=['AcrossSessionsResults.build_neuron_identities_df_for_CSV'], used_by=[], creation_date='2024-01-01 00:00', related_items=[])
def compute_and_export_session_extended_placefield_peak_information_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                             save_csv:bool=True, save_json:bool=False) -> dict:
    """  Extracts peak information for the placefields for each neuron. Responsible for outputting the combined neuron information CSV used in the final paper results by merging the three+ informationt tables into one `all_neuron_stats_table`
    
    
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import kdiba_session_post_fixup_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'compute_and_export_session_extended_placefield_peak_information_completion_function':
    callback_outputs = across_session_results_extended_dict['compute_and_export_session_extended_placefield_peak_information_completion_function']
    csv_output_path = callback_outputs['csv_output_path']
    
    
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('compute_and_export_session_extended_placefield_peak_information_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}


    Exports
        *_neuron_replay_stats_df.csv
        *_neuron_replay_stats_df.json
        
    """
    import sys
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults # for .build_neuron_identities_df_for_CSV

    # Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'compute_and_export_session_extended_placefield_peak_information_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')

    callback_outputs = {
        'csv_output_path': None,
        'json_output_path': None, #'t_end': t_end   
        
    }
    err = None

    # active_csv_parent_output_path = curr_active_pipeline.get_output_path().resolve()
    active_export_parent_output_path = self.collected_outputs_path.resolve()

    try:

        # 2025-06-09 18:22 - Added combined output helper:        
        all_neuron_stats_table: pd.DataFrame = AcrossSessionsResults.build_neuron_identities_df_for_CSV(curr_active_pipeline=curr_active_pipeline)
        # # Pre-2025-06-09 Export mode
        # rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        # minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        # included_qclu_values: List[int] = rank_order_results.included_qclu_values
        # directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        # track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        # print(f'included_qclu_values: {included_qclu_values}')
    
        # print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
        # jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
        # neuron_replay_stats_df: pd.DataFrame = deepcopy(jonathan_firing_rate_analysis_result.neuron_replay_stats_df)
        # neuron_replay_stats_df, all_pf2D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_peak_promenance_pf_peaks(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates)
        # neuron_replay_stats_df, all_pf1D_peaks_modified_columns = jonathan_firing_rate_analysis_result.add_directional_pf_maximum_peaks(track_templates=track_templates)
        # # both_included_neuron_stats_df = deepcopy(neuron_replay_stats_df[neuron_replay_stats_df['LS_pf_peak_x_diff'].notnull()]).drop(columns=['track_membership', 'neuron_type'])
        
        print(f'\t\t done (success).')

    except Exception as e:
        exception_info = sys.exc_info()
        err = CapturedException(e, exception_info)
        print(f"WARN: on_complete_success_execution_session: encountered exception {err} while trying to compute the extended placefield peak information and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
        # if self.fail_on_exception:
        #     raise e.exc
        # _out_inst_fr_comps = None
        all_neuron_stats_table = None
        pass
    

    if (all_neuron_stats_table is not None) and save_csv:
        print(f'\t try saving to CSV...')
        # Save DataFrame to CSV
        csv_output_path = active_export_parent_output_path.joinpath(f'{CURR_BATCH_OUTPUT_PREFIX}_neuron_replay_stats_df.csv').resolve()
        try:
            all_neuron_stats_table.to_csv(csv_output_path)
            print(f'\t saving to CSV: "{csv_output_path}" done.')
            callback_outputs['csv_output_path'] = csv_output_path

        except Exception as e:
            exception_info = sys.exc_info()
            err = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {err} while trying to save to CSV for {curr_session_context}")
            csv_output_path = None # set to None because it failed.
            if self.fail_on_exception:
                raise err.exc
    else:
        csv_output_path = None


    ## standalone saving:
    if (all_neuron_stats_table is not None) and save_json:
        print(f'\t try saving to JSON...')
        # Save DataFrame to JSON
        json_output_path = active_export_parent_output_path.joinpath(f'{CURR_BATCH_OUTPUT_PREFIX}_neuron_replay_stats_df.json').resolve()
        try:
            all_neuron_stats_table.to_json(json_output_path, orient='records', lines=True) ## This actually looks pretty good!
            print(f'\t saving to JSON: "{json_output_path}" done.')
            callback_outputs['json_output_path'] = json_output_path

        except Exception as e:
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



@function_attributes(short_name=None, tags=['posterior', 'marginal', 'CSV', 'non-PBE', 'epochs', 'decoding'], input_requires=[], output_provides=[], uses=['GenericDecoderDictDecodedEpochsDictResult', 'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions.EpochComputationFunctions.perform_compute_non_PBE_epochs'], used_by=[], creation_date='2025-03-09 16:35', related_items=['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'])
def generalized_decode_epochs_dict_and_export_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict, epochs_decoding_time_bin_size:float=0.025, force_recompute:bool=True, debug_print:bool=True) -> dict:
    """ Aims to generally:
    1. Build a dict of decoders (usually 1D) built on several different subsets of input epochs (long_LR_laps-only, long_laps-only, long_non_PBE-only, ...etc
    2. Use these decoders and the neural data to decode posteriors for a variety of parameters (e.g. cell types, epochs-to-be-decoded, time_bin_sizes, etc)
    3. Compute Pseudo2D (Context x Position) versions of these sets of decoders and decode using thse
    4. Compute a variety of marginals for each result (track_ID marginals, run_dir_marginals, etc)
    5. Export all the results to .CSV for later plotting and across-session analysis 
    
    Calls ['non_PBE_epochs_results', 'generalized_specific_epochs_decoding'] global computation functions
    
    
    USES: `GenericDecoderDictDecodedEpochsDictResult`
    
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import generalized_decode_epochs_dict_and_export_results_completion_function
    
    Results can be extracted from batch output by 
    
    # Extracts the callback results 'determine_session_t_delta_completion_function':
    extracted_callback_fn_results = {a_sess_ctxt:a_result.across_session_results.get('determine_session_t_delta_completion_function', {}) for a_sess_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

    ['basepath', 'session_spec', 'session_name', 'session_context', 'format_name', 'preprocessing_parameters', 'absolute_start_timestamp', 'position_sampling_rate_Hz', 'microseconds_to_seconds_conversion_factor', 'pix2cm', 'x_midpoint', 'loaded_track_limits', 'is_resolved', 'resolved_required_filespecs_dict', 'resolved_optional_filespecs_dict', 'x_unit_midpoint', 'first_valid_pos_time', 'last_valid_pos_time']

    
    OUTPUTS:
    
        callback_outputs = _across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function'] # 'PostHocPipelineFixup'
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = callback_outputs['a_new_fully_generic_result']
        csv_save_paths_dict: Dict[str, Path] = callback_outputs['csv_save_paths_dict']
        a_new_fully_generic_result

    """
    from typing import Literal
    from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe, ensure_Epoch, TimeColumnAliasesProtocol
    from pyphocorehelpers.print_helpers import get_now_rounded_time_str
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer, DecodingResultND, Compute_NonPBE_Epochs, KnownFilterEpochs, GeneralDecoderDictDecodedEpochsDictResult
    from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult, TrainTestLapsSplitting, CustomDecodeEpochsResult, decoder_name, epoch_split_key, get_proper_global_spikes_df, DirectionalPseudo2DDecodersResult
    from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult #, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType
    from pyphocorehelpers.assertion_helpers import Assert
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_evaluate_required_computations, batch_extended_computations

    # ==================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                  #
    # ==================================================================================================================== #

    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'generalized_decode_epochs_dict_and_export_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}, force_recompute: {force_recompute}, ...)')

    # ==================================================================================================================== #
    # New 2025-03-11 Generic Result:                                                                                       #
    # ==================================================================================================================== #

    if ('generalized_decode_epochs_dict_and_export_results_completion_function' in across_session_results_extended_dict) and force_recompute:
        ## drop the existing
        print(f'\tWARN: dropping the existing `generalized_decode_epochs_dict_and_export_results_completion_function` result because force_recompute is True')
        del across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function']

    # a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(curr_active_pipeline=curr_active_pipeline, force_recompute=force_recompute, time_bin_size=epochs_decoding_time_bin_size, debug_print=debug_print)
                    
    ## Unpack from pipeline:
    # valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
    fail_on_exception = True
    EpochComputations_result_needs_full_recompute: bool = False

    valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('EpochComputations', None)
    if valid_EpochComputations_result is None:
        print(f'\trecomputation is needed because "EpochComputations" is not in computed_data!')
        ## call computation function 'generalized_specific_epochs_decoding'        
        EpochComputations_result_needs_full_recompute = True
        
    else:
        ## already exists:
        extant_result_decoding_time_bin_size: float = deepcopy(valid_EpochComputations_result.epochs_decoding_time_bin_size) ## just get the standard size. Currently assuming all things are the same size!
        print(f'\textant_result_decoding_time_bin_size: {extant_result_decoding_time_bin_size}')
        EpochComputations_result_needs_full_recompute = (epochs_decoding_time_bin_size != extant_result_decoding_time_bin_size)
        if EpochComputations_result_needs_full_recompute:
            print(f"\t\tERROR: valid_EpochComputations_result.epochs_decoding_time_bin_size: {extant_result_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}.\n\tA FULL RECOMPUTE WILL BE NEEDED!")
            ## drop existing result
            dropped_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.pop('EpochComputations', None)
            print(f'\t\t dropped_EpochComputations_result removed!.')

        # assert epochs_decoding_time_bin_size == extant_result_decoding_time_bin_size, f"\tERROR: valid_EpochComputations_result.epochs_decoding_time_bin_size: {extant_result_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"

    if EpochComputations_result_needs_full_recompute:
        # a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(curr_active_pipeline=curr_active_pipeline, force_recompute=force_recompute, time_bin_size=epochs_decoding_time_bin_size, debug_print=debug_print)

        ## Next wave of computations
        extended_computations_include_includelist = ['split_to_directional_laps', 'non_PBE_epochs_results', 'generalized_specific_epochs_decoding',] # do only specified
        computation_kwargs_dict = {'non_PBE_epochs_results': dict(epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, drop_previous_result_and_compute_fresh=False, compute_2D=False), }

        # force_recompute_override_computations_includelist = deepcopy(extended_computations_include_includelist)
        force_recompute_override_computations_includelist = []
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)

        if debug_print:
            print(f'\tPost-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')

        # Post-hoc verification that the computations worked and that the validators reflect that. The list should be empty now.
        newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, 
                                                            computation_kwargs_dict=computation_kwargs_dict,
                                                            debug_print=False)

        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=force_recompute, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        if debug_print:
            print(f'\tPost-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')
        print(f'\t...recomputation done.')
        valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('EpochComputations', None)
        assert valid_EpochComputations_result is not None, f"valid_EpochComputations_result is still None ever after attempted recomputation!!"



    a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result

    if (a_new_fully_generic_result is None):
        print(f'WARN/ERROR: a_new_fully_generic_result is None! Doing last-ditch recomputation...')
        ## Next wave of computations
        extended_computations_include_includelist=['generalized_specific_epochs_decoding',] # do only specified
        force_recompute_override_computations_includelist = []
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=False, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        if debug_print:
            print(f'\tPost-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')
        # Post-hoc verification that the computations worked and that the validators reflect that. The list should be empty now.
        newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=fail_on_exception, progress_print=True,
                                                            force_recompute=False, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                            force_recompute=False, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
        if debug_print:
            print(f'\tPost-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')
        print(f'\t...a_generic_decoder_dict_decoded_epochs_dict_result recomputation done.')
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result
        assert a_new_fully_generic_result is not None, f"a_new_fully_generic_result is still None ever after attempted recomputation!!"
        
    ## now both are good!
    assert valid_EpochComputations_result is not None
    assert a_new_fully_generic_result is not None

    epochs_decoding_time_bin_size: float = valid_EpochComputations_result.epochs_decoding_time_bin_size ## just get the standard size. Currently assuming all things are the same size!
    print(f'\tepochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}')
    assert epochs_decoding_time_bin_size == valid_EpochComputations_result.epochs_decoding_time_bin_size, f"\tERROR: valid_EpochComputations_result.epochs_decoding_time_bin_size: {valid_EpochComputations_result.epochs_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"

    # ==================================================================================================================== #
    # Create and add the output                                                                                            #
    # ==================================================================================================================== #
    if 'generalized_decode_epochs_dict_and_export_results_completion_function' not in across_session_results_extended_dict:
        ## create
        print(f"\t creating new across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function']['a_new_fully_generic_result'] result.")
        across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function'] = {
            'a_new_fully_generic_result': deepcopy(a_new_fully_generic_result),
            'csv_save_paths_dict': {},
        }
    else:
        ## update the existing result
        print(f'\t updating existing result.')
        across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function']['a_new_fully_generic_result'] = deepcopy(a_new_fully_generic_result)

    ## Export to CSVs:
    decoding_time_bin_size: float = epochs_decoding_time_bin_size

    active_export_parent_output_path = self.collected_outputs_path.resolve()
    Assert.path_exists(active_export_parent_output_path)
    csv_save_paths_dict = a_new_fully_generic_result.default_export_all_CSVs(active_export_parent_output_path=active_export_parent_output_path, owning_pipeline_reference=curr_active_pipeline, decoding_time_bin_size=decoding_time_bin_size)
    across_session_results_extended_dict['generalized_decode_epochs_dict_and_export_results_completion_function']['csv_save_paths_dict'] = deepcopy(csv_save_paths_dict)
    print(f'csv_save_paths_dict: {csv_save_paths_dict}\n')
    print('\t\tdone.')

    # print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return across_session_results_extended_dict



@function_attributes(short_name=None, tags=['figure', 'posterior', 'hairly-plot'], input_requires=[], output_provides=[], uses=['_display_generalized_decoded_yellow_blue_marginal_epochs', '_display_decoded_trackID_marginal_hairy_position', '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay'], used_by=[], creation_date='2025-05-16 15:17', related_items=['generalized_decode_epochs_dict_and_export_results_completion_function'])
def figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict,
                                                                                        included_figures_names=['_display_directional_merged_pf_decoded_stacked_epoch_slices', '_display_generalized_decoded_yellow_blue_marginal_epochs', '_display_decoded_trackID_marginal_hairy_position', '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay', '_display_placefield_stable_formation_time_distribution', '_display_measured_vs_decoded_occupancy_distributions'],
                                                                                        extreme_threshold: float=0.8, opacity_max:float=0.7, thickness_ramping_multiplier:float=35.0,
                                                                                        **additional_marginal_overlaying_measured_position_kwargs) -> dict:
    """ Multi-purpose batch display function that just plots the figures so we don't have to wait for the entire batch_figures_plotting on 2025-04-16 15:22.
    corresponding to by `generalized_decode_epochs_dict_and_export_results_completion_function` 
    
    This is the global across-session marginal over trackID
    
    ## Getting outputs    
        _flattened_paths_dict = {} ## Outputs:

        _out_dict = _across_session_results_extended_dict.get('figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function', {}).get('_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay', {}) # FigureCollector 
        save_paths_dict = _out_dict.get('out_paths', {})
        for epoch_name, a_variant_paths_dict in save_paths_dict.items():
            ## loop over all variants:
            for a_variant_name, a_path in a_variant_paths_dict.items():
                if a_path is not None:
                    _curr_key = f"{epoch_name}.{a_variant_name}"
                    _flattened_paths_dict[_curr_key] = a_path


        _flattened_paths_dict


    """
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode	
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationDisplayFunctions
    from benedict import benedict
    from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting


    # 'trackID_weighted_position_posterior'
    if across_session_results_extended_dict is None:
        across_session_results_extended_dict = {}


    across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'] = {}


    assert self.collected_outputs_path.exists()
    curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}"
    print(f'CURR_BATCH_OUTPUT_PREFIX: {CURR_BATCH_OUTPUT_PREFIX}')
    
    print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    custom_figure_output_path = self.collected_outputs_path
    assert custom_figure_output_path.exists(), f"custom_figure_output_path: '{custom_figure_output_path}' does not exist!"
    
    custom_fig_man: FileOutputManager = FileOutputManager(figure_output_location=FigureOutputLocation.CUSTOM, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE, override_output_parent_path=custom_figure_output_path)
    
    # print(f'custom_figure_output_path: "{custom_figure_output_path}"')
    # test_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='display_long_short_laps')
    test_display_output_path = custom_fig_man.get_figure_save_file_path(curr_active_pipeline.get_session_context(), make_folder_if_needed=False)
    print(f'\ttest_display_output_path: "{test_display_output_path}"')

    curr_active_pipeline.reload_default_display_functions()



    # ==================================================================================================================================================================================================================================================================================== #
    # '_display_directional_merged_pf_decoded_stacked_epoch_slices'                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #
    ## this is the export of the separate 1D decoder posteriors to images
    if ('_display_directional_merged_pf_decoded_stacked_epoch_slices' in included_figures_names) or ('directional_decoded_stacked_epoch_slices' in included_figures_names):

        try:
            print(f'\t trying "_display_directional_merged_pf_decoded_stacked_epoch_slices"')
            a_params_kwargs = {}
            display_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='directional_decoded_stacked_epoch_slices')
            _out = curr_active_pipeline.display('_display_directional_merged_pf_decoded_stacked_epoch_slices', display_context, defer_render=True, save_figure=True,
                                                # override_fig_man=custom_fig_man, 
                                                parent_output_folder=custom_figure_output_path,
                                            )
            
            # _out = EpochComputationDisplayFunctions._display_directional_merged_pf_decoded_stacked_epoch_slices(curr_active_pipeline, None, None, None, include_includelist=None, save_figure=True)
            keys_to_convert_to_benedict = ['export_paths', 'out_custom_formats_dict']
            _out = {k:benedict(v) if (k in keys_to_convert_to_benedict) else v for k, v in _out.items()}
            
            across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                '_display_directional_merged_pf_decoded_stacked_epoch_slices': _out,
            })
            

        except Exception as e:
            print(f'\tfigures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(...): "_display_directional_merged_pf_decoded_stacked_epoch_slices" failed with error: {e}\n skipping.')
            raise
        




    # ==================================================================================================================================================================================================================================================================================== #
    # '_display_generalized_decoded_yellow_blue_marginal_epochs'                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #
    if '_display_generalized_decoded_yellow_blue_marginal_epochs' in included_figures_names:
        # _display_generalized_decoded_yellow_blue_marginal_epochs ___________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        try:
            print(f'\t trying "_display_generalized_decoded_yellow_blue_marginal_epochs"')
            _out = curr_active_pipeline.display('_display_generalized_decoded_yellow_blue_marginal_epochs', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=True, is_dark_mode=False, override_fig_man=custom_fig_man)
            across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                '_display_generalized_decoded_yellow_blue_marginal_epochs': _out,
            })

        except Exception as e:
            print(f'\tgeneralized_export_figures_customizazble_completion_function(...): "_display_generalized_decoded_yellow_blue_marginal_epochs" failed with error: {e}\n skipping.')
    ## END if '_display_generalized_decoded_yellow...


    # ==================================================================================================================================================================================================================================================================================== #
    # '_display_decoded_trackID_marginal_hairy_position'                                                                                                                                                                                                                                   #
    # ==================================================================================================================================================================================================================================================================================== #
    if '_display_decoded_trackID_marginal_hairy_position' in included_figures_names:
        print(f'\t trying "_display_decoded_trackID_marginal_hairy_position"')
        interesting_hair_parameter_kwarg_dict = {
            # 'defaults': dict(extreme_threshold=0.8, opacity_max=0.7, thickness_ramping_multiplier=35),
            'overrides': dict(extreme_threshold=extreme_threshold, opacity_max=opacity_max, thickness_ramping_multiplier=thickness_ramping_multiplier),
            # '50_sec_window_scale': dict(extreme_threshold=0.5, thickness_ramping_multiplier=50),
            'full_1700_sec_session_scale': dict(extreme_threshold=0.5, thickness_ramping_multiplier=12), ## really interesting, can see the low-magnitude endcap short-like firing
            # 'experimental': dict(extreme_threshold=0.8, thickness_ramping_multiplier=55),
        }
        
        # disable_all_grid_bin_bounds_lines: bool = additional_marginal_overlaying_measured_position_kwargs.get('disable_all_grid_bin_bounds_lines', False)
        if 'disable_all_grid_bin_bounds_lines' not in additional_marginal_overlaying_measured_position_kwargs:
            additional_marginal_overlaying_measured_position_kwargs['disable_all_grid_bin_bounds_lines'] = False ## show the lines by default for big figures


        ## loop through the configs:
        for a_plot_name, a_params_kwargs in interesting_hair_parameter_kwarg_dict.items():
        
            # _display_decoded_trackID_marginal_hairy_position ___________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            display_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='trackID_marginal_hairy_position')

            try:
                sub_context = display_context.adding_context('subplot', subplot_name=a_plot_name)
                _out = curr_active_pipeline.display('_display_decoded_trackID_marginal_hairy_position', sub_context, defer_render=True, save_figure=True, override_fig_man=custom_fig_man, 
                                                    # extreme_threshold=extreme_threshold, opacity_max=opacity_max, thickness_ramping_multiplier=thickness_ramping_multiplier, **additional_marginal_overlaying_measured_position_kwargs,
                                                    **(a_params_kwargs | additional_marginal_overlaying_measured_position_kwargs), ## expand passed params 
                                                    )
                # across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                #     '_display_decoded_trackID_marginal_hairy_position': _out,
                # })
                                
            except Exception as e:
                print(f'\tgeneralized_export_figures_customizazble_completion_function(...): "_display_decoded_trackID_marginal_hairy_position" failed with error: {e}\n skipping.')
    ## END if '_display_decoded_trackID_marginal_hairy_position...


    # ==================================================================================================================================================================================================================================================================================== #
    # '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay' -- NOTE: this does all posterior export formats, not just the MultiColorCoverlay (e.g. 'greyscale', 'greyscale_shared_norm', 'viridis_shared_norm', etc.             #
    # ==================================================================================================================================================================================================================================================================================== #

    if ('_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay' in included_figures_names) or ('trackID_weighted_position_posterior' in included_figures_names):
        print(f'\t trying "_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay"')
        try:
            a_params_kwargs = {}
            display_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='trackID_weighted_position_posterior')
            _out = curr_active_pipeline.display('_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay', display_context, defer_render=True, save_figure=True,
                                                # override_fig_man=custom_fig_man, 
                                                parent_output_folder=custom_figure_output_path,
                                            )
            
            # _out = EpochComputationDisplayFunctions._display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay(curr_active_pipeline, None, None, None, include_includelist=None, save_figure=True)
            keys_to_convert_to_benedict = ['out_paths', 'out_custom_formats_dict']
            _out = {k:benedict(v) if (k in keys_to_convert_to_benedict) else v for k, v in _out.items()}

            ## merge if we can:
            _prev_out_dict = across_session_results_extended_dict.get('figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function', {}).get('_display_directional_merged_pf_decoded_stacked_epoch_slices', {}) 
            _out['out_paths'].merge(_prev_out_dict.get('export_paths', {}))
            _out['out_custom_formats_dict'].merge(_prev_out_dict.get('out_custom_formats_dict', {}))

            across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay': _out,
            })
            

            out_custom_formats_dict = _out.get('out_custom_formats_dict', None)
            if out_custom_formats_dict is not None:
                custom_merge_layout_dict = [['greyscale'],
                    ['greyscale_shared_norm'],
                    # ['psuedo2D_ignore/raw_rgba'], ## Implicitly always appends the pseudo2D_ignore/raw_rgba image at the bottom row
                ]
                _out_final_merged_image_save_paths, _out_final_merged_images = PosteriorExporting.post_export_build_combined_images(out_custom_formats_dict=out_custom_formats_dict, custom_merge_layout_dict=custom_merge_layout_dict,
                                                                                                                    epoch_name_list=['ripple'], progress_print=True) ## currently skip laps, just do ripples
                _out['final_merged_image_save_paths'] = deepcopy(_out_final_merged_image_save_paths)
                # across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                #     '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay': _out,
                # })

        except Exception as e:
            print(f'\tfigures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(...): "_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay" failed with error: {e}\n skipping.')
            raise



    # ==================================================================================================================================================================================================================================================================================== #
    # `_display_placefield_stable_formation_time_distribution`                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    if ('_display_placefield_stable_formation_time_distribution' in included_figures_names) or ('pf_stable_formation_time' in included_figures_names):
        print(f'\t trying "_display_placefield_stable_formation_time_distribution"')
        try:
            display_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='pf_stable_formation_time')
            _out = curr_active_pipeline.display('_display_placefield_stable_formation_time_distribution', display_context, defer_render=True, save_figure=True,
                                                # override_fig_man=custom_fig_man, 
                                                parent_output_folder=custom_figure_output_path,
                                            )
            
            across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                '_display_placefield_stable_formation_time_distribution': _out,
            })
            

        except Exception as e:
            print(f'\tfigures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(...): "_display_placefield_stable_formation_time_distribution" failed with error: {e}\n skipping.')
            raise


    # ==================================================================================================================================================================================================================================================================================== #
    # `_display_measured_vs_decoded_occupancy_distributions`                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    if ('_display_measured_vs_decoded_occupancy_distributions' in included_figures_names) or ('meas_v_decoded_occupancy' in included_figures_names):
        print(f'\t trying "_display_measured_vs_decoded_occupancy_distributions"')
        try:
            display_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='meas_v_decoded_occupancy')
            _out = curr_active_pipeline.display('_display_measured_vs_decoded_occupancy_distributions', display_context, defer_render=True, save_figure=True,
                                                override_fig_man=custom_fig_man,
                                                # parent_output_folder=custom_figure_output_path,
                                                #  size=[6.5, 2], dpi=100,
                                                size=[3.5, 2], dpi=100,
                                                prepare_for_publication=False,
                                                # prepare_for_publication=True,
                                            )
            
            across_session_results_extended_dict['figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'].update({
                '_display_measured_vs_decoded_occupancy_distributions': _out,
            })
            

        except Exception as e:
            print(f'\tfigures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(...): "_display_measured_vs_decoded_occupancy_distributions" failed with error: {e}\n skipping.')
            raise



    print(f'>>\t done with {curr_session_context}')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # return True
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
                                    # "compute_and_export_marginals_dfs_completion_function": compute_and_export_marginals_dfs_completion_function,
                                    'determine_session_t_delta_completion_function': determine_session_t_delta_completion_function,
                                    'perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function': perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function,
                                    'compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function': compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function,
                                    # 'kdiba_session_post_fixup_completion_function': kdiba_session_post_fixup_completion_function,
                                    'kdiba_session_post_fixup_completion_function': kdiba_session_post_fixup_completion_function,
                                    'export_session_h5_file_completion_function': export_session_h5_file_completion_function,
                                    'compute_and_export_session_wcorr_shuffles_completion_function': compute_and_export_session_wcorr_shuffles_completion_function,
                                    'compute_and_export_session_instantaneous_spike_rates_completion_function': compute_and_export_session_instantaneous_spike_rates_completion_function,
                                    'compute_and_export_session_extended_placefield_peak_information_completion_function': compute_and_export_session_extended_placefield_peak_information_completion_function,
                                    'compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function': compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function,
                                    'backup_previous_session_files_completion_function': backup_previous_session_files_completion_function,
                                    'compute_and_export_session_trial_by_trial_performance_completion_function': compute_and_export_session_trial_by_trial_performance_completion_function,
                                    'save_custom_session_files_completion_function': save_custom_session_files_completion_function,
                                    'compute_and_export_cell_first_spikes_characteristics_completion_function': compute_and_export_cell_first_spikes_characteristics_completion_function,
                                    'figures_plot_cell_first_spikes_characteristics_completion_function': figures_plot_cell_first_spikes_characteristics_completion_function,
                                    'generalized_decode_epochs_dict_and_export_results_completion_function': generalized_decode_epochs_dict_and_export_results_completion_function,
                                    'figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function': figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function,
                                    # 'generalized_export_figures_customizazble_completion_function': generalized_export_figures_customizazble_completion_function,
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





# %%
