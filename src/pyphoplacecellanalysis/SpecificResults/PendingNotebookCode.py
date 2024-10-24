# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Dict, Tuple, Any, Union
from matplotlib import cm, pyplot as plt
from matplotlib.gridspec import GridSpec
from neuropy.core import Laps
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.result_context import IdentifyingContext
from nptyping import NDArray
import attrs
import matplotlib as mpl
import napari
from neuropy.core.epoch import Epoch, ensure_dataframe
from neuropy.analyses.placefields import PfND
import numpy as np
import pandas as pd
from attrs import asdict, astuple, define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes

from functools import wraps, partial
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult


# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)

import matplotlib.pyplot as plt
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# ==================================================================================================================== #
# 2024-10-09 - Building Custom Individual time_bin decoded posteriors                                                  #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['individual_time_bin', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-09 09:27', related_items=[])
def _perform_build_individual_time_bin_decoded_posteriors_df(curr_active_pipeline, track_templates, all_directional_laps_filter_epochs_decoder_result, transfer_column_names_list: Optional[List[str]]=None):
    """ 
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_build_individual_time_bin_decoded_posteriors_df
    filtered_laps_time_bin_marginals_df = _perform_build_individual_time_bin_decoded_posteriors_df(curr_active_pipeline, track_templates=track_templates, all_directional_laps_filter_epochs_decoder_result=all_directional_laps_filter_epochs_decoder_result)
    
    """
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import co_filter_epochs_and_spikes

    ## INPUTS: all_directional_laps_filter_epochs_decoder_result
    if transfer_column_names_list is None:
        transfer_column_names_list = []
    # transfer_column_names_list: List[str] = ['maze_id', 'lap_dir', 'lap_id']
    TIME_OVERLAP_PREVENTION_EPSILON: float = 1e-12
    (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                        additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])
    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = laps_track_identity_marginals_tuple
    non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = laps_non_marginalized_decoder_marginals_tuple
    laps_time_bin_marginals_df: pd.DataFrame = all_directional_laps_filter_epochs_decoder_result.build_per_time_bin_marginals_df(active_marginals_tuple=(laps_directional_marginals, laps_track_identity_marginals, non_marginalized_decoder_marginals),
                                                                                                                                columns_tuple=(['P_LR', 'P_RL'], ['P_Long', 'P_Short'], ['long_LR', 'long_RL', 'short_LR', 'short_RL']), transfer_column_names_list=transfer_column_names_list)
    laps_time_bin_marginals_df['start'] = laps_time_bin_marginals_df['start'] + TIME_OVERLAP_PREVENTION_EPSILON ## ENSURE NON-OVERLAPPING

    ## INPUTS: laps_time_bin_marginals_df
    # active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.33333333333333)
    active_min_num_unique_aclu_inclusions_requirement = None # must be none for individual `time_bin` periods
    filtered_laps_time_bin_marginals_df, active_spikes_df = co_filter_epochs_and_spikes(active_spikes_df=get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=curr_active_pipeline.global_computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz),
                                                                    active_epochs_df=laps_time_bin_marginals_df, included_aclus=track_templates.any_decoder_neuron_IDs, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement,
                                                                    epoch_id_key_name='lap_individual_time_bin_id', no_interval_fill_value=-1, add_unique_aclus_list_column=True, drop_non_epoch_spikes=True)
    return filtered_laps_time_bin_marginals_df


# ==================================================================================================================== #
# 2024-10-08 - Reliability and Active Cell Testing                                                                     #
# ==================================================================================================================== #

# appearing_or_disappearing_aclus, appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus
@function_attributes(short_name=None, tags=['performance'], input_requires=[], output_provides=[], uses=['_do_train_test_split_decode_and_evaluate'], used_by=[], creation_date='2024-10-08 00:00', related_items=[])
def _perform_run_rigorous_decoder_performance_assessment(curr_active_pipeline, included_neuron_IDs, active_laps_decoding_time_bin_size: float = 0.25):
    # Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult, TrainTestLapsSplitting, CustomDecodeEpochsResult, decoder_name, epoch_split_key, get_proper_global_spikes_df, DirectionalPseudo2DDecodersResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _do_train_test_split_decode_and_evaluate
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import PfND
    from neuropy.core.session.dataSession import Laps
    # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance

    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]

    def _add_extra_epochs_df_columns(epochs_df: pd.DataFrame):
        """ captures: global_session, t_start, t_delta, t_end
        """
        epochs_df = epochs_df.sort_values(['start', 'stop', 'label']).reset_index(drop=True) # Sort by columns: 'start' (ascending), 'stop' (ascending), 'label' (ascending)
        epochs_df = epochs_df.drop_duplicates(subset=['start', 'stop', 'label'])
        epochs_df = epochs_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        epochs_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df=epochs_df, global_session=deepcopy(global_session), replace_existing=True)
        return epochs_df

    directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data.get('TrainTestSplit', None)
    force_recompute_directional_train_test_split_result: bool = False
    if (directional_train_test_split_result is None) or force_recompute_directional_train_test_split_result:
        ## recompute
        print(f"'TrainTestSplit' not computed, recomputing...")
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_train_test_split'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data['TrainTestSplit']
        assert directional_train_test_split_result is not None, f"faiiled even after recomputation"
        print('\tdone.')

    training_data_portion: float = directional_train_test_split_result.training_data_portion
    test_data_portion: float = directional_train_test_split_result.test_data_portion
    print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

    test_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = directional_train_test_split_result.test_epochs_dict
    train_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = directional_train_test_split_result.train_epochs_dict
    train_lap_specific_pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = directional_train_test_split_result.train_lap_specific_pf1D_Decoder_dict
    # OUTPUTS: train_test_split_laps_df_dict
    
    # MAIN _______________________________________________________________________________________________________________ #
    
    complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results = _do_train_test_split_decode_and_evaluate(curr_active_pipeline=curr_active_pipeline, active_laps_decoding_time_bin_size=active_laps_decoding_time_bin_size,
                                                                                                                                                                                                                                                  included_neuron_IDs=included_neuron_IDs,
                                                                                                                                                                                                                                                  force_recompute_directional_train_test_split_result=False, compute_separate_decoder_results=True)
    (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
    print(f"percent_laps_track_identity_estimated_correctly: {round(percent_laps_track_identity_estimated_correctly*100.0, ndigits=3)}%")

    if _out_separate_decoder_results is not None:
        assert len(_out_separate_decoder_results) == 3, f"_out_separate_decoder_results: {_out_separate_decoder_results}"
        test_decoder_results_dict, train_decoded_results_dict, train_decoded_measured_diff_df_dict = _out_separate_decoder_results
        ## OUTPUTS: test_decoder_results_dict, train_decoded_results_dict
    # _remerged_laps_dfs_dict = {}
    # for a_decoder_name, a_test_epochs_df in test_epochs_dict.items():
    #     a_train_epochs_df = train_epochs_dict[a_decoder_name]
    #     a_train_epochs_df['test_train_epoch_type'] = 'train'
    #     a_test_epochs_df['test_train_epoch_type'] = 'test'
    #     _remerged_laps_dfs_dict[a_decoder_name] = pd.concat([a_train_epochs_df, a_test_epochs_df], axis='index')
    #     _remerged_laps_dfs_dict[a_decoder_name] = _add_extra_epochs_df_columns(epochs_df=_remerged_laps_dfs_dict[a_decoder_name])
        
    return (complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results) 


@function_attributes(short_name=None, tags=['long_short', 'firing_rate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-17 05:22', related_items=['determine_neuron_exclusivity_from_firing_rate'])
def compute_all_cells_long_short_firing_rate_df(global_spikes_df: pd.DataFrame):
    """ computes the firing rates for all cells (not just placecells or excitatory cells) for the long and short track periods, and then their differences
    These firing rates are not spatially binned because they aren't just place cells.
    
    columns: ['LS_diff_firing_rate_Hz']: will be positive for Short-preferring cells and negative for Long-preferring ones.
    
    Usage:
    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_cells_long_short_firing_rate_df

        df_combined = compute_all_cells_long_short_firing_rate_df(global_spikes_df=global_spikes_df)
        df_combined

        print(list(df_combined.columns)) # ['long_num_spikes_count', 'short_num_spikes_count', 'global_num_spikes_count', 'long_firing_rate_Hz', 'short_firing_rate_Hz', 'global_firing_rate_Hz', 'LS_diff_firing_rate_Hz', 'firing_rate_percent_diff']
        
    """
    ## Needs to consider not only place cells but interneurons as well
    # global_all_spikes_counts # 73 rows
    # global_spikes_df.aclu.unique() # 108

    ## Split into the pre- and post- delta epochs
    # global_spikes_df['t_rel_seconds']
    # global_spikes_df

    # is_theta, is_ripple, maze_id, maze_relative_lap

    ## Split on 'maze_id'
    # partition

    from pyphocorehelpers.indexing_helpers import partition_df, reorder_columns_relative
    # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import add_spikes_df_placefield_inclusion_columns


    ## INPUTS: global_spikes_df

    ## Compute effective epoch duration by finding the earliest and latest spike in epoch.
    def _add_firing_rates_from_computed_durations(a_df: pd.DataFrame):
        spike_times = a_df['t_rel_seconds'].values
        end_t = np.nanmax(spike_times)
        start_t = np.nanmin(spike_times)
        duration_t: float = end_t - start_t    
        return duration_t, (start_t, end_t)


    partitioned_dfs = dict(zip(*partition_df(global_spikes_df, partitionColumn='maze_id'))) # non-maze is also an option, right?
    long_all_spikes_df: pd.DataFrame = partitioned_dfs[1]
    short_all_spikes_df: pd.DataFrame = partitioned_dfs[2]
    
    ## sum total number of spikes over the entire duration
    # Performed 1 aggregation grouped on column: 'aclu'
    long_all_spikes_count_df = long_all_spikes_df.groupby(['aclu']).agg(num_spikes_count=('t_rel_seconds', 'count')).reset_index()[['aclu', 'num_spikes_count']].set_index('aclu')
    # Performed 1 aggregation grouped on column: 'aclu'
    short_all_spikes_count_df = short_all_spikes_df.groupby(['aclu']).agg(num_spikes_count=('t_rel_seconds', 'count')).reset_index()[['aclu', 'num_spikes_count']].set_index('aclu')

    ## TODO: exclude replay periods

    ## OUTPUTS: long_all_spikes_count_df, short_all_spikes_count_df
        
    long_duration_t, _long_start_end_tuple = _add_firing_rates_from_computed_durations(long_all_spikes_df)
    long_all_spikes_count_df['firing_rate_Hz'] = long_all_spikes_count_df['num_spikes_count'] / long_duration_t

    short_duration_t, _short_start_end_tuple = _add_firing_rates_from_computed_durations(short_all_spikes_df)
    short_all_spikes_count_df['firing_rate_Hz'] = short_all_spikes_count_df['num_spikes_count'] / short_duration_t

    global_duration_t: float = long_duration_t + short_duration_t
    
    ## OUTPUTS: long_all_spikes_count_df, short_all_spikes_count_df

    # long_all_spikes_count_df
    # short_all_spikes_count_df

    # Performed 2 aggregations grouped on column: 't_rel_seconds'
    # long_all_spikes_df[['t_rel_seconds']].agg(t_rel_seconds_min=('t_rel_seconds', 'min'), t_rel_seconds_max=('t_rel_seconds', 'max')).reset_index()

    # short_all_spikes_df[['t_rel_seconds']].agg(t_rel_seconds_min=('t_rel_seconds', 'min'), t_rel_seconds_max=('t_rel_seconds', 'max')).reset_index()
    # long_all_spikes_df = long_all_spikes_df.groupby(['t_rel_seconds']).agg(t_rel_seconds_min=('t_rel_seconds', 'min'), t_rel_seconds_max=('t_rel_seconds', 'max')).reset_index()

    # Add prefixes to column names
    df1_prefixed = long_all_spikes_count_df.add_prefix("long_")
    df2_prefixed = short_all_spikes_count_df.add_prefix("short_")

    # Combine along the index
    df_combined = pd.concat([df1_prefixed, df2_prefixed], axis=1)

    ## Move the "height" columns to the end
    # df_combined = reorder_columns_relative(df_combined, column_names=list(filter(lambda column: column.endswith('_firing_rate_Hz'), existing_columns)), relative_mode='end')
    # df_combined = reorder_columns_relative(df_combined, column_names=['long_firing_rate_Hz', 'short_firing_rate_Hz'], relative_mode='end')

    df_combined = reorder_columns_relative(df_combined, column_names=['long_num_spikes_count', 'short_num_spikes_count', 'long_firing_rate_Hz', 'short_firing_rate_Hz'], relative_mode='end')
    
    # ['long_firing_rate_Hz', 'short_firing_rate_Hz', 'long_num_spikes_count', 'short_num_spikes_count', 'LS_diff_firing_rate_Hz', 'firing_rate_percent_diff']
        
    ## Compare the differnece between the two periods
    df_combined['LS_diff_firing_rate_Hz'] = df_combined['long_firing_rate_Hz'] - df_combined['short_firing_rate_Hz']
    
    # Calculate the percent difference in firing rate
    df_combined["firing_rate_percent_diff"] = (df_combined['LS_diff_firing_rate_Hz'] / df_combined["long_firing_rate_Hz"]) * 100

    df_combined['global_num_spikes_count'] = df_combined['long_num_spikes_count'] + df_combined['short_num_spikes_count']
    df_combined['global_firing_rate_Hz'] = df_combined['global_num_spikes_count'] / global_duration_t
    
    df_combined = reorder_columns_relative(df_combined, column_names=['long_num_spikes_count', 'short_num_spikes_count', 'global_num_spikes_count', 'long_firing_rate_Hz', 'short_firing_rate_Hz', 'global_firing_rate_Hz'], relative_mode='start')\
        
        
    # df_combined["long_num_spikes_percent"] = (df_combined['long_num_spikes_count'] / df_combined["global_num_spikes_count"]) * 100
    # df_combined["short_num_spikes_percent"] = (df_combined['short_num_spikes_count'] / df_combined["global_num_spikes_count"]) * 100
    
    # df_combined["firing_rate_percent_diff"] = (df_combined['LS_diff_firing_rate_Hz'] / df_combined["long_firing_rate_Hz"]) * 100
    
    
    return df_combined

@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-08 10:37', related_items=['compute_all_cells_long_short_firing_rate_df'])
def determine_neuron_exclusivity_from_firing_rate(df_combined: pd.DataFrame, firing_rate_required_diff_Hz: float = 1.0, maximum_opposite_period_firing_rate_Hz: float = 1.0):
    """ 
    firing_rate_required_diff_Hz: float = 1.0 # minimum difference required for a cell to be considered Long- or Short-"preferring"
    maximum_opposite_period_firing_rate_Hz: float = 1.0 # maximum allowed firing rate in the opposite period to be considered exclusive

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_cells_long_short_firing_rate_df, determine_neuron_exclusivity_from_firing_rate

        df_combined = compute_all_cells_long_short_firing_rate_df(global_spikes_df=global_spikes_df)
        (LpC_df, SpC_df, LxC_df, SxC_df), (LpC_aclus, SpC_aclus, LxC_aclus, SxC_aclus) = determine_neuron_exclusivity_from_firing_rate(df_combined=df_combined, firing_rate_required_diff_Hz=firing_rate_required_diff_Hz, 
																															   maximum_opposite_period_firing_rate_Hz=maximum_opposite_period_firing_rate_Hz)

        ## Extract the aclus
        print(f'LpC_aclus: {LpC_aclus}')
        print(f'SpC_aclus: {SpC_aclus}')

        print(f'LxC_aclus: {LxC_aclus}')
        print(f'SxC_aclus: {SxC_aclus}')

    """
    # Sort by column: 'LS_diff_firing_rate_Hz' (ascending)
    df_combined = df_combined.sort_values(['LS_diff_firing_rate_Hz'])
    # df_combined = df_combined.sort_values(['firing_rate_percent_diff'])
    df_combined

    # df_combined['LS_diff_firing_rate_Hz']

    # df_combined['firing_rate_percent_diff']

    LpC_df = df_combined[df_combined['LS_diff_firing_rate_Hz'] > firing_rate_required_diff_Hz]
    SpC_df = df_combined[df_combined['LS_diff_firing_rate_Hz'] < -firing_rate_required_diff_Hz]

    LxC_df = LpC_df[LpC_df['short_firing_rate_Hz'] <= maximum_opposite_period_firing_rate_Hz]
    SxC_df = SpC_df[SpC_df['long_firing_rate_Hz'] <= maximum_opposite_period_firing_rate_Hz]


    ## Let's consider +/- 50% diff XxC cells
    # LpC_df = df_combined[df_combined['firing_rate_percent_diff'] > 50.0]
    # SpC_df = df_combined[df_combined['firing_rate_percent_diff'] < -50.0]

    ## Extract the aclus"
    LpC_aclus = LpC_df.index.values
    SpC_aclus = SpC_df.index.values

    print(f'LpC_aclus: {LpC_aclus}')
    print(f'SpC_aclus: {SpC_aclus}')

    LxC_aclus = LxC_df.index.values
    SxC_aclus = SxC_df.index.values

    print(f'LxC_aclus: {LxC_aclus}')
    print(f'SxC_aclus: {SxC_aclus}')
    
    ## OUTPUTS: LpC_df, SpC_df, LxC_df, SxC_df

    ## OUTPUTS: LpC_aclus, SpC_aclus, LxC_aclus, SxC_aclus
    return (LpC_df, SpC_df, LxC_df, SxC_df), (LpC_aclus, SpC_aclus, LxC_aclus, SxC_aclus)


# ==================================================================================================================== #
# 2024-10-04 - Parsing `ProgrammaticDisplayFunctionTesting` output folder                                              #
# ==================================================================================================================== #
from pyphocorehelpers.assertion_helpers import Assert
from pyphocorehelpers.indexing_helpers import partition_df_dict, partition
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types

ContextDescStr = NewType('ContextDescStr', str) # like '2023-07-11_kdiba_gor01_one'
ImageNameStr = NewType('ImageNameStr', str) # like '2006-6-07_11-26-53/kdiba_gor01_one_2006-6-07_11-26-53_maze1__display_1d_placefield_validations.pdf'

class ProgrammaticDisplayFunctionTestingFolderImageLoading:
    """ 
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import ProgrammaticDisplayFunctionTestingFolderImageLoading
    
    """

    @function_attributes(short_name=None, tags=['ProgrammaticDisplayFunctionTesting', 'parse', 'filesystem'], input_requires=[], output_provides=[], uses=[], used_by=['parse_ProgrammaticDisplayFunctionTesting_image_folder'], creation_date='2024-10-04 12:21', related_items=[])
    @classmethod
    def parse_image_path(cls, programmatic_display_function_testing_path: Path, file_path: Path, debug_print=False) -> Tuple[IdentifyingContext, str, datetime]:
        """ Parses `"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting"` 
        "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2023-04-11/kdiba/gor01/one/2006-6-09_1-22-43/kdiba_gor01_one_2006-6-09_1-22-43_batch_plot_test_long_only_[45].png"
        Write a function that parses the following path structure: "./2023-04-11/kdiba/gor01/one/2006-6-09_1-22-43/kdiba_gor01_one_2006-6-09_1-22-43_batch_plot_test_long_only_[45].png"
        Into the following variable names: `/image_export_day_date/format_name/animal/exper_name/session_name/image_name`
        """
        test_relative_image_path = file_path.relative_to(programmatic_display_function_testing_path) # .resolve() ## RESOLVE MESSES UP SYMLINKS!
        if debug_print:
            print(f'{test_relative_image_path = }')

        # Split the path into components
        # parts = file_path.strip("./").split(os.sep)
        # Convert to a list of path components
        parts = test_relative_image_path.parts
        if debug_print:
            print(f'parts: {parts}')
        
        if len(parts) < 6:
            raise ValueError(f'parsed path should have at least 6 parts, but this one only has: {len(parts)}.\nparts: {parts}')
        
        if len(parts) > 6:
            joined_final_part: str = '/'.join(parts[5:]) # return anything after that back into a str
            parts = parts[:5] + (joined_final_part, )
            
        Assert.len_equals(parts, 6)

        # Assign the variables from the path components
        image_export_day_date = parts[0]      # "./2023-04-11"
        format_name = parts[1]                # "kdiba"
        animal = parts[2]                     # "gor01"
        exper_name = parts[3]                 # "one"
        session_name = parts[4]               # "2006-6-09_1-22-43"
        image_name = parts[5]                 # "kdiba_gor01_one_2006-6-09_1-22-43_batch_plot_test_long_only_[45].png"

        session_context = IdentifyingContext(format_name=format_name, animal=animal, exper_name=exper_name, session_name=session_name)
        # Parse image_export_day_date as a date (YYYY-mm-dd)
        image_export_day_date: datetime = datetime.strptime(image_export_day_date, "%Y-%m-%d")
        
        # return image_export_day_date, format_name, animal, exper_name, session_name, image_name
        return session_context, image_name, image_export_day_date, file_path


    @function_attributes(short_name=None, tags=['ProgrammaticDisplayFunctionTesting', 'filesystem', 'images', 'load'], input_requires=[], output_provides=[], uses=['parse_image_path'], used_by=[], creation_date='2024-10-04 12:21', related_items=[])
    @classmethod
    def parse_ProgrammaticDisplayFunctionTesting_image_folder(cls, programmatic_display_function_testing_path: Path, save_csv: bool = True):
        """
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import ProgrammaticDisplayFunctionTestingFolderImageLoading

            programmatic_display_function_testing_path: Path = Path('/home/halechr/repos/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').resolve()
            programmatic_display_function_outputs_df, programmatic_display_function_outputs_tuples, csv_out_path = ProgrammaticDisplayFunctionTestingFolderImageLoading.parse_ProgrammaticDisplayFunctionTesting_image_folder(programmatic_display_function_testing_path=programmatic_display_function_testing_path)
            programmatic_display_function_outputs_df

        """
        # programmatic_display_function_outputs_dict: Dict[IdentifyingContext, List] = {}
        programmatic_display_function_outputs_tuples: List[Tuple[IdentifyingContext, str, datetime]] = []

        Assert.path_exists(programmatic_display_function_testing_path)

        # Recursively enumerate all files in the directory
        def enumerate_files(directory: Path):
            return [file for file in directory.rglob('*') if file.is_file()]

        # Example usage
        all_files = enumerate_files(programmatic_display_function_testing_path)

        for test_image_path in all_files:
            try:
                # image_export_day_date, format_name, animal, exper_name, session_name, image_name = parse_image_path(programmatic_display_function_testing_path, test_image_path)
                # session_context, image_name, image_export_day_date = parse_image_path(programmatic_display_function_testing_path, test_image_path)
                # print(image_export_day_date, format_name, animal, exper_name, session_name, image_name)
                programmatic_display_function_outputs_tuples.append(cls.parse_image_path(programmatic_display_function_testing_path, test_image_path))
            except ValueError as e:
                # couldn't parse, skipping
                pass
            except Exception as e:
                raise e

        programmatic_display_function_outputs_df: pd.DataFrame = pd.DataFrame.from_records(programmatic_display_function_outputs_tuples, columns=['context', 'image_name', 'export_date', 'file_path'])
        # Sort by columns: 'context' (ascending), 'image_name' (ascending), 'export_date' (descending)
        programmatic_display_function_outputs_df = programmatic_display_function_outputs_df.sort_values(['context', 'image_name', 'export_date'], ascending=[True, True, False], key=lambda s: s.apply(str) if s.name in ['context'] else s).reset_index(drop=True)
        if save_csv:
            csv_out_path = programmatic_display_function_testing_path.joinpath('../../PhoDibaPaper2024Book/data').resolve().joinpath('programmatic_display_function_image_paths.csv')
            programmatic_display_function_outputs_df.to_csv(csv_out_path)

        return programmatic_display_function_outputs_df, programmatic_display_function_outputs_tuples, csv_out_path


    # @classmethod
    # def load_saved_ProgrammaticDisplayFunctionTesting_csv_and_build_widget(cls, programmatic_display_function_outputs_df: pd.DataFrame):
    @classmethod
    def build_ProgrammaticDisplayFunctionTesting_browsing_widget(cls, programmatic_display_function_outputs_df: pd.DataFrame):
        """
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import parse_ProgrammaticDisplayFunctionTesting_image_folder

            programmatic_display_function_testing_path: Path = Path('/home/halechr/repos/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').resolve()
            programmatic_display_function_outputs_df, programmatic_display_function_outputs_tuples, csv_out_path = ProgrammaticDisplayFunctionTestingFolderImageLoading.parse_ProgrammaticDisplayFunctionTesting_image_folder(programmatic_display_function_testing_path=programmatic_display_function_testing_path)
            programmatic_display_function_outputs_df

        """
        from pyphocorehelpers.gui.Jupyter.JupyterImageNavigatorWidget import ImageNavigator, ContextSidebar, build_context_images_navigator_widget

        # Assert.path_exists(in_path)

        # _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, List[Tuple[str, str]]]] = {}
        _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, Dict[datetime, Path]]] = {}
        for ctx, a_ctx_df in partition_df_dict(programmatic_display_function_outputs_df, partitionColumn='context').items():
            _final_out_dict_dict[ctx] = {}
            for an_img_name, an_img_df in partition_df_dict(a_ctx_df, partitionColumn='image_name').items():
                # _final_out_dict_dict[ctx][an_img_name] = list(zip(an_img_df['export_date'].values, an_img_df['file_path'].values)) #partition_df_dict(an_img_df, partitionColumn='image_name') 
                _final_out_dict_dict[ctx][an_img_name] = {datetime.strptime(k, "%Y-%m-%d"):Path(v).resolve() for k, v in dict(zip(an_img_df['export_date'].values, an_img_df['file_path'].values)).items() if v.endswith('.png')}

        """
        {'2023-07-11_kdiba_gor01_one': {'2006-6-07_11-26-53/kdiba_gor01_one_2006-6-07_11-26-53_maze1__display_1d_placefield_validations.pdf': [('2023-07-11',
            'C:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\EXTERNAL\\Screenshots\\ProgrammaticDisplayFunctionTesting\\2023-07-11\\2023-07-11\\kdiba\\gor01\\one\\2006-6-07_11-26-53\\kdiba_gor01_one_2006-6-07_11-26-53_maze1__display_1d_placefield_validations.pdf')],
        '2006-6-07_11-26-53/kdiba_gor01_one_2006-6-07_11-26-53_maze2__display_1d_placefield_validations.pdf': [('2023-07-11',
            'C:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\EXTERNAL\\Screenshots\\ProgrammaticDisplayFunctionTesting\\2023-07-11\\2023-07-11\\kdiba\\gor01\\one\\2006-6-07_11-26-53\\kdiba_gor01_one_2006-6-07_11-26-53_maze2__display_1d_placefield_validations.pdf')],
            ...
        """
        ## INPUTS: _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, Dict[datetime, Path]]]
        context_tabs_dict = {curr_context_desc_str:build_context_images_navigator_widget(curr_context_images_dict, curr_context_desc_str=curr_context_desc_str, max_num_widget_debug=2) for curr_context_desc_str, curr_context_images_dict in list(_final_out_dict_dict.items())}
        sidebar = ContextSidebar(context_tabs_dict)
        

        return sidebar, context_tabs_dict, _final_out_dict_dict





# ==================================================================================================================== #
# 2024-08-21 Plotting Generated Transition Matrix Sequences                                                            #
# ==================================================================================================================== #
from matplotlib.colors import Normalize

def apply_colormap(image: np.ndarray, color: tuple) -> np.ndarray:
    colored_image = np.zeros((*image.shape, 3), dtype=np.float32)
    for i in range(3):
        colored_image[..., i] = image * color[i]
    return colored_image

def blend_images(images: list, cmap=None) -> np.ndarray:
    """ Tries to pre-combine images to produce an output image of the same size
    
    # 'coolwarm'
    images = [a_seq_mat.todense().T for i, a_seq_mat in enumerate(sequence_frames_sparse)]
    blended_image = blend_images(images)
    # blended_image = blend_images(images, cmap='coolwarm')
    blended_image

    # blended_image = Image.fromarray(blended_image, mode="RGB")
    # # blended_image = get_array_as_image(blended_image, desired_height=100, desired_width=None, skip_img_normalization=True)
    # blended_image


    """
    if cmap is None:
        # Non-colormap mode:
        # Ensure images are in the same shape
        combined_image = np.zeros_like(images[0], dtype=np.float32)
        
        for img in images:
            combined_image += img.astype(np.float32)

    else:
        # colormap mode
        # Define a colormap (blue to red)
        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=len(images) - 1)
        
        combined_image = np.zeros((*images[0].shape, 3), dtype=np.float32)
        
        for i, img in enumerate(images):
            color = cmap(norm(i))[:3]  # Get RGB color from colormap
            colored_image = apply_colormap(img, color)
            combined_image += colored_image    

    combined_image = np.clip(combined_image, 0, 255)  # Ensure pixel values are within valid range
    return combined_image.astype(np.uint8)

def visualize_multiple_image_items(images: list, threshold=1e-3) -> None:
    """ Sample multiple pg.ImageItems overlayed on one another
    
    # Example usage:
    image1 = np.random.rand(100, 100) * 100  # Example image 1
    image2 = np.random.rand(100, 100) * 100  # Example image 2
    image3 = np.random.rand(100, 100) * 100  # Example image 3

    image1
    # Define the threshold

    _out = visualize_multiple_image_items([image1, image2, image3], threshold=50)

    """
    app = pg.mkQApp('visualize_multiple_image_items')  # Initialize the Qt application
    win = pg.GraphicsLayoutWidget(show=True)
    view = win.addViewBox()
    view.setAspectLocked(True)
    
    for img in images:
        if threshold is not None:
            # Create a masked array, masking values below the threshold
            img = np.ma.masked_less(img, threshold)

        image_item = pg.ImageItem(img)
        view.addItem(image_item)

    # QtGui.QApplication.instance().exec_()
    return app, win, view


# ==================================================================================================================== #
# 2024-08-16 - Image Processing Techniques                                                                             #
# ==================================================================================================================== #
def plot_grad_quiver(sobel_x, sobel_y, downsample_step=1):
    """ 
    
    # Compute the magnitude of the gradient
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    """
    # Compute the magnitude of the gradient
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # Create a grid of coordinates for plotting arrows
    Y, X = np.meshgrid(np.arange(gradient_magnitude.shape[0]), np.arange(gradient_magnitude.shape[1]), indexing='ij')

    # Downsample the arrow plot for better visualization (optional)
    
    X_downsampled = X[::downsample_step, ::downsample_step]
    Y_downsampled = Y[::downsample_step, ::downsample_step]
    sobel_x_downsampled = sobel_x[::downsample_step, ::downsample_step]
    sobel_y_downsampled = sobel_y[::downsample_step, ::downsample_step]

    # Plotting the gradient magnitude and arrows representing the direction
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(gradient_magnitude, cmap='gray', origin='lower')
    plt.quiver(X_downsampled, Y_downsampled, sobel_x_downsampled, sobel_y_downsampled, 
            color='red', angles='xy', scale_units='xy') # , scale=5, width=0.01
    plt.title('Gradient Magnitude with Direction Arrows')
    plt.axis('off')
    plt.show()
    
    return fig



# ==================================================================================================================== #
# 2024-07-15 - Factored out of Across Session Notebook                                                                 #
# ==================================================================================================================== #
def build_single_time_bin_size_dfs(all_sessions_all_scores_epochs_df, all_sessions_epochs_df, all_sessions_epochs_time_bin_df, target_time_bin_size: float, included_columns = ['delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch']):
    """ Filters the epochs dataframe down to a single time_bin_size specified by `target_time_bin_size`. 
     
     from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_single_time_bin_size_dfs
      
    """
    from neuropy.utils.misc import add_explicit_dataframe_columns_from_lookup_df

    print(f'all_sessions_ripple_df.time_bin_size.unique(): {all_sessions_epochs_df.time_bin_size.unique()}')
    single_time_bin_size_all_sessions_epochs_df = deepcopy(all_sessions_epochs_df[np.isclose(all_sessions_epochs_df['time_bin_size'], target_time_bin_size)])
    print(f'np.shape(single_time_bin_size_all_sessions_ripple_df): {np.shape(single_time_bin_size_all_sessions_epochs_df)}')

    print(f'all_sessions_ripple_time_bin_df.time_bin_size.unique(): {all_sessions_epochs_time_bin_df.time_bin_size.unique()}')
    single_time_bin_size_all_sessions_epochs_time_bin_df = deepcopy(all_sessions_epochs_time_bin_df[np.isclose(all_sessions_epochs_time_bin_df['time_bin_size'], target_time_bin_size)])
    print(f'np.shape(single_time_bin_size_all_sessions_ripple_time_bin_df): {np.shape(single_time_bin_size_all_sessions_epochs_time_bin_df)}')

    # single_time_bin_size_all_sessions_ripple_time_bin_df
    # single_time_bin_size_all_sessions_ripple_df # has ['ripple_start_t']
    # all_sessions_all_scores_ripple_df

    ## recover the important columns (user-annotation, epoch validity) from the newer `all_sessions_all_scores_ripple_df` for use in 'single_time_bin_size_all_sessions_ripple_df'
    all_sessions_all_scores_epochs_df['delta_aligned_start_t'] = all_sessions_all_scores_epochs_df['delta_aligned_start_t'].astype(float)
    single_time_bin_size_all_sessions_epochs_df['delta_aligned_start_t'] = single_time_bin_size_all_sessions_epochs_df['delta_aligned_start_t'].astype(float)

    # Added 'delta_aligned_start_t' for the merge
    single_time_bin_size_all_sessions_epochs_df = add_explicit_dataframe_columns_from_lookup_df(single_time_bin_size_all_sessions_epochs_df, all_sessions_all_scores_epochs_df[included_columns], join_column_name='delta_aligned_start_t')
    single_time_bin_size_all_sessions_epochs_df.sort_values(by=['delta_aligned_start_t'], inplace=True) # Need to re-sort by timestamps once done
    single_time_bin_size_all_sessions_epochs_df

    single_time_bin_size_all_sessions_epochs_time_bin_df = add_explicit_dataframe_columns_from_lookup_df(single_time_bin_size_all_sessions_epochs_time_bin_df, all_sessions_all_scores_epochs_df[included_columns], join_column_name='delta_aligned_start_t')
    single_time_bin_size_all_sessions_epochs_time_bin_df.sort_values(by=['t_bin_center'], inplace=True) # Need to re-sort by timestamps once done
    
    ## Add plotly helper columns:
    for a_df in (all_sessions_all_scores_epochs_df, all_sessions_epochs_df, all_sessions_epochs_time_bin_df, single_time_bin_size_all_sessions_epochs_df, single_time_bin_size_all_sessions_epochs_time_bin_df):
        a_df['pre_post_delta_category'] = 'post-delta'
        a_df['pre_post_delta_category'][a_df['delta_aligned_start_t'] < 0.0] = 'pre-delta'

    ## OUTPUTS: single_time_bin_size_all_sessions_ripple_df, single_time_bin_size_all_sessions_ripple_time_bin_df
    return single_time_bin_size_all_sessions_epochs_df, single_time_bin_size_all_sessions_epochs_time_bin_df




# ==================================================================================================================== #
# 2024-06-26 - Shuffled WCorr Output with working histogram                                                            #
# ==================================================================================================================== #
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from typing import NewType

import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer

from neuropy.utils.mixins.indexing_helpers import get_dict_subset

from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names
ShuffleIdx = NewType('ShuffleIdx', int)

def finalize_output_shuffled_wcorr(a_curr_active_pipeline, decoder_names, custom_suffix: str):
    """
    Gets the shuffled wcorr results and outputs the final histogram for this session

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import finalize_output_shuffled_wcorr

        decoder_names = deepcopy(track_templates.get_decoder_names())
        wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path) = finalize_output_shuffled_wcorr(curr_active_pipeline=curr_active_pipeline,
                                                                                                                                        decoder_names=decoder_names, custom_suffix=custom_suffix)
    """
    from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str


    wcorr_shuffle_results: SequenceBasedComputationsContainer = a_curr_active_pipeline.global_computation_results.computed_data.get('SequenceBased', None)
    if wcorr_shuffle_results is not None:    
        wcorr_shuffles: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
        wcorr_shuffles: WCorrShuffle = WCorrShuffle(**get_dict_subset(wcorr_shuffles.to_dict(), subset_excludelist=['_VersionedResultMixin_version']))
        a_curr_active_pipeline.global_computation_results.computed_data.SequenceBased.wcorr_ripple_shuffle = wcorr_shuffles
        filtered_epochs_df: pd.DataFrame = deepcopy(wcorr_shuffles.filtered_epochs_df)
        print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles}')
    else:
        print(f'SequenceBased is not computed.')
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
    (_out_p, _out_p_dict), (_out_shuffle_wcorr_ZScore_LONG, _out_shuffle_wcorr_ZScore_SHORT), (total_n_shuffles_more_extreme_than_real_df, total_n_shuffles_more_extreme_than_real_dict), _out_shuffle_wcorr_arr = wcorr_shuffles.post_compute(decoder_names=deepcopy(decoder_names))
    wcorr_ripple_shuffle_all_df, all_shuffles_wcorr_df = wcorr_shuffles.build_all_shuffles_dataframes(decoder_names=deepcopy(decoder_names))
    ## Prepare for plotting in histogram:
    wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.dropna(subset=['start', 'stop'], how='any', inplace=False)
    wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.dropna(subset=['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL'], how='all', inplace=False)
    wcorr_ripple_shuffle_all_df = wcorr_ripple_shuffle_all_df.convert_dtypes()
    # {'long_best_dir_decoder_IDX': int, 'short_best_dir_decoder_IDX': int}
    wcorr_ripple_shuffle_all_df
    ## Gets the absolutely most extreme value from any of the four decoders and uses that
    best_wcorr_max_indices = np.abs(wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].values).argmax(axis=1)
    wcorr_ripple_shuffle_all_df[f'abs_best_wcorr'] = [wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].values[i, best_idx] for i, best_idx in enumerate(best_wcorr_max_indices)] #  np.where(direction_max_indices, wcorr_ripple_shuffle_all_df['long_LR'].filter_epochs[a_column_name].to_numpy(), wcorr_ripple_shuffle_all_df['long_RL'].filter_epochs[a_column_name].to_numpy())
    
    ## Add the worst direction for comparison (testing):
    _out_worst_dir_indicies = []
    _LR_indicies = [0, 2]
    _RL_indicies = [1, 3]

    for an_is_most_likely_direction_LR in wcorr_ripple_shuffle_all_df['is_most_likely_direction_LR']:
        if an_is_most_likely_direction_LR:
            _out_worst_dir_indicies.append(_RL_indicies)
        else:
            _out_worst_dir_indicies.append(_LR_indicies)

    _out_worst_dir_indicies = np.vstack(_out_worst_dir_indicies)
    # _out_best_dir_indicies

    wcorr_ripple_shuffle_all_df['long_worst_dir_decoder_IDX'] = _out_worst_dir_indicies[:,0]
    wcorr_ripple_shuffle_all_df['short_worst_dir_decoder_IDX'] = _out_worst_dir_indicies[:,1]

    best_decoder_index = wcorr_ripple_shuffle_all_df['long_best_dir_decoder_IDX'] ## Kamran specified to restrict to the long-templates only for now
    worst_decoder_index = wcorr_ripple_shuffle_all_df['long_worst_dir_decoder_IDX']

    ## INPUTS: wcorr_ripple_shuffle_all_df, best_decoder_index
    ## MODIFIES: wcorr_ripple_shuffle_all_df
    curr_score_col_decoder_col_names = [f"wcorr_{a_decoder_name}" for a_decoder_name in ['long_LR', 'long_RL', 'short_LR', 'short_RL']]
    wcorr_ripple_shuffle_all_df['wcorr_most_likely'] = [wcorr_ripple_shuffle_all_df[curr_score_col_decoder_col_names].to_numpy()[epoch_idx, a_decoder_idx] for epoch_idx, a_decoder_idx in zip(np.arange(np.shape(wcorr_ripple_shuffle_all_df)[0]), best_decoder_index.to_numpy())]
    wcorr_ripple_shuffle_all_df['abs_most_likely_wcorr'] = np.abs(wcorr_ripple_shuffle_all_df['wcorr_most_likely'])
    wcorr_ripple_shuffle_all_df['wcorr_least_likely'] = [wcorr_ripple_shuffle_all_df[curr_score_col_decoder_col_names].to_numpy()[epoch_idx, a_decoder_idx] for epoch_idx, a_decoder_idx in zip(np.arange(np.shape(wcorr_ripple_shuffle_all_df)[0]), worst_decoder_index.to_numpy())]
    # wcorr_ripple_shuffle_all_df[['wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL']].max(axis=1, skipna=True)

    ## OUTPUTS: wcorr_ripple_shuffle_all_df
    wcorr_ripple_shuffle_all_df


    all_shuffles_only_best_decoder_wcorr_df = pd.concat([all_shuffles_wcorr_df[np.logical_and((all_shuffles_wcorr_df['epoch_idx'] == epoch_idx), (all_shuffles_wcorr_df['decoder_idx'] == best_idx))] for epoch_idx, best_idx in enumerate(best_wcorr_max_indices)])

    ## OUTPUTS: wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df


    ## INPUTS: wcorr_ripple_shuffle
    # standalone save
    standalone_pkl_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_wcorr_ripple_shuffle_data_only_{wcorr_shuffles.n_completed_shuffles}.pkl' 
    standalone_pkl_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_pkl_filename).resolve() # Path("W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\2024-05-30_0925AM_standalone_wcorr_ripple_shuffle_data_only_1100.pkl")
    print(f'saving to "{standalone_pkl_filepath}"...')
    wcorr_shuffles.save_data(standalone_pkl_filepath)
    ## INPUTS: wcorr_ripple_shuffle
    standalone_mat_filename: str = f'{get_now_rounded_time_str()}{custom_suffix}_standalone_all_shuffles_wcorr_array.mat' 
    standalone_mat_filepath = a_curr_active_pipeline.get_output_path().joinpath(standalone_mat_filename).resolve() # r"W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43\output\2024-06-03_0400PM_standalone_all_shuffles_wcorr_array.mat"
    wcorr_shuffles.save_data_mat(filepath=standalone_mat_filepath, **{'session': a_curr_active_pipeline.get_session_context().to_dict()})

    try:
        active_context = a_curr_active_pipeline.get_session_context()
        session_name: str = a_curr_active_pipeline.session_name
        export_files_dict = wcorr_shuffles.export_csvs(parent_output_path=a_curr_active_pipeline.get_output_path().resolve(), active_context=active_context, session_name=session_name, curr_active_pipeline=a_curr_active_pipeline,
                                                    #    source='diba_evt_file',
                                                       source='compute_diba_quiescent_style_replay_events',
                                                       )
        ripple_WCorrShuffle_df_export_CSV_path = export_files_dict['ripple_WCorrShuffle_df']
        print(f'Successfully exported ripple_WCorrShuffle_df_export_CSV_path: "{ripple_WCorrShuffle_df_export_CSV_path}" with wcorr_shuffles.n_completed_shuffles: {wcorr_shuffles.n_completed_shuffles} unique shuffles.')
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

    return wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path)


@function_attributes(short_name=None, tags=['histogram', 'figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-27 22:43', related_items=[])
def plot_replay_wcorr_histogram(df: pd.DataFrame, plot_var_name: str, all_shuffles_only_best_decoder_wcorr_df: Optional[pd.DataFrame]=None, footer_annotation_text=None):
    """ Create horizontal histogram Takes outputs of finalize_output_shuffled_wcorr to plot a histogram like the Diba 2007 paper

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_replay_wcorr_histogram
        plot_var_name: str = 'abs_best_wcorr'
        footer_annotation_text = f'{curr_active_pipeline.get_session_context()}<br>{params_description_str}'

        fig = plot_replay_wcorr_histogram(df=wcorr_ripple_shuffle_all_df, plot_var_name=plot_var_name,
             all_shuffles_only_best_decoder_wcorr_df=all_shuffles_only_best_decoder_wcorr_df, footer_annotation_text=footer_annotation_text)

        # Save figure to disk:
        _out_result = curr_active_pipeline.output_figure(a_fig_context, fig=fig)
        _out_result

        # Show the figure
        fig.show()

    """
    import plotly.io as pio
    import plotly.express as px
    import plotly.graph_objects as go
    from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers


    resolution_multiplier = 1
    fig_size_kwargs = {'width': resolution_multiplier*1650, 'height': resolution_multiplier*480}
    is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)
    pio.templates.default = template

    # fig = px.histogram(df, x=plot_var_name) # , orientation='h'
    df = deepcopy(df) # pd.DataFrame(data)
    df = df.dropna(subset=[plot_var_name], how='any', inplace=False)

    histogram_kwargs = dict(histnorm='percent', nbinsx=30)
    fig = go.Figure()
    wcorr_ripple_hist_trace = fig.add_trace(go.Histogram(x=df[plot_var_name], name='Observed Replay', **histogram_kwargs))

    if all_shuffles_only_best_decoder_wcorr_df is not None:
        shuffle_trace = fig.add_trace(go.Histogram(x=all_shuffles_only_best_decoder_wcorr_df['shuffle_wcorr'], name='Shuffle', **histogram_kwargs))
    
    # Overlay both histograms
    fig = fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig = fig.update_traces(opacity=0.75)

    # Update layout for better visualization
    fig = fig.update_layout(
        title=f'Horizontal Histogram of "{plot_var_name}"',
        xaxis_title=plot_var_name,
        yaxis_title='Percent Count',
        # yaxis_title='Count',
    )

    ## Add the metadata for the replays being plotted:
    # new_replay_epochs.metadata
    if footer_annotation_text is None:
        footer_annotation_text = ''

    # Add footer text annotation
    fig = fig.update_layout(
        annotations=[
            dict(
                x=0,
                y=-0.25,
                xref='paper',
                yref='paper',
                text=footer_annotation_text,
                showarrow=False,
                xanchor='left',
                yanchor='bottom'
            )
        ]
    )

    fig = fig.update_layout(fig_size_kwargs)
    return fig



# ---------------------------------------------------------------------------- #
#      2024-06-25 - Diba 2009-style Replay Detection via Quiescent Period      #
# ---------------------------------------------------------------------------- #
@function_attributes(short_name=None, tags=['pure', 'pkl'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-28 13:23', related_items=[])
def helper_perform_pickle_pipeline(a_curr_active_pipeline, custom_save_filenames, custom_save_filepaths, enable_save_pipeline_pkl, enable_save_global_computations_pkl, enable_save_h5):
    """ Pickles the pipelines as needed

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import helper_perform_pickle_pipeline
    custom_save_filepaths = helper_perform_pickle_pipeline(a_curr_active_pipeline=_temp_curr_active_pipeline, custom_save_filenames=custom_save_filenames, custom_save_filepaths=custom_save_filepaths, enable_save_pipeline_pkl=True, enable_save_global_computations_pkl=False, enable_save_h5=False)

    """
    from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
    
    try:
        if enable_save_pipeline_pkl:
            custom_save_filepaths['pipeline_pkl'] = a_curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE, active_pickle_filename=custom_save_filenames['pipeline_pkl'])

        if enable_save_global_computations_pkl:
            custom_save_filepaths['global_computation_pkl'] = a_curr_active_pipeline.save_global_computation_results(override_global_pickle_filename=custom_save_filenames['global_computation_pkl'])

        if enable_save_h5:
            custom_save_filepaths['pipeline_h5'] = a_curr_active_pipeline.export_pipeline_to_h5(override_filename=custom_save_filenames['pipeline_h5'])
        
        print(f'custom_save_filepaths: {custom_save_filepaths}')

    except BaseException as e:
        print(f'failed pickling in `helper_perform_pickle_pipeline(...)` with error: {e}')
        pass
    
    return custom_save_filepaths


@function_attributes(short_name=None, tags=['replay', 'epochs'], input_requires=[], output_provides=[], uses=[], used_by=['overwrite_replay_epochs_and_recompute'], creation_date='2024-06-26 21:10', related_items=[])
def replace_replay_epochs(curr_active_pipeline, new_replay_epochs: Epoch):
    """ 
    Replaces each session's replay epochs and their `preprocessing_parameters.epoch_estimation_parameters.replays` config
    
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import replace_replay_epochs


    """
    _backup_session_replay_epochs = {}
    _backup_session_configs = {}
    
    if isinstance(new_replay_epochs, pd.DataFrame):
        new_replay_epochs = Epoch.from_dataframe(new_replay_epochs) # ensure it is an epoch object
        
    new_replay_epochs = new_replay_epochs.get_non_overlapping() # ensure non-overlapping

    ## Get the estimation parameters:
    replay_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays
    assert replay_estimation_parameters is not None
    _bak_replay_estimation_parameters = deepcopy(replay_estimation_parameters) ## backup original
    ## backup original values:
    _backup_session_replay_epochs['sess'] = deepcopy(_bak_replay_estimation_parameters)
    _backup_session_configs['sess'] = deepcopy(curr_active_pipeline.sess.replay)

    ## Check if they changed
    did_change: bool = False
    did_change = did_change or np.any(ensure_dataframe(_backup_session_configs['sess']).to_numpy() != ensure_dataframe(new_replay_epochs).to_numpy())

    ## Set new:
    replay_estimation_parameters.epochs_source = new_replay_epochs.metadata.get('epochs_source', None)
    # replay_estimation_parameters.require_intersecting_epoch = None # don't actually purge these as I don't know what they are used for
    replay_estimation_parameters.min_inclusion_fr_active_thresh = new_replay_epochs.metadata.get('minimum_inclusion_fr_Hz', 1.0)
    replay_estimation_parameters.min_num_unique_aclu_inclusions = new_replay_epochs.metadata.get('min_num_active_neurons', 5)

    did_change = did_change or (get_dict_subset(_bak_replay_estimation_parameters, ['epochs_source', 'min_num_unique_aclu_inclusions', 'min_inclusion_fr_active_thresh']) != get_dict_subset(replay_estimation_parameters, ['epochs_source', 'min_num_unique_aclu_inclusions', 'min_inclusion_fr_active_thresh']))
    ## Assign the new parameters:
    curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays = deepcopy(replay_estimation_parameters)

    assert curr_active_pipeline.sess.basepath.exists()
    ## assign the new replay epochs:
    curr_active_pipeline.sess.replay = deepcopy(new_replay_epochs)
    for k, a_filtered_session in curr_active_pipeline.filtered_sessions.items():
        ## backup original values:
        _backup_session_replay_epochs[k] = deepcopy(a_filtered_session.config.preprocessing_parameters.epoch_estimation_parameters.replays)
        _backup_session_configs[k] = deepcopy(a_filtered_session.replay)

        ## assign the new replay epochs:
        a_filtered_session.replay = deepcopy(new_replay_epochs).time_slice(a_filtered_session.t_start, a_filtered_session.t_stop)
        assert curr_active_pipeline.sess.basepath.exists()
        a_filtered_session.config.basepath = deepcopy(curr_active_pipeline.sess.basepath)
        assert a_filtered_session.config.basepath.exists()
        a_filtered_session.config.preprocessing_parameters.epoch_estimation_parameters.replays = deepcopy(replay_estimation_parameters)

        # print(a_filtered_session.replay)
        # a_filtered_session.start()

    print(f'did_change: {did_change}')

    return did_change, _backup_session_replay_epochs, _backup_session_configs


def _get_custom_suffix_for_replay_filename(new_replay_epochs: Epoch, *extras_strings) -> str:
    """
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _get_custom_suffix_for_replay_filename
    custom_suffix = _get_custom_suffix_for_replay_filename(new_replay_epochs=new_replay_epochs)

    print(f'custom_suffix: "{custom_suffix}"')

    """
    assert new_replay_epochs.metadata is not None
    metadata = deepcopy(new_replay_epochs.metadata)
    extras_strings = []

    epochs_source = metadata.get('epochs_source', None)
    assert epochs_source is not None
    # print(f'epochs_source: {epochs_source}')

    valid_epochs_source_values = ['compute_diba_quiescent_style_replay_events', 'diba_evt_file', 'initial_loaded', 'normal_computed']
    assert epochs_source in valid_epochs_source_values, f"epochs_source: '{epochs_source}' is not in valid_epochs_source_values: {valid_epochs_source_values}"

    if epochs_source == 'compute_diba_quiescent_style_replay_events':
        custom_suffix: str = '_withNewComputedReplays'
        # qclu = new_replay_epochs.metadata.get('qclu', "[1,2]")
        custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', '[1,2]')}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])

    elif epochs_source == 'diba_evt_file':
        custom_suffix: str = '_withNewKamranExportedReplays'
        custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', '[1,2]')}", f"frateThresh_{metadata.get('minimum_inclusion_fr_Hz', 5.0):.1f}", *extras_strings])
        # qclu = new_replay_epochs.metadata.get('qclu', "[1,2]") # Diba export files are always qclus [1, 2]

    elif epochs_source == 'initial_loaded':
        custom_suffix: str = '_withOldestImportedReplays'
        custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', 'XX')}", f"frateThresh_{metadata.get('minimum_inclusion_fr_Hz', 0.1):.1f}", *extras_strings])

    elif epochs_source == 'normal_computed':
        custom_suffix: str = '_withNormalComputedReplays'
        custom_suffix = '-'.join([custom_suffix, f"qclu_{metadata.get('included_qclu_values', '[1,2]')}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
    else:
        raise NotImplementedError(f'epochs_source: {epochs_source} is of unknown type or is missing metadata.')    

    # with np.printoptions(precision=1, suppress=True, threshold=5):
    #     # score_text = f"score: " + str(np.array([epoch_score])).lstrip("[").rstrip("]") # output is just the number, as initially it is '[0.67]' but then the [ and ] are stripped.   
    #     return '-'.join([f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
    #     # return '-'.join([f"replaySource_{metadata['epochs_source']}", f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
        
    return custom_suffix


def _custom_replay_str_for_filename(new_replay_epochs: Epoch, *extras_strings):
    assert new_replay_epochs.metadata is not None
    with np.printoptions(precision=1, suppress=True, threshold=5):
        metadata = deepcopy(new_replay_epochs.metadata)
        # score_text = f"score: " + str(np.array([epoch_score])).lstrip("[").rstrip("]") # output is just the number, as initially it is '[0.67]' but then the [ and ] are stripped.   
        return '-'.join([f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
        # return '-'.join([f"replaySource_{metadata['epochs_source']}", f"qclu_{metadata['included_qclu_values']}", f"frateThresh_{metadata['minimum_inclusion_fr_Hz']:.1f}", *extras_strings])
        



@function_attributes(short_name=None, tags=['replay', 'new_replay', 'top'], input_requires=[], output_provides=[], uses=['replace_replay_epochs'], used_by=[], creation_date='2024-06-25 22:49', related_items=['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function'])
def overwrite_replay_epochs_and_recompute(curr_active_pipeline, new_replay_epochs: Epoch, ripple_decoding_time_bin_size: float = 0.025, 
                                          num_wcorr_shuffles: int=25, fail_on_exception=True,
                                          enable_save_pipeline_pkl: bool=True, enable_save_global_computations_pkl: bool=False, enable_save_h5: bool = False, user_completion_dummy=None):
    """ Recomputes the replay epochs using a custom implementation of the criteria in Diba 2007.

    , included_qclu_values=[1,2], minimum_inclusion_fr_Hz=5.0


    #TODO 2024-07-04 10:52: - [ ] Need to add the custom processing suffix to `BATCH_DATE_TO_USE`

    
    If `did_change` == True,
        ['merged_directional_placefields', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring']
        ['wcorr_shuffle_analysis']

        are updated

    Otherwise:
        ['wcorr_shuffle_analysis'] can be updated

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import overwrite_replay_epochs_and_recompute

        did_change, custom_save_filenames, custom_save_filepaths = overwrite_replay_epochs_and_recompute(curr_active_pipeline=curr_active_pipeline, new_replay_epochs=evt_epochs)

    """
    from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
    from neuropy.utils.debug_helpers import parameter_sweeps
    from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function
    # SimpleBatchComputationDummy = make_class('SimpleBatchComputationDummy', attrs=['BATCH_DATE_TO_USE', 'collected_outputs_path'])

    # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderComputationsContainer

    # rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
    # minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
    # included_qclu_values: List[int] = rank_order_results.included_qclu_values
    # ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
    # directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    # track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?

    # spikes_df = get_proper_global_spikes_df(curr_active_pipeline)
    # (qclu_included_aclus, active_track_templates, active_spikes_df, quiescent_periods), (new_replay_epochs_df, new_replay_epochs) = compute_diba_quiescent_style_replay_events(curr_active_pipeline=curr_active_pipeline, directional_laps_results=directional_laps_results,
    #                       
    # 'epochs_source'
    custom_suffix: str = _get_custom_suffix_for_replay_filename(new_replay_epochs=new_replay_epochs)
    print(f'custom_suffix: "{custom_suffix}"')

    assert (user_completion_dummy is not None), f"2024-07-04 - `user_completion_dummy` must be provided with a modified .BATCH_DATE_TO_USE to include the custom suffix!"

    additional_session_context = None
    try:
        if custom_suffix is not None:
            additional_session_context = IdentifyingContext(custom_suffix=custom_suffix)
            print(f'Using custom suffix: "{custom_suffix}" - additional_session_context: "{additional_session_context}"')
    except NameError as err:
        additional_session_context = None
        print(f'NO CUSTOM SUFFIX.')    
        

    ## OUTPUTS: new_replay_epochs, new_replay_epochs_df
    did_change, _backup_session_replay_epochs, _backup_session_configs = replace_replay_epochs(curr_active_pipeline=curr_active_pipeline, new_replay_epochs=new_replay_epochs)

    custom_save_filenames = {
        'pipeline_pkl':f'loadedSessPickle{custom_suffix}.pkl',
        'global_computation_pkl':f"global_computation_results{custom_suffix}.pkl",
        'pipeline_h5':f'pipeline{custom_suffix}.h5',
    }
    print(f'custom_save_filenames: {custom_save_filenames}')
    custom_save_filepaths = {k:v for k, v in custom_save_filenames.items()}

    if not did_change:
        print(f'no changes!')
        curr_active_pipeline.reload_default_computation_functions()
    
        ## wcorr shuffle:
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['wcorr_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': num_wcorr_shuffles}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

    else:
        print(f'replay epochs changed!')

        curr_active_pipeline.reload_default_computation_functions()

        metadata = deepcopy(new_replay_epochs.metadata)
        minimum_inclusion_fr_Hz = metadata.get('minimum_inclusion_fr_Hz', None)
        included_qclu_values = metadata.get('included_qclu_values', None)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields','perform_rank_order_shuffle_analysis'],
                                                           computation_kwargs_list=[{'laps_decoding_time_bin_size': None, 'ripple_decoding_time_bin_size': ripple_decoding_time_bin_size},
                                                                                    {'num_shuffles': num_wcorr_shuffles, 'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz, 'included_qclu_values': included_qclu_values, 'skip_laps': True}],
                                                         enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
        

        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['perform_rank_order_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': num_wcorr_shuffles, 'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz, 'included_qclu_values': included_qclu_values, 'skip_laps': True}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation

        global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['DirectionalDecodersEpochsEvaluations'], debug_print=True)

        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_evaluate_epochs',  'directional_decoders_epoch_heuristic_scoring'],
                        computation_kwargs_list=[{'should_skip_radon_transform': False}, {}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
        

        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields',
        #                                                                                            'directional_decoders_evaluate_epochs',
        #                                                                                              'directional_decoders_epoch_heuristic_scoring'],
        #                 computation_kwargs_list=[{'laps_decoding_time_bin_size': None, 'ripple_decoding_time_bin_size': ripple_decoding_time_bin_size},
        #                                          {'should_skip_radon_transform': False},
        #                                             {}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False) # 'laps_decoding_time_bin_size': None prevents laps recomputation
        



        ## Export these new computations to .csv for across-session analysis:
        # Uses `perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function` to compute the new outputs:

        # BEGIN normal data Export ___________________________________________________________________________________________ #
        return_full_decoding_results: bool = False
        # desired_laps_decoding_time_bin_size = [None] # doesn't work
        desired_laps_decoding_time_bin_size = [1.5] # large so it doesn't take long
        desired_ripple_decoding_time_bin_size = [0.010, 0.020]

        custom_all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(desired_laps_decoding_time_bin_size=desired_laps_decoding_time_bin_size,
                                                                                       desired_ripple_decoding_time_bin_size=desired_ripple_decoding_time_bin_size,
                                                                                use_single_time_bin_per_epoch=[False],
                                                                                minimum_event_duration=[desired_ripple_decoding_time_bin_size[-1]])


        ## make sure that the exported .csv and .h5 files have unique names based on the unique replays used. Also avoid unduely recomputing laps each time.
        _across_session_results_extended_dict = {}
        ## Combine the output of `perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function` into two dataframes for the laps, one per-epoch and one per-time-bin
        _across_session_results_extended_dict = _across_session_results_extended_dict | perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function(user_completion_dummy, None,
                                                        curr_session_context=curr_active_pipeline.get_session_context(), curr_session_basedir=curr_active_pipeline.sess.basepath.resolve(), curr_active_pipeline=curr_active_pipeline,
                                                        across_session_results_extended_dict=_across_session_results_extended_dict, return_full_decoding_results=return_full_decoding_results,
                                                        save_hdf=True, save_csvs=True,
                                                        # desired_shared_decoding_time_bin_sizes = np.linspace(start=0.030, stop=0.5, num=4),
                                                        custom_all_param_sweep_options=custom_all_param_sweep_options, # directly provide the parameter sweeps
                                                        additional_session_context=additional_session_context
                                                    )
        # with `return_full_decoding_results == False`
        out_path, output_laps_decoding_accuracy_results_df, output_extracted_result_tuples, combined_multi_timebin_outputs_tuple = _across_session_results_extended_dict['perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function']
        (several_time_bin_sizes_laps_df, laps_out_path, several_time_bin_sizes_time_bin_laps_df, laps_time_bin_marginals_out_path), (several_time_bin_sizes_ripple_df, ripple_out_path, several_time_bin_sizes_time_bin_ripple_df, ripple_time_bin_marginals_out_path) = combined_multi_timebin_outputs_tuple

        _out_file_paths_dict = {
            'ripple_h5_out_path': out_path,
            'ripple_csv_out_path': ripple_out_path,
            'ripple_csv_time_bin_marginals': ripple_time_bin_marginals_out_path,
            
            ## Laps:
            'laps_csv_out_path': laps_out_path,
            'laps_csv_time_bin_marginals_out_path': laps_time_bin_marginals_out_path,
        }

        for a_name, a_path in _out_file_paths_dict.items():
            custom_save_filepaths[a_name] = a_path

        # custom_save_filepaths['csv_out_path'] = out_path # ends up being the .h5 path for some reason
        # custom_save_filepaths['csv_out_path'] = out_path # ends up being the .h5 path for some reason
        # custom_save_filepaths['ripple_csv_out_path'] = ripple_out_path

        # END Normal data Export _____________________________________________________________________________________________ #




        ## Long/Short Stuff:
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['long_short_decoding_analyses','long_short_fr_indicies_analyses','jonathan_firing_rate_analysis',
        #             'long_short_post_decoding','long_short_inst_spike_rate_groups','long_short_endcap_analysis'], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

        ## Rank-Order Shuffle
        ## try dropping result and recomputing:
        global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['SequenceBased'], debug_print=True)

        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['rank_order_shuffle_analysis',], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['rank_order_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': 10, 'skip_laps': True}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

        ## Pickle first thing after changes:
        # custom_save_filepaths = helper_perform_pickle_pipeline(curr_active_pipeline=curr_active_pipeline, custom_save_filepaths=custom_save_filepaths)

        ## wcorr shuffle:
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['wcorr_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': num_wcorr_shuffles}], enabled_filter_names=None, fail_on_exception=fail_on_exception, debug_print=False)

        ## Pickle again after recomputing:
        custom_save_filepaths = helper_perform_pickle_pipeline(a_curr_active_pipeline=curr_active_pipeline, custom_save_filenames=custom_save_filenames, custom_save_filepaths=custom_save_filepaths,
                                                                enable_save_pipeline_pkl=enable_save_pipeline_pkl, enable_save_global_computations_pkl=enable_save_global_computations_pkl, enable_save_h5=enable_save_h5)

    try:
        decoder_names = deepcopy(TrackTemplates.get_decoder_names())
        wcorr_ripple_shuffle_all_df, all_shuffles_only_best_decoder_wcorr_df, (standalone_pkl_filepath, standalone_mat_filepath, ripple_WCorrShuffle_df_export_CSV_path) = finalize_output_shuffled_wcorr(a_curr_active_pipeline=curr_active_pipeline,
                                                                                                                                        decoder_names=decoder_names, custom_suffix=custom_suffix)
        custom_save_filepaths['standalone_wcorr_pkl'] = standalone_pkl_filepath
        custom_save_filepaths['standalone_mat_pkl'] = standalone_mat_filepath
        print(f'completed overwrite_replay_epochs_and_recompute(...). custom_save_filepaths: {custom_save_filepaths}\n')
        custom_save_filenames['standalone_wcorr_pkl'] = standalone_pkl_filepath.name
        custom_save_filenames['standalone_mat_pkl'] = standalone_mat_filepath.name
        
    except BaseException as e:
        print(f'failed doing `finalize_output_shuffled_wcorr(...)` with error: {e}')
        if user_completion_dummy.fail_on_exception:
            print(f'did_change: {did_change}, custom_save_filenames: {custom_save_filenames}, custom_save_filepaths: {custom_save_filepaths}')
            raise e

    # global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop=['SequenceBased', 'RankOrder', 'long_short_fr_indicies_analysis', 'long_short_leave_one_out_decoding_analysis', 'jonathan_firing_rate_analysis', 'DirectionalMergedDecoders', 'DirectionalDecodersDecoded', 'DirectionalDecodersEpochsEvaluations', 'DirectionalDecodersDecoded'], debug_print=True)    
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=[
    #     'merged_directional_placefields', 
    #     'long_short_decoding_analyses',
    #     'jonathan_firing_rate_analysis',
    #     'long_short_fr_indicies_analyses',
    #     'short_long_pf_overlap_analyses',
    #     'long_short_post_decoding',
    #     'long_short_rate_remapping',
    #     'long_short_inst_spike_rate_groups',
    #     'long_short_endcap_analysis',
    #     ], enabled_filter_names=None, fail_on_exception=False, debug_print=False) # , computation_kwargs_list=[{'should_skip_radon_transform': False}]

    # 2024-06-25 - Save all custom _______________________________________________________________________________________ #
    ## INPUTS: custom_suffix

    # custom_save_filepaths['pipeline_pkl'] = Path('W:/Data/KDIBA/gor01/two/2006-6-07_16-40-19/loadedSessPickle_withNewKamranExportedReplays.pkl').resolve()
    # custom_save_filepaths['global_computation_pkl'] = Path(r'W:\Data\KDIBA\gor01\two\2006-6-07_16-40-19\output\global_computation_results_withNewKamranExportedReplays.pkl').resolve()

    return did_change, custom_save_filenames, custom_save_filepaths


# Replay Loading/Estimation Methods __________________________________________________________________________________ #

@function_attributes(short_name=None, tags=['replay', 'epochs', 'import', 'diba_evt_file'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-26 21:06', related_items=[])
def try_load_neuroscope_EVT_file_epochs(curr_active_pipeline, ext:str='bst') -> Optional[Epoch]:
    """ loads the replay epochs from an exported .evt file

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import try_load_neuroscope_EVT_file_epochs

        evt_epochs = try_load_neuroscope_EVT_file_epochs(curr_active_pipeline)

        ## load a previously exported to .ebt computed replays:
        evt_epochs = try_load_neuroscope_EVT_file_epochs(curr_active_pipeline, ext='PHONEW')
        evt_epochs.metadata['epochs_source'] = 'compute_diba_quiescent_style_replay_events'

    """
    ## FROM .evt file
    evt_filepath = curr_active_pipeline.sess.basepath.joinpath(f'{curr_active_pipeline.session_name}.{ext}.evt').resolve()
    evt_epochs = None
    if evt_filepath.exists():
        # assert evt_filepath.exists(), f"evt_filepath: '{evt_filepath}' does not exist!"
        evt_epochs: Epoch = Epoch.from_neuroscope(in_filepath=evt_filepath, metadata={'epochs_source': 'diba_evt_file'}).get_non_overlapping()
        evt_epochs.filename = str(evt_filepath) ## set the filepath
    return evt_epochs


@function_attributes(short_name=None, tags=['helper'], input_requires=[], output_provides=[], uses=[], used_by=['compute_diba_quiescent_style_replay_events'], creation_date='2024-06-27 22:16', related_items=[])
def check_for_and_merge_overlapping_epochs(quiescent_periods: pd.DataFrame, debug_print=False) -> pd.DataFrame:
    """
    Checks for overlaps in the quiescent periods and merges them if necessary.

    Parameters:
    quiescent_periods (pd.DataFrame): DataFrame containing quiescent periods with 'start' and 'stop' columns.

    Returns:
    pd.DataFrame: DataFrame with non-overlapping quiescent periods.

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_diba_quiescent_style_replay_events, find_quiescent_windows, check_for_overlaps
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

        quiescent_periods = find_quiescent_windows(active_spikes_df=get_proper_global_spikes_df(curr_active_pipeline), silence_duration=0.06)
        quiescent_periods

    """
    non_overlapping_periods = []
    last_stop = -float('inf')

    for idx, row in quiescent_periods.iterrows():
        if (last_stop is not None):        
            if (row['start'] > last_stop):
                non_overlapping_periods.append(row)
                last_stop = row['stop']
            else:
                # Optionally, you can log or handle the overlapping intervals here
                if debug_print:
                    print(f"Overlap detected: {row['start']} - {row['stop']} overlaps with last stop {last_stop}")
                non_overlapping_periods[-1]['stop'] = row['stop'] # update the last event, don't add a new one
                last_stop = row['stop']

        else:
            non_overlapping_periods.append(row)
            last_stop = row['stop']

    non_overlapping_periods_df = pd.DataFrame(non_overlapping_periods)
    non_overlapping_periods_df["time_diff"] = non_overlapping_periods_df["stop"] - non_overlapping_periods_df["start"]
    non_overlapping_periods_df["duration"] = non_overlapping_periods_df["stop"] - non_overlapping_periods_df["start"]
    non_overlapping_periods_df = non_overlapping_periods_df.reset_index(drop=True)
    non_overlapping_periods_df["label"] = non_overlapping_periods_df.index.astype('str', copy=True)

    return non_overlapping_periods_df


@function_attributes(short_name=None, tags=['replay', 'compute', 'compute_diba_quiescent_style_replay_events'], input_requires=[], output_provides=[], uses=['check_for_and_merge_overlapping_epochs'], used_by=[], creation_date='2024-06-25 12:54', related_items=[])
def compute_diba_quiescent_style_replay_events(curr_active_pipeline, spikes_df, included_qclu_values=[1,2], minimum_inclusion_fr_Hz=5.0, silence_duration:float=0.06, firing_window_duration:float=0.3,
            enable_export_to_neuroscope_EVT_file:bool=True):
    """ Method to find putative replay events similar to the Diba 2007 paper: by finding quiet periods and then getting the activity for 300ms after them.

    if 'included_qclu_values' and 'minimum_inclusion_fr_Hz' don't change, the templates and directional lap results aren't required it seems. 
    All of this is just in service of getting the properly filtered `active_spikes_df` to determine the quiescent periods.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_diba_quiescent_style_replay_events

    """
    # ==================================================================================================================== #
    # BEGIN SUBFUNCTIONS                                                                                                   #
    # ==================================================================================================================== #

    def find_quiescent_windows(active_spikes_df: pd.DataFrame, silence_duration:float=0.06) -> pd.DataFrame:
        """
        # Define the duration for silence and firing window
        silence_duration = 0.06  # 60 ms
        firing_window_duration = 0.3  # 300 ms
        min_unique_neurons = 14

        CAPTURES NOTHING
        """
        ## INPUTS: active_spikes_df

        # Ensure the DataFrame is sorted by the event times
        spikes_df = deepcopy(active_spikes_df)[['t_rel_seconds']].sort_values(by='t_rel_seconds').reset_index(drop=True).drop_duplicates(subset=['t_rel_seconds'], keep='first')

        # Drop rows with duplicate values in the 't_rel_seconds' column, keeping the first occurrence
        spikes_df = spikes_df.drop_duplicates(subset=['t_rel_seconds'], keep='first')

        # Calculate the differences between consecutive event times
        spikes_df['time_diff'] = spikes_df['t_rel_seconds'].diff()

        # Find the indices where the time difference is greater than 60ms (0.06 seconds)
        quiescent_periods = spikes_df[spikes_df['time_diff'] > silence_duration]

        # Extract the start and end times of the quiescent periods
        # quiescent_periods['start'] = spikes_df['t_rel_seconds'].shift(1)
        quiescent_periods['stop'] = quiescent_periods['t_rel_seconds']
        quiescent_periods['start'] = quiescent_periods['stop'] - quiescent_periods['time_diff']

        # Drop the NaN values that result from the shift operation
        quiescent_periods = quiescent_periods.dropna(subset=['start'])

        # Select the relevant columns
        quiescent_periods = quiescent_periods[['start', 'stop', 'time_diff']]
        # quiescent_periods["label"] = quiescent_periods.index.astype('str', copy=True)
        # quiescent_periods["duration"] = quiescent_periods["stop"] - quiescent_periods["start"] 
        quiescent_periods = check_for_and_merge_overlapping_epochs(quiescent_periods=quiescent_periods)
        # print(quiescent_periods)
        return quiescent_periods

    def find_active_epochs_preceeded_by_quiescent_windows(active_spikes_df, silence_duration:float=0.06, firing_window_duration:float=0.3, min_unique_neurons:int=14):
        """
        # Define the duration for silence and firing window
        silence_duration = 0.06  # 60 ms
        firing_window_duration = 0.3  # 300 ms
        min_unique_neurons = 14

        CAPTURES NOTHING
        """
        ## INPUTS: active_spikes_df

        # Ensure the DataFrame is sorted by the event times
        spikes_df = deepcopy(active_spikes_df).sort_values(by='t_rel_seconds').reset_index(drop=True)
        # Calculate the differences between consecutive event times
        spikes_df['time_diff'] = spikes_df['t_rel_seconds'].diff()

        ## INPUTS: quiescent_periods
        quiescent_periods = find_quiescent_windows(active_spikes_df=active_spikes_df, silence_duration=silence_duration)

        # List to hold the results
        results = []

        # Variable to keep track of the end time of the last valid epoch
        # last_epoch_end = -float('inf')

        # Iterate over each quiescent period
        for idx, row in quiescent_periods.iterrows():
            silence_end = row['stop']
            window_start = silence_end
            window_end = silence_end + firing_window_duration
            
            # Check if there's another quiescent period within the current window
            if (idx + 1) < len(quiescent_periods):
                next_row = quiescent_periods.iloc[idx + 1]
                next_quiescent_start = next_row['start']
                if next_quiescent_start < window_end:
                    window_end = next_quiescent_start
                    # break
        
            # Filter events that occur in the 300-ms window after the quiescent period
            window_events = spikes_df[(spikes_df['t_rel_seconds'] >= window_start) & (spikes_df['t_rel_seconds'] <= window_end)]
            
            # Count unique neurons firing in this window
            unique_neurons = window_events['aclu'].nunique()
            
            # Check if at least 14 unique neurons fired in this window
            if unique_neurons >= min_unique_neurons:
                results.append({
                    'quiescent_start': row['start'],
                    'quiescent_end': silence_end,
                    'window_start': window_start,
                    'window_end': window_end,
                    'unique_neurons': unique_neurons
                })
                # Variable to keep track of the end time of the last valid epoch
                # last_epoch_end = window_end


        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        results_df["label"] = results_df.index.astype('str', copy=True)
        results_df["duration"] = results_df["window_end"] - results_df["window_start"] 
        return results_df, quiescent_periods

    # ==================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                  #
    # ==================================================================================================================== #

    ## INPUTS: curr_active_pipeline, directional_laps_results, rank_order_results
    # track_templates.determine_decoder_aclus_filtered_by_frate(5.0)
    # qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
    qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=included_qclu_values)
    # qclu_included_aclus
    
    directional_laps_results: DirectionalLapsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'])
    modified_directional_laps_results = directional_laps_results.filtered_by_included_aclus(qclu_included_aclus)
    active_track_templates: TrackTemplates = deepcopy(modified_directional_laps_results.get_templates(minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)) # Here is where the firing rate matters
    # active_track_templates

    any_decoder_neuron_IDs = deepcopy(active_track_templates.any_decoder_neuron_IDs)
    n_neurons: int = len(any_decoder_neuron_IDs)
    # min_num_active_neurons: int = max(int(round(0.3 * float(n_neurons))), 5)
    min_num_active_neurons: int = active_track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline) # smarter, considers the minimum template

    print(f'n_neurons: {n_neurons}, min_num_active_neurons: {min_num_active_neurons}')
    # get_templates(5.0)
    active_spikes_df: pd.DataFrame = deepcopy(spikes_df)
    active_spikes_df = active_spikes_df.spikes.sliced_by_neuron_id(any_decoder_neuron_IDs)
    # active_spikes_df

    ## OUTPUTS: active_spikes_df
    new_replay_epochs_df, quiescent_periods = find_active_epochs_preceeded_by_quiescent_windows(active_spikes_df, silence_duration=silence_duration, firing_window_duration=firing_window_duration, min_unique_neurons=min_num_active_neurons)
    new_replay_epochs_df = new_replay_epochs_df.rename(columns={'window_start': 'start', 'window_end': 'stop',})

    new_replay_epochs: Epoch = Epoch.from_dataframe(new_replay_epochs_df, metadata={'epochs_source': 'compute_diba_quiescent_style_replay_events',
                                                                                    'included_qclu_values': included_qclu_values, 'minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
                                                                                     'silence_duration': silence_duration, 'firing_window_duration': firing_window_duration,
                                                                                     'qclu_included_aclus': qclu_included_aclus, 'min_num_active_neurons': min_num_active_neurons})
    
    if enable_export_to_neuroscope_EVT_file:
        ## Save computed epochs out to a neuroscope .evt file:
        filename = f"{curr_active_pipeline.session_name}"
        filepath = curr_active_pipeline.get_output_path().joinpath(filename).resolve()
        ## set the filename of the Epoch:
        new_replay_epochs.filename = filepath
        filepath = new_replay_epochs.to_neuroscope(ext='PHONEW')
        assert filepath.exists()
        print(F'saved out newly computed epochs to "{filepath}".')

    return (qclu_included_aclus, active_track_templates, active_spikes_df, quiescent_periods), (new_replay_epochs_df, new_replay_epochs)

@function_attributes(short_name=None, tags=['MAIN', 'replay'], input_requires=[], output_provides=[], uses=['compute_diba_quiescent_style_replay_events', 'try_load_neuroscope_EVT_file_epochs', 'try_load_neuroscope_EVT_file_epochs'], used_by=['compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function'], creation_date='2024-07-03 06:12', related_items=[])
def compute_all_replay_epoch_variations(curr_active_pipeline) -> Dict[str, Epoch]:
    """ Computes alternative replays (such as loading them from Diba-exported files, computing using the quiescent periods before the event, etc)
    
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_replay_epoch_variations

    replay_epoch_variations = compute_all_replay_epoch_variations(curr_active_pipeline)
    replay_epoch_variations

    """
    # 
    from neuropy.core.epoch import Epoch, ensure_Epoch, ensure_dataframe
    from pyphocorehelpers.exception_helpers import ExceptionPrintingContext
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

    # print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # print(f'compute_all_replay_epoch_variations(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
    
    suppress_exceptions: bool = True # allow some alternative replay computations to fail


    # ==================================================================================================================== #
    # Compute Alternative Replays: `replay_epoch_variations`                                                               #
    # ==================================================================================================================== #

    ## Compute new epochs: 
    replay_epoch_variations = {}


    with ExceptionPrintingContext(suppress=True, exception_print_fn=(lambda formatted_exception_str: print(f'\t"initial_loaded" failed with error: {formatted_exception_str}. Skipping.'))):
        replay_epoch_variations.update({
            'initial_loaded': ensure_Epoch(deepcopy(curr_active_pipeline.sess.replay_backup), metadata={'epochs_source': 'initial_loaded'}),
        })
        
    with ExceptionPrintingContext(suppress=suppress_exceptions, exception_print_fn=(lambda formatted_exception_str: print(f'\t"normal_computed" failed with error: {formatted_exception_str}. Skipping.'))):
        ## Get the estimation parameters:
        replay_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays
        assert replay_estimation_parameters is not None

        ## get the epochs computed normally:
        replay_epoch_variations.update({
            'normal_computed': ensure_Epoch(deepcopy(curr_active_pipeline.sess.replay), metadata={'epochs_source': 'normal_computed', 'minimum_inclusion_fr_Hz': replay_estimation_parameters['min_inclusion_fr_active_thresh'], 'min_num_active_neurons': replay_estimation_parameters['min_num_unique_aclu_inclusions']}),
        })

    with ExceptionPrintingContext(suppress=suppress_exceptions, exception_print_fn=(lambda formatted_exception_str: print(f'\t"diba_quiescent_method_replay_epochs" failed with error: {formatted_exception_str}. Skipping.'))):
        ## Compute new epochs: 
        included_qclu_values = [1,2]
        minimum_inclusion_fr_Hz = 5.0
        spikes_df = get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        (qclu_included_aclus, active_track_templates, active_spikes_df, quiescent_periods), (diba_quiescent_method_replay_epochs_df, diba_quiescent_method_replay_epochs) = compute_diba_quiescent_style_replay_events(curr_active_pipeline=curr_active_pipeline,
                                                                                                                                                                                    included_qclu_values=included_qclu_values, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, spikes_df=spikes_df)
        ## OUTPUTS: diba_quiescent_method_replay_epochs
        replay_epoch_variations.update({
            'diba_quiescent_method_replay_epochs': deepcopy(diba_quiescent_method_replay_epochs),
        })
        
    with ExceptionPrintingContext(suppress=True, exception_print_fn=(lambda formatted_exception_str: print(f'\t"diba_evt_file" failed with error: {formatted_exception_str}. Skipping.'))):
        ## FROM .evt file
        ## Load exported epochs from a neuroscope .evt file:
        diba_evt_file_replay_epochs: Epoch = try_load_neuroscope_EVT_file_epochs(curr_active_pipeline)
        if diba_evt_file_replay_epochs is not None:
            replay_epoch_variations.update({
                'diba_evt_file': deepcopy(diba_evt_file_replay_epochs),
            })
        

    print(f'completed replay extraction, have: {list(replay_epoch_variations.keys())}')
    
    ## OUTPUT: replay_epoch_variations
    return replay_epoch_variations    




# ---------------------------------------------------------------------------- #
#                      2024-06-15 - Significant Remapping                      #
# ---------------------------------------------------------------------------- #
@function_attributes(short_name=None, tags=['remapping'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-15 00:00', related_items=[])
def _add_cell_remapping_category(neuron_replay_stats_df, loaded_track_limits: Dict, x_midpoint: float=72.0):
    """ yanked from `_perform_long_short_endcap_analysis to compute within the batch processing notebook

    'has_long_pf', 'has_short_pf'

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_cell_remapping_category

        neuron_replay_stats_df = deepcopy(neuron_replay_stats_table)
        neuron_replay_stats_df, (non_disappearing_endcap_cells_df, disappearing_endcap_cells_df, minorly_changed_endcap_cells_df, significant_distant_remapping_endcap_aclus) = _add_cell_remapping_category(neuron_replay_stats_df=neuron_replay_stats_df,
                                                            loaded_track_limits = {'long_xlim': np.array([59.0774, 228.69]), 'short_xlim': np.array([94.0156, 193.757]), 'long_ylim': np.array([138.164, 146.12]), 'short_ylim': np.array([138.021, 146.263])},
        )
        neuron_replay_stats_df
    
    """
    # `loaded_track_limits` = deepcopy(owning_pipeline_reference.sess.config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
    # x_midpoint: float = owning_pipeline_reference.sess.config.x_midpoint
    # pix2cm: float = owning_pipeline_reference.sess.config.pix2cm

    ## INPUTS: loaded_track_limits
    print(f'loaded_track_limits: {loaded_track_limits}')

    if 'has_long_pf' not in neuron_replay_stats_df.columns:
        neuron_replay_stats_df['has_long_pf'] = np.logical_not(np.isnan(neuron_replay_stats_df['long_pf_peak_x']))
    if 'has_short_pf' not in neuron_replay_stats_df.columns:
        neuron_replay_stats_df['has_short_pf'] = np.logical_not(np.isnan(neuron_replay_stats_df['short_pf_peak_x']))

    long_xlim = loaded_track_limits['long_xlim']
    # long_ylim = loaded_track_limits['long_ylim']
    short_xlim = loaded_track_limits['short_xlim']
    # short_ylim = loaded_track_limits['short_ylim']

    occupancy_midpoint: float = x_midpoint # 142.7512402496278 # 150.0
    left_cap_x_bound: float = (long_xlim[0] - x_midpoint) #-shift by midpoint - 72.0 # on long track
    right_cap_x_bound: float = (long_xlim[1] - x_midpoint) # 72.0 # on long track
    min_significant_remapping_x_distance: float = 50.0 # from long->short track
    # min_significant_remapping_x_distance: float = 100.0

    ## STATIC:
    # occupancy_midpoint: float = 142.7512402496278 # 150.0
    # left_cap_x_bound: float = -72.0 # on long track
    # right_cap_x_bound: float = 72.0 # on long track
    # min_significant_remapping_x_distance: float = 40.0 # from long->short track

    # Extract the peaks of the long placefields to find ones that have peaks outside the boundaries
    long_pf_peaks = neuron_replay_stats_df[neuron_replay_stats_df['has_long_pf']]['long_pf_peak_x'] - occupancy_midpoint # this shift of `occupancy_midpoint` is to center the midpoint of the track at 0. 
    is_left_cap = (long_pf_peaks < left_cap_x_bound)
    is_right_cap = (long_pf_peaks > right_cap_x_bound)
    # is_either_cap =  np.logical_or(is_left_cap, is_right_cap)

    # Adds ['is_long_peak_left_cap', 'is_long_peak_right_cap', 'is_long_peak_either_cap'] columns: 
    neuron_replay_stats_df['is_long_peak_left_cap'] = False
    neuron_replay_stats_df['is_long_peak_right_cap'] = False
    neuron_replay_stats_df.loc[is_left_cap.index, 'is_long_peak_left_cap'] = is_left_cap # True
    neuron_replay_stats_df.loc[is_right_cap.index, 'is_long_peak_right_cap'] = is_right_cap # True

    neuron_replay_stats_df['is_long_peak_either_cap'] = np.logical_or(neuron_replay_stats_df['is_long_peak_left_cap'], neuron_replay_stats_df['is_long_peak_right_cap'])

    # adds ['LS_pf_peak_x_diff'] column
    neuron_replay_stats_df['LS_pf_peak_x_diff'] = neuron_replay_stats_df['long_pf_peak_x'] - neuron_replay_stats_df['short_pf_peak_x']

    cap_cells_df: pd.DataFrame = neuron_replay_stats_df[np.logical_and(neuron_replay_stats_df['has_long_pf'], neuron_replay_stats_df['is_long_peak_either_cap'])]
    num_total_endcap_cells: int = len(cap_cells_df)

    # "Disppearing" cells fall below the 1Hz firing criteria on the short track:
    disappearing_endcap_cells_df: pd.DataFrame = cap_cells_df[np.logical_not(cap_cells_df['has_short_pf'])]
    num_disappearing_endcap_cells: int = len(disappearing_endcap_cells_df)
    print(f'num_disappearing_endcap_cells/num_total_endcap_cells: {num_disappearing_endcap_cells}/{num_total_endcap_cells}')

    non_disappearing_endcap_cells_df: pd.DataFrame = cap_cells_df[cap_cells_df['has_short_pf']] # "non_disappearing" cells are those with a placefield on the short track as well
    num_non_disappearing_endcap_cells: int = len(non_disappearing_endcap_cells_df)
    print(f'num_non_disappearing_endcap_cells/num_total_endcap_cells: {num_non_disappearing_endcap_cells}/{num_total_endcap_cells}')

    # Classify the non_disappearing cells into two groups:
    # 1. Those that exhibit significant remapping onto somewhere else on the track
    non_disappearing_endcap_cells_df['has_significant_distance_remapping'] = (np.abs(non_disappearing_endcap_cells_df['LS_pf_peak_x_diff']) >= min_significant_remapping_x_distance) # The most a placefield could translate intwards would be (35 + (pf_width/2.0)) I think.
    num_significant_position_remappping_endcap_cells: int = len(non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == True])
    print(f'num_significant_position_remappping_endcap_cells/num_non_disappearing_endcap_cells: {num_significant_position_remappping_endcap_cells}/{num_non_disappearing_endcap_cells}')

    # 2. Those that seem to remain where they were on the long track, perhaps being "sampling-clipped" or translated adjacent to the platform. These two subcases can be distinguished by a change in the placefield's length (truncated cells would be a fraction of the length, although might need to account for scaling with new track length)
    significant_distant_remapping_endcap_cells_df: pd.DataFrame = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == True] ## why only endcap cells?
    minorly_changed_endcap_cells_df: pd.DataFrame = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == False]
    # significant_distant_remapping_endcap_aclus = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping']].index # Int64Index([3, 5, 7, 11, 14, 38, 41, 53, 57, 61, 62, 75, 78, 79, 82, 83, 85, 95, 98, 100, 102], dtype='int64')
    
    return neuron_replay_stats_df, (non_disappearing_endcap_cells_df, disappearing_endcap_cells_df, minorly_changed_endcap_cells_df, significant_distant_remapping_endcap_cells_df,)








# ==================================================================================================================== #
# 2024-05-23 - Restoring 'is_user_annotated_epoch' and 'is_valid_epoch' columns                                        #
# ==================================================================================================================== #

















# ==================================================================================================================== #
# Older                                                                                                                #
# ==================================================================================================================== #


@function_attributes(short_name=None, tags=['batch', 'matlab_mat_file', 'multi-session', 'grid_bin_bounds'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-10 07:41', related_items=[])
def batch_build_user_annotation_grid_bin_bounds_from_exported_position_info_mat_files(search_parent_path: Path, platform_side_length: float = 22.0):
    """ finds all *.position_info.mat files recurrsively in the search_parent_path, then try to load them and parse their parent directory as a session to build an IdentifyingContext that can be used as a key in UserAnnotations.
    
    Builds a list of grid_bin_bounds user annotations that can be pasted into `specific_session_override_dict`

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import batch_build_user_annotation_grid_bin_bounds_from_exported_position_info_mat_files

        _out_user_annotations_add_code_lines, loaded_configs = batch_build_user_annotation_grid_bin_bounds_from_exported_position_info_mat_files(search_parent_path=Path(r'W:\\Data\\Kdiba'))
        loaded_configs

    """
    # from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
    from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphocorehelpers.geometry_helpers import BoundsRect
    from pyphocorehelpers.Filesystem.path_helpers import discover_data_files

    found_any_position_info_mat_files: list[Path] = discover_data_files(search_parent_path, glob_pattern='*.position_info.mat', recursive=True)
    found_any_position_info_mat_files
    ## INPUTS: found_any_position_info_mat_files
    
    user_annotations = UserAnnotationsManager.get_user_annotations()
    _out_user_annotations_add_code_lines: List[str] = []

    loaded_configs = {}
    for a_path in found_any_position_info_mat_files:
        if a_path.exists():
            file_prefix: str = a_path.name.split('.position_info.mat', maxsplit=1)[0]
            print(f'file_prefix: {file_prefix}')
            session_path = a_path.parent.resolve()
            session_name = KDibaOldDataSessionFormatRegisteredClass.get_session_name(session_path)
            print(f'session_name: {session_name}')
            if session_name == file_prefix:
                try:
                    _updated_config_dict = {}
                    _test_session = KDibaOldDataSessionFormatRegisteredClass.build_session(session_path) # Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')
                    _test_session_context = _test_session.get_context()
                    print(f'_test_session_context: {_test_session_context}')
                    
                    _updated_config_dict = KDibaOldDataSessionFormatRegisteredClass.perform_load_position_info_mat(session_position_mat_file_path=a_path, config_dict=_updated_config_dict)
                    _updated_config_dict['session_context'] = _test_session_context

                    loaded_track_limits = _updated_config_dict.get('loaded_track_limits', None) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
                    if loaded_track_limits is not None:
                        long_xlim = loaded_track_limits['long_xlim']
                        long_ylim = loaded_track_limits['long_ylim']
                        # short_xlim = loaded_track_limits['short_xlim']
                        # short_ylim = loaded_track_limits['short_ylim']

                        ## Build full grid_bin_bounds:
                        from_mat_lims_grid_bin_bounds = BoundsRect(xmin=(long_xlim[0]-platform_side_length), xmax=(long_xlim[1]+platform_side_length), ymin=long_ylim[0], ymax=long_ylim[1])
                        # print(f'from_mat_lims_grid_bin_bounds: {from_mat_lims_grid_bin_bounds}')
                        # from_mat_lims_grid_bin_bounds # BoundsRect(xmin=37.0773897438341, xmax=250.69004399129707, ymin=116.16397564990257, ymax=168.1197529956474)
                        _updated_config_dict['new_grid_bin_bounds'] = from_mat_lims_grid_bin_bounds.extents

                        ## Build Update:


                        # curr_active_pipeline.sess.config.computation_config['pf_params'].grid_bin_bounds = new_grid_bin_bounds

                        active_context = _test_session_context # curr_active_pipeline.get_session_context()
                        final_context = active_context.adding_context_if_missing(user_annotation='grid_bin_bounds')
                        user_annotations[final_context] = from_mat_lims_grid_bin_bounds.extents
                        # Updates the context. Needs to generate the code.

                        ## Generate code to insert int user_annotations:
                        
                        ## new style:
                        # print(f"user_annotations[{final_context.get_initialization_code_string()}] = {(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}")
                        _a_code_line: str = f"user_annotations[{active_context.get_initialization_code_string()}] = dict(grid_bin_bounds=({(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}))"
                        # print(_a_code_line)
                        _out_user_annotations_add_code_lines.append(_a_code_line)

                    loaded_configs[a_path] = _updated_config_dict
                except (AssertionError, BaseException) as e:
                    print(f'e: {e}. Skipping.')
                    pass
                    
    # loaded_configs

    print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
    print('\n'.join(_out_user_annotations_add_code_lines))
    return _out_user_annotations_add_code_lines, loaded_configs

def _build_new_grid_bin_bounds_from_mat_exported_xlims(curr_active_pipeline, platform_side_length: float = 22.0, build_user_annotation_string:bool=False):
    """ 2024-04-10 -
     
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _build_new_grid_bin_bounds_from_mat_exported_xlims

    """
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphocorehelpers.geometry_helpers import BoundsRect

    ## INPUTS: curr_active_pipeline
    loaded_track_limits = deepcopy(curr_active_pipeline.sess.config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
    x_midpoint: float = curr_active_pipeline.sess.config.x_midpoint
    pix2cm: float = curr_active_pipeline.sess.config.pix2cm

    ## INPUTS: loaded_track_limits
    print(f'loaded_track_limits: {loaded_track_limits}')

    long_xlim = loaded_track_limits['long_xlim']
    long_ylim = loaded_track_limits['long_ylim']
    short_xlim = loaded_track_limits['short_xlim']
    short_ylim = loaded_track_limits['short_ylim']


    ## Build full grid_bin_bounds:
    from_mat_lims_grid_bin_bounds = BoundsRect(xmin=(long_xlim[0]-platform_side_length), xmax=(long_xlim[1]+platform_side_length), ymin=long_ylim[0], ymax=long_ylim[1])
    # print(f'from_mat_lims_grid_bin_bounds: {from_mat_lims_grid_bin_bounds}')
    # from_mat_lims_grid_bin_bounds # BoundsRect(xmin=37.0773897438341, xmax=250.69004399129707, ymin=116.16397564990257, ymax=168.1197529956474)

    if build_user_annotation_string:
        ## Build Update:
        user_annotations = UserAnnotationsManager.get_user_annotations()

        # curr_active_pipeline.sess.config.computation_config['pf_params'].grid_bin_bounds = new_grid_bin_bounds

        active_context = curr_active_pipeline.get_session_context()
        final_context = active_context.adding_context_if_missing(user_annotation='grid_bin_bounds')
        user_annotations[final_context] = from_mat_lims_grid_bin_bounds.extents
        # Updates the context. Needs to generate the code.

        ## Generate code to insert int user_annotations:
        print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
        ## new style:
        # print(f"user_annotations[{final_context.get_initialization_code_string()}] = {(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}")
        print(f"user_annotations[{active_context.get_initialization_code_string()}] = dict(grid_bin_bounds=({(from_mat_lims_grid_bin_bounds.xmin, from_mat_lims_grid_bin_bounds.xmax), (from_mat_lims_grid_bin_bounds.ymin, from_mat_lims_grid_bin_bounds.ymax)}))")

    return from_mat_lims_grid_bin_bounds


# ==================================================================================================================== #
# 2024-04-05 - Back to the laps                                                                                        #
# ==================================================================================================================== #

# ==================================================================================================================== #
# 2024-01-17 - Lap performance validation                                                                              #
# ==================================================================================================================== #
from neuropy.analyses.placefields import PfND
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult, _check_result_laps_epochs_df_performance
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CompleteDecodedContextCorrectness

def add_groundtruth_information(curr_active_pipeline, a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult):
    """    takes 'laps_df' and 'result_laps_epochs_df' to add the ground_truth and the decoded posteriors:

        a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = alt_directional_merged_decoders_result


        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_groundtruth_information

        result_laps_epochs_df = add_groundtruth_information(curr_active_pipeline, a_directional_merged_decoders_result=a_directional_merged_decoders_result, result_laps_epochs_df=result_laps_epochs_df)


    """
    from neuropy.core import Laps

    ## Inputs: a_directional_merged_decoders_result, laps_df
    
    ## Get the most likely direction/track from the decoded posteriors:
    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = a_directional_merged_decoders_result.laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = a_directional_merged_decoders_result.laps_track_identity_marginals_tuple

    result_laps_epochs_df: pd.DataFrame = a_directional_merged_decoders_result.laps_epochs_df

    # Ensure it has the 'lap_track' column
    ## Compute the ground-truth information using the position information:
    # adds columns: ['maze_id', 'is_LR_dir']
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_df = laps_obj.to_dataframe()
    laps_df: pd.DataFrame = Laps._update_dataframe_computed_vars(laps_df=laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=curr_active_pipeline.sess) # NOTE: .sess is used because global_session is missing the last two laps
    
    ## 2024-01-17 - Updates the `a_directional_merged_decoders_result.laps_epochs_df` with both the ground-truth values and the decoded predictions
    result_laps_epochs_df['maze_id'] = laps_df['maze_id'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching
    ## add the 'is_LR_dir' groud-truth column in:
    result_laps_epochs_df['is_LR_dir'] = laps_df['is_LR_dir'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching

    ## Add the decoded results to the laps df:
    result_laps_epochs_df['is_most_likely_track_identity_Long'] = laps_is_most_likely_track_identity_Long
    result_laps_epochs_df['is_most_likely_direction_LR'] = laps_is_most_likely_direction_LR_dir


    ## Update the source:
    a_directional_merged_decoders_result.laps_epochs_df = result_laps_epochs_df

    return result_laps_epochs_df


@function_attributes(short_name=None, tags=['laps', 'groundtruth', 'context-decoding', 'context-discrimination'], input_requires=[], output_provides=[], uses=[], used_by=['perform_sweep_lap_groud_truth_performance_testing'], creation_date='2024-04-05 18:40', related_items=['DirectionalPseudo2DDecodersResult'])
def _perform_variable_time_bin_lap_groud_truth_performance_testing(owning_pipeline_reference,
                                                                    desired_laps_decoding_time_bin_size: float = 0.5, desired_ripple_decoding_time_bin_size: Optional[float] = None, use_single_time_bin_per_epoch: bool=False,
                                                                    included_neuron_ids: Optional[NDArray]=None) -> Tuple[DirectionalPseudo2DDecodersResult, pd.DataFrame, CompleteDecodedContextCorrectness]:
    """ 2024-01-17 - Pending refactor from ReviewOfWork_2024-01-17.ipynb 

    Makes a copy of the 'DirectionalMergedDecoders' result and does the complete process of re-calculation for the provided time bin sizes. Finally computes the statistics about correctly computed contexts from the laps.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_variable_time_bin_lap_groud_truth_performance_testing

        a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple = _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size=1.5, desired_ripple_decoding_time_bin_size=None)
        (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple

        ## Filtering with included_neuron_ids:
        a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple = _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size=1.5, included_neuron_ids=included_neuron_ids)
        
    
    """
    from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays, find_desired_sort_indicies
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import PfND

    ## Copy the default result:
    directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = owning_pipeline_reference.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)

    if included_neuron_ids is not None:
        # prune the decoder by the provided `included_neuron_ids`

        ## Due to a limitation of the merged pseudo2D decoder (`alt_directional_merged_decoders_result.all_directional_pf1D_Decoder`) that prevents `.get_by_id(...)` from working, we have to rebuild the pseudo2D decoder from the four pf1D decoders:
        restricted_all_directional_decoder_pf1D_dict: Dict[str, BasePositionDecoder] = deepcopy(alt_directional_merged_decoders_result.all_directional_decoder_dict) # copy the dictionary
        ## Filter the dictionary using .get_by_id(...)
        restricted_all_directional_decoder_pf1D_dict = {k:v.get_by_id(included_neuron_ids) for k,v in restricted_all_directional_decoder_pf1D_dict.items()}

        restricted_all_directional_pf1D = PfND.build_merged_directional_placefields(restricted_all_directional_decoder_pf1D_dict, debug_print=False)
        restricted_all_directional_pf1D_Decoder = BasePositionDecoder(restricted_all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        
        # included_neuron_ids = intersection_of_arrays(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.neuron_IDs, included_neuron_ids)
        # is_aclu_included_list = np.isin(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap.neuron_ids, included_neuron_ids)
        # included_aclus = np.array(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap.neuron_ids)[is_aclu_included_list
        # modified_decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.get_by_id(included_aclus)

        ## Set the result:
        alt_directional_merged_decoders_result.all_directional_pf1D_Decoder = restricted_all_directional_pf1D_Decoder

        # ratemap method:
        # modified_decoder_pf_ratemap = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap.get_by_id(included_aclus)
        # alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap = modified_decoder_pf_ratemap

        # alt_directional_merged_decoders_result.all_directional_pf1D_Decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.get_by_id(included_neuron_ids, defer_compute_all=True)
        
    all_directional_pf1D_Decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder

    # Modifies alt_directional_merged_decoders_result, a copy of the original result, with new timebins
    long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
    # t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()

    if use_single_time_bin_per_epoch:
        print(f'WARNING: use_single_time_bin_per_epoch=True so time bin sizes will be ignored.')
        
    ## Decode Laps:
    global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
    min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(global_any_laps_epochs_obj.to_dataframe()['duration'].to_numpy())
    laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002
    if use_single_time_bin_per_epoch:
        laps_decoding_time_bin_size = None
    
    alt_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

    ## Decode Ripples:
    if desired_ripple_decoding_time_bin_size is not None:
        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
        min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
        ripple_decoding_time_bin_size: float = min(desired_ripple_decoding_time_bin_size, min_possible_time_bin_size) # 10ms # 0.002
        if use_single_time_bin_per_epoch:
            ripple_decoding_time_bin_size = None
        alt_directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(owning_pipeline_reference.sess.spikes_df), global_replays, decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch)
        
    ## Post Compute Validations:
    alt_directional_merged_decoders_result.perform_compute_marginals()

    result_laps_epochs_df: pd.DataFrame = add_groundtruth_information(owning_pipeline_reference, a_directional_merged_decoders_result=alt_directional_merged_decoders_result)

    laps_decoding_time_bin_size = alt_directional_merged_decoders_result.laps_decoding_time_bin_size
    print(f'laps_decoding_time_bin_size: {laps_decoding_time_bin_size}')

    ## Uses only 'result_laps_epochs_df'
    complete_decoded_context_correctness_tuple = _check_result_laps_epochs_df_performance(result_laps_epochs_df)

    # Unpack like:
    # (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple

    return alt_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple


@function_attributes(short_name=None, tags=['laps', 'groundtruith', 'sweep', 'context-decoding', 'context-discrimination'], input_requires=[], output_provides=[], uses=['_perform_variable_time_bin_lap_groud_truth_performance_testing'], used_by=[], creation_date='2024-04-05 22:47', related_items=[])
def perform_sweep_lap_groud_truth_performance_testing(curr_active_pipeline, included_neuron_ids_list: List[NDArray], desired_laps_decoding_time_bin_size:float=1.5):
    """ Sweeps through each `included_neuron_ids` in the provided `included_neuron_ids_list` and calls `_perform_variable_time_bin_lap_groud_truth_performance_testing(...)` to get its laps ground-truth performance.
    Can be used to assess the contributes of each set of cells (exclusive, rate-remapping, etc) to the discrimination decoding performance.
    
    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import perform_sweep_lap_groud_truth_performance_testing

        desired_laps_decoding_time_bin_size: float = 1.0
        included_neuron_ids_list = [short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset]

        _output_tuples_list = perform_sweep_lap_groud_truth_performance_testing(curr_active_pipeline, 
                                                                                included_neuron_ids_list=included_neuron_ids_list,
                                                                                desired_laps_decoding_time_bin_size=desired_laps_decoding_time_bin_size)

        percent_laps_correctness_df: pd.DataFrame = pd.DataFrame.from_records([complete_decoded_context_correctness_tuple.percent_correct_tuple for (a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple) in _output_tuples_list],
                                columns=("track_ID_correct", "dir_correct", "complete_correct"))
        percent_laps_correctness_df

                                                                        
    
    Does not modifiy the curr_active_pipeline (pure)


    """
    _output_tuples_list = []
    for included_neuron_ids in included_neuron_ids_list:
        a_lap_ground_truth_performance_testing_tuple = _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size=desired_laps_decoding_time_bin_size, included_neuron_ids=included_neuron_ids)
        _output_tuples_list.append(a_lap_ground_truth_performance_testing_tuple)

        # ## Unpacking `a_lap_ground_truth_performance_testing_tuple`:
        # a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple = a_lap_ground_truth_performance_testing_tuple
        # (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple

        # a_directional_merged_decoders_result, result_laps_epochs_df, (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = a_lap_ground_truth_performance_testing_tuple
    return _output_tuples_list


# ---------------------------------------------------------------------------- #
#             2024-03-29 - Rigorous Decoder Performance assessment             #
# ---------------------------------------------------------------------------- #
# Quantify cell contributions to decoders
# Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result

import portion as P # Required for interval search: portion~=2.3.0
from neuropy.core.epoch import Epoch, ensure_dataframe
from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df, _convert_start_end_tuples_list_to_PortionInterval
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
from sklearn.metrics import mean_squared_error

## Get custom decoder that is only trained on a portion of the laps
## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
# long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]

## Restrict the data post-hoc?

## Time-dependent decoder?

## Split the lap epochs into training and test periods.
##### Ideally we could test the lap decoding error by sampling randomly from the time bins and omitting 1/6 of time bins from the placefield building (effectively the training data). These missing bins will be used as the "test data" and the decoding error will be computed by decoding them and subtracting the actual measured position during these bins.



# ==================================================================================================================== #
# 2024-03-09 - Filtering                                                                                               #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult, filter_and_update_epochs_and_spikes
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import HeuristicReplayScoring
from neuropy.core.epoch import ensure_dataframe, find_data_indicies_from_epoch_times

@function_attributes(short_name=None, tags=['filtering'], input_requires=[], output_provides=[], uses=[], used_by=['_perform_filter_replay_epochs'], creation_date='2024-04-25 06:38', related_items=[])
def _apply_filtering_to_marginals_result_df(active_result_df: pd.DataFrame, filtered_epochs_df: pd.DataFrame, filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult]):
    """ after filtering the epochs (for user selections, validity, etc) apply the same filtering to a results df. 

    Applied to `filtered_decoder_filter_epochs_decoder_result_dict` to build a dataframe
    
    """
    ## INPUTS: active_result_df, filtered_epochs_df

    # found_data_indicies = find_data_indicies_from_epoch_times(active_result_df, epoch_times=np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t'], atol=1e-2)
    # found_data_indicies = find_data_indicies_from_epoch_times(active_result_df, epoch_times=np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t'], atol=1e-2)
    found_data_indicies = find_data_indicies_from_epoch_times(active_result_df, epoch_times=np.squeeze(filtered_epochs_df['start'].to_numpy()), t_column_names=['ripple_start_t'], atol=1e-3)

    # ripple_all_epoch_bins_marginals_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)

    active_result_df = active_result_df.loc[found_data_indicies].copy().reset_index(drop=True)
    direction_max_indices = active_result_df[['P_LR', 'P_RL']].values.argmax(axis=1)
    track_identity_max_indices = active_result_df[['P_Long', 'P_Short']].values.argmax(axis=1)

    ## INPUTS: filtered_decoder_filter_epochs_decoder_result_dict

    df_column_names = [list(a_df.filter_epochs.columns) for a_df in filtered_decoder_filter_epochs_decoder_result_dict.values()]
    print(f"df_column_names: {df_column_names}")
    selected_df_column_names = ['wcorr', 'pearsonr']

    # merged_dfs_dict = {a_name:a_df.filter_epochs[selected_df_column_names].add_suffix(f"_{a_name}") for a_name, a_df in filtered_decoder_filter_epochs_decoder_result_dict.items()}
    # merged_dfs_dict = pd.concat([a_df.filter_epochs[selected_df_column_names].add_suffix(f"_{a_name}") for a_name, a_df in filtered_decoder_filter_epochs_decoder_result_dict.items()], axis='columns')
    # merged_dfs_dict

    # filtered_decoder_filter_epochs_decoder_result_dict['short_LR'][a_column_name], filtered_decoder_filter_epochs_decoder_result_dict['short_RL'][a_column_name]

    ## BEST/COMPARE OUT DF:
    # active_result_df = deepcopy(active_result_df)

    # Get only the best direction long/short values for each metric:
    for a_column_name in selected_df_column_names: # = 'wcorr'
        assert len(direction_max_indices) == len(filtered_decoder_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)
        active_result_df[f'long_best_{a_column_name}'] = np.where(direction_max_indices, filtered_decoder_filter_epochs_decoder_result_dict['long_LR'].filter_epochs[a_column_name].to_numpy(), filtered_decoder_filter_epochs_decoder_result_dict['long_RL'].filter_epochs[a_column_name].to_numpy())
        active_result_df[f'short_best_{a_column_name}'] = np.where(direction_max_indices, filtered_decoder_filter_epochs_decoder_result_dict['short_LR'].filter_epochs[a_column_name].to_numpy(), filtered_decoder_filter_epochs_decoder_result_dict['short_RL'].filter_epochs[a_column_name].to_numpy())
        active_result_df[f'{a_column_name}_abs_diff'] = active_result_df[f'long_best_{a_column_name}'].abs() - active_result_df[f'short_best_{a_column_name}'].abs()


    ## ['wcorr_abs_diff', 'pearsonr_abs_diff']
    return active_result_df

## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict
@function_attributes(short_name=None, tags=['filter', 'replay', 'IMPORTANT'], input_requires=[], output_provides=[], uses=['_apply_filtering_to_marginals_result_df'], used_by=[], creation_date='2024-04-24 18:03', related_items=[])
def _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], ripple_all_epoch_bins_marginals_df: pd.DataFrame, ripple_decoding_time_bin_size: float,
            should_only_include_user_selected_epochs:bool=True, **additional_selections_context):
    """ the main replay epochs filtering function.
    
    if should_only_include_user_selected_epochs is True, it only includes user selected (annotated) ripples


    Usage:
        from neuropy.core.epoch import find_data_indicies_from_epoch_times
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_filter_replay_epochs

        filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict, filtered_ripple_all_epoch_bins_marginals_df = _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict, ripple_all_epoch_bins_marginals_df, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)
        filtered_epochs_df

    """
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, HeuristicScoresTuple

    # 2024-03-04 - Filter out the epochs based on the criteria:
    filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
    ## 2024-03-08 - Also constrain the user-selected ones (just to try it):
    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates, **additional_selections_context)

    ## filter the epochs by something and only show those:
    # INPUTS: filtered_epochs_df
    # filtered_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(active_epochs_df[['start', 'stop']].to_numpy())
    ## Update the `decoder_ripple_filter_epochs_decoder_result_dict` with the included epochs:
    filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) for a_name, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()} # working filtered
    # print(f"any_good_selected_epoch_times.shape: {any_good_selected_epoch_times.shape}") # (142, 2)
    ## Constrain again now by the user selections
    if should_only_include_user_selected_epochs:
        filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(any_good_selected_epoch_times) for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
    # filtered_decoder_filter_epochs_decoder_result_dict

    # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores`
    filtered_decoder_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)

    if should_only_include_user_selected_epochs:
        filtered_epochs_df = filtered_epochs_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)
    
    ## OUT: filtered_decoder_filter_epochs_decoder_result_dict, filtered_epochs_df

    # `ripple_all_epoch_bins_marginals_df`
    filtered_ripple_all_epoch_bins_marginals_df = deepcopy(ripple_all_epoch_bins_marginals_df)
    filtered_ripple_all_epoch_bins_marginals_df = _apply_filtering_to_marginals_result_df(filtered_ripple_all_epoch_bins_marginals_df, filtered_epochs_df=filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)
    assert len(filtered_epochs_df) == len(filtered_ripple_all_epoch_bins_marginals_df), f"len(filtered_epochs_df): {len(filtered_epochs_df)} != len(active_result_df): {len(filtered_ripple_all_epoch_bins_marginals_df)}"

    df = filtered_ripple_all_epoch_bins_marginals_df

    # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
    session_name: str = curr_active_pipeline.session_name
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, t_start=t_start, curr_session_t_delta=t_delta, t_end=t_end, time_col='ripple_start_t')
    df["time_bin_size"] = ripple_decoding_time_bin_size
    df['is_user_annotated_epoch'] = True # if it's filtered here, it's true
    return filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict, filtered_ripple_all_epoch_bins_marginals_df



@function_attributes(short_name=None, tags=['filter', 'epoch_selection', 'export', 'h5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-08 13:28', related_items=[])
def export_numpy_testing_filtered_epochs(curr_active_pipeline, global_epoch_name, track_templates, required_min_percentage_of_active_cells: float = 0.333333, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, **additional_selections_context):
    """ Save testing variables to file 'NeuroPy/tests/neuropy_pf_testing.h5'
    exports: original_epochs_df, filtered_epochs_df, active_spikes_df

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import export_numpy_testing_filtered_epochs

    finalized_output_cache_file = export_numpy_testing_filtered_epochs(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
    finalized_output_cache_file

    """
    from neuropy.core import Epoch
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

    global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
    if isinstance(global_replays, pd.DataFrame):
        global_replays = Epoch(global_replays.epochs.get_valid_df())
    original_spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)
    original_spikes_df = original_spikes_df.spikes.sliced_by_neuron_id(track_templates.any_decoder_neuron_IDs)
    # Start Filtering
    original_epochs_df = deepcopy(global_replays.to_dataframe())

    # 2024-03-04 - Filter out the epochs based on the criteria:
    filtered_epochs_df, filtered_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)


    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates, **additional_selections_context)
    print(f"any_good_selected_epoch_times.shape: {any_good_selected_epoch_times.shape}") # (142, 2)

    ## Save for NeuroPy testing:
    finalized_output_cache_file='../NeuroPy/tests/neuropy_epochs_testing.h5'
    sess_identifier_key='sess'
    original_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/original_epochs_df', format='table')
    filtered_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/filtered_epochs_df', format='table')
    # selected_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/selected_epochs_df', format='table')
    any_good_selected_epoch_times_df = pd.DataFrame(any_good_selected_epoch_times, columns=['start', 'stop'])
    any_good_selected_epoch_times_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/any_good_selected_epoch_times_df', format='table')


    # basic_epoch_column_names = ['start', 'stop', 'label', 'duration', 'ripple_idx', 'P_Long']
    # test_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df[basic_epoch_column_names])
    # test_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/test_df', format='table')
    return finalized_output_cache_file


# ==================================================================================================================== #
# Old type display helpers                                                                                             #
# ==================================================================================================================== #

def register_type_display(func_to_register, type_to_register):
    """ adds the display function (`func_to_register`) it decorates to the class (`type_to_register) as a method


    """
    @wraps(func_to_register)
    def wrapper(*args, **kwargs):
        return func_to_register(*args, **kwargs)

    function_name: str = func_to_register.__name__ # get the name of the function to be added as the property
    setattr(type_to_register, function_name, wrapper) # set the function as a method with the same name as the decorated function on objects of the class.	
    return wrapper



# ==================================================================================================================== #
# 2024-02-15 - Radon Transform / Weighted Correlation, etc helpers                                                     #
# ==================================================================================================================== #


# ==================================================================================================================== #
# 2024-02-08 - Plot Single ACLU Heatmaps for Each Decoder                                                              #
# ==================================================================================================================== #
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle

@function_attributes(short_name=None, tags=['plot', 'heatmap', 'peak'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-08 00:00', related_items=[])
def plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin, point_dict=None, ax_dict=None, extra_decoder_values_dict=None, tuning_curves_dict=None, include_tuning_curves=False):
    """ 2024-02-06 - Plots the four position-binned-activity maps (for each directional decoding epoch) as a 4x4 subplot grid using matplotlib. 

    """
    from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap
    if tuning_curves_dict is None:
        assert include_tuning_curves == False
    
    # figure_kwargs = dict(layout="tight")
    figure_kwargs = dict(layout="none")

    if ax_dict is None:
        if not include_tuning_curves:
            # fig = plt.figure(layout="constrained", figsize=[9, 7], dpi=220, clear=True) # figsize=[Width, height] in inches.
            fig = plt.figure(figsize=[8, 7], dpi=220, clear=True, **figure_kwargs)
            long_width_ratio = 1
            ax_dict = fig.subplot_mosaic(
                [
                    ["ax_long_LR", "ax_long_RL"],
                    ["ax_short_LR", "ax_short_RL"],
                ],
                # set the height ratios between the rows
                # set the width ratios between the columns
                width_ratios=[long_width_ratio, long_width_ratio],
                sharex=True, sharey=False,
                gridspec_kw=dict(wspace=0.027, hspace=0.112) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )
        else:
            # tuning curves mode:
            fig = plt.figure(figsize=[9, 7], dpi=220, clear=True, **figure_kwargs)
            long_width_ratio = 1
            ax_dict = fig.subplot_mosaic(
                [
                    ["ax_long_LR_curve", "ax_long_RL_curve"],
                    ["ax_long_LR", "ax_long_RL"],
                    ["ax_short_LR", "ax_short_RL"],
                    ["ax_short_LR_curve", "ax_short_RL_curve"],
                ],
                # set the height ratios between the rows
                # set the width ratios between the columns
                width_ratios=[long_width_ratio, long_width_ratio],
                height_ratios=[1, 7, 7, 1], # tuning curves are smaller than laps
                sharex=True, sharey=False,
                gridspec_kw=dict(wspace=0.027, hspace=0.112) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )
            curve_ax_names = ["ax_long_LR_curve", "ax_long_RL_curve", "ax_short_LR_curve", "ax_short_RL_curve"]

    else:
        if not include_tuning_curves:
            # figure already exists, reuse the axes
            assert len(ax_dict) == 4
            assert list(ax_dict.keys()) == ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]
        else:
            # tuning curves mode:
            assert len(ax_dict) == 8
            assert list(ax_dict.keys()) == ["ax_long_LR_curve", "ax_long_RL_curve", "ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL", "ax_short_LR_curve", "ax_short_RL_curve"]
        

    
    # Get the colormap to use and set the bad color
    cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
    cmap.set_bad(color='black')

    # Compute extents for imshow:
    imshow_kwargs = {
        'origin': 'lower',
        # 'vmin': 0,
        # 'vmax': 1,
        'cmap': cmap,
        'interpolation':'nearest',
        'aspect':'auto',
        'animated':True,
        'show_xticks':False,
    }

    _old_data_to_ax_mapping = dict(zip(['maze1_odd', 'maze1_even', 'maze2_odd', 'maze2_even'], ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]))
    data_to_ax_mapping = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]))

    # ['long_LR', 'long_RL', 'short_LR', 'short_RL']

    
    for k, v in curr_aclu_z_scored_tuning_map_matrix_dict.items():
        # is_first_item = (k == list(curr_aclu_z_scored_tuning_map_matrix_dict.keys())[0])
        is_last_item = (k == list(curr_aclu_z_scored_tuning_map_matrix_dict.keys())[-1])
        
        curr_ax = ax_dict[data_to_ax_mapping[k]]
        curr_ax.clear()
        
        # hist_data = np.random.randn(1_500)
        # xbin_centers = np.arange(len(hist_data))+0.5
        # ax_dict["ax_LONG_pf_tuning_curve"] = plot_placefield_tuning_curve(xbin_centers, (-1.0 * curr_cell_normalized_tuning_curve), ax_dict["ax_LONG_pf_tuning_curve"], is_horizontal=True)

        n_epochs:int = np.shape(v)[1]
        epoch_indicies = np.arange(n_epochs)

        # Posterior distribution heatmaps at each point.
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], epoch_indicies[0], epoch_indicies[-1])           
        imshow_kwargs['extent'] = (xmin, xmax, ymin, ymax)

        # plot heatmap:
        curr_ax.set_xticklabels([])
        curr_ax.set_yticklabels([])
        fig, ax, im = visualize_heatmap(v.copy(), ax=curr_ax, title=f'{k}', layout='none', defer_show=True, **imshow_kwargs) # defer_show so it doesn't produce a separate figure for each!
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))


        if include_tuning_curves:
            tuning_curve = tuning_curves_dict[k]
            curr_curve_ax = ax_dict[f"{data_to_ax_mapping[k]}_curve"]
            curr_curve_ax.clear()

            if tuning_curve is not None:
                # plot curve heatmap:
                if not is_last_item:
                    curr_curve_ax.set_xticklabels([])
                    # Leave the position x-ticks on for the last item

                curr_curve_ax.set_yticklabels([])
                ymin, ymax = 0, 1
                imshow_kwargs['extent'] = (xmin, xmax, 0, 1)
                fig, curr_curve_ax, im = visualize_heatmap(tuning_curve.copy(), ax=curr_curve_ax, title=f'{k}', defer_show=True, **imshow_kwargs) # defer_show so it doesn't produce a separate figure for each!
                curr_curve_ax.set_xlim((xmin, xmax))
                curr_curve_ax.set_ylim((0, 1))
                
            point_ax = curr_curve_ax # draw the lines on the tuning curve axis
            
        else:
            point_ax = ax

        if point_dict is not None:
            if k in point_dict:
                # have points to plot
                point_ax.vlines(point_dict[k], ymin=ymin, ymax=ymax, colors='r', label=f'{k}_peak')
                

    # fig.tight_layout()
    # NOTE: these layout changes don't seem to take effect until the window containing the figure is resized.
    # fig.set_layout_engine('compressed') # TAKEWAY: Use 'compressed' instead of 'constrained'
    fig.set_layout_engine('none') # disabling layout engine. Strangely still allows window to resize and the plots scale, so I'm not sure what the layout engine is doing.


    # ax_dict["ax_SHORT_activity_v_time"].plot([1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 1, 2, 3, 3, 1, 2, 0, 0])
    # ax_dict["ax_SHORT_pf_tuning_curve"] = plot_placefield_tuning_curve(xbin_centers, curr_cell_normalized_tuning_curve, ax_dict["ax_SHORT_pf_tuning_curve"], is_horizontal=True)
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_xticklabels([])
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_yticklabels([])
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_box

    return fig, ax_dict

# INPUTS: directional_active_lap_pf_results_dicts, test_aclu: int = 26, xbin_centers, decoder_aclu_peak_location_df_merged

def plot_single_heatmap_set_with_points(directional_active_lap_pf_results_dicts, xbin_centers, xbin, decoder_aclu_peak_location_df_merged: pd.DataFrame, aclu: int = 26, **kwargs):
    """ 2024-02-06 - Plot all four decoders for a single aclu, with overlayed red lines for the detected peaks. 
    
    plot_single_heatmap_set_with_points

    plot_cell_position_binned_activity_over_time

    Usage:

        decoders_tuning_curves_dict = track_templates.decoder_normalized_tuning_curves_dict_dict.copy()

        extra_decoder_values_dict = {'tuning_curves': decoders_tuning_curves_dict, 'points': decoder_aclu_peak_location_df_merged}

        # decoders_tuning_curves_dict
        xbin_centers = deepcopy(active_pf_dt.xbin_centers)
        xbin = deepcopy(active_pf_dt.xbin)
        fig, ax_dict = plot_single_heatmap_set_with_points(directional_active_lap_pf_results_dicts, xbin_centers, xbin, extra_decoder_values_dict=extra_decoder_values_dict, aclu=4, 
                                                        decoders_tuning_curves_dict=decoders_tuning_curves_dict, decoder_aclu_peak_location_df_merged=decoder_aclu_peak_location_df_merged,
                                                            active_context=curr_active_pipeline.build_display_context_for_session('single_heatmap_set_with_points'))
                                                            
    """
    from neuropy.utils.result_context import IdentifyingContext

    ## TEst: Look at a single aclu value
    # test_aclu: int = 26
    # test_aclu: int = 28
    
    active_context: IdentifyingContext = kwargs.get('active_context', IdentifyingContext())
    active_context = active_context.overwriting_context(aclu=aclu)

    decoders_tuning_curves_dict = kwargs.get('decoders_tuning_curves_dict', None)
    
    matching_aclu_df = decoder_aclu_peak_location_df_merged[decoder_aclu_peak_location_df_merged.aclu == aclu].copy()
    assert len(matching_aclu_df) > 0, f"matching_aclu_df: {matching_aclu_df} for aclu == {aclu}"
    new_peaks_dict: Dict = list(matching_aclu_df.itertuples(index=False))[0]._asdict() # {'aclu': 28, 'long_LR_peak': 185.29063638457257, 'long_RL_peak': nan, 'short_LR_peak': 176.75276643746625, 'short_RL_peak': nan, 'LR_peak_diff': 8.537869947106316, 'RL_peak_diff': nan}
        
    # long_LR_name, long_RL_name, short_LR_name, short_RL_name
    curr_aclu_z_scored_tuning_map_matrix_dict = {}
    curr_aclu_mean_epoch_peak_location_dict = {}
    curr_aclu_median_peak_location_dict = {}
    curr_aclu_extracted_decoder_peak_locations_dict = {}

    ## Find the peak location for each epoch:
    for a_name, a_decoder_directional_active_lap_pf_result in directional_active_lap_pf_results_dicts.items():
        # print(f'a_name: {a_name}')
        matrix_idx = a_decoder_directional_active_lap_pf_result.aclu_to_matrix_IDX_map[aclu]
        curr_aclu_z_scored_tuning_map_matrix = a_decoder_directional_active_lap_pf_result.z_scored_tuning_map_matrix[:,matrix_idx,:] # .shape (22, 80, 56)
        curr_aclu_z_scored_tuning_map_matrix_dict[a_name] = curr_aclu_z_scored_tuning_map_matrix

        # curr_aclu_mean_epoch_peak_location_dict[a_name] = np.nanmax(curr_aclu_z_scored_tuning_map_matrix, axis=-1)
        assert np.shape(curr_aclu_z_scored_tuning_map_matrix)[-1] == len(xbin_centers), f"np.shape(curr_aclu_z_scored_tuning_map_matrix)[-1]: {np.shape(curr_aclu_z_scored_tuning_map_matrix)} != len(xbin_centers): {len(xbin_centers)}"
        curr_peak_value = new_peaks_dict[f'{a_name}_peak']
        # print(f'curr_peak_value: {curr_peak_value}')
        curr_aclu_extracted_decoder_peak_locations_dict[a_name] = curr_peak_value

        curr_aclu_mean_epoch_peak_location_dict[a_name] = np.nanargmax(curr_aclu_z_scored_tuning_map_matrix, axis=-1)
        curr_aclu_mean_epoch_peak_location_dict[a_name] = xbin_centers[curr_aclu_mean_epoch_peak_location_dict[a_name]] # convert to actual positions instead of indicies
        curr_aclu_median_peak_location_dict[a_name] = np.nanmedian(curr_aclu_mean_epoch_peak_location_dict[a_name])

    # curr_aclu_mean_epoch_peak_location_dict # {'maze1_odd': array([ 0, 55, 54, 55, 55, 53, 50, 55, 52, 52, 55, 53, 53, 52, 51, 52, 55, 55, 53, 55, 55, 54], dtype=int64), 'maze2_odd': array([46, 45, 43, 46, 45, 46, 46, 46, 45, 45, 44, 46, 44, 45, 46, 45, 44, 44, 45, 45], dtype=int64)}


    if decoders_tuning_curves_dict is not None:
        curr_aclu_tuning_curves_dict = {name:v.get(aclu, None) for name, v in decoders_tuning_curves_dict.items()}
    else:
        curr_aclu_tuning_curves_dict = None
                
    # point_value = curr_aclu_median_peak_location_dict
    point_value = curr_aclu_extracted_decoder_peak_locations_dict
    fig, ax_dict = plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin=xbin, point_dict=point_value, tuning_curves_dict=curr_aclu_tuning_curves_dict, include_tuning_curves=True)
    # Set window title and plot title
    perform_update_title_subtitle(fig=fig, ax=None, title_string=f"Position-Binned Activity per Lap - aclu {aclu}", subtitle_string=None, active_context=active_context, use_flexitext_titles=True)

    # fig, ax_dict = plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin=xbin, point_dict=curr_aclu_extracted_decoder_peak_locations_dict) # , defer_show=True
    
    # fig.show()
    return fig, ax_dict


# ==================================================================================================================== #
# Usability/Conveninece Helpers                                                                                        #
# ==================================================================================================================== #

