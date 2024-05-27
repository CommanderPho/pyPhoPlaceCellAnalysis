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


# ==================================================================================================================== #
# 2024-05-26 - Pre-05-27 Stuff                                                                                         #
# ==================================================================================================================== #

# def plot_histograms( data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str ) -> None:
#     # get the pre-delta epochs
#     pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
#     post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

#     descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
#     # plot pre-delta histogram
#     pre_delta_df.hist(column='P_Long')
#     plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
#     plt.show()

#     # plot post-delta histogram
#     post_delta_df.hist(column='P_Long')
#     plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
#     plt.show()
    

def plot_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str) -> None:
    """ plots a stacked histogram of the many time-bin sizes 
    
    Usage:
    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_histograms
        # # You can use it like this:
        # plot_histograms('Laps', 'All Sessions', all_sessions_laps_time_bin_df, "75 ms")
        # plot_histograms('Ripples', 'All Sessions', all_sessions_ripple_time_bin_df, "75 ms")
    """
    assert 'delta_aligned_start_t' in data_results_df.columns
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    time_bin_sizes = pre_delta_df['time_bin_size'].unique()
    
    figure_identifier: str = f"{descriptor_str}_preDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=str(time_bin_size)) 
    
    plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
    plt.legend()
    plt.show()

    # plot post-delta histogram
    time_bin_sizes = post_delta_df['time_bin_size'].unique()
    figure_identifier: str = f"{descriptor_str}_postDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=str(time_bin_size)) 
    
    plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
    plt.legend()
    plt.show()





# ==================================================================================================================== #
# 2024-05-24 - Factored out stats tests from Across Session Point and YellowBlue ...                                   #
# ==================================================================================================================== #

from pyphocorehelpers.indexing_helpers import partition, partition_df, partition_df_dict
import scipy.stats as stats


def _perform_stats_tests(valid_ripple_df, n_shuffles:int=10000, stats_level_of_significance_alpha: float = 0.05, stats_variable_name:str='short_best_wcorr'):
    """
    Splits the passed df into pre and post delta periods
    Take the difference between the pre and post delta means

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_stats_tests


        shuffle_results, p_value, f_value, (dof1, dof2), (variance1, variance2) = _perform_stats_tests(valid_ripple_df, stats_variable_name='short_best_wcorr')

    """
    ## INPUTS: stats_variable_name, valid_ripple_df


    # ['pre-delta', 'post-delta']
    analysis_df = deepcopy(valid_ripple_df[["delta_aligned_start_t", "pre_post_delta_category", stats_variable_name]]).dropna(subset=["pre_post_delta_category", stats_variable_name])
    # partition_df(analysis_df, partitionColumn='pre_post_delta_category')

    # _partition_values, (_pre_delta_df, _post_delta_df) = partition(analysis_df, partitionColumn='pre_post_delta_category') # use `valid_ripple_df` instead of the original dataframe to only get those which are valid.
    _pre_post_delta_partition_df_dict = partition_df_dict(analysis_df, partitionColumn='pre_post_delta_category')
    _pre_delta_df = _pre_post_delta_partition_df_dict['pre-delta']
    _post_delta_df = _pre_post_delta_partition_df_dict['post-delta']

    actual_diff_means = np.nanmean(_post_delta_df[stats_variable_name].to_numpy()) - np.nanmean(_pre_delta_df[stats_variable_name].to_numpy())
    print(f'stats_variable_name: "{stats_variable_name}" -- actual_diff_means: {actual_diff_means}')

    # _pre_delta_df
    # _post_delta_df
    
    # stats_variable_name: str = 'P_Short'

    ## INPUTS: analysis_df, n_shuffles

    shuffle_results = []
    ## INPUT: n_shuffles, analysis_df, stats_variable_name
    shuffled_analysis_df = deepcopy(analysis_df)
    for i in np.arange(n_shuffles):
        # shuffled_analysis_df[stats_variable_name] = shuffled_analysis_df[stats_variable_name].sample(frac=1).to_numpy() # .reset_index(drop=True)
        shuffled_analysis_df['pre_post_delta_category'] = shuffled_analysis_df.sample(frac=1)['pre_post_delta_category'].to_numpy()
        _shuffled_pre_post_delta_partition_df_dict = partition_df_dict(shuffled_analysis_df, partitionColumn='pre_post_delta_category')
        _shuffled_pre_delta_df = _shuffled_pre_post_delta_partition_df_dict['pre-delta']
        _shuffled_post_delta_df = _shuffled_pre_post_delta_partition_df_dict['post-delta']

        _diff_mean = np.nanmean(_shuffled_post_delta_df[stats_variable_name].to_numpy()) - np.nanmean(_shuffled_pre_delta_df[stats_variable_name].to_numpy())
        shuffle_results.append(_diff_mean)

    shuffle_results = np.array(shuffle_results)
    shuffle_results

    ## OUTPUTS: shuffle_results

    ## count the number which exceed the actual mean

    
    np.sum(actual_diff_means > np.abs(shuffle_results))
    # Create the data for two groups
    # group1 = np.random.rand(25)
    # group2 = np.random.rand(20)

    ## INPUTS: stats_variable_name
    print(f'stats_variable_name: {stats_variable_name}')

    group1 = _pre_delta_df[stats_variable_name].to_numpy()
    group2 = _post_delta_df[stats_variable_name].to_numpy()

    # perform mann whitney test 
    stat, p_value = stats.mannwhitneyu(group1, group2) 
    print('Statistics=%.2f, p=%.2f' % (stat, p_value)) 

    # Level of significance 
    
    # conclusion 
    if p_value < stats_level_of_significance_alpha: 
        print('Reject Null Hypothesis (Significant difference between two samples)') 
    else: 
        print('Do not Reject Null Hypothesis (No significant difference between two samples)')

    # Calculate the sample variances
    variance1 = np.var(group1, ddof=1)
    variance2 = np.var(group2, ddof=1)
    
    print('Variance 1:',variance1)
    print('Variance 2:',variance2)

    # Calculate the F-statistic
    f_value = variance1 / variance2
    
    # Calculate the degrees of freedom
    dof1 = len(group1) - 1
    dof2 = len(group2) - 1
    
    # Calculate the p-value
    p_value = stats.f.cdf(f_value, dof1, dof2)
    
    # Print the results
    print('Degree of freedom 1:',dof1)
    print('Degree of freedom 2:',dof2)
    print("F-statistic:", f_value)
    print("p-value:", p_value)

    return shuffle_results, p_value, f_value, (dof1, dof2), (variance1, variance2) 


    # Statistics=351933.00, p=0.00
    # Reject Null Hypothesis (Significant difference between two samples)
    # Variance 1: 0.017720245875104713
    # Variance 2: 0.02501347759017487
    # Degree of freedom 1: 1104
    # Degree of freedom 2: 1282
    # F-statistic: 0.7084279189577826
    # p-value: 1.882791591520268e-09


    # stats_variable_name: short_best_wcorr
    # Statistics=770405.00, p=0.00
    # Reject Null Hypothesis (Significant difference between two samples)
    # Variance 1: 0.13962063046395118
    # Variance 2: 0.15575146845969287
    # Degree of freedom 1: 1108
    # Degree of freedom 2: 1281
    # F-statistic: 0.8964321931904211
    # p-value: 0.030077963036698012


# ==================================================================================================================== #
# 2024-05-23 - Restoring 'is_user_annotated_epoch' and 'is_valid_epoch' columns                                        #
# ==================================================================================================================== #

## INPUTS: an_active_df, all_sessions_all_scores_df, a_time_column_names = 'ripple_start_t'
@function_attributes(short_name=None, tags=['IMPORTANT', 'missing-columns'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-23 18:10', related_items=[])
def recover_user_annotation_and_is_valid_columns(an_active_df, all_sessions_all_scores_df, a_time_column_names:str='ripple_start_t'):
    """ Gets the proper 'is_user_annotated_epoch' and 'is_valid_epoch' columns for the epochs passed in 'an_active_df' evaluated with different time_bin_sizes
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import recover_user_annotation_and_is_valid_columns
        
        ## epoch-based ones:
        all_sessions_ripple_df = recover_user_annotation_and_is_valid_columns(all_sessions_ripple_df, all_sessions_all_scores_df=all_sessions_all_scores_df, a_time_column_names='ripple_start_t')
        all_sessions_ripple_df

        # ## can't do the time_bin ones yet because it doesn't have 'ripple_start_t' to match on:
        # an_active_df = all_sessions_ripple_time_bin_df
        # a_time_column_names = 'delta_aligned_start_t'
        # all_sessions_all_scores_df = all_sessions_all_scores_ripple_df
    """
    from neuropy.utils.misc import numpyify_array
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

    # ## METHOD 0:
    # annotations_man = UserAnnotationsManager()
    # user_annotations = annotations_man.get_user_annotations()

    # # [k.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys()) for k, v in user_annotations.items()]

    # ## recover/build the annotation contexts to find the annotations:
    # recovered_session_contexts = [IdentifyingContext(**dict(zip(IdentifyingContext._get_session_context_keys(), k.split('_', maxsplit=3)))) for k in an_active_df.session_name.unique()]

    # epochs_name = 'ripple'

    # _out_any_good_selected_epoch_times = []

    # for a_ctxt in recovered_session_contexts:    
    #     loaded_selections_context_dict = {a_name:a_ctxt.adding_context_if_missing(display_fn_name='DecodedEpochSlices', epochs=epochs_name, decoder=a_name, user_annotation='selections') for a_name in ('long_LR','long_RL','short_LR','short_RL')}
    #     decoder_user_selected_epoch_times_dict = {a_name:numpyify_array(user_annotations.get(a_selections_ctx, [])) for a_name, a_selections_ctx in loaded_selections_context_dict.items()}
    #     _out_any_good_selected_epoch_times.extend(decoder_user_selected_epoch_times_dict.values())

    # # Find epochs that are present in any of the decoders:
    # concatenated_selected_epoch_times = np.concatenate([v for v in _out_any_good_selected_epoch_times if (np.size(v) > 0)], axis=0)
    # any_good_selected_epoch_times: NDArray = np.unique(concatenated_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
    # print(f'METHOD 0: any_good_selected_epoch_times: {np.shape(any_good_selected_epoch_times)}')

    # `is_user_annotated` ________________________________________________________________________________________________ #
    # did_update_user_annotation_col = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(an_active_df, any_good_selected_epoch_times=any_good_selected_epoch_times, t_column_names=[a_time_column_names,])

    ## Option 2 - the `all_sessions_all_scores_ripple_df` df column-based approach. get only the valid rows from the `all_sessions_all_scores_ripple_df` df
    any_good_selected_epoch_times: NDArray = all_sessions_all_scores_df[all_sessions_all_scores_df['is_user_annotated_epoch']][['start', 'stop']].to_numpy()
    any_good_selected_epoch_times = np.unique(any_good_selected_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
    # print(f'METHOD 1: any_good_selected_epoch_times: {np.shape(any_good_selected_epoch_times)}') # interesting: difference of 1: (436, 2) v. (435, 2) 

    did_update_user_annotation_col = DecoderDecodedEpochsResult.try_add_is_user_annotated_epoch_column(an_active_df, any_good_selected_epoch_times=any_good_selected_epoch_times, t_column_names=[a_time_column_names,])

    # `is_valid_epoch` ___________________________________________________________________________________________________ #
    # get only the valid rows from the `all_sessions_all_scores_ripple_df` df
    any_good_is_valid_epoch_times: NDArray = all_sessions_all_scores_df[all_sessions_all_scores_df['is_valid_epoch']][['start', 'stop']].to_numpy()
    any_good_is_valid_epoch_times = np.unique(any_good_is_valid_epoch_times, axis=0) # drops duplicate rows (present in multiple decoders), and sorts them ascending
    did_update_is_valid = DecoderDecodedEpochsResult.try_add_is_valid_epoch_column(an_active_df, any_good_selected_epoch_times=any_good_is_valid_epoch_times, t_column_names=[a_time_column_names,])

    ## OUTPUTS: an_active_df
    return an_active_df



# ==================================================================================================================== #
# 2024-04-25 - Factoring Out Helper GUI Code                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger

@function_attributes(short_name=None, tags=['raster', 'attached'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-29 17:14', related_items=[])
def _build_attached_raster_viewer(paginated_multi_decoder_decoded_epochs_window, track_templates, active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df):
    """ creates a new RankOrderRastersDebugger for use by `paginated_multi_decoder_decoded_epochs_window`.
    
    You can middle-click on any epoch heatmap in `paginated_multi_decoder_decoded_epochs_window` to display that corresponding epoch in the RankOrderRastersDebugger
    
    

    paginated_multi_decoder_decoded_epochs_window

    Captures: paginated_multi_decoder_decoded_epochs_window
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _build_attached_raster_viewer
        _out_ripple_rasters = _build_attached_raster_viewer(paginated_multi_decoder_decoded_epochs_window, track_templates=track_templates, active_spikes_df=active_spikes_df, filtered_ripple_simple_pf_pearson_merged_df=filtered_ripple_simple_pf_pearson_merged_df)
    
    
    """
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
    # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
    global_spikes_df = deepcopy(active_spikes_df)
    _out_ripple_rasters: RankOrderRastersDebugger = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, deepcopy(filtered_ripple_simple_pf_pearson_merged_df),
                                                                                                    track_templates, None,
                                                                                                        None, None,
                                                                                                        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right']))),
                                                                                                        )
    _out_ripple_rasters.set_top_info_bar_visibility(False)
    _out_ripple_rasters.set_bottom_controls_visibility(False)

    ## Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior.
    def update_attached_raster_viewer_epoch_callback(self, event, clicked_ax, clicked_data_index, clicked_epoch_is_selected, clicked_epoch_start_stop_time):
        """ Enable programmatically updating the rasters viewer to the clicked epoch index when middle clicking on a posterior. 
        called when the user middle-clicks an epoch 
        
        captures: _out_ripple_rasters
        """
        print(f'update_attached_raster_viewer_epoch_callback(clicked_data_index: {clicked_data_index}, clicked_epoch_is_selected: {clicked_epoch_is_selected}, clicked_epoch_start_stop_time: {clicked_epoch_start_stop_time})')
        if clicked_epoch_start_stop_time is not None:
            if len(clicked_epoch_start_stop_time) == 2:
                start_t, end_t = clicked_epoch_start_stop_time
                print(f'start_t: {start_t}')
                _out_ripple_rasters.programmatically_update_epoch_IDX_from_epoch_start_time(start_t)

    ## Attach the update to the pagination controllers:
    for a_name, a_pagination_controller in paginated_multi_decoder_decoded_epochs_window.pagination_controllers.items():
        # a_pagination_controller.params.debug_print = True
        if not a_pagination_controller.params.has_attr('on_middle_click_item_callbacks'):
            a_pagination_controller.params['on_middle_click_item_callbacks'] = {}
        a_pagination_controller.params.on_middle_click_item_callbacks['update_attached_raster_viewer_epoch_callback'] = update_attached_raster_viewer_epoch_callback

        
    return _out_ripple_rasters












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
        restricted_all_directional_decoder_pf1D_dict = deepcopy(alt_directional_merged_decoders_result.all_directional_decoder_dict) # copy the dictionary
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
def _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], ripple_all_epoch_bins_marginals_df: pd.DataFrame, ripple_decoding_time_bin_size: float):
    """ the main replay epochs filtering function.
    
    Only includes user selected (annotated) ripples


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
    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates)

    ## filter the epochs by something and only show those:
    # INPUTS: filtered_epochs_df
    # filtered_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(active_epochs_df[['start', 'stop']].to_numpy())
    ## Update the `decoder_ripple_filter_epochs_decoder_result_dict` with the included epochs:
    filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) for a_name, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()} # working filtered
    # print(f"any_good_selected_epoch_times.shape: {any_good_selected_epoch_times.shape}") # (142, 2)
    ## Constrain again now by the user selections
    filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(any_good_selected_epoch_times) for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
    # filtered_decoder_filter_epochs_decoder_result_dict

    # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores`
    filtered_decoder_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)

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
    df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, curr_session_t_delta=t_delta, time_col='ripple_start_t')
    # df = _add_maze_id_to_epochs(df, t_delta)
    df["time_bin_size"] = ripple_decoding_time_bin_size
    df['is_user_annotated_epoch'] = True # if it's filtered here, it's true


    return filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict, filtered_ripple_all_epoch_bins_marginals_df

@function_attributes(short_name=None, tags=['filter', 'epoch_selection', 'export', 'h5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-08 13:28', related_items=[])
def export_numpy_testing_filtered_epochs(curr_active_pipeline, global_epoch_name, track_templates, required_min_percentage_of_active_cells: float = 0.333333, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1):
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


    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates)
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

def pho_jointplot(*args, **kwargs):
    """ wraps sns.jointplot to allow adding titles/axis labels/etc.
    
    Example:
        import seaborn as sns
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import pho_jointplot
        sns.set_theme(style="ticks")
        common_kwargs = dict(ylim=(0,1), hue='time_bin_size') # , marginal_kws=dict(bins=25, fill=True)
        # sns.jointplot(data=a_laps_all_epoch_bins_marginals_df, x='lap_start_t', y='P_Long', kind="scatter", color="#4CB391")
        pho_jointplot(data=several_time_bin_sizes_laps_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Laps: per epoch') #color="#4CB391")
        pho_jointplot(data=several_time_bin_sizes_ripple_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Ripple: per epoch')
        pho_jointplot(data=several_time_bin_sizes_time_bin_ripple_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Ripple: per time bin')
        pho_jointplot(data=several_time_bin_sizes_time_bin_laps_df, x='delta_aligned_start_t', y='P_Long', kind="scatter", **common_kwargs, title='Laps: per time bin')
    
    """
    import seaborn as sns
    title = kwargs.pop('title', None)
    _out = sns.jointplot(*args, **kwargs)
    if title is not None:
        plt.suptitle(title)
    return _out


def plot_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str) -> None:
    """ plots a stacked histogram of the many time-bin sizes 

    Usage:    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_histograms

        # You can use it like this:
        _out0: "MatplotlibRenderPlots" = plot_histograms(data_type='Laps', session_spec='All Sessions', data_results_df=all_sessions_laps_time_bin_df, time_bin_duration_str="75 ms")
        _out1: "MatplotlibRenderPlots" = plot_histograms(data_type='Ripples', session_spec='All Sessions', data_results_df=all_sessions_ripple_time_bin_df, time_bin_duration_str="75 ms")


    """
    # get the pre-delta epochs
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]

    descriptor_str: str = '|'.join([data_type, session_spec, time_bin_duration_str])
    
    # plot pre-delta histogram
    time_bin_sizes = pre_delta_df['time_bin_size'].unique()
    
    figure_identifier: str = f"{descriptor_str}_preDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = pre_delta_df[pre_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=f'{float(time_bin_size):.2f}') 
        

    
    plt.title(f'{descriptor_str} - pre-$\Delta$ time bins')
    plt.legend()
    plt.show()

    # plot post-delta histogram
    time_bin_sizes = post_delta_df['time_bin_size'].unique()
    figure_identifier: str = f"{descriptor_str}_postDelta"
    plt.figure(num=figure_identifier, clear=True, figsize=(6, 2))
    for time_bin_size in time_bin_sizes:
        df_tbs = post_delta_df[post_delta_df['time_bin_size']==time_bin_size]
        df_tbs['P_Long'].hist(alpha=0.5, label=f'{float(time_bin_size):.2f}') 
    
    plt.title(f'{descriptor_str} - post-$\Delta$ time bins')
    plt.legend()
    plt.show()


def pho_plothelper(data, **kwargs):
    """ 2024-02-06 - Provides an interface like plotly's classes provide to extract keys fom DataFrame columns or dicts and generate kwargs to pass to a plotting function.
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import pho_plothelper
            extracted_value_kwargs = pho_plothelper(data=an_aclu_conv_overlap_output['valid_subset'], x='x', y='normalized_convolved_result')
            extracted_value_kwargs

    """
    # data is a pd.DataFrame or Dict-like
    extracted_value_kwargs = {}
    for k,v in kwargs.items():
        extracted_value_kwargs[k] = data[v]
    # end up with `extracted_value_kwargs` containing the real values to plot.
    return extracted_value_kwargs






# ==================================================================================================================== #
# 2024-01-29 - Ideal Pho Plotting Interface - UNFINISHED                                                               #
# ==================================================================================================================== #
def map_dataframe_to_plot(df: pd.DataFrame, **kwargs):
    """ 2024-01-29 - My ideal desired function that allows the user to map any column in a dataframe to a plot command, including rows/columns.
    Not yet finished.
     maps any column in the dataframe to a property in a plot. 
     
     Usage:
         fully_resolved_kwargs = map_dataframe_to_plot(df=all_sessions_laps_df, x='delta_aligned_start_t', y='P_Long', color='session_name', size='time_bin_size') # , title=f"Laps - {laps_title_string_suffix}"
        fully_resolved_kwargs

    """
    all_column_names: List[str] = list(df.columns)
    all_kwargs_keys: List[str] = list(kwargs.keys())
    all_kwargs_values: List[Union[str, Any]] = list(kwargs.values()) # expected to be either a column name to map or a literal.
    num_rows: int = len(df)
    
    should_fully_extract_dataframe_values: bool = True # if True, extracts the values from the dataframe as an array
    fully_resolved_kwargs = {}
    
    # for a_key in all_kwargs_keys:
    # 	assert a_key in df.columns, f'key "{a_key}" specified in kwargs is not a column in df! \n\tdf.columns: {list(df.columns)}'
    known_keys = ['x', 'y', 'color', 'size', 'row', 'column', 'page', 'xlabel', 'ylabel', 'title']
    for a_key, a_value in kwargs.items():
        if a_key not in known_keys:
            print(f'WARN: key "{a_key}" is not in the known keys list: known_keys: {known_keys}')
        if not isinstance(a_value, str):
            # not a string
            raise ValueError(f"value {a_value} is not a string and its length is not equal to the length of the dataframe.")
            #TODO 2024-01-29 23:45: - [ ] Allow passing literal list-like values with the correct length to be passed directly
            assert (len(a_value) == num_rows), f"(len(a_value) == num_rows) but (len(a_value): {len(a_value)} == num_rows: {num_rows})"
            fully_resolved_kwargs[a_key] = a_value # Set the passed value directly
            
        else:
            # it is a string, assume that it's a column in the dataframe
            assert a_value in all_column_names, f'key:value pair <"{a_key}":"{a_value}"> specified in kwargs has a value that is not a valid column in df! \n\tspecified_value: {a_value}\n\tdf.columns: {list(df.columns)}'
            if should_fully_extract_dataframe_values:
                fully_resolved_kwargs[a_key] = df[a_value].to_numpy()
            else:
                # leave as the validated column name
                fully_resolved_kwargs[a_key] = a_value
                
    return fully_resolved_kwargs


def _embed_in_subplots(scatter_fig):
    import plotly.subplots as sp
    import plotly.graph_objs as go
    # creating subplots
    fig = sp.make_subplots(rows=1, cols=3, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01)

    # adding first histogram
    # Calculate the histogram data
    hist1, bins1 = np.histogram(X[:split], bins='auto')

    # Adding the first histogram as a bar graph and making x negative
    fig.add_trace(
        go.Bar(
            x=-bins1[:-1],
            y=hist1,
            marker_color='#EB89B5',
            name='first half',
            orientation='h',
        ),
        row=1, col=1
    )


    # adding scatter plot
    fig.add_trace(scatter_fig, row=1, col=2)
    # fig.add_trace(
    #     go.Scatter(
    #         x=X,
    #         y=Y,
    #         mode='markers',
    #         marker_color='rgba(152, 0, 0, .8)',
    #     ),
    #     row=1, col=2
    # )

    # adding the second histogram

    # Calculate the histogram data for second half
    hist2, bins2 = np.histogram(X[split:], bins='auto')

    # Adding the second histogram
    fig.add_trace(
        go.Bar(
            x=bins2[:-1],
            y=hist2,
            marker_color='#330C73',
            name='second half',
            orientation='h',
        ),
        row=1, col=3
    )
    return fig


# ==================================================================================================================== #
# 2024-01-23 - Writes the posteriors out to file                                                                       #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['figure', 'save', 'IMPORTANT', 'marginal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-23 00:00', related_items=[])
def save_posterior(raw_posterior_laps_marginals, laps_directional_marginals, laps_track_identity_marginals, collapsed_per_lap_epoch_marginal_dir_point, collapsed_per_lap_epoch_marginal_track_identity_point, parent_array_as_image_output_folder: Path, epoch_id_identifier_str: str = 'lap', epoch_id: int = 9):
    """ 2024-01-23 - Writes the posteriors out to file 
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import save_posterior

        collapsed_per_lap_epoch_marginal_track_identity_point = laps_marginals_df[['P_Long', 'P_Short']].to_numpy().astype(float)
        collapsed_per_lap_epoch_marginal_dir_point = laps_marginals_df[['P_LR', 'P_RL']].to_numpy().astype(float)

        for epoch_id in np.arange(laps_filter_epochs_decoder_result.num_filter_epochs):
            raw_tuple, marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple = save_posterior(raw_posterior_laps_marginals, laps_directional_marginals, laps_track_identity_marginals, collapsed_per_lap_epoch_marginal_dir_point, collapsed_per_lap_epoch_marginal_track_identity_point,
                                                                                        parent_array_as_image_output_folder=parent_array_as_image_output_folder, epoch_id_identifier_str='lap', epoch_id=epoch_id)

    """
    from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image

    assert parent_array_as_image_output_folder.exists()
    
    epoch_id_str = f"{epoch_id_identifier_str}[{epoch_id}]"
    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_raw_marginal.png').resolve()
    img_data = raw_posterior_laps_marginals[epoch_id]['p_x_given_n'].astype(float)  # .shape: (4, n_curr_epoch_time_bins) - (63, 4, 120)
    raw_tuple = save_array_as_image(img_data, desired_height=100, desired_width=None, skip_img_normalization=True, out_path=_img_path)
    # image_raw, path_raw = raw_tuple

    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_dir.png').resolve()
    img_data = laps_directional_marginals[epoch_id]['p_x_given_n'].astype(float)
    marginal_dir_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)
    # image_marginal_dir, path_marginal_dir = marginal_dir_tuple

    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_track_identity.png').resolve()
    img_data = laps_track_identity_marginals[epoch_id]['p_x_given_n'].astype(float)
    marginal_track_identity_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)
    # image_marginal_track_identity, path_marginal_track_identity = marginal_track_identity_tuple


    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_track_identity_point.png').resolve()
    img_data = np.atleast_2d(collapsed_per_lap_epoch_marginal_track_identity_point[epoch_id,:]).T
    marginal_dir_point_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)

    _img_path = parent_array_as_image_output_folder.joinpath(f'{epoch_id_str}_marginal_dir_point.png').resolve()
    img_data = np.atleast_2d(collapsed_per_lap_epoch_marginal_dir_point[epoch_id,:]).T
    marginal_track_identity_point_tuple = save_array_as_image(img_data, desired_height=50, desired_width=None, skip_img_normalization=True, out_path=_img_path)


    return raw_tuple, marginal_dir_tuple, marginal_track_identity_tuple, marginal_dir_point_tuple, marginal_track_identity_point_tuple
    
