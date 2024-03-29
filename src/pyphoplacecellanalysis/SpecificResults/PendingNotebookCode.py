# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Dict, Tuple, Any, Union
import attrs
import matplotlib as mpl
import napari
from neuropy.analyses.placefields import PfND
import numpy as np
import pandas as pd
from attrs import asdict, define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes

from functools import wraps, partial
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult


# ---------------------------------------------------------------------------- #
#             2024-03-29 - Rigorous Decoder Performance assessment             #
# ---------------------------------------------------------------------------- #
# Quantify cell contributions to decoders
# Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder

import portion as P # Required for interval search: portion~=2.3.0
from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df, _convert_start_end_tuples_list_to_PortionInterval

## Get custom decoder that is only trained on a portion of the laps
## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
# long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]

## Restrict the data post-hoc?

## Time-dependent decoder?

## Split the lap epochs into training and test periods.
##### Ideally we could test the lap decoding error by sampling randomly from the time bins and omitting 1/6 of time bins from the placefield building (effectively the training data). These missing bins will be used as the "test data" and the decoding error will be computed by decoding them and subtracting the actual measured position during these bins.


@function_attributes(short_name=None, tags=['testing', 'split', 'laps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-29 15:37', related_items=[])
def split_laps_training_and_test(laps_df: pd.DataFrame, training_data_portion: float=5.0/6.0, debug_print: bool = False):
    """
    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import split_laps_training_and_test

        ### Get the laps to train on
        training_data_portion: float = 5.0/6.0
        test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration

        print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

        laps_df: pd.DataFrame = deepcopy(global_any_laps_epochs_obj.to_dataframe())

        laps_training_df, laps_test_df = split_laps_training_and_test(laps_df=laps_df, training_data_portion=training_data_portion, debug_print=False)

        laps_df
        laps_training_df
        laps_test_df 

    """
    from neuropy.core.epoch import Epoch, ensure_dataframe

    def _subfn_sample_random_period_from_lap(lap_start, lap_stop, training_data_portion: float, *additional_lap_columns):
        """ 
        randomly sample a portion of each lap. Draw a random period of duration (duration[i] * training_data_portion) from the lap.

        """
        total_lap_duration = lap_stop - lap_start
        training_duration = total_lap_duration * training_data_portion

        ## new method:
        # I'd like to randomly choose a test_start_t period from any time during the interval.
        # TRAINING data split mode:
        training_start_t = np.random.uniform(lap_start, lap_stop)
        training_end_t = training_start_t + training_duration
        # Wrap around if training_end_t is beyond the period
        training_wrap_duration = np.abs(lap_stop - training_end_t) 

        if training_wrap_duration > 0.0:
            training_end_t = lap_stop # training spans to the end of the lap
            ## new period is crated for training at start of lap
            second_training_start_t = lap_start
            second_training_stop_t = lap_start + training_wrap_duration
            return [(training_start_t, training_end_t, *additional_lap_columns), (second_training_start_t, second_training_stop_t, *additional_lap_columns)]

        else:
            return [(training_start_t, training_end_t, *additional_lap_columns)]
        

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

    additional_lap_identity_column_names = ['label', 'lap_id', 'lap_dir']

    # Randomly sample a portion of each lap. Draw a random period of duration (duration[i] * training_data_portion) from the lap.
    train_rows = []
    test_rows = []

    for lap_id, group in laps_df.groupby('lap_id'):
        lap_start = group['start'].min()
        lap_stop = group['stop'].max()
        if debug_print:
            print(f'lap_id: {lap_id} - group: {group}')
        curr_additional_lap_column_values = [group[a_col].to_numpy()[0] for a_col in additional_lap_identity_column_names]
        # lap_stop = group['lap_id']
        # lap_stop = group['stop']
        if debug_print:
            print(f'\tcurr_additional_lap_column_values: {curr_additional_lap_column_values}')
        # Get the random training start and stop times for the lap.
        # training_start, training_stop = sample_random_period_from_lap(lap_start, lap_stop, training_data_portion)
        # Define your period as an interval
        curr_lap_period = P.closed(lap_start, lap_stop)

        epoch_start_stop_tuple_list = _subfn_sample_random_period_from_lap(lap_start, lap_stop, training_data_portion, *curr_additional_lap_column_values)

        # _intermediate_portions_interval: P.Interval = _convert_start_end_tuples_list_to_PortionInterval(epoch_start_stop_tuple_list)
        # filtered_epochs_df = convert_PortionInterval_to_epochs_df(_intermediate_portions_interval)
        combined_intervals = P.empty()
        for an_epoch_start_stop_tuple in epoch_start_stop_tuple_list:
            combined_intervals = combined_intervals.union(P.closed(an_epoch_start_stop_tuple[0], an_epoch_start_stop_tuple[1]))
            train_rows.append(an_epoch_start_stop_tuple)
        
        # Calculate the difference between the period and the combined interval
        complement_intervals = curr_lap_period.difference(combined_intervals)
        _temp_test_epochs_df = convert_PortionInterval_to_epochs_df(complement_intervals)
        _temp_test_epochs_df[additional_lap_identity_column_names] = curr_additional_lap_column_values ## add in the additional columns
        test_rows.append(_temp_test_epochs_df)

    ## INPUTS: laps_df, laps_df

    # train_rows
    # Convert to DataFrame and reset indices
    laps_training_df = pd.DataFrame(train_rows, columns=['start', 'stop', *additional_lap_identity_column_names])
    laps_training_df['duration'] = laps_training_df['stop'] - laps_training_df['start']

    # ## Use Porition to find the test interval location:
    # _laps_Portion_obj: P.Interval = laps_df.epochs.to_PortionInterval()
    # _laps_training_Portion_obj: P.Interval = laps_training_df.epochs.to_PortionInterval()
    # _laps_test_Portion_obj: P.Interval = _laps_Portion_obj.difference(_laps_training_Portion_obj)
    # laps_test_df: pd.DataFrame = Epoch.from_PortionInterval(_laps_test_Portion_obj).to_dataframe() 

    # laps_test_df: Epoch = Epoch(Epoch.from_PortionInterval(laps_training_Portion_obj.complement()).time_slice(t_start=laps_df.epochs.t_start, t_stop=laps_df.epochs.t_stop).to_dataframe()[:-1]).to_dataframe() #[:-1] # any period except the replay ones, drop the infinite last entry

    # Convert to DataFrame and reset indices
    # laps_training_df = pd.DataFrame(train_rows)
    # laps_test_df = pd.DataFrame(test_rows)
    laps_test_df = pd.concat(test_rows)
    laps_training_df.reset_index(drop=True, inplace=True)
    laps_test_df.reset_index(drop=True, inplace=True)

    # assert np.shape(laps_test_df)[0] == np.shape(laps_df)[0], f"np.shape(laps_test_df)[0]: {np.shape(laps_test_df)[0]} != np.shape(laps_df)[0]: {np.shape(laps_df)[0]}"

    ## OUTPUTS: laps_training_df, laps_test_df
    # laps_df
    # laps_training_df
    # laps_test_df

    return laps_training_df, laps_test_df




## INPUTS: laps_df, laps_training_df, laps_test_df
@function_attributes(short_name=None, tags=['matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-29 15:46', related_items=[])
def debug_draw_laps_train_test_split_epochs(laps_df, laps_training_df, laps_test_df, fignum=1, fig=None, ax=None, active_context=None):
    """ 
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import debug_draw_laps_train_test_split_epochs

        fig, ax = debug_draw_laps_train_test_split_epochs(laps_df, laps_training_df, laps_test_df, fignum=0)
        fig.show()
    """
    from neuropy.core.epoch import Epoch, ensure_dataframe
    from matplotlib.gridspec import GridSpec
    from neuropy.utils.matplotlib_helpers import build_or_reuse_figure, perform_update_title_subtitle
    from neuropy.utils.matplotlib_helpers import draw_epoch_regions


    def _prepare_epochs_df(laps_test_df: pd.DataFrame) -> Epoch:
        active_filter_epochs = deepcopy(laps_test_df)

        if not 'stop' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
            
        if not 'label' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

        active_filter_epoch_obj = Epoch(active_filter_epochs)
        return active_filter_epoch_obj


    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

    laps_Epoch_obj = _prepare_epochs_df(laps_df)
    laps_training_df_Epoch_obj = _prepare_epochs_df(laps_training_df)
    laps_test_df_Epoch_obj = _prepare_epochs_df(laps_test_df)

    if fignum is None:
        if f := plt.get_fignums():
            fignum = f[-1] + 1
        else:
            fignum = 1

    ## Figure Setup:
    if ax is None:
        fig = build_or_reuse_figure(fignum=fignum, fig=fig, fig_idx=0, figsize=(12, 4.2), dpi=None, clear=True, tight_layout=False)
        gs = GridSpec(1, 1, figure=fig)
        ax = plt.subplot(gs[0])

    else:
        # otherwise get the figure from the passed axis
        fig = ax.get_figure()

    # epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=('red','cyan'), alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
    laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(laps_Epoch_obj, ax, facecolor='black', edgecolors=None, labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False, label='laps')
    test_epochs_collection, test_epoch_labels = draw_epoch_regions(laps_test_df_Epoch_obj, ax, facecolor='orange', edgecolors='orange', labels_kwargs=None, defer_render=False, debug_print=True, label='test')
    train_epochs_collection, train_epoch_labels = draw_epoch_regions(laps_training_df_Epoch_obj, ax, facecolor='green', edgecolors='green', labels_kwargs=None, defer_render=False, debug_print=True, label='train')
    ax.autoscale()
    fig.legend()
    # plt.title('Lap epochs divided into separate training and test intervals')
    plt.xlabel('time (sec)')
    plt.ylabel('Lap Epochs')

    # Set window title and plot title
    perform_update_title_subtitle(fig=fig, ax=ax, title_string=f'Lap epochs divided into separate training and test intervals', subtitle_string=None, active_context=active_context, use_flexitext_titles=True)
    0
    return fig, ax


# ==================================================================================================================== #
# 2024-03-09 - Filtering                                                                                               #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult, filter_and_update_epochs_and_spikes
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import HeuristicReplayScoring
from neuropy.core.epoch import find_data_indicies_from_epoch_times


def _apply_filtering_to_marginals_result_df(active_result_df, filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict):
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
def _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict, ripple_all_epoch_bins_marginals_df, ripple_decoding_time_bin_size: float):
    """ the main replay epochs filtering function.
    
    filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict, filtered_ripple_all_epoch_bins_marginals_df = _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict, ripple_all_epoch_bins_marginals_df)

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

@function_attributes(short_name=None, tags=['filter', 'epoch_selection'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-08 13:28', related_items=[])
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
# Position Derivatives Plotting Helpers                                                                                #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['decode'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 14:30', related_items=[])
def _compute_pos_derivs(time_window_centers, position, decoding_time_bin_size, debug_print=False):
    """try recomputing velocties/accelerations
    
    decoding_time_bin_size = a_result.decoding_time_bin_size
    """ 
    position = deepcopy(position)
    a_first_order_diff = np.diff(position, n=1, prepend=[position[0]]) 
    velocity = a_first_order_diff / float(decoding_time_bin_size) # velocity with real world units of cm/sec
    acceleration = np.diff(velocity, n=1, prepend=[velocity[0]])

    position_derivatives_df: pd.DataFrame = pd.DataFrame({'t': time_window_centers, 'x': position, 'vel_x': velocity, 'accel_x': acceleration})
    if debug_print:
        print(f'time_window_centers: {time_window_centers}')
        print(f'position: {position}')
        print(f'velocity: {velocity}')
        print(f'acceleration: {acceleration}')

    position_derivative_column_names = ['x', 'vel_x', 'accel_x']
    position_derivative_means = position_derivatives_df.mean(axis='index')[position_derivative_column_names].to_numpy()
    position_derivative_medians = position_derivatives_df.median(axis='index')[position_derivative_column_names].to_numpy()
    # position_derivative_medians = position_derivatives_df(axis='index')[position_derivative_column_names].to_numpy()
    if debug_print:
        print(f'\tposition_derivative_means: {position_derivative_means}')
        print(f'\tposition_derivative_medians: {position_derivative_medians}')
    return position_derivatives_df


@function_attributes(short_name=None, tags=['helper','matplotlib','figure', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=[], used_by=['debug_plot_position_and_derivatives_figure'], creation_date='2024-03-07 18:23', related_items=['debug_plot_position_and_derivatives_figure'])
def debug_plot_helper_add_position_and_derivatives(time_window_centers, position, velocity, acceleration, debug_plot_axs=None, debug_plot_name=None, common_plot_kwargs=None):
        """ HELPER to `debug_plot_position_and_derivatives_figure`: Renders a single series (measured, a_specific_decoder, ...)'s values for all 3 subplots: [position, velocity, acceleration]

        Usage:
            fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(time_window_centers, position, velocity, acceleration, debug_plot_axs=None, debug_plot_name=None, common_plot_kwargs=None)

            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import debug_plot_helper_add_position_and_derivatives

            
            enable_debug_plot = True
            if enable_debug_plot:
                fig, debug_plot_axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            else:
                debug_plot_axs = None
                debug_plot_name = None


            ## Plot measured
            fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(new_measured_pos_df['t'].to_numpy(), new_measured_pos_df['x'].to_numpy(), new_measured_pos_df['vel_x'].to_numpy(), new_measured_pos_df['accel_x'].to_numpy(),
                                                                                    debug_plot_axs=axs, debug_plot_name='measured', common_plot_kwargs=dict(color='k', markersize='2', marker='.', linestyle='None', alpha=0.35))

            ## Plot decoded
            for a_name, a_df in all_epochs_position_derivatives_df_dict.items():
                fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(a_df['t'].to_numpy(), a_df['x'].to_numpy(), a_df['vel_x'].to_numpy(), a_df['accel_x'].to_numpy(),
                                                                                    debug_plot_axs=debug_plot_axs, debug_plot_name=a_name, common_plot_kwargs=dict(marker='o', markersize=3, linestyle='None', alpha=0.6))

                                                                                
        """
        # Setup the figure and subplots
        if debug_plot_axs is None:
            fig, debug_plot_axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        else:
            fig = debug_plot_axs[0].get_figure()

        if debug_plot_name is None:
            debug_plot_name = ''

        common_plot_kwargs = common_plot_kwargs or {}
        common_plot_kwargs = common_plot_kwargs or dict(marker='o', linestyle='None', alpha=0.6)

        # Plot the position data on the first subplot
        debug_plot_axs[0].plot(time_window_centers, position, label=f'{debug_plot_name}_Position', **common_plot_kwargs) # , color='blue'
        debug_plot_axs[0].set_ylabel('Position (m)')
        debug_plot_axs[0].legend()

        # Plot the velocity data on the second subplot
        debug_plot_axs[1].plot(time_window_centers, velocity, label=f'{debug_plot_name}_Velocity', **common_plot_kwargs) # , color='orange'
        debug_plot_axs[1].set_ylabel('Velocity (m/s)')
        debug_plot_axs[1].legend()

        # Plot the acceleration data on the third subplot
        debug_plot_axs[2].plot(time_window_centers, acceleration, label=f'{debug_plot_name}_Acceleration', **common_plot_kwargs) # , color='green'
        debug_plot_axs[2].set_ylabel('Acceleration (m/s²)')
        debug_plot_axs[2].set_xlabel('Time (s)')
        debug_plot_axs[2].legend()

        # # Set a shared title for the subplots
        plt.suptitle('Position, Velocity and Acceleration vs. Time')

        # # Adjust the layout so the subplots fit nicely
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle at the top

        # # Show the subplots
        # plt.show()

        return fig, debug_plot_axs


@function_attributes(short_name=None, tags=['matplotlib','figure', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=['debug_plot_helper_add_position_and_derivatives'], used_by=[], creation_date='2024-03-07 18:35', related_items=['debug_plot_position_derivatives_stack'])
def debug_plot_position_and_derivatives_figure(new_measured_pos_df, all_epochs_position_derivatives_df_dict, debug_plot_axs=None, debug_figure_title=None, enable_debug_plot = True): # , common_plot_kwargs=None
    """ Renders a single matplotlib figure with a stack of 3 subplots: [position, velocity, acceleration] with both measured and decoded values. Plots measured vs. decoded positions.

    VARIANT: A Matplotlib variant of `debug_plot_position_derivatives_stack`


    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import debug_plot_position_and_derivatives_figure


        ## INPUTS: new_measured_pos_df, all_epochs_position_derivatives_df_dict
        fig, debug_plot_axs = debug_plot_position_and_derivatives_figure(new_measured_pos_df, all_epochs_position_derivatives_df_dict, debug_plot_axs=None, debug_figure_title=None, enable_debug_plot = True, common_plot_kwargs=None)


    """
    
    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    if enable_debug_plot:
        fig, debug_plot_axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    else:
        debug_plot_axs = None
        fig = None

    ## Plot measured
    fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(new_measured_pos_df['t'].to_numpy(), new_measured_pos_df['x'].to_numpy(), new_measured_pos_df['vel_x'].to_numpy(), new_measured_pos_df['accel_x'].to_numpy(),
                                                                            debug_plot_axs=debug_plot_axs, debug_plot_name='measured', common_plot_kwargs=dict(color='k', markersize='2', marker='.', linestyle='None', alpha=0.35))

    ## Plot decoded
    for a_name, a_df in all_epochs_position_derivatives_df_dict.items():
        fig, debug_plot_axs = debug_plot_helper_add_position_and_derivatives(a_df['t'].to_numpy(), a_df['x'].to_numpy(), a_df['vel_x'].to_numpy(), a_df['accel_x'].to_numpy(),
                                                                            debug_plot_axs=debug_plot_axs, debug_plot_name=a_name, common_plot_kwargs=dict(marker='o', markersize=3, linestyle='None', alpha=0.6))

    if debug_figure_title is not None:
        plt.suptitle(debug_figure_title)

    return fig, debug_plot_axs



# ==================================================================================================================== #
# 2024-03-06 - measured vs. decoded position distribution comparison                                                   #
# ==================================================================================================================== #
## basically: does the distribution of positions/velocities/accelerations differ between the correct vs. incorrect decoder? Is it reliable enough to determine whether the decoder is correct or not?
## for example using the wrong decoder might lead to wildly-off velocities.
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
template: str = 'plotly_dark' # set plotl template
pio.templates.default = template

@function_attributes(short_name=None, tags=['plotly', 'figure', 'position', 'derivitives'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 18:20', related_items=['debug_plot_position_and_derivatives_figure'])
def debug_plot_position_derivatives_stack(new_measured_pos_df, all_epochs_position_derivatives_df_dict, show_scatter = False, subplot_height = 300, figure_width=1900):
    """ Renders a stack of 3 subplots: [position, velocity, acceleration]

    VARIANT: A Plotly variant of `debug_plot_position_and_derivatives_figure`

    Usage:
        show_scatter = False
        subplot_height = 300  # Height in pixels for each subplot; adjust as necessary
        figure_width = 1900


        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import debug_plot_position_derivatives_stack

        # fig = debug_plot_position_derivatives_stack(new_measured_pos_df, all_epochs_position_derivatives_df_dict)
        fig = debug_plot_position_derivatives_stack(new_measured_pos_df, all_epochs_position_derivatives_df_dict, show_scatter=True)
        fig

    """
    included_legend_entries_list = []

    def debug_plot_position_and_derivatives_figure_plotly(time_window_centers, position, velocity, acceleration, fig, series_idx=0, row_offset=0, debug_plot_name=None, color_palette=None, scatter_plot_kwargs=None, hist_kwargs=None, common_plot_kwargs=None, show_scatter=True):
        """ Plots a single series of positions (like those measured, decoded_long_LR, decoded_long_RL, ...) on the figure.

        Captures: included_legend_entries_list
        """
        # global included_legend_entries_list
        if debug_plot_name is None:
            debug_plot_name = ''
        legend_group_name = f'{debug_plot_name}'  # Define a legend group name
        row_names = [f'{debug_plot_name}_Position', f'{debug_plot_name}_Velocity', f'{debug_plot_name}_Acceleration']

        scatter_plot_kwargs = scatter_plot_kwargs or {}

        common_plot_kwargs = common_plot_kwargs or {}
        common_plot_kwargs = dict(hoverinfo='skip', legendgroup=legend_group_name) | common_plot_kwargs

        if debug_plot_name in included_legend_entries_list:
            ## only generate legend entries for the first series
            common_plot_kwargs['showlegend'] = False


        color = color_palette[series_idx % len(color_palette)] if color_palette else None
        common_plot_kwargs['marker_color'] = color

        hist_kwargs = hist_kwargs or {}
        hist_kwargs = hist_kwargs | dict(opacity=0.5, nbinsx=25, histfunc='count') # , range_y=[0.0, 1.0]

        # is_first_series: bool = (series_idx == 0)
        # if not is_first_series:
        #     ## only generate legend entries for the first series
        #     common_plot_kwargs['showlegend'] = False

        # scatter_fn = go.Scatter
        scatter_fn = go.Scattergl

        with fig.batch_update():
            ## Add the 3 plots (pos, velocity, accel) as the 3 rows
            for i, row, data in zip(np.arange(3), [row_offset + (i+1) for i in np.arange(3)], [position, velocity, acceleration]):
                col: int = 1
                # is_first_row: bool = (i == 0)
                # if not is_first_row:
                #     ## only generate legend entries for the first series AND the first row of that series
                #     common_plot_kwargs['showlegend'] = False
                common_plot_kwargs['showlegend'] = False

                if show_scatter:
                    fig.add_trace(
                        scatter_fn(x=time_window_centers, y=data, name=legend_group_name, **scatter_plot_kwargs, **common_plot_kwargs),
                        row=row, col=col
                    )
                    col += 1        

                # Add histograms to y-axis of existing scatter trace
                common_plot_kwargs['showlegend'] = (debug_plot_name not in included_legend_entries_list) # never show for the histogram
                # , ybins=dict(start=10, end=15, size=1)

                if show_scatter:
                    hist_kwargs.update(dict(y=data)) # plot vertically
                else:
                    hist_kwargs.update(dict(x=data)) # plot horizontally

                fig.add_histogram(name=legend_group_name, **common_plot_kwargs, **hist_kwargs, #marker_color='rgba(0, 0, 255, 0.5)',
                    row=row, col=col, 
                )
                # Set barmode to 'overlay' for overlaying histograms
                fig.update_layout(barmode='overlay')
                included_legend_entries_list.append(debug_plot_name) ## add to list of entries in legend so it isn't included again


        return fig


    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    ## INPUTS: new_measured_pos_df, all_epochs_position_derivatives_df_dict

    total_subplots = 4 # (1 + len(all_epochs_position_derivatives_df_dict))  # for measured and all decoded
    
    make_subplots_kwargs = dict(horizontal_spacing=0.01)
    if show_scatter:
        make_subplots_kwargs.update(dict(rows=total_subplots, cols=2, shared_xaxes=True, column_widths=[0.90, 0.10]))
    else:
        make_subplots_kwargs.update(dict(rows=total_subplots, cols=1))
    # Start with creating the overall figure layout with predefined subplots

    fig = make_subplots(**make_subplots_kwargs)

    # Define the height of each subplot and then calculate the total figure height
    total_height = subplot_height * total_subplots # Total figure height
    color_palette = ['white', 'red', 'green', 'blue', 'yellow']
    # color = color_palette[series_idx % len(color_palette)] if color_palette else None
    # common_plot_kwargs['marker_color'] = color

    with fig.batch_update():
        # Plot measured
        series_idx = 0
        fig = debug_plot_position_and_derivatives_figure_plotly(new_measured_pos_df['t'].to_numpy(),
                                                                new_measured_pos_df['x'].to_numpy(),
                                                                new_measured_pos_df['vel_x'].to_numpy(),
                                                                new_measured_pos_df['accel_x'].to_numpy(),
                                                                fig, series_idx,
                                                                debug_plot_name='measured',
                                                                color_palette=color_palette,
                                                                scatter_plot_kwargs=dict(mode='markers', marker=dict(size=5, opacity=0.35)), show_scatter=show_scatter) # , color=series_color
        
        # Add histograms to y-axis of existing scatter trace
        series_idx += 1

        # Plot decoded
        # row_offset = 0  # Increment the row offset for the next series of plots
        for a_name, a_df in all_epochs_position_derivatives_df_dict.items():
            fig = debug_plot_position_and_derivatives_figure_plotly(a_df['t'].to_numpy(),
                                                                    a_df['x'].to_numpy(),
                                                                    a_df['vel_x'].to_numpy(),
                                                                    a_df['accel_x'].to_numpy(),
                                                                    fig, series_idx,
                                                                    debug_plot_name=a_name,
                                                                    color_palette=color_palette,
                                                                    scatter_plot_kwargs=dict(mode='markers', marker=dict(size=3, opacity=0.6)), show_scatter=show_scatter)
            series_idx += 1
            
            
        # Update xaxis and yaxis properties if necessary
        # for i in range(1, total_subplots+1):
        #     fig.update_yaxes(title_text="Value", row=i, col=1)
        # fig.update_xaxes(title_text="Time (s)", row=total_subplots, col=1)  # Update only the last x-axis

        subplot_ylabel_text = ['pos', 'vel.', 'accel.']
        for i, a_label in enumerate(subplot_ylabel_text):
            fig = fig.update_yaxes(title_text=a_label, row=(i+1), col=1)
        ## only last one
        fig = fig.update_xaxes(title_text="Time (s)", row=total_subplots, col=1)  # Update only the last x-axis

        # Set the figure size here
        fig = fig.update_layout(
            height=total_height,  # Set the height of the figure
            width=figure_width,            # Set the width of the figure (or use your desired value)
            showlegend=True,     # You can turn off the legend if it's not needed
            margin=dict(
                l=50,  # Left margin
                r=50,  # Right margin
                t=50,  # Top margin
                b=50   # Bottom margin
            )
        )

    # Show the subplots
    # fig.show()
    return fig









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
# 2024-02-20 - Track Remapping Figures                                                                                 #
# ==================================================================================================================== #
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from pyphoplacecellanalysis.Pho2D.track_shape_drawing import _build_track_1D_verticies

@function_attributes(short_name=None, tags=['matplotlib', 'track', 'remapping', 'good', 'working'], input_requires=[], output_provides=[], uses=['pyphoplacecellanalysis.Pho2D.track_shape_drawing._build_track_1D_verticies'], used_by=[], creation_date='2024-02-22 11:12', related_items=[])
def _plot_track_remapping_diagram(LR_only_decoder_aclu_MAX_peak_maps_df, grid_bin_bounds, long_column_name:str='long_LR', short_column_name:str='short_LR'):
    """ Plots a single figure containing the long and short track outlines (flattened, overlayed) with single points on each corresponding to the peak location in 1D

    🔝🖼️🎨
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_track_remapping_diagram
    # grid_bin_bounds = BoundsRect.init_from_grid_bin_bounds(global_pf2D.config.grid_bin_bounds)
    fix, ax, _outputs_tuple = _plot_track_remapping_diagram(LR_only_decoder_aclu_MAX_peak_maps_df, long_peak_x, short_peak_x, peak_x_diff, grid_bin_bounds=long_pf2D.config.grid_bin_bounds)

    """
    # BUILDS TRACK PROPERTIES ____________________________________________________________________________________________ #
    from matplotlib.path import Path

    from pyphocorehelpers.geometry_helpers import BoundsRect
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackDimensions
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance

    ## Extract the quantities needed from the DF passed
    active_aclus = LR_only_decoder_aclu_MAX_peak_maps_df.index.to_numpy()
    long_peak_x = LR_only_decoder_aclu_MAX_peak_maps_df[long_column_name].to_numpy()
    short_peak_x = LR_only_decoder_aclu_MAX_peak_maps_df[short_column_name].to_numpy()
    # peak_x_diff = LR_only_decoder_aclu_MAX_peak_maps_df['peak_diff'].to_numpy()

    grid_bin_bounds = BoundsRect.init_from_grid_bin_bounds(grid_bin_bounds)
    # display(grid_bin_bounds)

    # long_track_dims = LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds)
    # short_track_dims = LinearTrackDimensions.init_from_grid_bin_bounds(grid_bin_bounds)

    long_track_dims = LinearTrackDimensions(track_length=170.0)
    short_track_dims = LinearTrackDimensions(track_length=100.0)

    common_1D_platform_height = 0.25
    common_1D_track_height = 0.1
    long_track_dims.minor_axis_platform_side_width = common_1D_platform_height
    long_track_dims.track_width = common_1D_track_height # (short_track_dims.minor_axis_platform_side_width

    short_track_dims.minor_axis_platform_side_width = common_1D_platform_height
    short_track_dims.track_width = common_1D_track_height # (short_track_dims.minor_axis_platform_side_width

    # instances:
    long_track = LinearTrackInstance(long_track_dims, grid_bin_bounds=grid_bin_bounds)
    short_track = LinearTrackInstance(short_track_dims, grid_bin_bounds=grid_bin_bounds)

    print(long_track_dims)
    print(short_track_dims)

    # BEGIN PLOTTING _____________________________________________________________________________________________________ #
    long_path = _build_track_1D_verticies(platform_length=22.0, track_length=170.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=long_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=-1.0, debug_print=True)
    short_path = _build_track_1D_verticies(platform_length=22.0, track_length=100.0, track_1D_height=1.0, platform_1D_height=1.1, track_center_midpoint_x=short_track.grid_bin_bounds.center_point[0], track_center_midpoint_y=1.0, debug_print=True)

    ## Create the remapping figure:
    fig, ax = plt.subplots()
    long_patch = patches.PathPatch(long_path, facecolor='orange', alpha=0.5, lw=2)
    ax.add_patch(long_patch)

    short_patch = patches.PathPatch(short_path, facecolor='green', alpha=0.5, lw=2)
    ax.add_patch(short_patch)
    ax.autoscale()

    ## INPUTS: LR_only_decoder_aclu_MAX_peak_maps_df, long_peak_x, short_peak_x, peak_x_diff

    
    # Define a colormap to map your unique integer indices to colors
    colormap = plt.cm.viridis  # or any other colormap
    normalize = mcolors.Normalize(vmin=active_aclus.min(), vmax=active_aclus.max())
    scalar_map = cm.ScalarMappable(norm=normalize, cmap=colormap)

    random_y_jitter = np.random.ranf((np.shape(active_aclus)[0], )) * 0.05
    # random_y_jitter = np.zeros((np.shape(active_aclus)[0], )) # no jitter

    long_y = (np.full_like(long_peak_x, 0.1)+random_y_jitter)
    short_y = (np.full_like(short_peak_x, 0.75)+random_y_jitter)

    _out = ax.scatter(long_peak_x, y=long_y, c=scalar_map.to_rgba(active_aclus), alpha=0.9, label='LR long_peak_x')
    _out2 = ax.scatter(short_peak_x, y=short_y, c=scalar_map.to_rgba(active_aclus), alpha=0.9, label='LR short_peak_x')

    # Add text labels to scatter points
    for i, aclu_val in enumerate(active_aclus):
        ax.text(long_peak_x[i], long_y[i], str(aclu_val), color='black', fontsize=8)
        ax.text(short_peak_x[i], short_y[i], str(aclu_val), color='black', fontsize=8)

    # Draw arrows from the first set of points to the second set
    arrows_output = {}
    for idx in range(len(long_peak_x)):
        # Starting point coordinates
        start_x = long_peak_x[idx]
        # start_y = 0.1 + random_y_jitter[idx]
        start_y = long_y[idx]
        # End point coordinates
        end_x = short_peak_x[idx]
        # end_y = 0.75 + random_y_jitter[idx]
        end_y = short_y[idx]
        # Calculate the change in x and y for the arrow
        # dx = end_x - start_x
        # dy = end_y - start_y

        # Get the corresponding color for the current index using the colormap
        arrow_color = scalar_map.to_rgba(active_aclus[idx])
        
        # Annotate the plot with arrows; adjust the properties according to your needs
        arrows_output[idx] = ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=dict(arrowstyle="->", color=arrow_color, alpha=0.6))

    # Show the plot
    # plt.legend()
    plt.show()
    return fig, ax, (arrows_output)





# ==================================================================================================================== #
# 2024-02-15 - Radon Transform / Weighted Correlation, etc helpers                                                     #
# ==================================================================================================================== #


# ==================================================================================================================== #
# 2024-02-08 - Plot Single ACLU Heatmaps for Each Decoder                                                              #
# ==================================================================================================================== #
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle

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
    """ wraps sns.jointplot to allow adding titles/axis labels/etc."""
    import seaborn as sns
    title = kwargs.pop('title', None)
    _out = sns.jointplot(*args, **kwargs)
    if title is not None:
        plt.suptitle(title)
    return _out


def plot_histograms(data_type: str, session_spec: str, data_results_df: pd.DataFrame, time_bin_duration_str: str) -> None:
    """ plots a stacked histogram of the many time-bin sizes """
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
# 2024-02-02 - Napari Export Helpers - Batch export all images                                                         #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_set_time_windw_index



import napari
from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict


def napari_add_aclu_slider(viewer, neuron_ids):
    """ adds a neuron aclu index overlay

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_add_aclu_slider


    """
    def on_update_slider(event):
        """ captures: viewer, neuron_ids
        
        Adds a little text label to the bottom right corner
        
        """
        # only trigger if update comes from first axis (optional)
        # print('inside')
        #ind_lambda = viewer.dims.indices[0]

        time = viewer.dims.current_step[0]
        matrix_aclu_IDX = int(time)
        # find the aclu value for this index:
        aclu: int = neuron_ids[matrix_aclu_IDX]
        viewer.text_overlay.text = f"aclu: {aclu}, IDX: {matrix_aclu_IDX}"
        
        # viewer.text_overlay.text = f"{time:1.1f} time"


    # viewer = napari.Viewer()
    # viewer.add_image(np.random.random((5, 5, 5)), colormap='red', opacity=0.8)
    viewer.text_overlay.visible = True
    _connected_on_update_slider_event = viewer.dims.events.current_step.connect(on_update_slider)
    # viewer.dims.events.current_step.disconnect(on_update_slider)
    return _connected_on_update_slider_event


def napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts, include_trial_by_trial_correlation_matrix:bool = True):
    """ Plots the directional trial-by-trial activity visualization:
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_plot_directional_trial_by_trial_activity_viz
        
        directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict = napari_plot_directional_trial_by_trial_activity_viz(directional_active_lap_pf_results_dicts)
    
    """
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict

    custom_direction_split_layers_dict = {}
    layers_list_sort_order = ['maze1_odd_z_scored_tuning_maps', 'maze1_odd_C_trial_by_trial_correlation_matrix', 
    'maze1_even_z_scored_tuning_maps', 'maze1_even_C_trial_by_trial_correlation_matrix',
    'maze2_odd_z_scored_tuning_maps', 'maze2_odd_C_trial_by_trial_correlation_matrix', 
    'maze2_even_z_scored_tuning_maps', 'maze2_even_C_trial_by_trial_correlation_matrix']

    ## Build the image data layers for each
    # for an_epoch_name, (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids) in directional_active_lap_pf_results_dicts.items():
    for an_epoch_name, active_trial_by_trial_activity_obj in directional_active_lap_pf_results_dicts.items():
        # (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids)
        z_scored_tuning_map_matrix = active_trial_by_trial_activity_obj.z_scored_tuning_map_matrix
        custom_direction_split_layers_dict[f'{an_epoch_name}_z_scored_tuning_maps'] = dict(blending='translucent', colormap='viridis', name=f'{an_epoch_name}_z_scored_tuning_maps', img_data=z_scored_tuning_map_matrix.transpose(1, 0, 2)) # reshape to be compatibile with C_i's dimensions
        if include_trial_by_trial_correlation_matrix:
            C_trial_by_trial_correlation_matrix = active_trial_by_trial_activity_obj.C_trial_by_trial_correlation_matrix
            custom_direction_split_layers_dict[f'{an_epoch_name}_C_trial_by_trial_correlation_matrix'] = dict(blending='translucent', colormap='viridis', name=f'{an_epoch_name}_C_trial_by_trial_correlation_matrix', img_data=C_trial_by_trial_correlation_matrix)

    # custom_direction_split_layers_dict

    # directional_viewer, directional_image_layer_dict = napari_trial_by_trial_activity_viz(None, None, layers_dict=custom_direction_split_layers_dict)

    ## sort the layers dict:
    custom_direction_split_layers_dict = {k:custom_direction_split_layers_dict[k] for k in reversed(layers_list_sort_order) if k in custom_direction_split_layers_dict}

    directional_viewer, directional_image_layer_dict = napari_from_layers_dict(layers_dict=custom_direction_split_layers_dict, title='Directional Trial-by-Trial Activity', axis_labels=('aclu', 'lap', 'xbin'))
    if include_trial_by_trial_correlation_matrix:
        directional_viewer.grid.shape = (-1, 4)
    else:
        directional_viewer.grid.shape = (2, -1)

    return directional_viewer, directional_image_layer_dict, custom_direction_split_layers_dict


def napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix, layers_dict=None, **viewer_kwargs):
    """ Visualizes position binned activity matrix beside the trial-by-trial correlation matrix.
    

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import napari_trial_by_trial_activity_viz
        viewer, image_layer_dict = napari_trial_by_trial_activity_viz(z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix)

        
        image_layer_dict
        # can find peak spatial shift distance by performing convolution and finding time of maximum value?
        _layer_z_scored_tuning_maps = image_layer_dict['z_scored_tuning_maps']
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 56]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 55.5]]), step=array([1, 1, 1]))

        _layer_C_trial_by_trial_correlation_matrix = image_layer_dict['C_trial_by_trial_correlation_matrix']
        _layer_C_trial_by_trial_correlation_matrix.extent

        # _layer_z_scored_tuning_maps.extent
        # Extent(data=array([[0, 0, 0],
        #        [80, 84, 84]]), world=array([[-0.5, -0.5, -0.5],
        #        [79.5, 83.5, 83.5]]), step=array([1, 1, 1]))

        # array([0, 0, 0])

    , title='Trial-by-trial Correlation Matrix C', axis_labels=('aclu', 'lap', 'xbin')
        
    Viewer properties:
        # viewer.grid # GridCanvas(stride=1, shape=(-1, -1), enabled=True)
        viewer.grid.enabled = True
        https://napari.org/0.4.15/guides/preferences.html
        https://forum.image.sc/t/dividing-the-display-in-the-viewer-window/42034
        https://napari.org/stable/howtos/connecting_events.html
        https://napari.org/stable/howtos/headless.html
        https://forum.image.sc/t/napari-how-add-a-text-label-time-always-in-the-same-spot-in-viewer/52932/3
        https://napari.org/stable/tutorials/segmentation/annotate_segmentation.html
        https://napari.org/stable/gallery/add_labels.html
        
    """
    from pyphoplacecellanalysis.GUI.Napari.napari_helpers import napari_from_layers_dict
    

    # inputs: z_scored_tuning_map_matrix, C_trial_by_trial_correlation_matrix
    image_layer_dict = {}
    if layers_dict is None:
        # build the default from the values:
        layers_dict = {
            'z_scored_tuning_maps': dict(blending='translucent', colormap='viridis', name='z_scored_tuning_maps', img_data=z_scored_tuning_map_matrix.transpose(1, 0, 2)), # reshape to be compatibile with C_i's dimensions
            'C_trial_by_trial_correlation_matrix': dict(blending='translucent', colormap='viridis', name='C_trial_by_trial_correlation_matrix', img_data=C_trial_by_trial_correlation_matrix),
        }

    viewer = None
    for i, (a_name, layer_dict) in enumerate(layers_dict.items()):
        img_data = layer_dict.pop('img_data').astype(float) # assumes integrated img_data in the layers dict
        if viewer is None: #i == 0:
            # viewer = napari.view_image(img_data) # rgb=True
            viewer = napari.Viewer(**viewer_kwargs)

        image_layer_dict[a_name] = viewer.add_image(img_data, **(dict(name=a_name)|layer_dict))

    viewer.dims.axis_labels = ('aclu', 'lap', 'xbin')
    viewer.grid.enabled = True # enables the grid layout of the data so adjacent data is displayed next to each other

    # outputs: viewer, image_layer_dict
    return viewer, image_layer_dict


def napari_export_image_sequence(viewer: napari.viewer.Viewer, imageseries_output_directory='output/videos/imageseries/', slider_axis_IDX: int = 0, build_filename_from_viewer_callback_fn=None):
    """ 
    
    Based off of `napari_export_video_frames`
    
    Usage:
            
        desired_save_parent_path = Path('/home/halechr/Desktop/test_napari_out').resolve()
        imageseries_output_directory = napari_export_image_sequence(viewer=viewer, imageseries_output_directory=desired_save_parent_path, slider_axis_IDX=0, build_filename_from_viewer_callback_fn=build_filename_from_viewer)

    
    """
    # Get the slide info:
    slider_min, slider_max, slider_step = viewer.dims.range[slider_axis_IDX]
    slider_range = np.arange(start=slider_min, step=slider_step, stop=slider_max)

    # __MAX_SIMPLE_EXPORT_COUNT: int = 5
    n_time_windows = np.shape(slider_range)[0]
    # n_time_windows = min(__MAX_SIMPLE_EXPORT_COUNT, n_time_windows) ## Limit the export to 5 items for testing

    if not isinstance(imageseries_output_directory, Path):
        imageseries_output_directory: Path = Path(imageseries_output_directory).resolve()
        
    for window_idx in np.arange(n_time_windows):
        # napari_set_time_windw_index(viewer, window_idx+1)
        napari_set_time_windw_index(viewer, window_idx)
        
        if build_filename_from_viewer_callback_fn is not None:
            image_out_path = build_filename_from_viewer_callback_fn(viewer, desired_save_parent_path=imageseries_output_directory, slider_axis_IDX=slider_axis_IDX)
        else:
            image_out_path = imageseries_output_directory.joinpath(f'screenshot_{window_idx}.png').resolve()
                
        screenshot = viewer.screenshot(path=image_out_path, canvas_only=True, flash=False)

    return imageseries_output_directory



# ==================================================================================================================== #
# 2024-02-02 - Trial-by-trial Correlation Matrix C                                                                     #
# ==================================================================================================================== #

from nptyping import NDArray
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
@define(slots=False)
class TrialByTrialActivity:
    """ 2024-02-12 - Computes lap-by-lap placefields and helps display correlation matricies and such.
    
    """
    active_epochs_df: pd.DataFrame = field()
    C_trial_by_trial_correlation_matrix: NDArray = field()
    z_scored_tuning_map_matrix: NDArray = field()
    aclu_to_matrix_IDX_map: Dict = field() # factory=Factory(dict)
    neuron_ids: NDArray = field()
    
    @property 
    def stability_score(self) -> NDArray:
        return np.nanmedian(self.C_trial_by_trial_correlation_matrix, axis=(1,2))
    
    @property 
    def aclu_to_stability_score_dict(self) -> Dict[int, NDArray]:
        return dict(zip(self.neuron_ids, self.stability_score))
    


    @classmethod
    def compute_spatial_binned_activity_via_pfdt(cls, active_pf_dt: PfND_TimeDependent, epochs_df: pd.DataFrame, included_neuron_IDs=None):
        """ 2024-02-01 - Use pfND_dt to compute spatially binned activity during the epochs.
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_spatial_binned_activity_via_pfdt
            
            if 'pf1D_dt' not in curr_active_pipeline.computation_results[global_epoch_name].computed_data:
                # if `KeyError: 'pf1D_dt'` recompute
                curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['pfdt_computation'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)


            active_pf_1D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf1D_dt'])
            active_pf_2D_dt: PfND_TimeDependent = deepcopy(curr_active_pipeline.computation_results[global_epoch_name].computed_data['pf2D_dt'])


            laps_df = deepcopy(global_any_laps_epochs_obj.to_dataframe())
            n_laps = len(laps_df)

            active_pf_dt: PfND_TimeDependent = deepcopy(active_pf_1D_dt)
            # active_pf_dt = deepcopy(active_pf_2D_dt) # 2D
            historical_snapshots = compute_spatial_binned_activity_via_pfdt(active_pf_dt=active_pf_dt, epochs_df=laps_df)

        """
        use_pf_dt_obj = False

        if included_neuron_IDs is None:
            included_neuron_IDs = deepcopy(active_pf_dt.included_neuron_IDs) # this may be under-included. Is there like an "all-times-neuron_IDs?"
            
        if isinstance(epochs_df, pd.DataFrame):
            # dataframes are treated weird by PfND_dt, convert to basic numpy array of shape (n_epochs, 2)
            time_intervals = epochs_df[['start', 'stop']].to_numpy() # .shape # (n_epochs, 2)
        else:
            time_intervals = epochs_df # assume already a numpy array
            
        assert np.shape(time_intervals)[-1] == 2
        n_epochs: int = np.shape(time_intervals)[0]
            
        ## Entirely independent computations for binned_times:
        if use_pf_dt_obj:
            active_pf_dt.reset()

        # if included_neuron_IDs is not None:
        #     # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        #     active_pf_dt.all_time_filtered_spikes_df = active_pf_dt.all_time_filtered_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        #     active_pf_dt.all_time_filtered_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_dt.all_time_filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        
        if not use_pf_dt_obj:
            historical_snapshots = {} # build a dict<float:PlacefieldSnapshot>

        for start_t, end_t in time_intervals:
            ## Inline version that reuses active_pf_1D_dt directly:
            if use_pf_dt_obj:
                # active_pf_1D_dt.update(end_t, should_snapshot=True) # use this because it correctly integrates over [0, end_t] instead of [start_t, end_t]
                # active_pf_1D_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=True, should_snapshot=True)
                historical_snapshots[float(end_t)] = active_pf_dt.complete_time_range_computation(start_t, end_t, assign_results_to_member_variables=False, should_snapshot=False) # Integrates each [start_t, end_t] independently
            else:
                # Static version that calls PfND_TimeDependent.perform_time_range_computation(...) itself using just the computed variables of `active_pf_1D_dt`:
                all_time_filtered_spikes_df: pd.DataFrame = deepcopy(active_pf_dt.all_time_filtered_spikes_df).spikes.sliced_by_neuron_id(included_neuron_IDs)
                historical_snapshots[float(end_t)] = PfND_TimeDependent.perform_time_range_computation(all_time_filtered_spikes_df, active_pf_dt.all_time_filtered_pos_df, position_srate=active_pf_dt.position_srate,
                                                                            xbin=active_pf_dt.xbin, ybin=active_pf_dt.ybin,
                                                                            start_time=start_t, end_time=end_t,
                                                                            included_neuron_IDs=included_neuron_IDs, active_computation_config=active_pf_dt.config, override_smooth=active_pf_dt.smooth)

        # {1.9991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x16c2b74fb20>, 2.4991045125061646: <neuropy.analyses.time_dependent_placefields.PlacefieldSnapshot at 0x168acfb3bb0>, ...}
        if use_pf_dt_obj:
            historical_snapshots = active_pf_dt.historical_snapshots

        epoch_pf_results_dict = {'historical_snapshots': historical_snapshots}
        epoch_pf_results_dict['num_position_samples_occupancy'] = np.stack([placefield_snapshot.num_position_samples_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['seconds_occupancy'] = np.stack([placefield_snapshot.seconds_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['normalized_occupancy'] = np.stack([placefield_snapshot.normalized_occupancy for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['spikes_maps_matrix'] = np.stack([placefield_snapshot.spikes_maps_matrix for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        epoch_pf_results_dict['occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in epoch_pf_results_dict['historical_snapshots'].values()])
        # active_lap_pf_results_dict['snapshot_occupancy_weighted_tuning_maps'] = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in active_lap_pf_results_dict['historical_snapshots'].values()])

        # len(historical_snapshots)
        return epoch_pf_results_dict


    @classmethod
    def compute_trial_by_trial_correlation_matrix(cls, active_pf_dt: PfND_TimeDependent, occupancy_weighted_tuning_maps_matrix: NDArray, included_neuron_IDs=None):
        """ 2024-02-02 - computes the Trial-by-trial Correlation Matrix C 
        
        Returns:
            C_trial_by_trial_correlation_matrix: .shape (n_aclus, n_epochs, n_epochs) - (80, 84, 84)
            z_scored_tuning_map_matrix

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_trial_by_trial_correlation_matrix

            C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix = compute_trial_by_trial_correlation_matrix(active_pf_dt, occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix)

        """
        if included_neuron_IDs is None:
            neuron_ids = deepcopy(np.array(active_pf_dt.ratemap.neuron_ids))
        else:
            neuron_ids = np.array(included_neuron_IDs)
            

        n_aclus = len(neuron_ids)
        n_xbins = len(active_pf_dt.xbin_centers)

        assert np.shape(occupancy_weighted_tuning_maps_matrix)[1] == n_aclus
        assert np.shape(occupancy_weighted_tuning_maps_matrix)[2] == n_xbins

        epsilon_value: float = 1e-12
        # Assuming 'occupancy_weighted_tuning_maps_matrix' is your dataset with shape (trials, positions)
        # Z-score along the position axis (axis=1)
        position_axis_idx: int = 2
        z_scored_tuning_map_matrix: NDArray = (occupancy_weighted_tuning_maps_matrix - np.nanmean(occupancy_weighted_tuning_maps_matrix, axis=position_axis_idx, keepdims=True)) / ((np.nanstd(occupancy_weighted_tuning_maps_matrix, axis=position_axis_idx, keepdims=True))+epsilon_value)

        # trial-by-trial correlation matrix C
        M = float(n_xbins)
        C_list = []
        for i, aclu in enumerate(neuron_ids):
            A_i = np.squeeze(z_scored_tuning_map_matrix[:,i,:])
            C_i = (1/(M-1)) * (A_i @ A_i.T) # Perform matrix multiplication using the @ operator
            # C_i.shape # (n_epochs, n_epochs) - (84, 84) - gives the correlation between each epoch and the others
            C_list.append(C_i)
        # occupancy_weighted_tuning_maps_matrix

        C_trial_by_trial_correlation_matrix = np.stack(C_list, axis=0) # .shape (n_aclus, n_epochs, n_epochs) - (80, 84, 84)
        # outputs: C_trial_by_trial_correlation_matrix

        # n_laps: int = len(laps_unique_ids)
        aclu_to_matrix_IDX_map = dict(zip(neuron_ids, np.arange(n_aclus)))

        return C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map

    ## MAIN CALL:
    @classmethod
    def directional_compute_trial_by_trial_correlation_matrix(cls, active_pf_dt: PfND_TimeDependent, directional_lap_epochs_dict, included_neuron_IDs=None) -> Dict[str, "TrialByTrialActivity"]:
        """ 
        
        2024-02-02 - 10pm - Have global version working but want seperate directional versions. Seperately do `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`:
        
        """
        directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = {}

        # # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        # if included_neuron_IDs is not None:
        #     active_pf_dt.all_time_filtered_spikes_df = active_pf_dt.all_time_filtered_spikes_df.spikes.sliced_by_neuron_id(included_neuron_IDs)
        #     active_pf_dt.all_time_filtered_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_pf_dt.all_time_filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()


        # Seperately do (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj):
        for an_epoch_name, active_laps_epoch in directional_lap_epochs_dict.items():
            active_laps_df = deepcopy(active_laps_epoch.to_dataframe())
            active_lap_pf_results_dict = cls.compute_spatial_binned_activity_via_pfdt(active_pf_dt=active_pf_dt, epochs_df=active_laps_df, included_neuron_IDs=included_neuron_IDs)
            # Unpack the variables:
            historical_snapshots = active_lap_pf_results_dict['historical_snapshots']
            occupancy_weighted_tuning_maps_matrix = active_lap_pf_results_dict['occupancy_weighted_tuning_maps'] # .shape: (n_epochs, n_aclus, n_xbins) - (84, 80, 56)
            # 2024-02-02 - Trial-by-trial Correlation Matrix C
            C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map = cls.compute_trial_by_trial_correlation_matrix(active_pf_dt, occupancy_weighted_tuning_maps_matrix=occupancy_weighted_tuning_maps_matrix, included_neuron_IDs=included_neuron_IDs)
            neuron_ids = np.array(list(aclu_to_matrix_IDX_map.keys()))
            
            # directional_active_lap_pf_results_dicts[an_epoch_name] = (active_laps_df, C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map, neuron_ids) # currently discards: occupancy_weighted_tuning_maps_matrix, historical_snapshots, active_lap_pf_results_dict, active_laps_df
            directional_active_lap_pf_results_dicts[an_epoch_name] = TrialByTrialActivity(active_epochs_df=active_laps_df, C_trial_by_trial_correlation_matrix=C_trial_by_trial_correlation_matrix, z_scored_tuning_map_matrix=z_scored_tuning_map_matrix, aclu_to_matrix_IDX_map=aclu_to_matrix_IDX_map, neuron_ids=neuron_ids)
            
        return directional_active_lap_pf_results_dicts


# ==================================================================================================================== #
# 2024-02-01 - Spatial Information                                                                                     #
# ==================================================================================================================== #

from neuropy.analyses.placefields import PfND
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

def _perform_calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy):
    """ function to calculate Spatial Information (SI) score
    
    # f_i is the trial-averaged activity per position bin i -- sounds like the average number of spikes in each position bin within the trial

    # f is the mean activity rate over the whole session, computed as the sum of f_i * p_i over all N (position) bins

    ## What they call "p_i" - "occupancy probability per position bin per trial" ([Sosa et al., 2023, p. 23](zotero://select/library/items/I5FLMP5R)) ([pdf](zotero://open-pdf/library/items/C3Y8AKEB?page=23&annotation=GAHX9PYH))
    occupancy_probability = a_spikes_bin_counts_mat.copy()
    occupancy_probability = occupancy_probability / occupancy_probability.sum(axis=1, keepdims=True) # quotient is "total number of samples in each trial"
    occupancy_probability

    # We then summed the occupancy probabilities across trials and divided by the total per session to get an occupancy probability per position bin per session

    # To get the spatial “tuning curve” over the session, we averaged the activity in each bin across trials

    Usage:    
    SI = calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy)
    """
    ## SI Calculator: fi/<f>
    p_i = probability_normalized_occupancy.copy()

    # f_rate_over_all_session = global_all_spikes_counts['rate_Hz'].to_numpy()
    # f_rate_over_all_session
    check_f = np.nansum((p_i *  epoch_averaged_activity_per_pos_bin), axis=-1) # a check for f (rate over all session)
    f_rate_over_all_session = check_f # temporarily use check_f instead of the real f_rate

    fi_over_mean_f = epoch_averaged_activity_per_pos_bin / f_rate_over_all_session.reshape(-1, 1) # the `.reshape(-1, 1)` fixes the broadcasting

    log_base_2_of_fi_over_mean_f = np.log2(fi_over_mean_f) ## Here is where some entries become -np.inf

    _summand = (p_i * fi_over_mean_f * log_base_2_of_fi_over_mean_f) # _summand.shape # (77, 56)

    SI = np.nansum(_summand, axis=1)
    return SI


def compute_spatial_information(all_spikes_df: pd.DataFrame, an_active_pf: PfND, global_session_duration:float):
    """ Calculates the spatial information (SI) for each cell and returns all intermediates.

    Usage: 
        global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df).drop(columns=['neuron_type'], inplace=False)
        an_active_pf = deepcopy(global_pf1D)
        SI, all_spikes_df, epoch_averaged_activity_per_pos_bin, global_all_spikes_counts = compute_spatial_information(all_spikes_df=global_spikes_df, an_active_pf=an_active_pf, global_session_duration=global_session.duration)


    """
    from neuropy.core.flattened_spiketrains import SpikesAccessor
    from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns

    #  Inputs: global_spikes_df: pd.DataFrame, an_active_pf: PfND, 
    # Build the aclu indicies:
    # neuron_IDs = global_spikes_df.aclu.unique()
    # n_aclus = global_spikes_df.aclu.nunique()
    neuron_IDs = deepcopy(np.array(an_active_pf.ratemap.neuron_ids))
    n_aclus = len(neuron_IDs)

    all_spikes_df = deepcopy(all_spikes_df).spikes.sliced_by_neuron_id(neuron_IDs)
    all_spikes_df, neuron_id_to_new_IDX_map = all_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
    all_spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(all_spikes_df, bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
    # global_spikes_df


    # Get <f> for each sell, the rate over the entire session.
    global_all_spikes_counts = all_spikes_df.groupby(['aclu']).agg(t_count=('t', 'count')).reset_index()
    global_all_spikes_counts['rate_Hz'] = global_all_spikes_counts['t_count'] / global_session_duration
    # global_all_spikes_counts

    assert len(global_all_spikes_counts) == n_aclus
    
    ## Next need epoch-averaged activity per position bin:

    # Build the full matrix:
    global_per_position_bin_spikes_counts = all_spikes_df.groupby(['aclu', 'binned_x', 'binned_y']).agg(t_count=('t', 'count')).reset_index()
    a_spikes_df_bin_grouped = global_per_position_bin_spikes_counts.groupby(['aclu', 'binned_x']).agg(t_count_sum=('t_count', 'sum')).reset_index() ## for 1D plotting mode, collapse over all y-bins
    # a_spikes_df_bin_grouped

    assert n_aclus is not None
    n_xbins = len(an_active_pf.xbin_centers)
    # n_ybins = len(an_active_pf.ybin_centers)

    print(f'{n_aclus = }, {n_xbins = }')

    # a_spikes_bin_counts_mat = np.zeros((n_laps, n_xbins)) # for this single cell
    epoch_averaged_activity_per_pos_bin = np.zeros((n_aclus, n_xbins)) # for this single cell

    ## Update the matrix:
    for index, row in a_spikes_df_bin_grouped.iterrows():
        # lap = int(row['lap'])
        aclu = int(row['aclu'])
        neuron_fragile_IDX: int = neuron_id_to_new_IDX_map[aclu]
        binned_x = int(row['binned_x'])
        count = row['t_count_sum']
        # a_spikes_bin_counts_mat[lap - 1][binned_x - 1] = count
        epoch_averaged_activity_per_pos_bin[neuron_fragile_IDX - 1][binned_x - 1] = count

    # an_active_pf.occupancy.shape # (n_xbins,) - (56,)
    # epoch_averaged_activity_per_pos_bin.shape # (n_aclus, n_xbins) - (77, 56)
    assert np.shape(an_active_pf.occupancy)[0] == np.shape(epoch_averaged_activity_per_pos_bin)[1]
        
    ## Compute actual Spatial Information for each cell:
    SI = _perform_calc_SI(epoch_averaged_activity_per_pos_bin, probability_normalized_occupancy=an_active_pf.ratemap.probability_normalized_occupancy)

    return SI, all_spikes_df, epoch_averaged_activity_per_pos_bin, global_all_spikes_counts


def permutation_test(position_data, rate_maps, occupancy_maps, n_permutations=100):
    """ Not yet implemented. 2024-02-01
    
    Based off of the following quote:
    To determine the significance of the SI scores, we created a null distribution by circularly permuting the position data relative to the timeseries of each cell, by a random amount of at least 1 sec and a maximum amount of the length of the trial, independently on each trial. SI was calculated from the trial-averaged activity of each shuffle, and this shuffle procedure was repeated 100 times per cell. A cell’s true SI was considered significant if it exceeded 95% of the SI scores from all shuffles within animal (i.e. shuffled scores were pooled across cells within animal to produce this threshold, which is more stringent than comparing to the shuffle of each individual cell
    
    Usage:
        # True place field rate maps for all cells
        rate_maps = np.array('your rate maps')
        # True occupancy maps for all cells
        occupancy_maps = np.array('your occupancy maps')
        # Your position data
        position_data = np.array('your position data')

        # Call the permutation test function with the given number of permutations
        sig_cells = permutation_test(position_data, rate_maps, occupancy_maps, n_permutations=100)

        print(f'Indices of cells with significant SI: {sig_cells}')

    
    """
    # function to calculate Spatial Information (SI) score
    def calc_SI(rate_map, occupancy):
        # Place your existing SI calculation logic here
        pass

    # function to calculate rate map for given position data
    def calc_rate_map(position_data):
        # logic to calculate rate map
        pass

    # function to calculate occupancy map for given position data
    def calc_occupancy_map(position_data):
        # logic to calculate occupancy map
        pass

    n_cells = rate_maps.shape[0]  # number of cells
    si_scores = np.empty((n_cells, n_permutations))  # Initialize container for SI scores per cell per permutation
    true_si_scores = np.empty(n_cells)  # Initialize container for true SI scores per cell
   
    for cell_idx in range(n_cells):
        true_si_scores[cell_idx] = calc_SI(rate_maps[cell_idx], occupancy_maps[cell_idx])
        
        for perm_idx in range(n_permutations):
            shift_val = np.random.randint(1, len(position_data))  # A random shift amount
            shuffled_position_data = np.roll(position_data, shift_val)  # Shift the position data
        
            shuffled_rate_map = calc_rate_map(shuffled_position_data)
            shuffled_occupancy_map = calc_occupancy_map(shuffled_position_data)

            si_scores[cell_idx][perm_idx] = calc_SI(shuffled_rate_map, shuffled_occupancy_map)
   
    pooled_scores = si_scores.flatten() # Pool scores within animal
    threshold = np.percentile(pooled_scores, 95)  # Get the 95th percentile of the pooled scores

    return np.where(true_si_scores > threshold)  # Return indices where true SI scores exceed 95 percentile


def compute_activity_by_lap_by_position_bin_matrix(a_spikes_df: pd.DataFrame, lap_id_to_matrix_IDX_map: Dict, n_xbins: int): # , an_active_pf: Optional[PfND] = None
    """ 2024-01-31 - Note that this does not take in position tracking information, so it cannot compute real occupancy. 
    
    Plots for a single neuron.
    
    an_active_pf: is just so we have access to the placefield's properties later
    
    
    Currently plots raw spikes counts (in number of spikes).
    
    """
    # Filter rows based on column: 'binned_x'
    a_spikes_df = a_spikes_df[a_spikes_df['binned_x'].astype("string").notna()]
    # a_spikes_df_bin_grouped = a_spikes_df.groupby(['binned_x', 'binned_y']).agg(t_seconds_count=('t_seconds', 'count')).reset_index()
    a_spikes_df_bin_grouped = a_spikes_df.groupby(['binned_x', 'binned_y', 'lap']).agg(t_seconds_count=('t_seconds', 'count')).reset_index()
    # a_spikes_df_bin_grouped

    ## for 1D plotting mode, collapse over all y-bins:
    a_spikes_df_bin_grouped = a_spikes_df_bin_grouped.groupby(['binned_x', 'lap']).agg(t_seconds_count_sum=('t_seconds_count', 'sum')).reset_index()
    # a_spikes_df_bin_grouped
    assert n_xbins is not None
    assert lap_id_to_matrix_IDX_map is not None
    n_laps: int = len(lap_id_to_matrix_IDX_map)
    
    a_spikes_bin_counts_mat = np.zeros((n_laps, n_xbins)) # for this single cell

    ## Update the matrix:
    for index, row in a_spikes_df_bin_grouped.iterrows():
        lap_id = int(row['lap'])
        lap_IDX = lap_id_to_matrix_IDX_map[lap_id]
        
        binned_x = int(row['binned_x'])
        count = row['t_seconds_count_sum']
        a_spikes_bin_counts_mat[lap_IDX][binned_x - 1] = count
        
    # active_out_matr = occupancy_probability
    
    # active_out_matr = a_spikes_bin_counts_mat
    # “calculated the occupancy (number of imaging samples) in each bin on each trial, and divided this by the total number of samples in each trial to get an occupancy probability per position bin per trial” 
    return a_spikes_bin_counts_mat

def compute_spatially_binned_activity(an_active_pf: PfND): # , global_any_laps_epochs_obj
    """ 
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_spatially_binned_activity
        
        # a_spikes_df = None
        # a_spikes_df: pd.DataFrame = deepcopy(long_one_step_decoder_1D.spikes_df) #.drop(columns=['neuron_type'], inplace=False)

        # an_active_pf = deepcopy(global_pf2D)
        # an_active_pf = deepcopy(global_pf1D)
        # an_active_pf.linear_pos_obj

        # an_active_pf = active_pf_2D_dt
        an_active_pf = active_pf_1D_dt
        position_binned_activity_matr_dict, split_spikes_df_dict, (neuron_id_to_new_IDX_map, lap_id_to_matrix_IDX_map) = compute_spatially_binned_activity(an_active_pf)
        # 14.8s
    """
    from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
    # from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # needed to add laps column

    ## need global laps positions now.

    # # Position:
    # position_df: pd.DataFrame = deepcopy(an_active_pf.filtered_pos_df) # .drop(columns=['neuron_type'], inplace=False)
    # position_df, (xbin,), bin_infos = build_df_discretized_binned_position_columns(position_df, bin_values=(an_active_pf.xbin,), position_column_names=('lin_pos',), binned_column_names=('binned_x',), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
    # if 'lap' not in position_df:
    #     position_df = add_epochs_id_identity(position_df, epochs_df=deepcopy(global_any_laps_epochs_obj.to_dataframe()), epoch_id_key_name='lap', epoch_label_column_name='lap_id', no_interval_fill_value=-1, override_time_variable_name='t')
    #     # drop the -1 indicies because they are below the speed:
    #     position_df = position_df[position_df['lap'] != -1] # Drop all non-included spikes
    # position_df

    neuron_IDs = deepcopy(np.array(an_active_pf.ratemap.neuron_ids))
    n_aclus = len(neuron_IDs)

    # all_spikes_df: pd.DataFrame = deepcopy(all_spikes_df) # require passed-in value
    # a_spikes_df: pd.DataFrame = deepcopy(an_active_pf.spikes_df)
    # a_spikes_df: pd.DataFrame = deepcopy(an_active_pf.filtered_spikes_df)
    all_spikes_df: pd.DataFrame = deepcopy(an_active_pf.spikes_df) # Use placefields all spikes 
    all_spikes_df = all_spikes_df.spikes.sliced_by_neuron_id(neuron_IDs)
    all_spikes_df = all_spikes_df[all_spikes_df['lap'] > -1] # get only the spikes within a lap
    all_spikes_df, neuron_id_to_new_IDX_map = all_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
    all_spikes_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(all_spikes_df, bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)

    split_spikes_dfs_list = all_spikes_df.spikes.get_split_by_unit()
    split_spikes_df_dict = dict(zip(neuron_IDs, split_spikes_dfs_list))
    
    laps_unique_ids = all_spikes_df.lap.unique()
    n_laps: int = len(laps_unique_ids)
    lap_id_to_matrix_IDX_map = dict(zip(laps_unique_ids, np.arange(n_laps)))

    # n_laps: int = position_df.lap.nunique()
    n_xbins = len(an_active_pf.xbin_centers)
    # n_ybins = len(an_active_pf.ybin_centers)
    
    # idx: int = 9
    # aclu: int = neuron_IDs[idx]
    # print(f'aclu: {aclu}')
    
    position_binned_activity_matr_dict = {}

    # for a_spikes_df in split_spikes_dfs:
    for aclu, a_spikes_df in split_spikes_df_dict.items():
        # split_spikes_df_dict[aclu], (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(a_spikes_df.drop(columns=['neuron_type'], inplace=False), bin_values=(an_active_pf.xbin, an_active_pf.ybin), active_computation_config=deepcopy(an_active_pf.config), force_recompute=True, debug_print=False)
        a_position_binned_activity_matr = compute_activity_by_lap_by_position_bin_matrix(a_spikes_df=a_spikes_df, lap_id_to_matrix_IDX_map=lap_id_to_matrix_IDX_map, n_xbins=n_xbins)
        position_binned_activity_matr_dict[aclu] = a_position_binned_activity_matr
        
    # output: split_spikes_df_dict
    return position_binned_activity_matr_dict, split_spikes_df_dict, (neuron_id_to_new_IDX_map, lap_id_to_matrix_IDX_map)





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
    
