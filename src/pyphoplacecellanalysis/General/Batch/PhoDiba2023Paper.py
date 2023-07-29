import sys
from copy import deepcopy
from typing import Dict, List, Any, Tuple, Optional, Callable
from attrs import define, field, Factory, asdict
import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # used for plot_epoch_track_assignments
from flexitext import flexitext ## flexitext version

from enum import Enum
from neuropy.utils.mixins.enum_helpers import ExtendedEnum # used in TrackAssignmentState
from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update

from pyphocorehelpers.mixins.key_value_hashable import KeyValueHashableObject
from pyphocorehelpers.indexing_helpers import partition # needed by `AssigningEpochs` to partition the dataframe by aclus

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences

## Laps Stuff:
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass

from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.result_context import overwriting_display_context, providing_context
from neuropy.core.user_annotations import UserAnnotationsManager
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import determine_long_short_pf1D_indicies_sort_by_peak
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from neuropy.utils.matplotlib_helpers import FormattedFigureText


from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots, BatchPhoJonathanFiguresHelper
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends
from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import interactive_good_epoch_selections

import pyphoplacecellanalysis.External.pyqtgraph as pg # pyqtgraph
import matplotlib.pyplot as plt


# Testing:
from pyphocorehelpers.plotting.figure_management import capture_new_figures_decorator


_bak_rcParams = mpl.rcParams.copy()
# mpl.rcParams['toolbar'] = 'None' # disable toolbars
# %matplotlib qt

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for item in list_of_dicts:
        for key, value in item.items():
            if key in dict_of_lists:
                dict_of_lists[key].append(value)
            else:
                dict_of_lists[key] = [value]
    return dict_of_lists


# ==================================================================================================================== #
# 2023-06-29 - Build Properly sorted ratemaps with potentially missing entries                                         #
# ==================================================================================================================== #

from neuropy.plotting.ratemaps import _help_plot_ratemap_neuronIDs # required for build_shared_sorted_neuronIDs

@function_attributes(short_name=None, tags=['active', 'sort', 'neuron_ids'], input_requires=[], output_provides=[], uses=['_help_plot_ratemap_neuronIDs'], used_by=[], creation_date='2023-06-29 07:46', related_items=[])
def build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind):
    """ 
    
    `sort_ind` should be the indicies to sort `included_unit_neuron_IDs`.
        `included_unit_neuron_IDs` may include entries not present in `ratemap`.
    
    Usage:
        ratemap = long_pf1D.ratemap
        included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
        rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())
    """
    if not isinstance(sort_ind, np.ndarray):
        sort_ind = np.array(sort_ind)
    assert np.size(included_unit_neuron_IDs) == np.size(sort_ind), f"`sort_ind` should be the indicies to sort `included_unit_neuron_IDs`."

    #TODO 2023-06-29 06:50: - [ ] SOOO CLOSE. This is the right way to do it... the way that's done in `neuropy.plotting.ratemaps.plot_ratemap_1D`, but because i'm trying to plot the ratemaps as a heatmap I need to fill the missing entries with appropriately sized np.nans or something.
    active_maps, title_substring, included_unit_indicies = _help_plot_ratemap_neuronIDs(ratemap, included_unit_neuron_IDs=included_unit_neuron_IDs, debug_print=True)
    n_neurons = len(included_unit_indicies) # n_neurons includes Non-active neurons without a placefield if they're provided in included_unit_indicies.
    if not isinstance(included_unit_indicies, np.ndarray):
        included_unit_indicies = np.array(included_unit_indicies)
    included_unit_indicies

    needed_empty_map_shape = np.shape(active_maps)[1:]

    sorted_included_unit_indicies = included_unit_indicies[sort_ind]
    rediculous_final_sorted_all_included_pfmap = []
    rediculous_final_sorted_all_included_neuron_ID = []

    for i, curr_included_unit_index in enumerate(sorted_included_unit_indicies):
        # `curr_included_unit_index` is either an index into the `included_unit_neuron_IDs` array or None
        ### Three things must be considered for each "row" of the plot: 1. the pfmap curve values, 2. the cell id label displayed to the left of the row, 3. the color which is used for the row.
        if curr_included_unit_index is not None:
            # valid neuron ID, access like normal
            pfmap = active_maps[curr_included_unit_index]
            # normal (non-shared mode)
            curr_ratemap_relative_neuron_IDX = curr_included_unit_index
            curr_neuron_ID = ratemap.neuron_ids[curr_ratemap_relative_neuron_IDX]
            
        else:
            # invalid neuron ID, generate blank entry
            curr_ratemap_relative_neuron_IDX = None # This neuron_ID doesn't correspond to a neuron_IDX in the current ratemap, so we'll mark this value as None
            assert included_unit_neuron_IDs is not None
            curr_neuron_ID = included_unit_neuron_IDs[sort_ind[i]]

            # pfmap = np.zeros((np.shape(active_maps)[1],)) # fully allocated new array of zeros
            pfmap = np.zeros(needed_empty_map_shape) # fully allocated new array of zeros
            
        rediculous_final_sorted_all_included_pfmap.append(pfmap)
        rediculous_final_sorted_all_included_neuron_ID.append(curr_neuron_ID)

    rediculous_final_sorted_all_included_neuron_ID = np.array(rediculous_final_sorted_all_included_neuron_ID)
    
    rediculous_final_sorted_all_included_pfmap = np.vstack(rediculous_final_sorted_all_included_pfmap)
    # rediculous_final_sorted_all_included_pfmap.shape # (68, 117)
    return rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap

# ==================================================================================================================== #
# 2023-06-27 - Paper Figure 1 Code                                                                                     #
# ==================================================================================================================== #


# ==================================================================================================================== #
# 2023-07-14 - Iterative Epoch Track Assignment                                                                        #
# ==================================================================================================================== #
class TrackAssignmentState(ExtendedEnum):
    """Docstring for TrackAssignmentState."""
    UNASSIGNED = "unassigned"
    LONG_TRACK = "long_track"
    SHORT_TRACK = "short_track"
    NEITHER = "neither"
    

    @property
    def displayColor(self):
        return self.displayColorList()[self]
    
    @classmethod
    def displayColorList(cls):
        return cls.build_member_value_dict(['grey', 'blue', 'red', 'black'])
    

    # track_assignment = epoch['track_assignment'].decision.name
    # # Set the color of the epoch based on the track assignment
    # if track_assignment == 'UNASSIGNED':
    #     color = 'grey'
    # elif track_assignment == 'LONG_TRACK':
    #     color = 'blue'
    # elif track_assignment == 'SHORT_TRACK':
    #     color = 'red'
    # elif track_assignment == 'NEITHER':
    #     color = 'black'

@define(slots=False, frozen=True)
class TrackAssignmentDecision(KeyValueHashableObject):
    """ represents the decision made to assign a particular epoch to a track (long or short) or to neither track"""
    decision: TrackAssignmentState
    confidence: float # a value between 0 and 1

@define(slots=False)
class AssigningEpochs:
    """ class responsible for iteratively applying various criteria to assign epochs to a particular track starting with the unassigned set of all epochs (in this analysis replay events) """
    filter_epochs_df: pd.DataFrame
    

    @property
    def unassigned_epochs_df(self):
        """A convenience accessor for only the unassigned epochs remaining in the self.filter_epochs_df."""
        is_unassigned = np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])
        return self.filter_epochs_df[is_unassigned]
    @unassigned_epochs_df.setter
    def unassigned_epochs_df(self, value):
        self.filter_epochs_df[np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])] = value
        

    def _subfn_find_example_epochs(self, spikes_df: pd.DataFrame, exclusive_aclus, included_neuron_ids=None):
        """ aims to find epochs for use as examples in Figure 1:
            1. Contain spikes from aclus that are exclusive to one of the two tracks (either short or long)
            2. Are labeled as "good replays" by my user annotations
            
            Adds: 'contains_one_exclusive_aclu' column to filter_epochs_df
            
            Usage:
                from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _find_example_epochs
                # included_neuron_ids = deepcopy(exclusive_aclus)
                # included_neuron_ids = None
                included_neuron_ids = EITHER_subset.track_exclusive_aclus
                spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
                # filter_epochs = deepcopy(curr_active_pipeline.sess.replay)
                # spikes_df = deepcopy(long_results_obj.spikes_df) # LeaveOneOutDecodingAnalysisResult
                # spikes_df[np.isin(spikes_df.aclu, included_neuron_ids)]
                filter_epochs = deepcopy(long_results_obj.active_filter_epochs)

                filter_epoch_spikes_df, filter_epochs_df = _find_example_epochs(spikes_df, filter_epochs, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids)

        """

        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids
            
        filter_epoch_spikes_df = deepcopy(spikes_df)
        filter_epoch_spikes_df = _prepare_spikes_df_from_filter_epochs(filter_epoch_spikes_df, filter_epochs=self.filter_epochs_df, included_neuron_ids=included_neuron_ids, epoch_id_key_name='replay_epoch_id', debug_print=False) # replay_epoch_id

        unique_replay_epoch_id_values, epoch_split_spikes_df_list = partition(filter_epoch_spikes_df, 'replay_epoch_id')
        # filter_epoch_spikes_df.groupby('replay_epoch_id')
        # unique_replay_epoch_id_values # (198,)
        epoch_spikes_unique_aclus_list = [] # a list of each aclu that fires at least once in the epoch
        epoch_contains_any_exclusive_aclus = []

        for an_epoch_id, epoch_specific_spikes_df in zip(unique_replay_epoch_id_values, epoch_split_spikes_df_list):
            # Loop through the epochs
            # find epochs containing at least one in `exclusive_aclus`:
            epoch_spikes_unique_aclus = epoch_specific_spikes_df.aclu.unique()
            epoch_spikes_unique_aclus_list.append(epoch_spikes_unique_aclus)
            # epoch_spikes_unique_aclus # finds lists of all the active aclus in the given epoch
            if len(epoch_spikes_unique_aclus) > 0:	
                epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
            else:
                epoch_contains_any_exclusive_aclus.append(False)
            # epoch_contains_any_exclusive_aclus # returns True if the epoch contains one ore more spikes from the search list `exclusive_aclus`

        assert len(epoch_contains_any_exclusive_aclus) == np.shape(self.filter_epochs_df)[0]
        self.filter_epochs_df['contains_one_exclusive_aclu'] = epoch_contains_any_exclusive_aclus # adds the appropriate column to the filter_epochs_df
        self.filter_epochs_df['active_unique_aclus'] = epoch_spikes_unique_aclus_list
        return filter_epoch_spikes_df, self.filter_epochs_df


    def determine_if_contains_active_set_exclusive_cells(self, spikes_df: pd.DataFrame, exclusive_aclus, short_exclusive, long_exclusive, included_neuron_ids=None):
        """ NOTE: also updates `filter_epoch_spikes_df` which is returned
    
        adds ['contains_one_exclusive_aclu', 'active_unique_aclus', 'has_SHORT_exclusive_aclu', 'has_LONG_exclusive_aclu'] columns to `filter_epochs_df`
        """
        filter_epoch_spikes_df, self.filter_epochs_df = self._subfn_find_example_epochs(spikes_df, exclusive_aclus=exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
        # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
        self.filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in self.filter_epochs_df['active_unique_aclus']]
        self.filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in self.filter_epochs_df['active_unique_aclus']]
        return filter_epoch_spikes_df, self.filter_epochs_df


    def filter_by_user_selections(self, curr_active_pipeline, allow_interactive_selection:bool = False):
        """Get the manual user annotations to determine the good replays for both long/short decoding
        
        Adds the ['long_is_user_included', 'short_is_user_included'] columns to the `filter_epochs_df` DataFrame
        """
        user_annotation_man = UserAnnotationsManager()
        user_annotations = user_annotation_man.get_user_annotations()

        final_context_L = curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj')
        final_context_S = curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='short_results_obj')
        # _out_pagination_controller.params.active_identifying_figure_ctx.adding_context(None,  user_annotation="selections")
        selections_context_L = final_context_L.adding_context(None,  user_annotation="selections")
        selections_context_S = final_context_S.adding_context(None,  user_annotation="selections")
        
        ## try to get the user annotations for this session:
        try:
            selection_idxs_L = user_annotations[selections_context_L]
            selection_idxs_S = user_annotations[selections_context_S]
        except KeyError as e:
            if allow_interactive_selection:
                print(f'user annotations <good replay selections> are not found. Creating them interactively...')
                user_annotations = interactive_good_epoch_selections(annotations_man=user_annotation_man, curr_active_pipeline=curr_active_pipeline)  # perform interactive selection. Should block here.
                selection_idxs_L = user_annotations[selections_context_L]
                selection_idxs_S = user_annotations[selections_context_S]
            else:
                print(f'interactive annotation is not permitted. Failing.')
                raise e
        except Exception as e:
            print('Unhandled exception: {e}')
            raise

        # for updating the filter_epochs_df (`filter_epochs_df`) from the selections:
        self.filter_epochs_df['long_is_user_included'] = np.isin(self.filter_epochs_df.index, selection_idxs_L)
        self.filter_epochs_df['short_is_user_included'] = np.isin(self.filter_epochs_df.index, selection_idxs_S)


    def _debug_print_assignment_statuses(self, debug_print=True):
        is_unassigned = np.array([v.decision == TrackAssignmentState.UNASSIGNED for v in self.filter_epochs_df.track_assignment])
        is_disregarded = np.array([v.decision == TrackAssignmentState.NEITHER for v in self.filter_epochs_df.track_assignment])
        num_unassigned = np.sum(is_unassigned)
        num_disregarded = np.sum(is_disregarded)
        num_assigned = len(self.filter_epochs_df) - (num_unassigned + num_disregarded)
        if debug_print:
            print(f'num_unassigned: {num_unassigned}, num_disregarded: {num_disregarded}, num_assigned: {num_assigned}')
        return num_unassigned, num_disregarded, num_assigned
    

    def _subfn_plot_epoch_track_assignments(self, axis_idx:int, fig=None, axs=None, defer_render=False):
        """ Plots a figure that represents each epoch as a little box that can be colored in based on the track assignment: grey for Unassigned, blue for Long, red for Short, black for Neither.

        Args:
        - assigning_epochs_obj: an object of type AssigningEpochs

        Returns:
        - None
        """
        def _subfn_setup_each_individual_row_axis(ax, is_bottom_row=False):
            """ captures self 
            """
            # Set the x and y limits of the axis
            ax.set_xlim([0, len(self.filter_epochs_df)])
            ax.set_ylim([0, 1])
            
            # Add vertical grid lines
            for i in range(len(self.filter_epochs_df)):
                ax.axvline(x=i, color='white', linewidth=0.5)

            # Set the title and axis labels
            if is_bottom_row:
                ax.set_title('Replay Track Assignments')
                ax.set_xlabel('Replay Epoch Index')
                
            ax.set_ylabel('') # Track Assignment

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            return ax            

        # Create a figure and axis object if not provided
        if fig is None or axs is None:
            fig, axs = plt.subplots(ncols=1, nrows=5, sharex=True, sharey=True, figsize=(20, 10))
            
            # Set up each individual row axis
            for ax in axs:
                ax = _subfn_setup_each_individual_row_axis(ax, is_bottom_row=(ax == axs[-1]))
            
            # Add legend (only to the bottom axis)
            grey_patch = mpatches.Patch(color='grey', label='Unassigned')
            blue_patch = mpatches.Patch(color='blue', label='Long')
            red_patch = mpatches.Patch(color='red', label='Short')
            black_patch = mpatches.Patch(color='black', label='Neither')
            plt.legend(handles=[grey_patch, blue_patch, red_patch, black_patch], loc='lower center', ncol=4)


        # Get the axis object
        ax = axs[axis_idx]

        # Iterate over each epoch in the filter_epochs_df
        for i, epoch in self.filter_epochs_df.iterrows():
            # Get the track assignment of the epoch
            # track_assignment = epoch['track_assignment'].decision.name
            
            track_assignment = epoch['track_assignment'].decision
            track_assignment_color = track_assignment.displayColor

            # # Set the color of the epoch based on the track assignment
            # if track_assignment == 'UNASSIGNED':
            #     color = 'grey'
            # elif track_assignment == 'LONG_TRACK':
            #     color = 'blue'
            # elif track_assignment == 'SHORT_TRACK':
            #     color = 'red'
            # elif track_assignment == 'NEITHER':
            #     color = 'black'

            # Draw a rectangle for the epoch
            ax.add_patch(plt.Rectangle((i, len(ax.get_yticks())), 1, 1, color=track_assignment_color))

        # Show the plot
        if not defer_render:
            plt.show()
            
        return fig, axs
    

    @classmethod
    def main_plot_iterative_epoch_track_assignments(cls, curr_active_pipeline, defer_render=False):
        """ 2023-06-27 - Test how many can actually be sorted by active_set criteria:
        
        """
        ## First filter by only those user-included replays (in the future this won't be done)

        # from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import TrackAssignmentDecision, TrackAssignmentState, AssigningEpochs
        (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)

        ## Initialize to unassigned
        axis_idx:int = 0
        assigning_epochs_obj.filter_epochs_df['track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.UNASSIGNED, 0.0)
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, defer_render=True)

        # Partition based on whether the user included the epoch in the long or short track in the user-included epochs:
        is_user_exclusive_L = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']))
        is_user_exclusive_S = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included']))
        is_user_unassigned_in_both_epochs = np.logical_and(np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']), np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'])) # the user said it was bad in both epochs, so assign it to neither with high confidence

        # NOTE: be sure to assign to the filter_epochs_df, not the unassigned_epochs_df, because the unassigned_epochs_df is a subset of the filter_epochs_df, and we want to assign to the filter_epochs_df so that the unassigned_epochs_df will be a subset of the filter_epochs_df:
        # assign the user_exclusive_L to long_track:
        assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_L, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 1.0)
        # assign the user_exclusive_S to short_track:
        assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_S, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 1.0)
        # assign the user_unassigned_in_both_epochs to neither:
        assigning_epochs_obj.filter_epochs_df.loc[is_user_unassigned_in_both_epochs, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.NEITHER, 1.0)

        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_S]
        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_L]
        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)


        # Filter based on the active_set cells (LxC, SxC):
        assigning_epochs_obj.unassigned_epochs_df.loc[np.logical_and(assigning_epochs_obj.unassigned_epochs_df['has_LONG_exclusive_aclu'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['has_SHORT_exclusive_aclu'])), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 1.0)
        assigning_epochs_obj._debug_print_assignment_statuses()
        # assign the user_exclusive_S to short_track:
        assigning_epochs_obj.unassigned_epochs_df.loc[np.logical_and(np.logical_not(assigning_epochs_obj.unassigned_epochs_df['has_LONG_exclusive_aclu']), assigning_epochs_obj.unassigned_epochs_df['has_SHORT_exclusive_aclu']), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 1.0)
        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)
        
        if not defer_render:
            fig.show() 
            
        return fig, axs


# ==================================================================================================================== #
# 2023-07-14 - Paper Figure 1                                                                                          #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['FIGURE1', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 21:40', related_items=[])
def PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline, allow_interactive_good_epochs_selection=False, debug_print=False):
    """ 
    
    # general approach copied from `pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations._epoch_unit_avg_firing_rates`
    
    # requires: selection_idxs_L, selection_idxs_S
        - Uses the `long_short_decoding_analyses` global result to access `long_results_obj.active_filter_epochs`:
        - Uses the `JonathanFiringRateAnalysisResult` global result to get info about the long/short placefields:
    
    long_results_obj, short_results_obj 
    """
   
    

    ## Use the `long_short_decoding_analyses` global result to access `long_results_obj.active_filter_epochs`:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    ## Use the `JonathanFiringRateAnalysisResult` to get info about the long/short placefields:
    jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
    neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions()

    assigning_epochs_obj = AssigningEpochs(filter_epochs_df=deepcopy(long_results_obj.active_filter_epochs.to_dataframe()))

    # included_neuron_ids = deepcopy(exclusive_aclus)
    # included_neuron_ids = None
    included_neuron_ids = EITHER_subset.track_exclusive_aclus
    spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
    # filter_epochs_df = deepcopy(long_results_obj.active_filter_epochs.to_dataframe())
    filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj.determine_if_contains_active_set_exclusive_cells(spikes_df, exclusive_aclus=EITHER_subset.track_exclusive_aclus, short_exclusive=short_exclusive, long_exclusive=long_exclusive, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'    
    # filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj._find_example_epochs(spikes_df, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
    # # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
    # filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]
    # filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]

    # Get the manual user annotations to determine the good replays for both long/short decoding:
    assigning_epochs_obj.filter_by_user_selections(curr_active_pipeline=curr_active_pipeline, allow_interactive_selection=allow_interactive_good_epochs_selection)
    

    #### Finally, get only the epochs that meet the criteria:

    # # only include those with one or more exclusive aclu:
    # considered_filter_epochs_df = filter_epochs_df[filter_epochs_df.contains_one_exclusive_aclu].copy()
    # # require inclusion for long or short:
    # considered_filter_epochs_df = considered_filter_epochs_df[np.logical_xor(filter_epochs_df['long_is_user_included'], filter_epochs_df['short_is_user_included'])]

    # Get separate long-side/short-side candidate replays:
    epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'] & filter_epochs_df['long_is_user_included'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
    epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'] & filter_epochs_df['short_is_user_included'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.

    # Common for all rasters:
    new_all_aclus_sort_indicies = determine_long_short_pf1D_indicies_sort_by_peak(curr_active_pipeline=curr_active_pipeline, curr_any_context_neurons=EITHER_subset.track_exclusive_aclus)

    # Build one spikes_df for Long and Short:
    filter_epoch_spikes_df_L, filter_epoch_spikes_df_S = [_prepare_spikes_df_from_filter_epochs(filter_epoch_spikes_df, filter_epochs=an_epochs_df, included_neuron_ids=EITHER_subset.track_exclusive_aclus, epoch_id_key_name='replay_epoch_id', debug_print=False) for an_epochs_df in (epochs_df_L, epochs_df_S)]    

    # requires epochs_df_L, epochs_df_S from `PAPER_FIGURE_figure_1_add_replay_epoch_rasters`
    # requires epochs_df_L, epochs_df_S from `PAPER_FIGURE_figure_1_add_replay_epoch_rasters`
    # get the good epoch indicies from epoch_df_L.Index
    good_example_epoch_indicies_L = epochs_df_L.index.to_numpy()
    # good_epoch_indicies_L = np.array([15, 49])
    good_example_epoch_indicies_S = epochs_df_S.index.to_numpy()
    # good_epoch_indicies_S = np.array([ 31,  49,  55,  68,  70,  71,  73,  77,  78,  89,  94, 100, 104, 105, 111, 114, 117, 118, 122, 123, 131])
    if debug_print:
        print(f'good_example_epoch_indicies_L: {good_example_epoch_indicies_L}')
        print(f'good_example_epoch_indicies_S: {good_example_epoch_indicies_S}')


    return (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj


@function_attributes(short_name=None, tags=['FINAL', 'publication', 'figure', 'combined'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-21 14:33', related_items=[])
def PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=False, save_figure=True, should_plot_pf1d_compare=True, should_plot_example_rasters=True, should_plot_stacked_epoch_slices=True, should_plot_pho_jonathan_figures=True):
    ## long_short_decoding_analyses:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    ## Extract variables from results object:
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    # (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, recalculate_anyway=True)
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
    long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D
    decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333

    ## Get global 'long_short_post_decoding' results:
    # curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    # expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
    # rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
    # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result['Flat_epoch_time_bins_mean'], expected_v_observed_result['Flat_decoder_time_bin_centers'], expected_v_observed_result['num_neurons'], expected_v_observed_result['num_timebins_in_epoch'], expected_v_observed_result['num_total_flat_timebins'], expected_v_observed_result['is_short_track_epoch'], expected_v_observed_result['is_long_track_epoch'], expected_v_observed_result['short_short_diff'], expected_v_observed_result['long_long_diff']

    # ==================================================================================================================== #
    # Show 1D Placefields for both Short and Long (top half of the figure)                                                 #
    # ==================================================================================================================== #
    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

    if should_plot_pf1d_compare:
        long_single_cell_pfmap_processing_fn = None
        short_single_cell_pfmap_processing_fn = None
        # sort_idx = None
        # sort_idx = sort_ind
        # sort_idx = sort_indicies
        # sort_idx = new_all_aclus_sort_indicies
        sort_idx = 'peak_long'
        # sort_idx = 'bad'

        pf1d_compare_graphics = curr_active_pipeline.display('_display_short_long_pf1D_comparison', active_identifying_session_ctx, single_figure=False, debug_print=False, fignum='Short v Long pf1D Comparison',
                                        long_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': long_single_cell_pfmap_processing_fn},
                                        short_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': short_single_cell_pfmap_processing_fn, 'curve_hatch_style': {'hatch':'///', 'edgecolor':'k'}},
                                        shared_kwargs={'cmap': 'hsv'},
                                        save_figure=save_figure,
                                        defer_render=defer_show
                                        )
                                        
    else:
        pf1d_compare_graphics = None


    # ==================================================================================================================== #
    # Show Example Replay Epochs containing the long or short only cells                                                                  #
    # ==================================================================================================================== #

    (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


    if should_plot_example_rasters:
        # unit_colors_list = None # default rainbow of colors for the raster plots
        neuron_qcolors_list = [pg.mkColor('black') for aclu in EITHER_subset.track_exclusive_aclus] # solid green for all
        unit_colors_list = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)

        # Copy and modify the colors for the cells that are long/short exclusive:
        unit_colors_list_L = deepcopy(unit_colors_list)
        is_L_exclusive = np.isin(EITHER_subset.track_exclusive_aclus, long_exclusive.track_exclusive_aclus) # get long exclusive
        unit_colors_list_L[0, is_L_exclusive] = 255 # [1.0, 0.0, 0.0, 1.0]
        unit_colors_list_L[1, is_L_exclusive] = 0.0
        unit_colors_list_L[2, is_L_exclusive] = 0.0
        
        unit_colors_list_S = deepcopy(unit_colors_list)
        is_S_exclusive = np.isin(EITHER_subset.track_exclusive_aclus, short_exclusive.track_exclusive_aclus) # get short exclusive
        unit_colors_list_S[0, is_S_exclusive] = 0.0 # [1.0, 0.0, 0.0, 1.0]
        unit_colors_list_S[1, is_S_exclusive] = 0.0
        unit_colors_list_S[2, is_S_exclusive] = 255.0
        
        
        ## Build scatterplot args:    
        # Common Tick Label
        vtick = pg.QtGui.QPainterPath()

        # Thicker Tick Label:
        tick_width = 0.25
        # tick_width = 10.0
        half_tick_width = 0.5 * tick_width
        vtick.moveTo(-half_tick_width, -0.5)
        vtick.addRect(-half_tick_width, -0.5, tick_width, 1.0) # x, y, width, height
        pen = {'color': 'white', 'width': 1}
        override_scatter_plot_kwargs = dict(pxMode=True, symbol=vtick, size=6, pen=pen)

        # Not sure if needed:
        # filter_epoch_spikes_df_L.spikes.rebuild_fragile_linear_neuron_IDXs()

        example_epoch_rasters_L = plot_multiple_raster_plot(epochs_df_L, filter_epoch_spikes_df_L, included_neuron_ids=EITHER_subset.track_exclusive_aclus, unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_L, scatter_plot_kwargs=override_scatter_plot_kwargs,
                                            epoch_id_key_name='replay_epoch_id', scatter_app_name="Long Decoded Example Replays", defer_show=defer_show,
                                            active_context=curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track='long', epoch='example_replays'))
        app_L, win_L, plots_L, plots_data_L = example_epoch_rasters_L
        if save_figure:
            curr_active_pipeline.output_figure(plots_data_L.active_context, win_L) 

        example_epoch_rasters_S = plot_multiple_raster_plot(epochs_df_S, filter_epoch_spikes_df_S, included_neuron_ids=EITHER_subset.track_exclusive_aclus, unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_S, scatter_plot_kwargs=override_scatter_plot_kwargs,
                                                                        epoch_id_key_name='replay_epoch_id', scatter_app_name="Short Decoded Example Replays", defer_show=defer_show,
                                                                        active_context=curr_active_pipeline.build_display_context_for_session('plot_multiple_raster_plot', fig=1, track='short', epoch='example_replays'))
        app_S, win_S, plots_S, plots_data_S = example_epoch_rasters_S
        if save_figure:
            curr_active_pipeline.output_figure(plots_data_S.active_context, win_S)
    else:
        example_epoch_rasters_L, example_epoch_rasters_S = None, None
        

    ## Stacked Epoch Plot: this is the slowest part of the figure by far as it renders in a scrollable pyqtgraph window.
    if should_plot_stacked_epoch_slices:
        example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=defer_show, save_figure=save_figure)
    else: 
        example_stacked_epoch_graphics = None


    # ==================================================================================================================== #
    # Fig 1c) 2023-07-14 - LxC and SxC PhoJonathanSession plots                                                            #
    # ==================================================================================================================== #

    ## Get global 'jonathan_firing_rate_analysis' results:
    curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis['neuron_replay_stats_df'], curr_jonathan_firing_rate_analysis['rdf']['rdf'], curr_jonathan_firing_rate_analysis['rdf']['aclu_to_idx'], curr_jonathan_firing_rate_analysis['irdf']['irdf']

    if should_plot_pho_jonathan_figures:
        fig_1c_figures_out_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=XOR_subset.track_exclusive_aclus, n_max_page_rows=20, write_vector_format=False, write_png=save_figure) # active_out_figures_dict: {IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'BatchPhoJonathanReplayFRC', 'long_only', '(12,21,48)')>: <Figure size 1920x660 with 12 Axes>, IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'BatchPhoJonathanReplayFRC', 'short_only', '(18,19,65)')>: <Figure size 1920x660 with 12 Axes>}
    else:
        fig_1c_figures_out_dict = None

    return pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics, fig_1c_figures_out_dict

# ==================================================================================================================== #
# 2023-06-26 - Paper Figure 2 Code                                                                                     #
# ==================================================================================================================== #

@define(slots=False)
class SingleBarResult:
    """ a simple replacement for the tuple that's current passed """
    mean: float
    std: float
    values: np.ndarray
    LxC_aclus: np.ndarray # the list of long-eXclusive cell aclus
    SxC_aclus: np.ndarray # the list of short-eXclusive cell aclus
    LxC_scatter_props: Optional[Dict]
    SxC_scatter_props: Optional[Dict]


# Instantaneous versions:
@define(slots=False)
class InstantaneousSpikeRateGroupsComputation:
    """ class to handle spike rate computations 

    from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import InstantaneousSpikeRateGroupsComputation

    """
    instantaneous_time_bin_size_seconds: float = 0.01  # 20ms
    active_identifying_session_ctx: IdentifyingContext = field(init=False)

    LxC_aclus: np.ndarray = field(init=False) # the list of long-eXclusive cell aclus
    SxC_aclus: np.ndarray = field(init=False) # the list of short-eXclusive cell aclus
    
    Fig2_Replay_FR: List[SingleBarResult] = field(init=False) # a list of the four single-bar results.
    Fig2_Laps_FR: List[SingleBarResult] = field(init=False) # a list of the four single-bar results.

    LxC_ReplayDeltaMinus: SpikeRateTrends = field(init=False, repr=False, default=None)
    LxC_ReplayDeltaPlus: SpikeRateTrends = field(init=False, repr=False, default=None)
    SxC_ReplayDeltaMinus: SpikeRateTrends = field(init=False, repr=False, default=None)
    SxC_ReplayDeltaPlus: SpikeRateTrends = field(init=False, repr=False, default=None)

    LxC_ThetaDeltaMinus: SpikeRateTrends = field(init=False, repr=False, default=None)
    LxC_ThetaDeltaPlus: SpikeRateTrends = field(init=False, repr=False, default=None)
    SxC_ThetaDeltaMinus: SpikeRateTrends = field(init=False, repr=False, default=None)
    SxC_ThetaDeltaPlus: SpikeRateTrends = field(init=False, repr=False, default=None)

    def compute(self, curr_active_pipeline, **kwargs):
        """ full instantaneous computations for both Long and Short epochs:
        
        Can access via:
            from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import InstantaneousSpikeRateGroupsComputation

            _out_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
        
        """
        sess = curr_active_pipeline.sess 
        # Get the provided context or use the session context:
        active_context = kwargs.get('active_context', sess.get_context()) 

        self.active_identifying_session_ctx = active_context
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # only uses global_session
        
        ## Use the `JonathanFiringRateAnalysisResult` to get info about the long/short placefields:
        jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
        neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions()

        long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
        long_laps, long_replays, short_laps, short_replays, global_laps, global_replays = [long_short_fr_indicies_analysis_results[k] for k in ['long_laps', 'long_replays', 'short_laps', 'short_replays', 'global_laps', 'global_replays']]

        # Store the Long and Short exclusive ACLUs:
        self.LxC_aclus = long_exclusive.track_exclusive_aclus
        self.SxC_aclus = short_exclusive.track_exclusive_aclus

        # Replays: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_replays`, `short_replays`
        # LxC: `long_exclusive.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        LxC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ReplayDeltaPlus: `short_replays`
        LxC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_replays, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        # SxC: `short_exclusive.track_exclusive_aclus`
        # ReplayDeltaMinus: `long_replays`
        SxC_ReplayDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ReplayDeltaPlus: `short_replays`
        SxC_ReplayDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_replays, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        self.LxC_ReplayDeltaMinus, self.LxC_ReplayDeltaPlus, self.SxC_ReplayDeltaMinus, self.SxC_ReplayDeltaPlus = LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus

        # Note that in general LxC and SxC might have differing numbers of cells.
        # self.Fig2_Replay_FR: list[tuple[Any, Any]] = [(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list) for v in (LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus)]
        self.Fig2_Replay_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, self.LxC_aclus, self.SxC_aclus, None, None) for v in (LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus)]
        
        # Laps/Theta: Uses `global_session.spikes_df`, `long_exclusive.track_exclusive_aclus, `short_exclusive.track_exclusive_aclus`, `long_laps`, `short_laps`
        # LxC: `long_exclusive.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        LxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_laps, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ThetaDeltaPlus: `short_laps`
        LxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_laps, included_neuron_ids=long_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        # SxC: `short_exclusive.track_exclusive_aclus`
        # ThetaDeltaMinus: `long_laps`
        SxC_ThetaDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_laps, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        # ThetaDeltaPlus: `short_laps`
        SxC_ThetaDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_laps, included_neuron_ids=short_exclusive.track_exclusive_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)

        self.LxC_ThetaDeltaMinus, self.LxC_ThetaDeltaPlus, self.SxC_ThetaDeltaMinus, self.SxC_ThetaDeltaPlus = LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus

        # Note that in general LxC and SxC might have differing numbers of cells.
        self.Fig2_Laps_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, self.LxC_aclus, self.SxC_aclus, None, None) for v in (LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus)]
        


# @overwriting_display_context(
@metadata_attributes(short_name=None, tags=['figure_2', 'paper', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-26 21:36', related_items=[])
@define(slots=False, repr=False)
class PaperFigureTwo(SerializedAttributesAllowBlockSpecifyingClass):
    """ full instantaneous firing rate computations for both Long and Short epochs
    
    Usage:
        _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
        _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline)
        _out_fig_2.display()

    """

    instantaneous_time_bin_size_seconds: float = 0.01 # 10ms
    
    computation_result: InstantaneousSpikeRateGroupsComputation = field(init=False)
    active_identifying_session_ctx: IdentifyingContext = field(init=False)

    _pipeline_file_callback_fn: Callable = field(init=False, repr=False, default=None) # this callback is 2737.983306 MB!!


    @classmethod
    def get_bar_colors(cls):
        return [(1.0, 0, 0, 1), (0.65, 0, 0, 1), (0, 0, 0.65, 1), (0, 0, 1.0, 1)]    # corresponding colors


    def compute(self, curr_active_pipeline, **kwargs):
        """ full instantaneous computations for both Long and Short epochs:
        
        Can access via:
            from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import PaperFigureTwo

            _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            out_inst_fr_comps = _out_fig_2.computation_result
            LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
        
        """
        self.computation_result = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds)
        self.computation_result.compute(curr_active_pipeline, **kwargs)
        self.active_identifying_session_ctx = self.computation_result.active_identifying_session_ctx

        # Set callback, the only self-specific property
        self._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)
        
    
    @classmethod
    def _build_formatted_title_string(cls, epochs_name) -> str:
        """ buidls the two line colored string that is passed into `flexitext`.
        """
        return (f"<size:22><weight:bold>{epochs_name}</> Firing Rates\n"
                "<size:14>for the "
                "<color:crimson, weight:bold>Long</>/<color:royalblue, weight:bold>Short</> eXclusive Cells on each track</></>"
                )


    @classmethod
    def create_plot(cls, x_labels, y_values, scatter_props, ylabel, title, fig_name, active_context, defer_show, title_modifier=None):
        """ """
        if title_modifier:
            title = title_modifier(title)

        # fig, ax, bars, scatter_plots, title_text_obj, footer_text_obj, plot_data = cls.create_bar_plot(x_labels, y_values, scatter_props, active_context, ylabel, title)
        text_formatter = FormattedFigureText()
        x = np.arange(len(x_labels))
        width = 0.3

        fig, ax = plt.subplots()
        text_formatter.setup_margins(fig)

        bars = ax.bar(x, [np.mean(yi) for yi in y_values], yerr=[np.std(yi) for yi in y_values], capsize=5, width=width, tick_label=x_labels, color=(0, 0, 0, 0), edgecolor=cls.get_bar_colors())

        scatter_plots = []
        x_values_list = []
        
        if hasattr(cls, 'scatter_props_fn'):
            # get individual scatter_props_Fn
            _output = cls.scatter_props_fn(active_context)
            print(f'_output: {_output}')


        for i in range(len(x)):
            x_values = (x[i] + np.random.random(y_values[i].size) * width - width / 2)
            scatter_plot = ax.scatter(x_values, y_values[i], color=cls.get_bar_colors()[i], **list_of_dicts_to_dict_of_lists(scatter_props[i])) # the np.random part is to spread the points out along the x-axis within their bar so they're visible and don't overlap.
            scatter_plots.append(scatter_plot)
            x_values_list.append(x_values)
            
        ax.set_xlabel('Groups')
        ax.set_ylabel(ylabel)
        # Hide the right and top spines (box components)
        ax.spines[['right', 'top']].set_visible(False)
        
        title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin,
                                cls._build_formatted_title_string(epochs_name=title), va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext(text_formatter.left_margin * 0.1, text_formatter.bottom_margin * 0.25,
                                    text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        plot_data = (x_values_list, y_values)
        
        if not defer_show:
            plt.show()

        return MatplotlibRenderPlots(name=fig_name, figures=[fig], axes=[ax], context=active_context,
                                    plot_objects={'bars': bars, 'scatter_plots': scatter_plots, 'text_objects': {'title': title_text_obj, 'footer': footer_text_obj}},
                                    plot_data=plot_data)


    # @providing_context(fig='F2', frs='Laps')
    @classmethod
    def fig_2_Theta_FR_matplotlib(cls, Fig2_Laps_FR, defer_show=False, **kwargs) -> MatplotlibRenderPlots:
        active_context = kwargs.get('active_context', None)
        assert active_context is not None

        x_labels = ['$L_x C$\t$\\theta_{\\Delta -}$', '$L_x C$\t$\\theta_{\\Delta +}$', '$S_x C$\t$\\theta_{\\Delta -}$', '$S_x C$\t$\\theta_{\\Delta +}$']
        all_data_points = np.array([v.values for v in Fig2_Laps_FR])
        # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 

        if Fig2_Laps_FR[0].LxC_scatter_props is not None:
            # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 
            all_scatter_props =  [Fig2_Laps_FR[0].LxC_scatter_props, Fig2_Laps_FR[1].LxC_scatter_props, Fig2_Laps_FR[2].SxC_scatter_props, Fig2_Laps_FR[3].SxC_scatter_props]

        else:
            all_scatter_props = [{}, {}, {}, {}]

        return cls.create_plot(x_labels, all_data_points, all_scatter_props, 'Laps Firing Rates (Hz)', 'Lap ($\\theta$)', 'fig_2_Theta_FR_matplotlib', active_context, defer_show, kwargs.get('title_modifier'))


    # @providing_context(fig='F2', frs='Replay')
    @classmethod
    def fig_2_Replay_FR_matplotlib(cls, Fig2_Replay_FR, defer_show=False, **kwargs) -> MatplotlibRenderPlots:
        active_context = kwargs.get('active_context', None)
        assert active_context is not None

        x_labels = ['$L_x C$\t$R_{\\Delta -}$', '$L_x C$\t$R_{\\Delta +}$', '$S_x C$\t$R_{\\Delta -}$', '$S_x C$\t$R_{\\Delta +}$']
        assert len(Fig2_Replay_FR) == 4
        all_data_points = np.array([v.values for v in Fig2_Replay_FR])
        all_scatter_props = Fig2_Replay_FR[0].LxC_scatter_props + Fig2_Replay_FR[1].LxC_scatter_props + Fig2_Replay_FR[2].SxC_scatter_props + Fig2_Replay_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 
        
        # label_list = [LxC_aclus, LxC_aclus, SxC_aclus, SxC_aclus]
        return cls.create_plot(x_labels, all_data_points, all_scatter_props, 'Replay Firing Rates (Hz)', 'Replay', 'fig_2_Replay_FR_matplotlib', active_context, defer_show, kwargs.get('title_modifier'))

    @classmethod
    def add_optional_aclu_labels(cls, a_fig_container, LxC_aclus, SxC_aclus, enable_hover_labels=True, enable_tiny_point_labels=True):
        """ Adds disambiguating labels to each of the scatterplot points. Important for specifying which ACLU is plotted.


        Parameters:
            enable_hover_labels = True # add interactive point hover labels using mplcursors
            enable_tiny_point_labels = True # add static tiny aclu labels beside each point

        Uses:
            a_fig_container['plot_objects']['scatter_plots']
            a_fig_container['plot_data']
            
        Modifies a_fig_container, adding:
            a_fig_container['plot_objects']['tiny_annotation_labels']
            a_fig_container['plot_objects']['hover_label_objects']
        """
        if enable_hover_labels:
            import mplcursors # for hover tooltips that specify the aclu of the selected point

        # LxC_aclus = _out_fig_2.computation_result.LxC_aclus
        # SxC_aclus = _out_fig_2.computation_result.SxC_aclus
        label_list = [LxC_aclus, LxC_aclus, SxC_aclus, SxC_aclus]
            
        fig = a_fig_container.figures[0]
        ax = a_fig_container.axes[0] # one shared axis per figure
        plot_data = a_fig_container['plot_data']
        x_values_list, y_values_list = plot_data # four lists of x and y values,

        assert len(x_values_list) == 4
        assert len(y_values_list) == 4
        
        # Loop through the four bars in the container:
        if enable_tiny_point_labels:
            a_fig_container['plot_objects']['tiny_annotation_labels'] = []
        if enable_hover_labels:
            a_fig_container['plot_objects']['hover_label_objects'] = []
        
        if enable_hover_labels or enable_tiny_point_labels:
            for bar_idx, a_scatter_plot, x_values, y_values, active_labels in zip(np.arange(4), a_fig_container['plot_objects']['scatter_plots'], x_values_list, y_values_list, label_list): # four scatter plots ( one for each group/bar)
                point_hover_labels = [f'{i}' for i in active_labels] # point_hover_labels will be added as tooltip annotations to the datapoints
                assert len(x_values) == len(y_values) and len(x_values) == len(point_hover_labels), f"len(x_values): {len(x_values)}, len(y_values): {len(y_values)}, len(point_hover_labels): {len(point_hover_labels)}"
                # add static tiny labels beside each point
                if enable_tiny_point_labels:
                    temp_annotation_labels_list = []
                    for i, (x, y, label) in enumerate(zip(x_values, y_values, point_hover_labels)):
                        # print(f'{i}, (x, y, label): ({x}, {y}, {label})')
                        annotation_item = ax.annotate(label, (x, y), textcoords="offset points", xytext=(2,2), ha='left', va='bottom', fontsize=8) # , color=rect.get_facecolor()
                        temp_annotation_labels_list.append(annotation_item)
                    a_fig_container['plot_objects']['tiny_annotation_labels'].append(temp_annotation_labels_list)
                    
                # add hover labels:
                # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
                # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib/21654635#21654635
                # add hover labels using mplcursors
                if enable_hover_labels:
                    hover_label_obj = mplcursors.cursor(a_scatter_plot, hover=True).connect("add", lambda sel: sel.annotation.set_text(point_hover_labels[sel.index]))
                    a_fig_container['plot_objects']['hover_label_objects'].append(hover_label_obj)

        return a_fig_container


    @providing_context(fig='2', display_fn_name='inst_FR_bar_graphs')
    def display(self, defer_show=False, save_figure=True, enable_tiny_point_labels=True, enable_hover_labels=False, **kwargs):
        """ 
        
        title_modifier: lambda original_title: f"{original_title} (all sessions)"

        """
        # Get the provided context or use the session context:
        active_context = kwargs.get('active_context', self.active_identifying_session_ctx)
        title_modifier = kwargs.get('title_modifier_fn', (lambda original_title: original_title))
        top_margin, left_margin, bottom_margin = kwargs.get('top_margin', 0.8), kwargs.get('left_margin', 0.090), kwargs.get('bottom_margin', 0.150)

        _fig_2_theta_out = self.fig_2_Theta_FR_matplotlib(self.computation_result.Fig2_Laps_FR, defer_show=defer_show,
                                                        active_context=active_context, top_margin=top_margin,
                                                        left_margin=left_margin, bottom_margin=bottom_margin,
                                                        title_modifier=title_modifier)

        _fig_2_replay_out = self.fig_2_Replay_FR_matplotlib(self.computation_result.Fig2_Replay_FR, defer_show=defer_show,
                                                            active_context=active_context, top_margin=top_margin,
                                                            left_margin=left_margin, bottom_margin=bottom_margin,
                                                            title_modifier=title_modifier)


        if enable_hover_labels or enable_tiny_point_labels:
            LxC_aclus = self.computation_result.LxC_aclus
            SxC_aclus = self.computation_result.SxC_aclus
            _fig_2_theta_out = self.add_optional_aclu_labels(_fig_2_theta_out, LxC_aclus, SxC_aclus, enable_tiny_point_labels=enable_tiny_point_labels, enable_hover_labels=enable_hover_labels)
            _fig_2_replay_out = self.add_optional_aclu_labels(_fig_2_replay_out, LxC_aclus, SxC_aclus, enable_tiny_point_labels=enable_tiny_point_labels, enable_hover_labels=enable_hover_labels)
        
        def _perform_write_to_file_callback():
            ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            return (self.perform_save(_fig_2_theta_out.context, _fig_2_theta_out.figures[0]), 
                    self.perform_save(_fig_2_replay_out.context, _fig_2_replay_out.figures[0]))

        if save_figure:
            _fig_2_theta_out['saved_figures'], _fig_2_replay_out['saved_figures'] = _perform_write_to_file_callback()
        else:
            _fig_2_theta_out['saved_figures'], _fig_2_replay_out['saved_figures'] = [], []

        # Merge the two (_fig_2_theta_out | _fig_2_replay_out)
        return (_fig_2_theta_out, _fig_2_replay_out)


    def perform_save(self, *args, **kwargs):
        """ used to save the figure without needing a hard reference to curr_active_pipeline """
        assert self._pipeline_file_callback_fn is not None
        return self._pipeline_file_callback_fn(*args, **kwargs) # call the saved callback


# 2023-07-21 - Across Sessions Aggregate Figure: __________________________________________________________________________________ #

def across_sessions_bar_graphs(across_session_inst_fr_computation: Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation], num_sessions:int, **kwargs):
    ## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.

    # num_sessions = len(across_sessions_instantaneous_fr_dict)
    print(f'num_sessions: {num_sessions}')

    global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.

    # To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).

    ## Display the aggregate across sessions:
    _out_aggregate_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # WARNING: we didn't save this info
    _out_aggregate_fig_2.computation_result = across_session_inst_fr_computation
    _out_aggregate_fig_2.active_identifying_session_ctx = across_session_inst_fr_computation.active_identifying_session_ctx
    # Set callback, the only self-specific property
    # _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)

    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FigureOutputManager, FigureOutputLocation, ContextToPathMode

    registered_output_files = {}

    def output_figure(final_context: IdentifyingContext, fig, write_vector_format:bool=False, write_png:bool=True, debug_print=True):
        """ outputs the figure using the provided context. """
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_and_write_to_file
        def register_output_file(output_path, output_metadata=None):
            """ registers a new output file for the pipeline """
            print(f'register_output_file(output_path: {output_path}, ...)')
            registered_output_files[output_path] = output_metadata or {}

        fig_out_man = FigureOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.HIERARCHY_UNIQUE)
        active_out_figure_paths = build_and_write_to_file(fig, final_context, fig_out_man, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=register_output_file)
        return active_out_figure_paths, final_context


    # Set callback, the only self-specific property
    _out_aggregate_fig_2._pipeline_file_callback_fn = output_figure

    # Showing
    matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
    # Perform interactive Matplotlib operations with 'Qt5Agg' backend
    _fig_2_theta_out, _fig_2_replay_out = _out_aggregate_fig_2.display(active_context=global_multi_session_context, title_modifier_fn=lambda original_title: f"{original_title} ({num_sessions} sessions)", save_figure=True, **kwargs)
        
    _out_aggregate_fig_2.perform_save()

    global_multi_session_context, _out_aggregate_fig_2

# ==================================================================================================================== #
# 2023-06-26 Figure 3                                                                                                  #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['figure_3', 'paper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-27 03:03', related_items=[])
def PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True):
    """ Firing rate index, Long|Short firing rate figures 

        Renders 3 Subfigures:
            a) Shows the firing rate index between the long and short track computed for two different periods: the laps along the x-axis and the replays along the y-axis.
            b) The ratio of lap to replay firing rate on the long track.
            c) The ratio of lap to replay firing rate on the short track.
    
    from PendingNotebookCode import PAPER_FIGURE_figure_3
    _out, _out2 = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=False, save_figure=True)
    
    """
    _out = curr_active_pipeline.display('_display_short_long_firing_rate_index_comparison', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)
    _out2 = curr_active_pipeline.display('_display_long_and_short_firing_rate_replays_v_laps', curr_active_pipeline.get_session_context(), defer_render=defer_render, save_figure=save_figure)

    return (_out, _out2)






# ==================================================================================================================== #
# MAIN RUN FUNCTION TO GENERATE ALL FIGURES                                                                            #
# ==================================================================================================================== #
def main_complete_figure_generations(curr_active_pipeline, enable_default_neptune_plots:bool=True, save_figures_only:bool=False, save_figure=True):
    """ main run function to generate all figures
    
        from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import main_complete_figure_generations
        main_complete_figure_generations(curr_active_pipeline)
        
    
    """

    if save_figures_only:
        defer_show = True
    else:
        defer_show = False
        
    curr_active_pipeline.reload_default_display_functions()

    # ==================================================================================================================== #
    # Perform `batch_perform_all_plots`                                                                                    #
    # ==================================================================================================================== #
    if enable_default_neptune_plots:
        neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True)

    # ==================================================================================================================== #
    # Extract Relevent Specific Data Needed for Figure Display                                                             #
    # ==================================================================================================================== #
    ## long_short_decoding_analyses:
    curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
    ## Extract variables from results object:
    long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

    # (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, recalculate_anyway=True)
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
    long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D
    decoding_time_bin_size = long_one_step_decoder_1D.time_bin_size # 1.0/30.0 # 0.03333333333333333

    ## Get global 'long_short_post_decoding' results:
    curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
    expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
    rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
    Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result['Flat_epoch_time_bins_mean'], expected_v_observed_result['Flat_decoder_time_bin_centers'], expected_v_observed_result['num_neurons'], expected_v_observed_result['num_timebins_in_epoch'], expected_v_observed_result['num_total_flat_timebins'], expected_v_observed_result['is_short_track_epoch'], expected_v_observed_result['is_long_track_epoch'], expected_v_observed_result['short_short_diff'], expected_v_observed_result['long_long_diff']

    jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
    (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


    # ==================================================================================================================== #
    # Figure 1) pf1D Ratemaps, Active set, etc                                                                             #
    # ==================================================================================================================== #
    _out_fig_1 = PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=defer_show, save_figure=save_figure) # did not display the pf1



    # Critical new code: Not used anyhwere
    ratemap = long_pf1D.ratemap
    included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
    rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())


    # ==================================================================================================================== #
    # Figure 2) Firing Rate Bar Graphs                                                                                     #
    # ==================================================================================================================== #


    # Instantaneous versions:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends

    _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # 10ms
    _out_fig_2.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
    _out_fig_2.display(defer_show=defer_show, save_figure=save_figure, active_context=curr_active_pipeline.sess.get_context())


    # ==================================================================================================================== #
    # Figure 3) `PAPER_FIGURE_figure_3`: Firing Rate Index and Long/Short Firing Rate Replays v. Laps                      #
    # ==================================================================================================================== #
    _out_fig_3_a, _out_fig_3_b = PAPER_FIGURE_figure_3(curr_active_pipeline, defer_render=defer_show, save_figure=save_figure)

    # # ==================================================================================================================== #
    # # HELPERS: Interactive Components                                                                                      #
    # # ==================================================================================================================== #
    # from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
    # # fig, ax, rect_selector, set_extents, reset_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)

    # grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True, should_apply_updates_to_pipeline=False)
    # print(f'grid_bin_bounds: {grid_bin_bounds}')
    # print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")

    # # ==================================================================================================================== #
    # # DEBUGGING:                                                                                                           #
    # # ==================================================================================================================== #
    # ### Testing `plot_kourosh_activity_style_figure` for debugging:
    # from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import plot_kourosh_activity_style_figure

    # # plot_aclus = EITHER_subset.track_exclusive_aclus.copy()
    # plot_aclus = EITHER_subset.track_exclusive_aclus[new_all_aclus_sort_indicies].copy()
    # _out_A = plot_kourosh_activity_style_figure(long_results_obj, long_session, plot_aclus, unit_sort_order=new_all_aclus_sort_indicies, epoch_idx=13, callout_epoch_IDXs=None, skip_rendering_callouts=False)
    # app, win, plots, plots_data = _out_A

    # Unwrapping:
    # pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = _out_fig_1
    if not save_figures_only:
        # only in active display mode is there something to return:
        return (_out_fig_1, _out_fig_2, _out_fig_3_a, _out_fig_3_b)
    
    # plots


