from copy import deepcopy
from pathlib import Path
from typing import Dict, Callable, List, Optional, Tuple
from attrs import define, field
from nptyping import NDArray
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.patches as mpatches # used for plot_epoch_track_assignments
from flexitext import flexitext ## flexitext version

from neuropy.utils.mixins.enum_helpers import ExtendedEnum # used in TrackAssignmentState
from neuropy.core.epoch import Epoch
from neuropy.core.user_annotations import UserAnnotationsManager, SessionCellExclusivityRecord

from pyphocorehelpers.mixins.key_value_hashable import KeyValueHashableObject
from pyphocorehelpers.indexing_helpers import partition # needed by `AssigningEpochs` to partition the dataframe by aclus

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences

## Laps Stuff:
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass

from neuropy.utils.result_context import IdentifyingContext, providing_context, DisplaySpecifyingIdentifyingContext
from neuropy.core.user_annotations import UserAnnotationsManager

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import determine_long_short_pf1D_indicies_sort_by_peak
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from neuropy.utils.matplotlib_helpers import FormattedFigureText


from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots, BatchPhoJonathanFiguresHelper
from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import interactive_good_epoch_selections

import pyphoplacecellanalysis.External.pyqtgraph as pg # pyqtgraph
import matplotlib.pyplot as plt

from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import fig_surprise_results, fig_remapping_cells
from pyphocorehelpers.indexing_helpers import list_of_dicts_to_dict_of_lists

from attrs import define, field, Factory  # Import the attrs library
import ipywidgets as widgets
from IPython.display import display
from copy import deepcopy
from IPython import get_ipython  # Import get_ipython to access the user namespace

# Testing:


_bak_rcParams = mpl.rcParams.copy()
# mpl.rcParams['toolbar'] = 'None' # disable toolbars
# %matplotlib qt


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


    @function_attributes(short_name=None, tags=['user-annotations','selections'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-04 15:30', related_items=[])
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

        # from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import TrackAssignmentDecision, TrackAssignmentState, AssigningEpochs
        (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)

        ## Initialize to unassigned
        axis_idx:int = 0
        assigning_epochs_obj.filter_epochs_df['track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.UNASSIGNED, 0.0)
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, defer_render=True)

        ## User Assignment:
        # # Partition based on whether the user included the epoch in the long or short track in the user-included epochs:
        # is_user_exclusive_L = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']))
        # is_user_exclusive_S = np.logical_and(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included'], np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included']))
        # is_user_unassigned_in_both_epochs = np.logical_and(np.logical_not(assigning_epochs_obj.unassigned_epochs_df['short_is_user_included']), np.logical_not(assigning_epochs_obj.unassigned_epochs_df['long_is_user_included'])) # the user said it was bad in both epochs, so assign it to neither with high confidence

        # # NOTE: be sure to assign to the filter_epochs_df, not the unassigned_epochs_df, because the unassigned_epochs_df is a subset of the filter_epochs_df, and we want to assign to the filter_epochs_df so that the unassigned_epochs_df will be a subset of the filter_epochs_df:
        # # assign the user_exclusive_L to long_track:
        # assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_L, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 1.0)
        # # assign the user_exclusive_S to short_track:
        # assigning_epochs_obj.filter_epochs_df.loc[is_user_exclusive_S, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 1.0)
        # # assign the user_unassigned_in_both_epochs to neither:
        # assigning_epochs_obj.filter_epochs_df.loc[is_user_unassigned_in_both_epochs, 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.NEITHER, 1.0)

        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_S]
        # assigning_epochs_obj.filter_epochs_df[is_user_exclusive_L]
        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)

        unassigned_epochs_df = assigning_epochs_obj.unassigned_epochs_df
        # Filter based on the active_set cells (LxC, SxC):
        unassigned_epochs_df.loc[np.logical_and(unassigned_epochs_df['has_LONG_exclusive_aclu'], np.logical_not(unassigned_epochs_df['has_SHORT_exclusive_aclu'])), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.LONG_TRACK, 0.85)
        assigning_epochs_obj._debug_print_assignment_statuses()
        # assign the user_exclusive_S to short_track:
        unassigned_epochs_df.loc[np.logical_and(np.logical_not(unassigned_epochs_df['has_LONG_exclusive_aclu']), unassigned_epochs_df['has_SHORT_exclusive_aclu']), 'track_assignment'] = TrackAssignmentDecision(TrackAssignmentState.SHORT_TRACK, 0.85)

        # Re assign
        assigning_epochs_obj.unassigned_epochs_df = unassigned_epochs_df

        axis_idx = axis_idx + 1
        assigning_epochs_obj._debug_print_assignment_statuses()
        fig, axs = assigning_epochs_obj._subfn_plot_epoch_track_assignments(axis_idx=axis_idx, fig=fig, axs=axs, defer_render=True)
        
        if not defer_render:
            fig.show() 
            
        return fig, axs, assigning_epochs_obj


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
    jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
    neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions(frs_index_inclusion_magnitude=0.5)

    assigning_epochs_obj = AssigningEpochs(filter_epochs_df=deepcopy(long_results_obj.active_filter_epochs.to_dataframe()))

    # included_neuron_ids = deepcopy(exclusive_aclus)
    # included_neuron_ids = None
    included_neuron_ids = EITHER_subset.track_exclusive_aclus
    spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df).spikes.sliced_by_neuron_type('pyr')
    # filter_epochs_df = deepcopy(long_results_obj.active_filter_epochs.to_dataframe())


    ## TODO: need to usee the actual LxC/SxCs (hand-picked) instead of the ones based on placefields.
    filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj.determine_if_contains_active_set_exclusive_cells(spikes_df, exclusive_aclus=EITHER_subset.track_exclusive_aclus, short_exclusive=short_exclusive, long_exclusive=long_exclusive, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'    

    # filter_epoch_spikes_df, filter_epochs_df = assigning_epochs_obj._find_example_epochs(spikes_df, EITHER_subset.track_exclusive_aclus, included_neuron_ids=included_neuron_ids) # adds 'active_unique_aclus'
    # # epoch_contains_any_exclusive_aclus.append(np.isin(epoch_spikes_unique_aclus, exclusive_aclus).any())
    # filter_epochs_df['has_SHORT_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, short_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]
    # filter_epochs_df['has_LONG_exclusive_aclu'] = [np.isin(epoch_spikes_unique_aclus, long_exclusive.track_exclusive_aclus).any() for epoch_spikes_unique_aclus in filter_epochs_df['active_unique_aclus']]

    # Get the manual user annotations to determine the good replays for both long/short decoding:
    # assigning_epochs_obj.filter_by_user_selections(curr_active_pipeline=curr_active_pipeline, allow_interactive_selection=allow_interactive_good_epochs_selection)

    #### Finally, get only the epochs that meet the criteria:

    # # only include those with one or more exclusive aclu:
    # considered_filter_epochs_df = filter_epochs_df[filter_epochs_df.contains_one_exclusive_aclu].copy()
    # # require inclusion for long or short:
    # considered_filter_epochs_df = considered_filter_epochs_df[np.logical_xor(filter_epochs_df['long_is_user_included'], filter_epochs_df['short_is_user_included'])]

    # Get separate long-side/short-side candidate replays:
    if 'long_is_user_included' not in filter_epochs_df:
        print(f'WARNING: PAPER_FIGURE_figure_1_add_replay_epoch_rasters(...): no user-assigned manually labeled replay epochs. Reeturning all epochs.')
        epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
        epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.
    else:
        epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'] & filter_epochs_df['long_is_user_included'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
        epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'] & filter_epochs_df['short_is_user_included'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.


    # epochs_df_L = filter_epochs_df[(filter_epochs_df['has_LONG_exclusive_aclu'] & filter_epochs_df['long_is_user_included'])].copy() # replay not considered good by user for short decoding, but it is for long decoding. Finally, has at least one LONG exclusive ACLU.
    # epochs_df_S = filter_epochs_df[(filter_epochs_df['has_SHORT_exclusive_aclu'] & filter_epochs_df['short_is_user_included'])].copy()  # replay not considered good by user for long decoding, but it is for short decoding. Finally, has at least one SHORT exclusive ACLU.

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
def PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=False, save_figure=True, should_plot_pf1d_compare=True, should_plot_example_rasters=False, should_plot_stacked_epoch_slices=False, should_plot_pho_jonathan_figures=True, show_only_refined_cells=True):
    """ 
    
    show_only_refined_cells: bool - added 2023-09-28 to output LxC and SxC values "refined" by their firing rate index.

    
    Usage:
        pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics, fig_1c_figures_out_dict = PAPER_FIGURE_figure_1_full(curr_active_pipeline) # did not display the pf1
    
    """
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
    # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff
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

        pf1d_compare_graphics = curr_active_pipeline.display('_display_long_short_pf1D_comparison', active_identifying_session_ctx, single_figure=False, debug_print=False, fignum='Short v Long pf1D Comparison',
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
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis.neuron_replay_stats_df, curr_jonathan_firing_rate_analysis.rdf.rdf, curr_jonathan_firing_rate_analysis.rdf.aclu_to_idx, curr_jonathan_firing_rate_analysis.irdf.irdf
    
    if should_plot_pho_jonathan_figures:
        if show_only_refined_cells:
            fig_1c_figures_out_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=XOR_subset.get_refined_track_exclusive_aclus(), n_max_page_rows=20, write_vector_format=False, write_png=save_figure, show_only_refined_cells=show_only_refined_cells, disable_top_row=True)
        else:
            fig_1c_figures_out_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=XOR_subset.track_exclusive_aclus, n_max_page_rows=20, write_vector_format=False, write_png=save_figure, disable_top_row=True) # active_out_figures_dict: {IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'BatchPhoJonathanReplayFRC', 'long_only', '(12,21,48)')>: <Figure size 1920x660 with 12 Axes>, IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'BatchPhoJonathanReplayFRC', 'short_only', '(18,19,65)')>: <Figure size 1920x660 with 12 Axes>}
    else:
        fig_1c_figures_out_dict = None

    return pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics, fig_1c_figures_out_dict

# ==================================================================================================================== #
# 2023-06-26 - Paper Figure 2 Code                                                                                     #
# ==================================================================================================================== #
# Shows the LxC/SxC metrics and firing rate indicies

# Instantaneous versions:

# @overwriting_display_context(
@metadata_attributes(short_name=None, tags=['figure_2', 'paper', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-26 21:36', related_items=[])
@define(slots=False, repr=False)
class PaperFigureTwo(SerializedAttributesAllowBlockSpecifyingClass):
    """ Plots the LxC/SxC bar graph containing full instantaneous firing rate computations for both Long and Short epochs
    
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
        # return [(1.0, 0, 0, 1), (0.65, 0, 0, 1), (0, 0, 0.65, 1), (0, 0, 1.0, 1)] # corresponding colors
        return [(0, 0, 1.0, 1), (0, 0, 0.65, 1), (0.65, 0, 0, 1), (1.0, 0, 0, 1)] # long=Blue, short=Red -- dimmer


    def compute(self, curr_active_pipeline, **kwargs):
        """ full instantaneous computations for both Long and Short epochs:
        
        Can access via:
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo

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
                "<color:royalblue, weight:bold>Long</>/<color:crimson, weight:bold>Short</> eXclusive Cells on each track</></>"
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
            scatter_props_kwargs: Dict = list_of_dicts_to_dict_of_lists(scatter_props[i])
            had_custom_marker_handling = False
            user_markers = scatter_props_kwargs.get('marker', None)
            if user_markers is not None:
                # if marker is just a scalar we can do the normal (flat) ax.scatter(...) command. Otherwise we have to loop.
                if isinstance(user_markers, (list, tuple, np.ndarray)):
                    ## have to loop. 
                    user_markers = scatter_props_kwargs.pop('marker') # remove from `scatter_props_kwargs`
                    scatter_plot = [] # scatter plot is just going to be a list here, hope that's okay.
                    for point_index, a_marker in enumerate(user_markers):
                        a_sub_scatter_plot = ax.scatter(x_values[point_index], y_values[i][point_index], color=cls.get_bar_colors()[i], marker=user_markers[point_index], **scatter_props_kwargs) # the np.random part is to spread the points out along the x-axis within their bar so they're visible and don't overlap.
                        scatter_plot.append(a_sub_scatter_plot)
                    had_custom_marker_handling = True # indicate that this was handled in this manner

            if not had_custom_marker_handling:
                # No special marker or just a scalar specific marker value, no need to loop
                scatter_plot = ax.scatter(x_values, y_values[i], color=cls.get_bar_colors()[i], **scatter_props_kwargs) # the np.random part is to spread the points out along the x-axis within their bar so they're visible and don't overlap.
    
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
        
        # delta_minus_str: str = '\\Delta -'        
        # delta_plus_str: str = '\\Delta +'
        delta_minus_str: str = '⬖'        
        delta_plus_str: str = '⬗'

        var_name:str = 'Laps'
        active_context = active_context.adding_context_if_missing(variable=var_name) # title='Laps'

        x_labels = ['$L_x C$\t$\\theta_{' + delta_minus_str + '}$', '$L_x C$\t$\\theta_{' + delta_plus_str + '}$', '$S_x C$\t$\\theta_{' + delta_minus_str + '}$', '$S_x C$\t$\\theta_{' + delta_plus_str + '}$']
        all_data_points = np.array([v.values for v in Fig2_Laps_FR])
        # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 

        if Fig2_Laps_FR[0].LxC_scatter_props is not None:
            # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 
            all_scatter_props =  [Fig2_Laps_FR[0].LxC_scatter_props, Fig2_Laps_FR[1].LxC_scatter_props, Fig2_Laps_FR[2].SxC_scatter_props, Fig2_Laps_FR[3].SxC_scatter_props]

        else:
            all_scatter_props = [{}, {}, {}, {}]

        all_scatter_props = [{}, {}, {}, {}] # override, 2023-10-03

        return cls.create_plot(x_labels, all_data_points, all_scatter_props, 'Laps Firing Rates (Hz)', 'Lap ($\\theta$)', 'fig_2_Theta_FR_matplotlib', active_context, defer_show, kwargs.get('title_modifier'))


    # @providing_context(fig='F2', frs='Replay')
    @classmethod
    def fig_2_Replay_FR_matplotlib(cls, Fig2_Replay_FR, defer_show=False, **kwargs) -> MatplotlibRenderPlots:
        active_context = kwargs.get('active_context', None)
        assert active_context is not None
        
        # delta_minus_str: str = '\\Delta -'        
        # delta_plus_str: str = '\\Delta +'
        delta_minus_str: str = '⬖'        
        delta_plus_str: str = '⬗'
        
        var_name:str = 'Replay'
        active_context = active_context.adding_context_if_missing(variable=var_name) # title='Laps'
        
        x_labels = ['$L_x C$\t$R_{' + delta_minus_str + '}$', '$L_x C$\t$R_{' + delta_plus_str + '}$', '$S_x C$\t$R_{' + delta_minus_str + '}$', '$S_x C$\t$R_{' + delta_plus_str + '}$']
        assert len(Fig2_Replay_FR) == 4
        all_data_points = np.array([v.values for v in Fig2_Replay_FR])
        # all_scatter_props = Fig2_Replay_FR[0].LxC_scatter_props + Fig2_Replay_FR[1].LxC_scatter_props + Fig2_Replay_FR[2].SxC_scatter_props + Fig2_Replay_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 

        if Fig2_Replay_FR[0].LxC_scatter_props is not None:
            # all_scatter_props =  Fig2_Laps_FR[0].LxC_scatter_props + Fig2_Laps_FR[1].LxC_scatter_props + Fig2_Laps_FR[2].SxC_scatter_props + Fig2_Laps_FR[3].SxC_scatter_props # the LxC_scatter_props and SxC_scatter_props are actually the same for all entries in this list, but get em like this anyway. 
            all_scatter_props =  [Fig2_Replay_FR[0].LxC_scatter_props, Fig2_Replay_FR[1].LxC_scatter_props, Fig2_Replay_FR[2].SxC_scatter_props, Fig2_Replay_FR[3].SxC_scatter_props]
        else:
            all_scatter_props = [{}, {}, {}, {}]
            
        all_scatter_props = [{}, {}, {}, {}] # override, 2023-10-03
        # label_list = [LxC_aclus, LxC_aclus, SxC_aclus, SxC_aclus]
        return cls.create_plot(x_labels, all_data_points, all_scatter_props, 'Replay Firing Rates (Hz)', 'Replay', 'fig_2_Replay_FR_matplotlib', active_context, defer_show, kwargs.get('title_modifier'))

    @providing_context(fig='2', display_fn_name='inst_FR_bar_graphs')
    def display(self, defer_show=False, save_figure=True, enable_tiny_point_labels=True, enable_hover_labels=False, enabled_point_connection_lines=True, **kwargs):
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


        if (enable_hover_labels or enable_tiny_point_labels or enabled_point_connection_lines):
            LxC_aclus = self.computation_result.LxC_aclus
            SxC_aclus = self.computation_result.SxC_aclus
            _fig_2_theta_out = self.add_optional_aclu_labels(_fig_2_theta_out, LxC_aclus, SxC_aclus, enable_tiny_point_labels=enable_tiny_point_labels, enable_hover_labels=enable_hover_labels, enabled_point_connection_lines=enabled_point_connection_lines)
            _fig_2_replay_out = self.add_optional_aclu_labels(_fig_2_replay_out, LxC_aclus, SxC_aclus, enable_tiny_point_labels=enable_tiny_point_labels, enable_hover_labels=enable_hover_labels, enabled_point_connection_lines=enabled_point_connection_lines)
        
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

    @classmethod
    def add_optional_aclu_labels(cls, a_fig_container, LxC_aclus, SxC_aclus, enable_hover_labels=True, enable_tiny_point_labels=True, enabled_point_connection_lines=True):
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
        if enabled_point_connection_lines:
            a_fig_container['plot_objects']['point_connection_lines'] = {} # aclu:(start_point, end_point)
            # start_points = []
            # end_points = []
            # Draw arrows from the first set of points to the second set _________________________________________________________ #
            # arrowprops_kwargs = dict(arrowstyle="->", alpha=0.6)
            # arrowprops_kwargs = dict(arrowstyle="simple", alpha=0.7)
            arrowprops_kwargs = dict(arrowstyle="fancy, head_length=0.25, head_width=0.25, tail_width=0.05", alpha=0.4)
            a_fig_container['plot_objects']['long_to_short_arrow'] = {}

        if (enable_hover_labels or enable_tiny_point_labels or enabled_point_connection_lines):
            
            for bar_idx, a_scatter_plot, x_values, y_values, active_labels in zip(np.arange(4), a_fig_container['plot_objects']['scatter_plots'], x_values_list, y_values_list, label_list): # four scatter plots ( one for each group/bar)
                point_hover_labels = [f'{i}' for i in active_labels] # point_hover_labels will be added as tooltip annotations to the datapoints
                assert len(x_values) == len(y_values) and len(x_values) == len(point_hover_labels), f"len(x_values): {len(x_values)}, len(y_values): {len(y_values)}, len(point_hover_labels): {len(point_hover_labels)}"
                # add static tiny labels beside each point
                if (enable_tiny_point_labels or enabled_point_connection_lines):
                    if enable_tiny_point_labels:
                        temp_annotation_labels_list = []
                    for i, (x, y, label) in enumerate(zip(x_values, y_values, point_hover_labels)):
                        # print(f'{i}, (x, y, label): ({x}, {y}, {label})')
                        if enable_tiny_point_labels:
                            annotation_item = ax.annotate(label, (x, y), textcoords="offset points", xytext=(2,2), ha='left', va='bottom', fontsize=8) # , color=rect.get_facecolor()
                            temp_annotation_labels_list.append(annotation_item)
                        if enabled_point_connection_lines:
                            if str(label) not in a_fig_container['plot_objects']['point_connection_lines']:
                                a_fig_container['plot_objects']['point_connection_lines'][str(label)] = [] ## create
                            a_fig_container['plot_objects']['point_connection_lines'][str(label)].append((x, y))
    
                    if enable_tiny_point_labels:
                        a_fig_container['plot_objects']['tiny_annotation_labels'].append(temp_annotation_labels_list)
                    
                # add hover labels:
                # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
                # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib/21654635#21654635
                # add hover labels using mplcursors
                if enable_hover_labels:
                    hover_label_obj = mplcursors.cursor(a_scatter_plot, hover=True).connect("add", lambda sel: sel.annotation.set_text(point_hover_labels[sel.index]))
                    a_fig_container['plot_objects']['hover_label_objects'].append(hover_label_obj)


        if enabled_point_connection_lines:
            for a_label, a_point_list in a_fig_container['plot_objects']['point_connection_lines'].items():
                print(f'a_label: {a_label}, a_point_list: {a_point_list}')
                assert len(a_point_list) == 2,f"len(a_point_list) should be 2, but it is {len(a_point_list)}"
                (start_x, start_y), (end_x, end_y) = a_point_list
                a_fig_container['plot_objects']['long_to_short_arrow'][a_label] = ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=dict(**arrowprops_kwargs, color='black'), label=str(label))
                # plt.plot(x,y,zorder=1) 

        return a_fig_container

# ==================================================================================================================== #
# 2023-06-26 Figure 3                                                                                                  #
""" Firing rate index, Long|Short firing rate figures 

        Renders 3 Subfigures:
            a) Shows the firing rate index between the long and short track computed for two different periods: the laps along the x-axis and the replays along the y-axis.
            b) The ratio of lap to replay firing rate on the long track.
            c) The ratio of lap to replay firing rate on the short track
"""
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
# Statistical Tests                                                                                                    #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['stats', 'binomial', 'FRI'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-10 16:55', related_items=[])
def pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table, x_col_name='x_frs_index', y_col_name='y_frs_index', extra_required_columns=['neuron_uid']):
    """ Performs a binomial test to see if the number of entries above/below the y=x diagnoal were greater than would be expected by chance.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_perform_diagonal_line_binomial_test

        binom_test_chance_result = pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table)
        binom_test_chance_result

    """
    # Drop column: 'index'
    # long_short_fr_indicies_analysis_table = long_short_fr_indicies_analysis_table.drop(columns=['index'])
    # Drop rows with missing data in columns: 'x_frs_index', 'y_frs_index', 'neuron_uid'
    # extra_required_columns = ['neuron_uid']
    long_short_fr_indicies_analysis_table = long_short_fr_indicies_analysis_table.dropna(subset=[x_col_name, y_col_name, *extra_required_columns])

    ## Find the values above/below the main y=x diagonal:
    x_minus_y_diff = (long_short_fr_indicies_analysis_table[x_col_name] - long_short_fr_indicies_analysis_table[y_col_name])
    assert np.sum(np.logical_not(np.isfinite(x_minus_y_diff))) == 0, f"ERROR: contains {np.sum(np.logical_not(np.isfinite(x_minus_y_diff)))} non-finite values"
    ## Find the counts for each:
    n_total = len(x_minus_y_diff) # 856
    n_below_diagonal = np.sum((0.0 > x_minus_y_diff)) # 365
    n_above_diagonal = np.sum((0.0 < x_minus_y_diff)) # 487
    n_exact_on_diagonal = np.sum((0.0 == x_minus_y_diff))
    print(f'n_total: {n_total}, n_above_diagonal: {n_above_diagonal}, n_exact_on_diagonal: {n_exact_on_diagonal}, n_below_diagonal: {n_below_diagonal}')
    assert (n_above_diagonal + n_below_diagonal + n_exact_on_diagonal) == n_total, f"they don't add up!" 
    binom_test_chance_result = stats.binomtest(n_above_diagonal, n=n_total, p=0.5) # p=0.5 random assignment on each trial, n=n_total trials
    return binom_test_chance_result



def pho_stats_paired_t_test(values1, values2):
    """ Paired (Dependent) T-Test of means

    degrees of freedom (dof): n -1

    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_paired_t_test


    """
    assert len(values1) == len(values2), f"this is supposed to be a paired t-test so the number of samples in values1 should equal values2!! but {np.shape(values1)} and {np.shape(values2)}"
    # # Manual Calculation:
    # n_samples = len(values1) # sample_size (number of neurons)
    # out_numerator = (np.mean(values1) - np.mean(values2))
    # out_denom = np.std(values1 - values2) / np.sqrt(n_samples)
    # T = out_numerator/out_denom
    T_result = stats.ttest_rel(values1, values2)
    # T_value = T_result.statistic
    return T_result


@function_attributes(short_name=None, tags=['stats', 'bar'], input_requires=[], output_provides=[], uses=['pho_stats_paired_t_test'], used_by=[], creation_date='2023-10-10 16:54', related_items=[])
def pho_stats_bar_graph_t_tests(across_session_inst_fr_computation):
    """ performs the statistical tests for the bar-graphs 

    Usage:
    
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_bar_graph_t_tests

        LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(across_session_inst_fr_computation)

    """
    ## Laps Bar Graph Statistics:
    LxC_Laps_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Laps_FR[0].values, across_session_inst_fr_computation.Fig2_Laps_FR[1].values)
    SxC_Laps_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Laps_FR[2].values, across_session_inst_fr_computation.Fig2_Laps_FR[3].values)
    print(f'LxC_Laps_T_result: {LxC_Laps_T_result}') # LxC_Laps_T_result: TtestResult(statistic=13.925882964152734, pvalue=2.3158087721181958e-10, df=16)
    print(f'SxC_Laps_T_result: {SxC_Laps_T_result}') # SxC_Laps_T_result: TtestResult(statistic=-12.402705609994197, pvalue=8.279901167065766e-08, df=11)

    ## Replay Bar Graph Statistics
    LxC_Replay_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Replay_FR[0].values, across_session_inst_fr_computation.Fig2_Replay_FR[1].values)
    SxC_Replay_T_result = pho_stats_paired_t_test(across_session_inst_fr_computation.Fig2_Replay_FR[2].values, across_session_inst_fr_computation.Fig2_Replay_FR[3].values)
    print(f'LxC_Replay_T_result: {LxC_Replay_T_result}') # LxC_Replay_T_result: TtestResult(statistic=-0.44250837706672874, pvalue=0.6640450004297094, df=16) # LxC_Replay_T_result is NOT p<0.05 significant (pvalue=0.6640450004297094)
    print(f'SxC_Replay_T_result: {SxC_Replay_T_result}') # SxC_Replay_T_result: TtestResult(statistic=-3.6555017036343607, pvalue=0.0037841961453242896, df=11) # SxC_Replay_T_result IS p<0.05 significant (pvalue=0.0037841961453242896)
    
    return LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result




@function_attributes(short_name=None, tags=['epoch', 'pbe'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-11 14:13', related_items=[])
def build_derived_epochs_dicts(owning_pipeline_reference):
    """ builds three dictionaries containing all of the Epoch objects for {global, long, short}. Contains epochs that don't normally exist on the session object
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import build_derived_epochs_dicts
        all_epochs, long_only_all_epochs, short_only_all_epochs = build_derived_epochs_dicts(curr_active_pipeline)
    
    """	
    
    long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
    long_epoch_obj, short_epoch_obj = [Epoch(owning_pipeline_reference.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name)) for an_epoch_name in [long_epoch_name, short_epoch_name]]

    ## Do all epoch computations on the original session. When done, should have: ['pbe', 'replay', 'laps', 'non_running_periods', 'non_replay_periods'
    # dictionary of all epoch objects across the session
    all_epochs: Dict[str,Epoch] = dict(
        laps = owning_pipeline_reference.sess.laps.as_epoch_obj(),
        pbe = owning_pipeline_reference.sess.pbe,
        replay = Epoch(owning_pipeline_reference.sess.replay),
        non_running_periods = Epoch.from_PortionInterval(owning_pipeline_reference.sess.laps.as_epoch_obj().to_PortionInterval().complement()),
        non_replay_periods = Epoch(Epoch.from_PortionInterval(owning_pipeline_reference.sess.replay.epochs.to_PortionInterval().complement()).time_slice(t_start=long_epoch_obj.t_start, t_stop=short_epoch_obj.t_stop).to_dataframe()[:-1]),  #[:-1] # any period except the replay ones, drop the infinite last entry
    )

    # Split into long/short only periods:
    long_only_all_epochs: Dict[str,Epoch] = {k:v.time_slice(t_start=long_epoch_obj.t_start, t_stop=long_epoch_obj.t_stop) for k,v in all_epochs.items()}
    short_only_all_epochs: Dict[str,Epoch] = {k:v.time_slice(t_start=short_epoch_obj.t_start, t_stop=short_epoch_obj.t_stop) for k,v in all_epochs.items()}

    return all_epochs, long_only_all_epochs, short_only_all_epochs


@function_attributes(short_name=None, tags=['inst_fr', 'spike_rate_Trends'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-11 14:13', related_items=[])
def add_extra_spike_rate_trends(curr_active_pipeline) -> InstantaneousSpikeRateGroupsComputation:
    """ independent of all other FR computations. Builds inst spike rate groups for the PBEs. 
    
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import add_extra_spike_rate_trends


    """
    temp = InstantaneousSpikeRateGroupsComputation()
    temp.active_identifying_session_ctx = curr_active_pipeline.sess.get_context()

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]] # only uses global_session

    ## Add additional Epochs:
    all_epochs, long_only_all_epochs, short_only_all_epochs = build_derived_epochs_dicts(curr_active_pipeline)

    ## Manual User-annotation mode:
    annotation_man: UserAnnotationsManager = UserAnnotationsManager()
    session_cell_exclusivity: SessionCellExclusivityRecord = annotation_man.annotations[temp.active_identifying_session_ctx].get('session_cell_exclusivity', None)
    if session_cell_exclusivity is not None:
        print(f'setting LxC_aclus/SxC_aclus from user annotation.')
        temp.LxC_aclus = session_cell_exclusivity.LxC
        temp.SxC_aclus = session_cell_exclusivity.SxC
    else:
        print(f'WARN: no user annotation for session_cell_exclusivity')

    are_LxC_empty: bool = (len(temp.LxC_aclus) == 0)
    are_SxC_empty: bool = (len(temp.SxC_aclus) == 0)

    temp.LxC_PBEsDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_only_all_epochs['pbe'], included_neuron_ids=temp.LxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)
    temp.LxC_PBEsDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_only_all_epochs['pbe'], included_neuron_ids=temp.LxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)
    temp.SxC_PBEsDeltaMinus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=long_only_all_epochs['pbe'], included_neuron_ids=temp.SxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)
    temp.SxC_PBEsDeltaPlus: SpikeRateTrends = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=global_session.spikes_df, filter_epochs=short_only_all_epochs['pbe'], included_neuron_ids=temp.SxC_aclus, instantaneous_time_bin_size_seconds=temp.instantaneous_time_bin_size_seconds)

    # Note that in general LxC and SxC might have differing numbers of cells.
    if (are_LxC_empty or are_SxC_empty):
        temp.Fig2_PBEs_FR: list[SingleBarResult] = []
        for v in (temp.LxC_PBEsDeltaMinus, temp.LxC_PBEsDeltaPlus, temp.SxC_PBEsDeltaMinus, temp.SxC_PBEsDeltaPlus):
            if v is not None:
                temp.Fig2_PBEs_FR.append(SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, temp.LxC_aclus, temp.SxC_aclus, None, None))
            else:
                temp.Fig2_PBEs_FR.append(SingleBarResult(None, None, np.array([], dtype=float), temp.LxC_aclus, temp.SxC_aclus, None, None))
    else:
        temp.Fig2_PBEs_FR: list[SingleBarResult] = [SingleBarResult(v.cell_agg_inst_fr_list.mean(), v.cell_agg_inst_fr_list.std(), v.cell_agg_inst_fr_list, temp.LxC_aclus, temp.SxC_aclus, None, None) for v in (temp.LxC_PBEsDeltaMinus, temp.LxC_PBEsDeltaPlus, temp.SxC_PBEsDeltaMinus, temp.SxC_PBEsDeltaPlus)]

    return temp



# ==================================================================================================================== #
# 2024-09-05 Firing Rate Regression Lines                                                                              #
# ==================================================================================================================== #
from sklearn.linear_model import LinearRegression

class LongShortFRRegression:
    """
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import LongShortFRRegression
    import matplotlib
    from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update

    _bak = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

    (model_L, model_S), (fig, ax) = LongShortFRRegression.compute_laps_replay_regression(neuron_replay_stats_table)
    (model_L, model_S), (fig, ax) = LongShortFRRegression.compute_laps_replay_opposite_regression(neuron_replay_stats_table)

    """
    @classmethod
    def compute_regression_line(cls, X, y, name: str, ax=None):
        from neuropy.utils.matplotlib_helpers import build_or_reuse_figure

        # Fit the regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        y_pred = model.predict(X)

        # Plot the results
        if ax is None:
            # fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
            
        # plt.figure(clear=True)
        ax.scatter(X, y, label=f'Data {name}', alpha=0.7) # , color='blue'
        ax.plot(X, y_pred, label=f'Regression {name}', alpha=0.7) # , color='red'
        plt.xlabel('Non-Replay Mean')
        plt.ylabel('Replay Mean')
        plt.legend()
        # ax.show()

        # Retrieve the slope
        slope = model.coef_[0]
        
        # Print the slope
        print("Slope:", slope)
        
        return model, (fig, ax)

    @classmethod
    def compute_laps_replay_regression(cls, neuron_replay_stats_table: pd.DataFrame, long_replay_mean_col: str = 'long_replay_mean', long_non_replay_mean_col: str = 'long_non_replay_mean', short_replay_mean_col: str = 'short_replay_mean', short_non_replay_mean_col: str = 'short_non_replay_mean') -> tuple:
        """ computes the laps vs. replay 
        
        """        
        # Specify columns to convert to float
        columns_to_convert = [long_replay_mean_col, long_non_replay_mean_col, short_replay_mean_col, short_non_replay_mean_col]

        # Convert specified columns to float
        neuron_replay_stats_table[columns_to_convert] = neuron_replay_stats_table[columns_to_convert].astype(float)

        # Long Track Replay|Laps FR Figure
        cleaned_df = neuron_replay_stats_table.dropna(subset=[long_replay_mean_col, long_non_replay_mean_col], inplace=False, how='any')
        np.sum(cleaned_df[[long_replay_mean_col, long_non_replay_mean_col, short_replay_mean_col, short_non_replay_mean_col]].isna())

        X = cleaned_df[[long_non_replay_mean_col]].to_numpy().astype(float)
        y = cleaned_df[long_replay_mean_col].to_numpy().astype(float)

        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0

        model_L, (fig, ax) = cls.compute_regression_line(X, y, name='long')

        # Short Track Replay|Laps FR Figure
        cleaned_df = neuron_replay_stats_table.dropna(subset=[short_replay_mean_col, short_non_replay_mean_col], inplace=False, how='any')[[short_replay_mean_col, short_non_replay_mean_col]]
        cleaned_df[[short_replay_mean_col, short_non_replay_mean_col]] = cleaned_df[[short_replay_mean_col, short_non_replay_mean_col]].astype(float)
        cleaned_df = cleaned_df.dropna(subset=[short_replay_mean_col, short_non_replay_mean_col], inplace=False, how='any')

        X = cleaned_df[[short_non_replay_mean_col]].to_numpy().astype(float)
        y = cleaned_df[short_replay_mean_col].to_numpy().astype(float)

        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0

        model_S, (fig, ax) = cls.compute_regression_line(X, y, name='short', ax=ax)
        ax.set_aspect(1.)
        plt.title('Non-PBE/PBE Firing Rate Gain Factor Regression for Long+Short')

        slopes_string = '\n'.join([f"{a_name}: {model.coef_[0]:.4f}" for a_name, model in dict(zip(('Long', 'Short'), (model_L, model_S))).items()])
        
        ax.text(0.05, 0.95, f'Slope:\n{slopes_string}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        fig.show()

        return (model_L, model_S), (fig, ax)




    @classmethod
    def compute_laps_replay_opposite_regression(cls, neuron_replay_stats_table):
        ## INPUTS: neuron_replay_stats_table

        ## Long Track Replay|Laps FR Figure
        # Specify columns to convert to float
        columns_to_convert = ['long_replay_mean', 'long_non_replay_mean', 'short_replay_mean', 'short_non_replay_mean']

        # Convert specified columns to float
        neuron_replay_stats_table[columns_to_convert] = neuron_replay_stats_table[columns_to_convert].astype(float)



        # first oness ________________________________________________________________________________________________________ #
        cleaned_df = neuron_replay_stats_table.dropna(subset=['long_replay_mean', 'short_non_replay_mean'], inplace=False, how='any')
        # Drop rows with NaN values
        # cleaned_df = neuron_replay_stats_df.dropna(subset=['long_replay_mean', 'long_non_replay_mean'])
        # np.sum(cleaned_df[['long_replay_mean', 'long_non_replay_mean', 'short_replay_mean', 'short_non_replay_mean']].isna())

        # Extract data
        # X = neuron_replay_stats_df['long_replay_mean'].values.astype(float)
        # X = cleaned_df[['long_replay_mean']].values.astype(float)
        # y = cleaned_df['long_non_replay_mean'].values.astype(float)

        X = cleaned_df[['long_replay_mean']].to_numpy().astype(float)
        y = cleaned_df['short_non_replay_mean'].to_numpy().astype(float)

        # X.dtype
        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0

        # X.shape
        # y.shape
        
        model_L, (fig, ax) = cls.compute_regression_line(X, y, name='long') # Slope: 0.3690231851911259

        ## Short Track Replay|Laps FR Figure
        # cleaned_df.dtypes
        cleaned_df = neuron_replay_stats_table.dropna(subset=['short_replay_mean', 'long_non_replay_mean'], inplace=False, how='any')[['short_replay_mean', 'long_non_replay_mean']]
        cleaned_df[['short_replay_mean', 'long_non_replay_mean']] = cleaned_df[['short_replay_mean', 'long_non_replay_mean']].astype(float)
        # cleaned_df.dtypes
        cleaned_df = cleaned_df.dropna(subset=['short_replay_mean', 'long_non_replay_mean'], inplace=False, how='any')
        # cleaned_df
        # cleaned_df.dtypes

        X = cleaned_df[['short_replay_mean']].to_numpy().astype(float)
        y = cleaned_df['long_non_replay_mean'].to_numpy().astype(float)

        assert np.sum(np.isnan(X)) == 0
        assert np.sum(np.isnan(y)) == 0
        # fig_S, ax_S, active_display_context_S = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=final_context.adding_context_if_missing(filter_name='short'), **common_scatter_kwargs)

        model_S, (fig, ax) = cls.compute_regression_line(X, y, name='short', ax=ax) # Slope: 0.3690231851911259
        # Set aspect of the main Axes.
        ax.set_aspect(1.)
        plt.title('Long/Short Opposite Gain Factor Regression')
        
            # Retrieve the slope
        slopes_string = '\n'.join([f"{a_name}: {model.coef_[0]:.4f}" for a_name, model in dict(zip(('Long', 'Short'), (model_L, model_S))).items()])
        
            # Add slope text to the plot
        ax.text(0.05, 0.95, f'Slope:\n{slopes_string}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        
        fig.show()
        # Slope: 0.3374416368231605

        return (model_L, model_S), (fig, ax)









# ==================================================================================================================== #
# MAIN RUN FUNCTION TO GENERATE ALL FIGURES                                                                            #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['export', 'output', 'images', 'final', 'figure', 'complete'], input_requires=[], output_provides=[], uses=['PAPER_FIGURE_figure_1_full', 'PaperFigureTwo', 'PAPER_FIGURE_figure_3', 'fig_remapping_cells'], used_by=[], creation_date='2024-08-29 18:07', related_items=[])
def main_complete_figure_generations(curr_active_pipeline, enable_default_neptune_plots:bool=True, save_figures_only:bool=False, save_figure=True):
    """ main run function to generate all figures
    
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import main_complete_figure_generations
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
    Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff
    jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis    
    (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)


    # ==================================================================================================================== #
    # Figure 1) pf1D Ratemaps, Active set, etc                                                                             #
    # ==================================================================================================================== #
    _out_fig_1 = PAPER_FIGURE_figure_1_full(curr_active_pipeline, defer_show=defer_show, save_figure=save_figure) # did not display the pf1


    # Critical new code: Not used anyhwere
    # ratemap = long_pf1D.ratemap
    # included_unit_neuron_IDs = EITHER_subset.track_exclusive_aclus
    # included_unit_neuron_IDs = EITHER_subset.get_refined_track_exclusive_aclus() # 2023-09-28 - "Refined"
    # included_aclus_sort_indicies = deepcopy(new_all_aclus_sort_indicies[np.isin(new_all_aclus_sort_indicies, included_unit_neuron_IDs)])
    # # rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=new_all_aclus_sort_indicies.copy())
    # rediculous_final_sorted_all_included_neuron_ID, rediculous_final_sorted_all_included_pfmap = build_shared_sorted_neuronIDs(ratemap, included_unit_neuron_IDs, sort_ind=included_aclus_sort_indicies.copy())

    # ==================================================================================================================== #
    # Figure 2) Firing Rate Bar Graphs                                                                                     #
    # ==================================================================================================================== #


    # Instantaneous versions:

    _out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.005) # 10ms
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


    # ==================================================================================================================== #
    # 2023-09-26 - Presentation Figures:                                                                                   #
    # ==================================================================================================================== #
    
    graphics_output_dict = {}

    try:
        # 2023-09-21 - Plot All
        graphics_output_dict = graphics_output_dict | fig_remapping_cells(curr_active_pipeline)
    except Exception:
        print(f'plotting `fig_remapping_cells(...)` failed. Continuing.') 


    try:
        # 2023-09-21 - Plot All
        graphics_outputs_list = fig_surprise_results(curr_active_pipeline)
    except Exception:
        print(f'plotting `fig_surprise_results(...)` failed. Continuing.')
        

    # Unwrapping:
    # pf1d_compare_graphics, (example_epoch_rasters_L, example_epoch_rasters_S), example_stacked_epoch_graphics = _out_fig_1
    if not save_figures_only:
        # only in active display mode is there something to return:
        return (_out_fig_1, _out_fig_2, _out_fig_3_a, _out_fig_3_b)
    
    # plots




# ==================================================================================================================== #
# Plotting Helpers 2024-10-29                                                                                          #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['matplotlib', 'good', 'stacked-hist', 'ACTIVE'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-23 18:51', related_items=[])
def _perform_dual_hist_plot(grainularity_desc: str, laps_df: pd.DataFrame, ripple_df: pd.DataFrame, is_dark_mode: bool=False, legend_groups_to_solo=None, legend_groups_to_hide = ['0.03', '0.044', '0.05']):
    """ plots the stacked histograms for both laps and ripples
    
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _perform_dual_hist_plot
    
    
    """
    from flexitext import flexitext
    from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors, FormattedFigureText ## flexitext version
    from pyphoplacecellanalysis.Pho2D.statistics_plotting_helpers import plot_histograms_across_sessions, plot_stacked_histograms

    variable_name: str = 'P_Short'
    y_baseline_level: float = 0.5 # for P(short), etc
    y_ylims = (0, 1)

    # y_baseline_level: float = 0.0 # for wcorr, etc
    # y_ylims = (-1, 1)

    if is_dark_mode:
        baseline_kwargs = dict(color=(0.8,0.8,0.8,.75), linewidth=2)
    else:
        baseline_kwargs = dict(color=(0.2,0.2,0.2,.75), linewidth=2)
        
    def _update_stacked_hist_post_plot(histogram_out):
        """ captures: y_baseline_level, y_ylims """
        for k, ax in histogram_out.axes.items():
            _tmp_line = ax.axhline(y_baseline_level, **baseline_kwargs) # draw baseline line (horizontally)
            ax.set_ylim(*y_ylims)
            
        ## add flexitext text:
        a_fig = histogram_out.figures[0]
        extracted_fig_titles_dict = MatplotlibFigureExtractors.extract_titles(fig=a_fig)
        suptitle: str = extracted_fig_titles_dict.get('suptitle', None) # 'Laps (by-time-bin)|2 Sessions|5 tbin sizes'
        subtitle_string = None

        # Clear the normal text:
        a_fig.suptitle('')
        # for k, ax in a_histogram_out.axes.items():
        # 	ax.set_title('')

        text_formatter = FormattedFigureText.init_from_margins() # top_margin=0.8
        text_formatter.setup_margins(a_fig)
        # active_config = deepcopy(self.config)
        # active_config.float_precision = 1

        # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
        full_title_str: str = f'<size:22><weight:bold>{suptitle}</></>'
        if (subtitle_string is not None) and (len(subtitle_string) > 0):
            full_title_str += f'\n<size:10>{subtitle_string}</>'

        # header_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, full_title_str, va="bottom", xycoords="figure fraction") # 0.90
        header_text_obj = flexitext(text_formatter.left_margin, 0.89, full_title_str, va="bottom", xycoords="figure fraction")
        return {'header_text_obj': header_text_obj}
            
    common_stacked_hist_kwargs = dict(figsize=(12, 3), column_name=variable_name)

    compare_decimal_precision_ndigits: int = 3
    if (legend_groups_to_solo is not None):
        assert legend_groups_to_hide is None, f"cannot provide both legend_groups_to_solo and legend_groups_to_hide"
        legend_groups_to_solo = np.array([round(float(v), ndigits=compare_decimal_precision_ndigits) for v in legend_groups_to_solo]) # convert to float
        laps_df = deepcopy(laps_df)
        ripple_df = deepcopy(ripple_df)
        # laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float), legend_groups_to_hide)]
        # ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float), legend_groups_to_hide)]
        laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_solo)]
        ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_solo)]		

    elif (legend_groups_to_hide is not None):
        legend_groups_to_hide = np.array([round(float(v), ndigits=compare_decimal_precision_ndigits) for v in legend_groups_to_hide]) # convert to float
        laps_df = deepcopy(laps_df)
        ripple_df = deepcopy(ripple_df)
        # laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float), legend_groups_to_hide)]
        # ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float), legend_groups_to_hide)]
        laps_df = laps_df[np.isin(laps_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_hide, invert=True)]
        ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float).round(decimals=compare_decimal_precision_ndigits), legend_groups_to_hide, invert=True)]		
        # laps_df = laps_df[np.isclose(laps_df.time_bin_size.astype(float), legend_groups_to_hide)]
        # ripple_df = ripple_df[np.isin(ripple_df.time_bin_size.astype(float), legend_groups_to_hide)]
        
    # You can use it like this:
    num_unique_sessions: int = laps_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    num_unique_time_bins: int = laps_df.time_bin_size.nunique(dropna=True)
    _laps_histogram_out = plot_stacked_histograms(laps_df, data_type=f'Laps ({grainularity_desc})', session_spec=f'{num_unique_sessions} Sessions', time_bin_duration_str=f"{num_unique_time_bins} tbin sizes", **common_stacked_hist_kwargs)
    _laps_flexitext_dict = _update_stacked_hist_post_plot(_laps_histogram_out)
    # fig_to_clipboard(_laps_histogram_out.figures[0], bbox_inches='tight')

    num_unique_sessions: int = ripple_df.session_name.nunique(dropna=True) # number of unique sessions, ignoring the NA entries
    num_unique_time_bins: int = ripple_df.time_bin_size.nunique(dropna=True)
    _ripple_histogram_out = plot_stacked_histograms(ripple_df, data_type=f'PBEs ({grainularity_desc})', session_spec=f'{num_unique_sessions} Sessions', time_bin_duration_str=f"{num_unique_time_bins} tbin sizes", **common_stacked_hist_kwargs)
    _ripple_flexitext_dict = _update_stacked_hist_post_plot(_ripple_histogram_out)
    # fig_to_clipboard(_ripple_histogram_out.figures[0], bbox_inches='tight')

    return _laps_histogram_out, _ripple_histogram_out


@function_attributes(short_name=None, tags=['MAIN', 'CRITICAL', 'FINAL', 'plotly'], input_requires=[], output_provides=[], uses=['plotly_pre_post_delta_scatter'], used_by=[], creation_date='2024-10-23 20:04', related_items=[])
def _perform_plot_pre_post_delta_scatter(data_context: IdentifyingContext, concatenated_ripple_df: pd.DataFrame, time_delta_tuple: Tuple[float, float, float], fig_size_kwargs: Dict, save_plotly: Callable, is_dark_mode: bool=False, enable_custom_widget_buttons:bool=True,
                                          extant_figure=None, custom_output_widget=None, legend_groups_to_hide=['0.03', '0.044', '0.05'], should_save: bool = True, variable_name = 'P_Short', y_baseline_level: float = 0.5, **kwargs):
    """ plots the stacked histograms for both laps and ripples

    Usage:
        from functools import partial
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_pre_post_delta_scatter
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _perform_plot_pre_post_delta_scatter

        is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)

        _perform_plot_pre_post_delta_scatter = partial(
            _perform_plot_pre_post_delta_scatter,
            time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end),
            fig_size_kwargs=fig_size_kwargs,
            is_dark_mode=is_dark_mode,
            save_plotly=save_plotly,
        )

        new_fig, new_fig_context, _extras_output_dict, figure_out_paths = _perform_plot_pre_post_delta_scatter(data_context=IdentifyingContext(epochs_name='laps', data_grain='per_epoch', title_prefix="Lap Per Epoch", dataframe_name='laps_df'), concatenated_ripple_df=deepcopy(all_sessions_laps_df))


    """
    from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers
    from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_helper_save_figures, _helper_build_figure, plotly_pre_post_delta_scatter, add_copy_save_action_buttons
    
    is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=is_dark_mode)
    
    if data_context is None:
        data_context = concatenated_ripple_df.attrs.get('data_context', None)
        assert data_context is not None, f"could not get context from dataframe's df.attrs.data_context either."
        

    histogram_bins = 25
    num_sessions = 1

    num_events: int = len(concatenated_ripple_df)
    # print(f'num_events: {num_events}')
    data_context.overwriting_context(n_events=num_events) # adds 'n_events' context
    # .025 .03 .044 .05 .058
    # Define the legend groups you want to hide on startup
    legend_groups_to_hide = kwargs.pop('legend_groups_to_hide', ['0.03', '0.044', '0.05'])  # '0.025', , '0.058'

    # y_baseline_level: float = 0.5 # for P(short), etc
    # y_baseline_level: float = 0.0 # for wcorr, etc

    # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color':"is_user_annotated_epoch", 'title': f"'{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
    px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'title': f"{data_context.get_description(subset_includelist=['dataframe_name', 'title_prefix', 'num_events'], separator=' - ')} - '{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
    px_scatter_kwargs['color'] = "time_bin_size"
    # px_scatter_kwargs.pop('color')


    # Controls scatterplot point size
    if 'dummy_column_for_size' not in concatenated_ripple_df.columns:
        concatenated_ripple_df['dummy_column_for_size'] = 2.0
        
    px_scatter_kwargs['size'] = "dummy_column_for_size"
    px_scatter_kwargs.setdefault('size_max', 5) # don't override tho
    
    # px_scatter_kwargs['marker'] = dict(line=dict(width=0))
    # Remove the white border around scatter points by setting line width to 0
    # px_scatter_kwargs['line_width'] = 0  # <---- Correct way for Plotly Express

    # px_scatter_kwargs.update(dict(marginal_x="histogram", marginal_y="rug"))

    hist_kwargs = dict(color="time_bin_size")
    # hist_kwargs = dict(color="is_user_annotated_epoch") # , histnorm='probability density'
    # hist_kwargs.pop('color')
    
    px_scatter_kwargs = px_scatter_kwargs | kwargs.pop('px_scatter_kwargs', {})
    hist_kwargs = hist_kwargs | kwargs.pop('hist_kwargs', {})

    figure_sup_huge_title_text: str = data_context.get_description(subset_includelist=['epochs_name', 'data_grain', 'dataframe_name', 'n_events'], separator=' | ')
    if data_context.has_keys(keys_list=['custom_suffix']):
        custom_suffix_description: str = data_context.get_description(subset_includelist=['custom_suffix'])
        figure_sup_huge_title_text = figure_sup_huge_title_text + f'\n{custom_suffix_description}'
    
    # filter_context = df_filter.filter_context # IdentifyingContext(time_bin_sizes=df_filter.time_bin_size, custom_suffix=df_filter.replay_name)
    figure_footer_text = data_context.get_description(separator='|', subset_excludelist=['time_bin_sizes'])

    new_fig, new_fig_context = plotly_pre_post_delta_scatter(data_results_df=concatenated_ripple_df, data_context=data_context,
                                                              extant_figure=extant_figure,
                            out_scatter_fig=None, histogram_bins=histogram_bins,
                            px_scatter_kwargs=px_scatter_kwargs, histogram_variable_name=variable_name, hist_kwargs=hist_kwargs, forced_range_y=None,
                            time_delta_tuple=time_delta_tuple, legend_title_text=None, is_dark_mode=is_dark_mode,
                            # figure_sup_huge_title_text=data_context.get_description(subset_excludelist=['title_prefix'], separator=' | '),
                            figure_sup_huge_title_text=figure_sup_huge_title_text, figure_footer_text=figure_footer_text,
                            **kwargs,
    )

    new_fig = new_fig.update_layout(fig_size_kwargs)

    if legend_groups_to_hide is not None:
        # Collect all unique legend groups you want to hide
        hidden_groups = set(legend_groups_to_hide)
        # Iterate over traces and hide those in the specified legend groups
        for trace in new_fig.data:
            if trace.legendgroup in hidden_groups:
                trace.visible = 'legendonly'
            
    # new_fig_laps.show()

    _extras_output_dict = {}
    if is_dark_mode:
        _extras_output_dict["y_mid_line"] = new_fig.add_hline(y=y_baseline_level, line=dict(color="rgba(0.8,0.8,0.8,.75)", width=2), row='all', col='all')
    else:
        _extras_output_dict["y_mid_line"] = new_fig.add_hline(y=y_baseline_level, line=dict(color="rgba(0.2,0.2,0.2,.75)", width=2), row='all', col='all')


    # fig_to_clipboard(new_fig_laps, **fig_size_kwargs)
    new_fig_context = new_fig_context.adding_context_if_missing(**data_context.get_subset(subset_includelist=['epochs_name', 'data_grain']).to_dict(), num_sessions=num_sessions, plot_type='scatter+hist', comparison='pre-post-delta', variable_name=variable_name)
    if should_save:
        figure_out_paths = save_plotly(a_fig=new_fig, a_fig_context=new_fig_context)
    else:
        figure_out_paths = None
        
    if enable_custom_widget_buttons:
        # _extras_output_dict['out_widget'] = add_copy_save_action_buttons(new_fig)
        # _extras_output_dict['out_container_widget'], _extras_output_dict['out_widget'] = add_copy_save_action_buttons(new_fig, output_widget=custom_output_widget)
        _extras_output_dict['custom_widget_buttons'] = add_copy_save_action_buttons(new_fig)
        

    return new_fig, new_fig_context, _extras_output_dict, figure_out_paths


# ==================================================================================================================== #
# SpecificPrePostDeltaScatter                                                                                          #
# ==================================================================================================================== #

from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_pre_post_delta_scatter
from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows


@metadata_attributes(short_name=None, tags=['plotly', 'scatter', 'notebook', 'figure', 'outputs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-08 15:10', related_items=[])
class SpecificPrePostDeltaScatter:
    """ not yet finished.
    """
    _generic_kwargs = {'fig_size_kwargs': None, 'is_dark_mode': False, 'histogram_bins': 25, 'num_sessions': 1}
    
    @classmethod
    def _pre_post_delta_scatter_laps_per_time_bin(cls, all_sessions_laps_time_bin_df, t_delta_df, t_delta_dict, earliest_delta_aligned_t_start, latest_delta_aligned_t_end):
        histogram_bins = 25
        num_sessions = 1

        # ==================================================================================================================== #
        # Laps                                                                                                                 #
        # ==================================================================================================================== #
        # all_sessions_laps_time_bin_df
        # all_sessions_simple_pearson_laps_df

        # Define the legend groups you want to hide on startup
        legend_groups_to_hide = ['0.030', '0.044', '0.050', '0.058'] # '0.025', 

        # data_context = IdentifyingContext(epochs_name='laps', data_grain='per_epoch', title_prefix="Lap Per Epoch")
        # concatenated_ripple_df = deepcopy(all_sessions_laps_df)
        data_context = IdentifyingContext(epochs_name='laps', data_grain='per_time_bin', title_prefix="Lap Individual Time Bins")
        concatenated_ripple_df = deepcopy(all_sessions_laps_time_bin_df)

        # data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_epoch', title_prefix="PBE Per Epoch")
        # concatenated_ripple_df = deepcopy(all_sessions_ripple_df)
        # data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_time_bin', title_prefix="PBE Individual Time Bins")
        # concatenated_ripple_df = deepcopy(all_sessions_ripple_time_bin_df)

        # concatenated_ripple_df = deepcopy(all_sessions_simple_pearson_laps_df) # ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'ripple_idx', 'ripple_start_t', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'most_likely_decoder_index', 'start', 'stop', 'label', 'duration', 'long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr', 'best_decoder_index', 'session_name', 'time_bin_size', 'delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch', 'custom_replay_name', 'epoch_idx', 'long_best_pf_peak_x_pearsonr', 'short_best_pf_peak_x_pearsonr', 'wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL', 'long_best_wcorr', 'short_best_wcorr', 'wcorr_abs_diff', 'pearsonr_abs_diff']
        # concatenated_ripple_df = deepcopy(all_sessions_laps_time_bin_df) # ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'ripple_idx', 'ripple_start_t', 'P_Long_LR', 'P_Long_RL', 'P_Short_LR', 'P_Short_RL', 'most_likely_decoder_index', 'start', 'stop', 'label', 'duration', 'long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr', 'best_decoder_index', 'session_name', 'time_bin_size', 'delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch', 'custom_replay_name', 'epoch_idx', 'long_best_pf_peak_x_pearsonr', 'short_best_pf_peak_x_pearsonr', 'wcorr_long_LR', 'wcorr_long_RL', 'wcorr_short_LR', 'wcorr_short_RL', 'long_best_wcorr', 'short_best_wcorr', 'wcorr_abs_diff', 'pearsonr_abs_diff']
        # print(f'concatenated_ripple_df.columns: {list(concatenated_ripple_df.columns)}')
        # concatenated_ripple_df

        # variable_name = 'P_Long'
        variable_name = 'P_Short' # Shows expected effect - short-only replay prior to delta and then split replays post-delta
        # variable_name = 'P_LR'

        y_baseline_level: float = 0.5 # for P(short), etc
        # y_baseline_level: float = 0.0 # for wcorr, etc

        # px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'color':"is_user_annotated_epoch", 'title': f"'{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
        px_scatter_kwargs = {'x': 'delta_aligned_start_t', 'y': variable_name, 'title': f"{data_context.get_description(subset_includelist=['title_prefix'])} - '{variable_name}'"} # , 'color': 'time_bin_size', 'range_y': [-1.0, 1.0], 'labels': {'session_name': 'Session', 'time_bin_size': 'tbin_size', 'is_user_annotated_epoch':'user_sel'}
        px_scatter_kwargs['color'] = "time_bin_size"
        # px_scatter_kwargs.pop('color')

        concatenated_ripple_df['dummy_column_for_size'] = 1.0
        px_scatter_kwargs['size'] = "dummy_column_for_size"
        px_scatter_kwargs['size_max'] = 3
        # px_scatter_kwargs['marker'] = dict(line=dict(width=0))
        # Remove the white border around scatter points by setting line width to 0
        # px_scatter_kwargs['line_width'] = 0  # <---- Correct way for Plotly Express

        # px_scatter_kwargs.update(dict(marginal_x="histogram", marginal_y="rug"))

        hist_kwargs = dict(color="time_bin_size")
        # hist_kwargs = dict(color="is_user_annotated_epoch") # , histnorm='probability density'
        # hist_kwargs.pop('color')
        new_fig_laps, new_fig_laps_context = plotly_pre_post_delta_scatter(data_results_df=concatenated_ripple_df, out_scatter_fig=None, histogram_bins=histogram_bins,
                                px_scatter_kwargs=px_scatter_kwargs, histogram_variable_name=variable_name, hist_kwargs=hist_kwargs, forced_range_y=None,
                                time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end), legend_title_text=None, is_dark_mode=is_dark_mode)


        new_fig_laps = new_fig_laps.update_layout(fig_size_kwargs)

        if legend_groups_to_hide is not None:
            # Collect all unique legend groups you want to hide
            hidden_groups = set(legend_groups_to_hide)

            # Iterate over traces and hide those in the specified legend groups
            for trace in new_fig_laps.data:
                if trace.legendgroup in hidden_groups:
                    trace.visible = 'legendonly'
                
        # new_fig_laps.show()

        _extras_output_dict = {}
        if is_dark_mode:
            _extras_output_dict["y_mid_line"] = new_fig_laps.add_hline(y=y_baseline_level, line=dict(color="rgba(0.8,0.8,0.8,.75)", width=2), row='all', col='all')
        else:
            _extras_output_dict["y_mid_line"] = new_fig_laps.add_hline(y=y_baseline_level, line=dict(color="rgba(0.2,0.2,0.2,.75)", width=2), row='all', col='all')


        # fig_to_clipboard(new_fig_laps, **fig_size_kwargs)
        new_fig_laps_context = new_fig_laps_context.adding_context_if_missing(**data_context.get_subset(subset_includelist=['epochs_name', 'data_grain']).to_dict(), num_sessions=num_sessions, plot_type='scatter+hist', comparison='pre-post-delta', variable_name=variable_name)
        # figure_out_paths = save_plotly(a_fig=new_fig_laps, a_fig_context=new_fig_laps_context)
        new_fig_laps



@function_attributes(short_name=None, tags=['across-session', 'time_bin_size'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-07-15 00:00', related_items=[])
def build_single_time_bin_size_dfs(all_sessions_all_scores_epochs_df, all_sessions_epochs_df, all_sessions_epochs_time_bin_df, target_time_bin_size: float, included_columns = ['delta_aligned_start_t', 'is_user_annotated_epoch', 'is_valid_epoch']):
    """ Filters the epochs dataframe down to a single time_bin_size specified by `target_time_bin_size`. 
     
     from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import build_single_time_bin_size_dfs
      
     History: 2024-07-15 - Factored out of Across Session Notebook
    """
    from neuropy.utils.indexing_helpers import PandasHelpers

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
    single_time_bin_size_all_sessions_epochs_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(single_time_bin_size_all_sessions_epochs_df, all_sessions_all_scores_epochs_df[included_columns], join_column_name='delta_aligned_start_t')
    single_time_bin_size_all_sessions_epochs_df.sort_values(by=['delta_aligned_start_t'], inplace=True) # Need to re-sort by timestamps once done
    single_time_bin_size_all_sessions_epochs_df

    single_time_bin_size_all_sessions_epochs_time_bin_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(single_time_bin_size_all_sessions_epochs_time_bin_df, all_sessions_all_scores_epochs_df[included_columns], join_column_name='delta_aligned_start_t')
    single_time_bin_size_all_sessions_epochs_time_bin_df.sort_values(by=['t_bin_center'], inplace=True) # Need to re-sort by timestamps once done
    
    ## Add plotly helper columns:
    for a_df in (all_sessions_all_scores_epochs_df, all_sessions_epochs_df, all_sessions_epochs_time_bin_df, single_time_bin_size_all_sessions_epochs_df, single_time_bin_size_all_sessions_epochs_time_bin_df):
        a_df['pre_post_delta_category'] = 'post-delta'
        a_df['pre_post_delta_category'][a_df['delta_aligned_start_t'] < 0.0] = 'pre-delta'

    ## OUTPUTS: single_time_bin_size_all_sessions_ripple_df, single_time_bin_size_all_sessions_ripple_time_bin_df
    return single_time_bin_size_all_sessions_epochs_df, single_time_bin_size_all_sessions_epochs_time_bin_df


from functools import partial
from pyphoplacecellanalysis.Pho2D.plotly.plotly_templates import PlotlyHelpers
import plotly.io as pio
import plotly.graph_objects as go
from ipydatagrid import Expr, DataGrid, TextRenderer, BarRenderer # for use in DataFrameFilter
from IPython.display import display, Javascript
import base64
import solara # `pip install "solara[assets]`
import json
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from pyphocorehelpers.gui.Jupyter.simple_widgets import CheckBoxListWidget
from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorExporting
from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
from pyphocorehelpers.assertion_helpers import Assert
from attrs import define, field, Factory
from pyphoplacecellanalysis.Pho2D.data_exporting import PosteriorPlottingDatasource, LoadedPosteriorContainer


def _build_solera_file_download_widget(fig, filename="figure-image.png", label="Save Figure"):
    """ 
    _file_download_widget = _build_solera_file_download_widget(fig=self.figure_widget, filename="figure-image.png")
    
    """
    png_bytes = pio.to_image(fig, format='png')
    mime_type="image/png"
    data = deepcopy(png_bytes)
    return solara.FileDownload.widget(data=data, filename=filename, label=label, mime_type=mime_type, )

@custom_define(slots=False, eq=False)
class DataframeFilterPredicates(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    is_enabled: bool = serialized_attribute_field(default=True)
    

@custom_define(slots=False, eq=False)
class DataFrameFilter(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ handles interactive filtering of dataframes by presenting a jupyter widget interface.
    
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import DataFrameFilter


        assert 'best_overall_quantile' in all_sessions_all_scores_ripple_df
        assert 'best_overall_quantile' in all_sessions_MultiMeasure_ripple_df

        min_wcorr_threshold: float = 0.7
        high_pearsonr_threshold: float = 0.9
        high_shuffle_score_threshold: float = 0.9
        high_shuffle_wcorr_z_score_threshold: float = 0.9

        additional_filter_predicates = {
            'high_wcorr': (lambda df: np.any((df[['long_best_wcorr', 'short_best_wcorr']].abs() > min_wcorr_threshold), axis=1)),
            'user_selected': lambda df: np.all((df[['is_user_annotated_epoch', 'is_valid_epoch']]), axis=1),
            'high_pearsonr_corr': (lambda df: np.any((df[['long_LR_pf_peak_x_pearsonr', 'long_RL_pf_peak_x_pearsonr', 'short_LR_pf_peak_x_pearsonr', 'short_RL_pf_peak_x_pearsonr']].abs() > high_pearsonr_threshold), axis=1)),
            'high_shuffle_percentile_score': (lambda df: (df['best_overall_quantile'].abs() > high_shuffle_score_threshold)),
            'high_shuffle_wcorr_z_score': (lambda df: (df['best_overall_wcorr_z'].abs() > high_shuffle_wcorr_z_score_threshold)),
        }

        ## ensure that the qclu is always before the frateThresh, reversing them if needed:
        replay_name: str = 'withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0'
        # time_bin_size: float = 0.025
        time_bin_size: Tuple[float] = (0.025, 0.058)

        _build_filter_changed_plotly_plotting_callback_fn = DataFrameFilter._build_plot_callback(earliest_delta_aligned_t_start=earliest_delta_aligned_t_start, latest_delta_aligned_t_end=latest_delta_aligned_t_end, save_plotly=save_plotly, resolution_multiplier=resolution_multiplier)

        df_filter: DataFrameFilter = DataFrameFilter(
            all_sessions_ripple_df=all_sessions_ripple_df,
            all_sessions_ripple_time_bin_df=all_sessions_ripple_time_bin_df,
            all_sessions_MultiMeasure_ripple_df=all_sessions_MultiMeasure_ripple_df,
            all_sessions_all_scores_ripple_df=all_sessions_all_scores_ripple_df,
            all_sessions_laps_df=all_sessions_laps_df,
            all_sessions_laps_time_bin_df=all_sessions_laps_time_bin_df,
            all_sessions_MultiMeasure_laps_df=all_sessions_MultiMeasure_laps_df,
            additional_filter_predicates=additional_filter_predicates,
            on_filtered_dataframes_changed_callback_fns={'build_filter_changed_plotly_plotting_callback_fn': _build_filter_changed_plotly_plotting_callback_fn},
            active_plot_df_name='filtered_all_sessions_all_scores_ripple_df',
        )

        # Set initial values: ________________________________________________________________________________________________ #
        df_filter.replay_name = replay_name # 'withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]'
        # df_filter.time_bin_size = (time_bin_size, )
        df_filter.time_bin_size = time_bin_size
        # df_filter.time_bin_size = (0.025, 0.058)
        df_filter.update_filters()
        df_filter.display()

    """
    # Original DataFrames passed during initialization
    all_sessions_ripple_df: pd.DataFrame = serialized_field()
    all_sessions_ripple_time_bin_df: pd.DataFrame = serialized_field()
    all_sessions_MultiMeasure_ripple_df: pd.DataFrame = serialized_field()
    all_sessions_all_scores_ripple_df: pd.DataFrame = serialized_field()

    # Original DataFrames for laps
    all_sessions_laps_df: pd.DataFrame = serialized_field()
    all_sessions_laps_time_bin_df: pd.DataFrame = serialized_field()
    all_sessions_MultiMeasure_laps_df: pd.DataFrame = serialized_field()

    # Filtered DataFrames (initialized to None)
    filtered_all_sessions_ripple_df: pd.DataFrame = non_serialized_field(init=False, default=None)
    filtered_all_sessions_ripple_time_bin_df: pd.DataFrame = non_serialized_field(init=False, default=None)
    filtered_all_sessions_MultiMeasure_ripple_df: pd.DataFrame = non_serialized_field(init=False, default=None)
    filtered_all_sessions_all_scores_ripple_df: pd.DataFrame = non_serialized_field(init=False, default=None)

    # Filtered DataFrames for laps
    filtered_all_sessions_laps_df: pd.DataFrame = non_serialized_field(init=False, default=None)
    filtered_all_sessions_laps_time_bin_df: pd.DataFrame = non_serialized_field(init=False, default=None)
    filtered_all_sessions_MultiMeasure_laps_df: pd.DataFrame = non_serialized_field(init=False, default=None)

    active_plot_df_name: str = serialized_attribute_field(default='filtered_all_sessions_all_scores_ripple_df')
    active_plot_variable_name: str = serialized_attribute_field(default='P_Short')
    active_plot_fn_kwargs: Dict = serialized_field(default=Factory(dict))
    # Add filename attribute
    filename: str = serialized_attribute_field(init=False, default='figure.png')
    

    additional_filter_predicates = non_serialized_field(default=Factory(dict)) # a list of boolean predicates to be applied as filters
    on_filtered_dataframes_changed_callback_fns = non_serialized_field(default=Factory(dict)) # a list of callables that will be called when the filters are changed. 
    
    selected_points = non_serialized_field(default=Factory(dict))

    # Widgets (will be initialized in __attrs_post_init__) _______________________________________________________________ #
    replay_name_widget = non_serialized_field(init=False)
    time_bin_size_widget = non_serialized_field(init=False)
    active_filter_predicate_selector_widget: CheckBoxListWidget = non_serialized_field(init=False)
    active_plot_df_name_selector_widget = non_serialized_field(init=False)
    active_plot_variable_name_widget = non_serialized_field(init=False)
    
    output_widget: widgets.Output = non_serialized_field(init=False)
    figure_widget: go.FigureWidget = non_serialized_field(init=False)
    table_widget: DataGrid = non_serialized_field(init=False)

    # Add button widgets as class attributes
    button_copy: widgets.Button = non_serialized_field(init=False)
    button_download: widgets.widget = non_serialized_field(init=False)
    filename_label: widgets.Label = non_serialized_field(init=False)
    # Add Output widget for JavaScript execution
    # js_output: widgets.Output = non_serialized_field(init=False)
    hover_posterior_preview_figure_widget: go.FigureWidget = non_serialized_field(init=False)
    # hover_posterior_data: LoadedPosteriorContainer = non_serialized_field()
    hover_posterior_data: PosteriorPlottingDatasource = non_serialized_field()
    

    # Begin Properties ___________________________________________________________________________________________________ #
    @property
    def replay_name(self) -> str:
        """The replay_name property."""
        return self.replay_name_widget.value
    @replay_name.setter
    def replay_name(self, value):
        self.replay_name_widget.value = value

    @property
    def time_bin_size(self) -> str:
        """The time_bin_size property."""
        return self.time_bin_size_widget.value
    @time_bin_size.setter
    def time_bin_size(self, value):
        # Combine all DataFrames to get unique options
        # combined_df = self.all_sessions_ripple_df.append(self.all_sessions_laps_df, ignore_index=True)
        # replay_name_options = sorted(combined_df['custom_replay_name'].unique())
        # time_bin_size_options = sorted(combined_df['time_bin_size'].unique())

        # Set default initial_time_bin_sizes if not provided
        if value is None:
            # Default to selecting all options or the first option
            # initial_time_bin_sizes = (time_bin_size_options[0],)
            value = tuple() # empty tuple
            pass
        else:
            # Ensure initial_time_bin_sizes is a tuple
            if isinstance(value, (float, int)):
                value = (value,)
            elif isinstance(value, list):
                value = tuple(value)
            elif isinstance(value, tuple):
                pass  # already a tuple
            else:
                raise ValueError("initial_time_bin_sizes must be a float, int, list, or tuple")

        try:
            self.time_bin_size_widget.value = value
        except Exception as e:
            print(f'value: {value} is no in the list of valid options: {self.time_bin_size_widget.options}')
            raise e
        

    
    @property
    def filter_context(self) -> IdentifyingContext:
        """The time_bin_size property."""
        return IdentifyingContext(time_bin_sizes=self.time_bin_size, custom_suffix=self.replay_name)
    

    @property
    def original_df_list(self) -> Tuple[pd.DataFrame]:
        """The original_df_list property."""
        return (
            self.all_sessions_ripple_df,
            self.all_sessions_ripple_time_bin_df,
            self.all_sessions_MultiMeasure_ripple_df,
            self.all_sessions_all_scores_ripple_df,
            self.all_sessions_laps_df,
            self.all_sessions_laps_time_bin_df,
            self.all_sessions_MultiMeasure_laps_df
        )
        
    @property
    def filtered_df_list(self) -> Tuple[pd.DataFrame]:
        """The original_df_list property."""
        return (
            self.filtered_all_sessions_ripple_df,
            self.filtered_all_sessions_ripple_time_bin_df,
            self.filtered_all_sessions_MultiMeasure_ripple_df,
            self.filtered_all_sessions_all_scores_ripple_df,
            self.filtered_all_sessions_laps_df,
            self.filtered_all_sessions_laps_time_bin_df,
            self.filtered_all_sessions_MultiMeasure_laps_df
        )


    @property
    def original_df_dict(self) -> Dict[str, pd.DataFrame]:
        """The original_df_list property."""
        return {k:v for k, v in dict(
            all_sessions_ripple_df=self.all_sessions_ripple_df,
            all_sessions_ripple_time_bin_df=self.all_sessions_ripple_time_bin_df,
            all_sessions_MultiMeasure_ripple_df=self.all_sessions_MultiMeasure_ripple_df,
            all_sessions_all_scores_ripple_df=self.all_sessions_all_scores_ripple_df,
            all_sessions_laps_df=self.all_sessions_laps_df,
            all_sessions_laps_time_bin_df=self.all_sessions_laps_time_bin_df,
            all_sessions_MultiMeasure_laps_df=self.all_sessions_MultiMeasure_laps_df
        ).items() if (v is not None)}
        
    @property
    def filtered_df_dict(self) -> Dict[str, pd.DataFrame]:
        """The original_df_list property."""
        return {k:v for k, v in dict(
            filtered_all_sessions_ripple_df=self.filtered_all_sessions_ripple_df,
            filtered_all_sessions_ripple_time_bin_df=self.filtered_all_sessions_ripple_time_bin_df,
            filtered_all_sessions_MultiMeasure_ripple_df=self.filtered_all_sessions_MultiMeasure_ripple_df,
            filtered_all_sessions_all_scores_ripple_df=self.filtered_all_sessions_all_scores_ripple_df,
            filtered_all_sessions_laps_df=self.filtered_all_sessions_laps_df,
            filtered_all_sessions_laps_time_bin_df=self.filtered_all_sessions_laps_time_bin_df,
            filtered_all_sessions_MultiMeasure_laps_df=self.filtered_all_sessions_MultiMeasure_laps_df
        ).items() if (v is not None)}


    @property
    def original_df_names(self) -> List[str]:
        return sorted(list(self.original_df_dict.keys()))
    
    @property
    def filtered_df_names(self) -> List[str]:
        return sorted(list(self.filtered_df_dict.keys()))
    
    @property
    def filtered_size_info_df(self) -> pd.DataFrame:
        """The size of the filtered dataframes with the current filters."""
        n_records_tuples = [(name, len(df)) for name, df in self.filtered_df_dict.items() if (df is not None)]
        return pd.DataFrame(n_records_tuples, columns=['df_name', 'n_elements'])


    @property
    def plot_variable_name_options(self) -> List[str]:
        return sorted(['P_Short', 'best_overall_quantile', 'best_overall_wcorr_z'])
    

    @property
    def active_plot_df(self) -> pd.DataFrame:
        """The selected filtered dataframe to use with the plot."""
        return self.filtered_df_dict[self.active_plot_df_name]




    # ==================================================================================================================== #
    # Initializers                                                                                                         #
    # ==================================================================================================================== #
    def __attrs_post_init__(self):
        # This method runs after the generated __init__
        self._init_filtered_dataframes()
        
        self._setup_widgets()
        # Initial filtering with default widget values
        self.update_filtered_dataframes(self.replay_name_widget.value, self.time_bin_size_widget.value)
        
        # Button Widget Initialize ___________________________________________________________________________________________ #
        # Set up the buttons after figure_widget is created
        self._setup_widgets_buttons()

    def _setup_widgets(self):
        import plotly.subplots as sp
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import PlotlyFigureContainer
        
        # Extract unique options for the widgets
        replay_name_options = sorted(self.active_plot_df['custom_replay_name'].astype(str).unique())
        time_bin_size_options = sorted(self.active_plot_df['time_bin_size'].unique())
        
        # Create dropdown widgets with adjusted layout and style
        self.replay_name_widget = widgets.Dropdown(
            options=replay_name_options,
            description='Replay Name:',
            disabled=False,
            layout=widgets.Layout(width='500px'),
            style={'description_width': 'initial'}
        )
        
        self.active_plot_df_name_selector_widget = widgets.Dropdown(
            options=sorted(self.filtered_df_names),
            description='Plot df Name:',
            disabled=False,
            layout=widgets.Layout(width='400px'),
            style={'description_width': 'initial'}
        )
        self.active_plot_df_name_selector_widget.value = self.active_plot_df_name
        
        self.active_plot_variable_name_widget = widgets.Dropdown(
            options=sorted(self.plot_variable_name_options),
            description='Plot Variable Name:',
            disabled=False,
            layout=widgets.Layout(width='300px'),
            style={'description_width': 'initial'}
        )
        self.active_plot_variable_name_widget.value = self.active_plot_variable_name
        
        # Use SelectMultiple widget for time_bin_size
        self.time_bin_size_widget = widgets.SelectMultiple(
            options=time_bin_size_options,
            description='Time Bin Size:',
            disabled=False,
            layout=widgets.Layout(width='300px', height='100px'),
            style={'description_width': 'initial'},
        )

        self.active_filter_predicate_selector_widget = CheckBoxListWidget(options_list=list(self.additional_filter_predicates.keys()))
            # description='Filter Predicates:',
            # disabled=False,
        # )

        self.output_widget = widgets.Output(layout=widgets.Layout(width='100%', # min_width='200px', height='100px',
                                                                  border='1px solid black'),
                                                                  ) #  {'border': '1px solid black'}
        self.figure_widget, did_create_new_figure = PlotlyFigureContainer._helper_build_pre_post_delta_figure_if_needed(extant_figure=None, use_latex_labels=False, main_title='test', figure_class=go.FigureWidget)
        
        ## initialize the preview widget:
        self.hover_posterior_preview_figure_widget = go.FigureWidget() # .set_subplots(rows=1, cols=1, column_widths=[0.10, 0.80, 0.10], horizontal_spacing=0.01, shared_yaxes=True, column_titles=[pre_delta_label, main_title, post_delta_label])
        # self.hover_posterior_preview_figure_widget.layout
        
        # Set up observers to handle changes in widget values
        self.replay_name_widget.observe(self._on_widget_change, names='value')
        self.time_bin_size_widget.observe(self._on_widget_change, names='value')
        self.active_filter_predicate_selector_widget.observe(self._on_widget_change, names='value')
        self.active_plot_df_name_selector_widget.observe(self._on_widget_change, names='value')
        self.active_plot_variable_name_widget.observe(self._on_widget_change, names='value')
        

        self.table_widget = DataGrid(self.filtered_size_info_df,
                                base_row_size=15, base_column_size=300, horizontal_stripes=True,
                                #  renderers=renderers,
                                )
        # self.table_widget.transform([{"type": "sort", "columnIndex": 2, "desc": True}])
        # self.table_widget.auto_fit_columns = True
        
        # Set layout properties
        # self.table_widget.layout = widgets.Layout(flex='0 1 auto', width='auto')
        # self.output_widget.layout = widgets.Layout(flex='1 1 auto', width='auto')

        # Combine in HBox
        # container = widgets.HBox([shrink_widget, grow_widget], layout=widgets.Layout(width='100%'))



    def _setup_widgets_buttons(self):
        """Sets up the copy and download buttons."""
        self.button_copy = widgets.Button(description="Copy to Clipboard", icon='copy')
        # self.button_download = widgets.Button(description="Download Image", icon='save')
        self.button_download =  _build_solera_file_download_widget(fig=self.figure_widget, filename="figure-image.png")
        
        # @solara.component
        # def Page():
        #     def get_data():
        #         # I run in a thread, so I can do some heavy processing
        #         time.sleep(3)
        #         # I only get called when the download is requested
        #         return "This is the content of the file"
        #     solara.FileDownload(get_data, "solara-lazy-download.txt")
            

        self.filename_label = widgets.Label()

        def on_copy_button_click(b):
            # Convert the figure to a PNG image
            # Retrieve width and height if set
            width = self.figure_widget.layout.width
            height = self.figure_widget.layout.height
            to_image_kwargs = {}
            print(f"Width: {width}, Height: {height}")
            if width is not None:
                to_image_kwargs['width'] = width
            if height is not None:
                to_image_kwargs['height'] = height

            png_bytes = pio.to_image(self.figure_widget, format='png', **to_image_kwargs)
            encoded_image = base64.b64encode(png_bytes).decode('utf-8')

            # JavaScript code to copy the image to the clipboard using the canvas element
            js_code = f'''
                const img = new Image();
                img.src = 'data:image/png;base64,{encoded_image}';
                img.onload = function() {{
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    canvas.toBlob(function(blob) {{
                        const item = new ClipboardItem({{ 'image/png': blob }});
                        navigator.clipboard.write([item]).then(function() {{
                            console.log('Image copied to clipboard');
                        }}).catch(function(error) {{
                            console.error('Error copying image to clipboard: ', error);
                        }});
                    }});
                }};
            '''

            display(Javascript(js_code))

        self.button_copy.on_click(on_copy_button_click)

        ## Finish setup:
        self.on_widget_update_filename()  # Initialize filename and label
        # Set up observers for figure changes
        # self.figure_widget.layout.on_change(self.on_fig_layout_change, 'title', 'meta')
        # Initialize the Output widget for JavaScript execution
        # self.js_output = widgets.Output()


        # self.button_copy.on_click(self.on_copy_button_click)
        # self.button_download.on_click(self.on_download_button_click)

    # ==================================================================================================================== #
    # Widget Update Functions                                                                                              #
    # ==================================================================================================================== #
    def on_widget_update_filename(self):
        """Updates the filename and label based on the figure's title or metadata."""
        fig = self.figure_widget
        preferred_filename = fig.layout.meta.get('preferred_filename') if fig.layout.meta else None
        if preferred_filename:
            self.filename = f"{preferred_filename}.png"
            self.filename_label.value = preferred_filename
        else:
            title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "figure"
            self.filename = f"{title.replace(' ', '_')}.png"
            self.filename_label.value = title

        ## rebuild the download widget with the current figure
        self.button_download =  _build_solera_file_download_widget(fig=self.figure_widget, filename=Path(self.filename).with_suffix('.png').as_posix())


    def on_fig_layout_change(self, layout, *args):
        """Callback for when the figure's layout changes."""
        self.on_widget_update_filename()


    def _on_widget_change(self, change):
        active_plot_df_name = self.active_plot_df_name_selector_widget.value
        self.active_plot_df_name = self.active_plot_df_name_selector_widget.value
        self.active_plot_variable_name = self.active_plot_variable_name_widget.value
        
        # Update filtered DataFrames when widget values change
        self.update_filtered_dataframes(self.replay_name_widget.value, self.time_bin_size_widget.value)


    def display(self):
        """Displays the widgets."""
        # Arrange your widgets as needed
        display(widgets.VBox([
            widgets.HBox([
                self.replay_name_widget, 
                self.time_bin_size_widget, 
                self.active_filter_predicate_selector_widget,
                # self.table_widget
            ], #layout=widgets.Layout(width='100%'),
            ),
            widgets.HBox([
                self.active_plot_df_name_selector_widget, 
                self.active_plot_variable_name_widget
            ]),
            self.figure_widget,
            widgets.HBox([self.button_copy, self.button_download, self.filename_label],
                        #   layout=widgets.Layout(width='100%'),
                          ),
            # self.js_output,  # Include the Output widget to allow the buttons to perform their actions
            widgets.HBox([self.output_widget, ],
                          layout=widgets.Layout(height='300px', width='100%'),
                          ),              
            widgets.HBox([self.table_widget,
                          self.hover_posterior_preview_figure_widget,
                          ], layout=widgets.Layout(height='300px', width='100%')),
        ]))
        

    # ==================================================================================================================== #
    # Plot Updating Functions                                                                                              #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['plotting'], input_requires=[], output_provides=[], uses=['_perform_plot_pre_post_delta_scatter'], used_by=[], creation_date='2024-11-20 13:08', related_items=[])
    @classmethod
    def _build_plot_callback(cls, earliest_delta_aligned_t_start, latest_delta_aligned_t_end, save_plotly, should_save: bool = False, resolution_multiplier=1, enable_debug_print=False, **extra_plot_kwargs):
        # fig_size_kwargs = {'width': 1650, 'height': 480}
        
        
                

        # fig_size_kwargs = {'width': resolution_multiplier*1650, 'height': resolution_multiplier*480}
        ## set up figure size
        fig_size_kwargs = {'width': (resolution_multiplier * 1800), 'height': (resolution_multiplier*480)}
        # fig_size_kwargs = {'width': (resolution_multiplier * 1080), 'height': resolution_multiplier*480}
        is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)
        pio.templates.default = template

        ## INPUTS: earliest_delta_aligned_t_start, latest_delta_aligned_t_end
        # should_save: bool = True        

        _new_perform_plot_pre_post_delta_scatter = partial(
            _perform_plot_pre_post_delta_scatter,
            time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end),
            fig_size_kwargs=fig_size_kwargs,
            is_dark_mode=is_dark_mode,
            save_plotly=save_plotly,
        )

        _new_perform_plot_pre_post_delta_scatter_with_embedded_context = partial(
            _new_perform_plot_pre_post_delta_scatter,
            data_context=None,
        )
        
        extra_plot_kwargs = deepcopy(extra_plot_kwargs)

        def _build_filter_changed_plotly_plotting_callback_fn(df_filter: "DataFrameFilter", should_save:bool=False, **kwargs):
            """ `filtered_all_sessions_all_scores_ripple_df` versions -
            captures: _perform_plot_pre_post_delta_scatter_with_embedded_context, should_save, extra_plot_kwargs
            
            """
            # def _plot_hoverred_heatmap_preview_posterior(df_filter: DataFrameFilter, last_selected_idx: Optional[int] = 0):
            def _plot_hoverred_heatmap_preview_posterior(df_filter: DataFrameFilter, a_heatmap_img: Optional[NDArray]):
                # if last_selected_idx is None:
                #     # df_filter.hover_posterior_data.ripple_img_dict
                #     # df_filter.hover_posterior_preview_figure_widget.add_heatmap()

                # a_heatmap_img = df_filter.hover_posterior_data.ripple_img_dict['long_LR'][last_selected_idx]    
                ## update the plot
                df_filter.hover_posterior_preview_figure_widget.add_heatmap(z=a_heatmap_img, showscale=False, name='selected_posterior', )



            # df_filter.output_widget.clear_output(wait=True)
            active_plot_df_name: str = df_filter.active_plot_df_name
            active_plot_df: pd.DataFrame = df_filter.active_plot_df
            plot_variable_name: str = df_filter.active_plot_variable_name
            
            assert plot_variable_name in active_plot_df.columns, f"plot_variable_name: '{plot_variable_name}' is not present in active_plot_df.columns! Cannot plot!"
            
            # extra_plot_kwargs = deepcopy(extra_plot_kwargs)
            active_plot_kwargs = (extra_plot_kwargs | kwargs) 
            fig, new_fig_context, _extras_output_dict, figure_out_paths = _new_perform_plot_pre_post_delta_scatter_with_embedded_context(concatenated_ripple_df=deepcopy(active_plot_df), is_dark_mode=False, should_save=should_save, extant_figure=df_filter.figure_widget,
                                                                                                                                    variable_name=plot_variable_name, **active_plot_kwargs) # , enable_custom_widget_buttons=True
            
            # Customize the hovertemplate
            fig.update_traces(
                # hovertemplate="<b>sess:</b> %{customdata[0]}<br>"
                #             # "<b>X:</b> %{x}<br>"
                #             "<b>start, duration:</b> %{customdata[3]}, %{customdata[4]}<br>"
                #             "<b>Y:</b> %{y}<br>"
                #             "<b>custom_replay:</b> %{customdata[1]}",
                
                hovertemplate="<b>sess:</b> %{customdata[0]} | <b>replay_name:</b> %{customdata[1]} | <b>time_bin_size:</b> %{customdata[2]}<br>"
                            "<b>start:</b> %{customdata[3]}<br>",
                customdata=active_plot_df[["session_name", "custom_replay_name", "time_bin_size", "start", "duration"]].values,
                hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.4)", font=dict(color="black")),
            )

            if df_filter.hover_posterior_data is not None:
                if df_filter.hover_posterior_data.plot_heatmap_fn is None:
                    df_filter.hover_posterior_data.plot_heatmap_fn = (lambda a_df_filter, a_heatmap_img, *args, **kwargs: a_df_filter.hover_posterior_preview_figure_widget.add_heatmap(z=a_heatmap_img, showscale=False, name='selected_posterior', ))
                    

            def on_click(trace, points, selector):
                if points.point_inds:
                    ## has selection:
                    ind = points.point_inds[0]
                    session_name = df_filter.active_plot_df['session_name'].iloc[ind]
                    custom_replay_name = df_filter.active_plot_df['custom_replay_name'].iloc[ind]
                    time_bin_size = df_filter.active_plot_df['time_bin_size'].iloc[ind]
                    start_t = df_filter.active_plot_df['start'].iloc[ind]
                    stop_t = df_filter.active_plot_df['stop'].iloc[ind]
                    
                    df_filter.output_widget.clear_output()
                    with df_filter.output_widget:
                        print(f"Clicked point index: {ind}")
                        print(f'start_t, stop_t: {start_t}, {stop_t}')
                        print(f"session_name: {session_name}")
                        print(f"custom_replay_name: {custom_replay_name}")
                        print(f"time_bin_size: {time_bin_size}")

                    selected_event_session_context: IdentifyingContext = IdentifyingContext(session_name=session_name, custom_replay_name=custom_replay_name, time_bin_size=time_bin_size)
                    
                    selected_event_context: IdentifyingContext = IdentifyingContext(start_t=start_t, stop_t=stop_t)
                    
                    curr_selected_points_dict = df_filter.selected_points.get(selected_event_session_context, None)
                    if curr_selected_points_dict is None:
                        df_filter.selected_points[selected_event_session_context] = [] # [start_t, ] ## add start_t only
                        

                    if selected_event_context not in df_filter.selected_points[selected_event_session_context]:
                        df_filter.selected_points[selected_event_session_context].append(selected_event_context) # add to the selection points list
                        
                    
                 ## update selected points
                    

                    ## try to update the selected heatmap posterior:
                    try:
                        ## try to update by start_t, stop_t
                        _heatmap_data = df_filter.hover_posterior_data.get_posterior_data(session_name=session_name, custom_replay_name=custom_replay_name,
                                                                          a_decoder_name='long_LR', last_selected_idx=ind)
                        
                        # _plot_hoverred_heatmap_preview_posterior(df_filter=df_filter, last_selected_idx=ind) #TODO 2024-11-26 07:32: - [ ] does this work?
                        _plot_hoverred_heatmap_preview_posterior(df_filter=df_filter, a_heatmap_img=_heatmap_data)

                    except Exception as e:
                        print(f'encountered exception when trying to call `_plot_hoverred_heatmap_preview_posterior(..., last_selected_idx={ind}, start_t: {start_t}, stop_t: {stop_t}). Error {e}. Skipping.')
                                            
                else:
                    ## no selection:
                    # print(f'NOPE! points: {points}, trace: {trace}')
                    # df_filter.output_widget.clear_output()
                    if enable_debug_print:
                        with df_filter.output_widget:
                            print(f'NOPE! points: {points}, trace: {trace}')
                    

            fig.layout.hovermode = 'closest'
            if len(fig.data) > 0:
                ## prevent IndexError: tuple index out of range
                fig.data[0].on_click(on_click)

            scatter_traces = list(fig.select_traces(selector=None, row=1, col=2))
            for trace in scatter_traces:
                trace.on_click(on_click)

            if fig is not None:
                df_filter.figure_widget = fig

        ## end def _build_filter_changed_plotly_plotting_callback_fn(...)
        
        return _build_filter_changed_plotly_plotting_callback_fn


    # ==================================================================================================================== #
    # Data Update Functions                                                                                                #
    # ==================================================================================================================== #
    
    def _init_filtered_dataframes(self, enable_overwrite_is_filter_included_column: bool=True):
        """ builds the filtered dataframes from the original_df_dict. Initially they are unfiltered. """
        ## Update the 'is_filter_included' column on the original dataframes
        for name, df in self.original_df_dict.items():
            if enable_overwrite_is_filter_included_column or ('is_filter_included' not in df.columns):
                df['is_filter_included'] = True  # Initialize with default value
            filtered_name: str = f"filtered_{name}"
            filtered_df = deepcopy(df[df['is_filter_included']])
            setattr(self, filtered_name, filtered_df)


    def update_filtered_dataframes(self, replay_name, time_bin_sizes, debug_print=True, enable_overwrite_is_filter_included_column: bool=True):
        """ Perform filtering on each DataFrame
        """
        if not time_bin_sizes:
            print("Please select at least one Time Bin Size.")
            return

        self.output_widget.clear_output()
        with self.output_widget:
            # Convert time_bin_sizes to a list if it's not already
            if isinstance(time_bin_sizes, (float, int)):
                time_bin_sizes = [time_bin_sizes]
            elif isinstance(time_bin_sizes, tuple):
                time_bin_sizes = list(time_bin_sizes)
                

            enabled_filter_predicate_list = self.active_filter_predicate_selector_widget.value
            did_applying_predicate_fail_for_df_dict = {}
            
            ## Update the 'is_filter_included' column on the original dataframes
            for name, df in self.original_df_dict.items():
                filtered_name: str = f"filtered_{name}"
                did_applying_predicate_fail_for_df_dict[filtered_name] = False ## start false
                
                if enable_overwrite_is_filter_included_column or ('is_filter_included' not in df.columns):
                    df['is_filter_included'] = True  # Initialize with default value

                # Update based on conditions
                df['is_filter_included'] = (df['custom_replay_name'] == replay_name) & (df['time_bin_size'].isin(time_bin_sizes))

                ## Apply predicates
                active_predicate_filter_name_modifier = []
                for a_predicate_name, a_predicate_fn in self.additional_filter_predicates.items():
                    if a_predicate_name in enabled_filter_predicate_list:
                        did_predicate_fail: bool = False # whether an error was encountered when trying to evaluate this predicate for this df
                        try:
                            is_predicate_true = a_predicate_fn(df)
                            did_predicate_fail = False
                        except KeyError as e:
                            if debug_print:
                                print(f'NOTE: failed to apply predicate "{a_predicate_name}" to df: {name}')                            
                            is_predicate_true = False
                            did_predicate_fail = True
                        except Exception as e:
                            did_predicate_fail = True
                            raise
                        if did_predicate_fail:
                            did_applying_predicate_fail_for_df_dict[filtered_name] = (did_applying_predicate_fail_for_df_dict[filtered_name] or did_predicate_fail)

                        df['is_filter_included'] = np.logical_and(df['is_filter_included'], is_predicate_true)
                        if not did_predicate_fail:
                            active_predicate_filter_name_modifier.append(a_predicate_name)
                # END for a_predicate_name, a_predicate_fn

                
                filtered_df = deepcopy(df[df['is_filter_included']])
                
                ## update df metadata to indicate that it is filtered
                df_context_dict = filtered_df.attrs['data_context'].to_dict() #  ## benedict dict
                df_context_dict.pop('filter', None) ## remove old filter
                if len(active_predicate_filter_name_modifier) > 0:
                    active_predicate_filter_name_modifier = '_'.join(active_predicate_filter_name_modifier) ## build string
                    df_context_dict['filter'] = active_predicate_filter_name_modifier
                filtered_df.attrs['data_context'] = IdentifyingContext.init_from_dict(df_context_dict) ## update
                filtered_df.attrs['did_filter_predicate_fail'] = did_applying_predicate_fail_for_df_dict[filtered_name]
                setattr(self, filtered_name, filtered_df)
            # END for name, df

            ## Update sizes table:
            self.table_widget.data = self.filtered_size_info_df
            # self.table_widget.auto_fit_columns = True
            
            if did_applying_predicate_fail_for_df_dict[self.active_plot_df_name]:
                print(f'!!! Warning!!! applying predicates failed for the current active plot df (self.active_plot_df_name: {self.active_plot_df_name})!\n\tthe plotted output has NOT been filtered!')
                

        ## end with self.output_widget
        for k, a_callback_fn in self.on_filtered_dataframes_changed_callback_fns.items():
            # print(f'k: {k}')
            try:
                a_callback_fn(self)
            except Exception as e:
                print(f'WARNING: callback_fn[{k}] failed with error: {e}, skipping.')
                raise
            
        ## Update the preferred_filename from the dataframe metadata:
        self.on_widget_update_filename()


        # ## rebuild the download widget with the current figure
        # self.button_download =  _build_solera_file_download_widget(fig=self.figure_widget, filename=Path(self.filename).with_suffix('.png').as_posix())
    
    def update_filters(self):
        self.update_filtered_dataframes(replay_name=self.replay_name, time_bin_sizes=self.time_bin_size)

    def update_calling_namespace_locals(self):
        """ dangerous!! Updates the calling namespace (such as a jupyter notebook cell) """
        # Update the variables in the notebook's user namespace
        ipython = get_ipython()
        user_ns = ipython.user_ns

        # Update ripple DataFrames
        user_ns['filtered_all_sessions_ripple_df'] = self.filtered_all_sessions_ripple_df
        user_ns['filtered_all_sessions_ripple_time_bin_df'] = self.filtered_all_sessions_ripple_time_bin_df
        user_ns['filtered_all_sessions_MultiMeasure_ripple_df'] = self.filtered_all_sessions_MultiMeasure_ripple_df
        user_ns['filtered_all_sessions_all_scores_ripple_df'] = self.filtered_all_sessions_all_scores_ripple_df
        
        # Update laps DataFrames
        user_ns['filtered_all_sessions_laps_df'] = self.filtered_all_sessions_laps_df
        user_ns['filtered_all_sessions_laps_time_bin_df'] = self.filtered_all_sessions_laps_time_bin_df
        user_ns['filtered_all_sessions_MultiMeasure_laps_df'] = self.filtered_all_sessions_MultiMeasure_laps_df

    # Update instance values from a dictionary
    def update_instance_from_dict(self, update_dict):
        # Filter the dictionary to match only fields in the class
        field_names = {field.name for field in self.__attrs_attrs__}
        filtered_dict = {k: v for k, v in update_dict.items() if k in field_names}
        for k, v in filtered_dict.items():
            setattr(self, k, v)
        # return evolve(self, **filtered_dict)


    # Accessor methods for the filtered DataFrames _______________________________________________________________________ #
    # Accessor methods for ripple DataFrames
    def get_filtered_all_sessions_ripple_df(self):
        return self.filtered_all_sessions_ripple_df
    
    def get_filtered_all_sessions_ripple_time_bin_df(self):
        return self.filtered_all_sessions_ripple_time_bin_df
    
    def get_filtered_all_sessions_MultiMeasure_ripple_df(self):
        return self.filtered_all_sessions_MultiMeasure_ripple_df
    
    def get_filtered_all_sessions_all_scores_ripple_df(self):
        return self.filtered_all_sessions_all_scores_ripple_df
    
    # Accessor methods for laps DataFrames
    def get_filtered_all_sessions_laps_df(self):
        return self.filtered_all_sessions_laps_df
    
    def get_filtered_all_sessions_laps_time_bin_df(self):
        return self.filtered_all_sessions_laps_time_bin_df
    
    def get_filtered_all_sessions_MultiMeasure_laps_df(self):
        return self.filtered_all_sessions_MultiMeasure_laps_df


    @classmethod
    def safe_boolean_predicate_wrapper(cls, predicate_fn, df):
        """ returns False if predicate can't be evaluated"""
        try:
            return predicate_fn(df)
        except Exception as e:
            # raise e
            print(f'failed to apply predicate to df')
            return False


    # ==================================================================================================================== #
    # Serialization                                                                                                        #
    # ==================================================================================================================== #
    #TODO 2024-11-20 08:26: - [ ] These were written by ChatGPT and I'm using them instead of my normal HDFSerialization stuff.    

    # Class variables for attribute names
    DATAFRAME_ATTR_NAMES = [
        'all_sessions_ripple_df',
        'all_sessions_ripple_time_bin_df',
        'all_sessions_MultiMeasure_ripple_df',
        'all_sessions_all_scores_ripple_df',
        'all_sessions_laps_df',
        'all_sessions_laps_time_bin_df',
        'all_sessions_MultiMeasure_laps_df',
        'filtered_all_sessions_ripple_df',
        'filtered_all_sessions_ripple_time_bin_df',
        'filtered_all_sessions_MultiMeasure_ripple_df',
        'filtered_all_sessions_all_scores_ripple_df',
        'filtered_all_sessions_laps_df',
        'filtered_all_sessions_laps_time_bin_df',
        'filtered_all_sessions_MultiMeasure_laps_df'
    ]

    SCALAR_ATTR_NAMES = [
        'active_plot_df_name',
        'active_plot_variable_name',
        'active_plot_fn_kwargs',
        'filename'
    ]


    def save_to_hdf(self, filename):
        """Saves the data (excluding widgets) to a single .hdf file."""
        import json
        with pd.HDFStore(filename, 'w') as store:
            # Save dataframes
            for attr_name in self.DATAFRAME_ATTR_NAMES:
                df = getattr(self, attr_name, None)
                if df is not None:
                    store.put(attr_name, df)
            # Save scalar attributes
            scalar_attrs = {}
            for attr_name in self.SCALAR_ATTR_NAMES:
                attr_value = getattr(self, attr_name)
                try:
                    json.dumps(attr_value)
                    scalar_attrs[attr_name] = attr_value
                except (TypeError, ValueError):
                    # Not serializable, skip or set to None
                    scalar_attrs[attr_name] = None

            # Serialize scalar_attrs to JSON and store in root attributes
            store.get_storer('all_sessions_ripple_df').attrs.scalar_attrs = json.dumps(scalar_attrs)
            # Save additional_filter_predicates keys (since we can't save functions)
            store.get_storer('all_sessions_ripple_df').attrs.additional_filter_predicates_keys = list(self.additional_filter_predicates.keys())


    @classmethod
    def load_from_hdf(cls, filename) -> "DataFrameFilter":
        """Loads data from a .hdf file and returns a new instance."""
        import json
        with pd.HDFStore(filename, 'r') as store:
            # Load dataframes
            dataframes = {}
            for attr_name in cls.DATAFRAME_ATTR_NAMES:
                if attr_name in store:
                    dataframes[attr_name] = store[attr_name]
                else:
                    dataframes[attr_name] = None

            # Load scalar attributes from the attributes of the first dataframe
            attrs = store.get_storer('all_sessions_ripple_df').attrs
            scalar_attrs = json.loads(attrs.scalar_attrs)

            # Create a new instance with loaded data
            instance = cls(
                all_sessions_ripple_df=dataframes.get('all_sessions_ripple_df', None),
                all_sessions_ripple_time_bin_df=dataframes.get('all_sessions_ripple_time_bin_df', None),
                all_sessions_MultiMeasure_ripple_df=dataframes.get('all_sessions_MultiMeasure_ripple_df', None),
                all_sessions_all_scores_ripple_df=dataframes.get('all_sessions_all_scores_ripple_df', None),
                all_sessions_laps_df=dataframes.get('all_sessions_laps_df', None),
                all_sessions_laps_time_bin_df=dataframes.get('all_sessions_laps_time_bin_df', None),
                all_sessions_MultiMeasure_laps_df=dataframes.get('all_sessions_MultiMeasure_laps_df', None),
                additional_filter_predicates={}  # Functions can't be serialized
            )

            # Set filtered dataframes
            for attr_name in cls.DATAFRAME_ATTR_NAMES:
                if attr_name.startswith('filtered_'):
                    setattr(instance, attr_name, dataframes.get(attr_name, None))

            # Set scalar attributes
            for attr_name in cls.SCALAR_ATTR_NAMES:
                if attr_name in scalar_attrs:
                    setattr(instance, attr_name, scalar_attrs[attr_name])
                else:
                    setattr(instance, attr_name, None)

            return instance
        
