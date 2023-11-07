from copy import deepcopy
from typing import Any, List, Tuple
from matplotlib.colors import ListedColormap
from pathlib import Path
from neuropy.core import Epoch
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt


from nptyping import NDArray
from attrs import define, field, Factory, astuple
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import pho_stats_paired_t_test
from neuropy.utils.mixins.time_slicing import add_epochs_id_identity
import scipy.stats
from scipy import ndimage
from neuropy.utils.misc import build_shuffled_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.print_helpers import print_array
import matplotlib.pyplot as plt


from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange


from scipy import stats # _recover_samples_per_sec_from_laps_df
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _build_default_tick, build_scatter_plot_kwargs
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import RasterScatterPlotManager, UnitSortOrderManager, _build_default_tick, _build_scatter_plotting_managers, _prepare_spikes_df_from_filter_epochs, _subfn_build_and_add_scatterplot_row
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _plot_multi_sort_raster_browser


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used in TrackTemplates


# ==================================================================================================================== #
# 2023-10-20 - Close-to-working Rank Order Strategy:                                                                   #
# ==================================================================================================================== #
"""

🟢 2023-10-21 - Z-Score Comparisons with Neuron_ID Shuffled templates
1. Take the intersection of the long and short templates to get only the common cells
2. Determine the long and short "tempaltes": this is done by ranking the aclus for each by their placefields' center of mass. `compute_placefield_center_of_masses`
    2a. `long_pf_peak_ranks`, `short_pf_peak_ranks` - there are one of each of these for each shared aclu.
3. Generate the unit_id shuffled (`shuffled_aclus`, `shuffle_IDXs`) ahead of time to use to shuffle the two templates during the epochs.
4. For each replay event, take each shuffled template
    4a. Iterate through each shuffle and obtain the shuffled templates like `long_pf_peak_ranks[epoch_specific_shuffled_indicies]`, `short_pf_peak_ranks[epoch_specific_shuffled_indicies]`
    4b. compute the spearman rank-order of the event and each shuffled template, and accumulate the results in `long_spearmanr_rank_stats_results`, `short_spearmanr_rank_stats_results`

5. After we're done with the shuffle loop, accumulate the results and convert to the right output format.

6. When all epochs are done, loop through the results (the epochs again) and compute the z-scores for each epoch so they can be compared to each other. Keep track of the means and std_dev for comparisons later, and subtract the two sets of z-scores (long/short) to get the delta_Z for each template.

7. TODO: Next figure out what to do with the array of z-scores and delta_Z. We have:
    n_epochs sets of results
        n_shuffles scores of delta_Z


Usage:
z



"""



@function_attributes(short_name=None, tags=['rank_order', 'shuffle', 'renormalize'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-23 13:05', related_items=[])
def relative_re_ranking(rank_array: NDArray, filter_indicies: NDArray, debug_checking=False, disable_re_ranking: bool=False) -> NDArray:
    """ Re-index the rank_array once filtered flat to extract the global ranks. 
        
    Idea: During each ripple event epoch, only a subset of all cells are active. As a result, we need to extract valid ranks from the epoch's subset so they can be compared directly to the ranks within that epoch.

    """
    if disable_re_ranking:
        # Disable any re-ranking, just return the original ranks
        return rank_array[filter_indicies]
    else:
        if debug_checking:
            global_max_rank = np.max(rank_array)
            global_min_rank = np.min(rank_array) # should be 1.0
            print(f'global_max_rank: {global_max_rank}, global_min_rank: {global_min_rank}')
        subset_rank_array = rank_array[filter_indicies]
        if debug_checking:
            subset_max_rank = np.max(subset_rank_array)
            subset_min_rank = np.min(subset_rank_array)
            print(f'subset_rank_array: {subset_rank_array}, subset_max_rank: {subset_max_rank}, subset_min_rank: {subset_min_rank}')
        subset_rank_array = scipy.stats.rankdata(subset_rank_array) # re-rank the subset 
        if debug_checking:
            re_subset_max_rank = np.max(subset_rank_array)
            re_subset_min_rank = np.min(subset_rank_array)
            print(f're_subset_rank_array: {subset_rank_array}, re_subset_max_rank: {re_subset_max_rank}, re_subset_min_rank: {re_subset_min_rank}')
        return subset_rank_array


def compute_placefield_center_of_masses(tuning_curves):
    return np.squeeze(np.array([ndimage.center_of_mass(x) for x in tuning_curves]))

@define(slots=False, repr=False)
class TrackTemplates:
    """ Holds the four directional templates for direction placefield analysis.
    from PendingNotebookCode import TrackTemplates
    
    History:
        Based off of `ShuffleHelper` on 2023-10-27
        TODO: eliminate functional overlap with `ShuffleHelper`
    """
    long_LR_decoder: BasePositionDecoder = field()
    long_RL_decoder: BasePositionDecoder = field()
    short_LR_decoder: BasePositionDecoder = field()
    short_RL_decoder: BasePositionDecoder = field()
    
    # ## Computed properties
    shared_LR_aclus_only_neuron_IDs: NDArray = field()
    is_good_LR_aclus: NDArray = field()
    
    shared_RL_aclus_only_neuron_IDs: NDArray = field()
    is_good_RL_aclus: NDArray = field()

    ## Computed properties
    decoder_LR_pf_peak_ranks_list: List = field()
    decoder_RL_pf_peak_ranks_list: List = field()


    @classmethod
    def init_from_paired_decoders(cls, LR_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], RL_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder]) -> "ShuffleHelper":
        """ 2023-10-31 - Extract from pairs
        
        """        
        long_LR_decoder, short_LR_decoder = LR_decoder_pair
        long_RL_decoder, short_RL_decoder = RL_decoder_pair
            
        shared_LR_aclus_only_neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs)
        shared_RL_aclus_only_neuron_IDs = deepcopy(long_RL_decoder.neuron_IDs)

    
        # is_good_aclus = np.logical_not(np.isin(shared_aclus_only_neuron_IDs, bimodal_exclude_aclus))
        # shared_aclus_only_neuron_IDs = shared_aclus_only_neuron_IDs[is_good_aclus]

        ## 2023-10-11 - Get the long/short peak locations
        # decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        ## Compute the ranks:
        # decoder_pf_peak_ranks_list = [scipy.stats.rankdata(a_peaks_com, method='dense') for a_peaks_com in decoder_peak_coms_list]
        
        return cls(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, None, shared_RL_aclus_only_neuron_IDs, None,
                    decoder_LR_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense') for a_decoder in (long_LR_decoder, short_LR_decoder)],
                    decoder_RL_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense') for a_decoder in (long_RL_decoder, short_RL_decoder)] )




@define()
class ShuffleHelper:
    """ holds the result of shuffling templates. Used for rank-order analyses """
    shared_aclus_only_neuron_IDs = field()
    is_good_aclus = field()
    shuffled_aclus = field()
    shuffle_IDX = field()
    
    decoder_pf_peak_ranks_list = field()
    
    # long_pf_peak_ranks = field()
    # short_pf_peak_ranks = field()

    @property
    def long_pf_peak_ranks(self):
        """ 2023-10-27 - for backwards compat. """
        assert len(self.decoder_pf_peak_ranks_list) >= 2
        return self.decoder_pf_peak_ranks_list[0]

    @property
    def short_pf_peak_ranks(self):
        """ 2023-10-27 - for backwards compat. """
        assert len(self.decoder_pf_peak_ranks_list) >= 2
        return self.decoder_pf_peak_ranks_list[1]



    def to_tuple(self):
        """ 
        shared_aclus_only_neuron_IDs, is_good_aclus, long_pf_peak_ranks, short_pf_peak_ranks, shuffled_aclus, shuffle_IDXs = a_shuffle_helper.to_tuple()
        """
        return astuple(self)
    
    @classmethod
    def _compute_ranks_template(a_decoder):
        """ computes the rank template from a decoder such as `long_shared_aclus_only_decoder` """
        return scipy.stats.rankdata(compute_placefield_center_of_masses(a_decoder.pf.ratemap.pdf_normalized_tuning_curves), method='dense')

    @classmethod
    def init_from_shared_aclus_only_decoders(cls, *decoder_args, num_shuffles: int = 100, bimodal_exclude_aclus=[]) -> "ShuffleHelper":
        assert len(decoder_args) > 0
        
        shared_aclus_only_neuron_IDs = deepcopy(decoder_args[0].neuron_IDs)

        # Exclude the bimodal cells:
        if bimodal_exclude_aclus is None:
            bimodal_exclude_aclus = []

        is_good_aclus = np.logical_not(np.isin(shared_aclus_only_neuron_IDs, bimodal_exclude_aclus))
        shared_aclus_only_neuron_IDs = shared_aclus_only_neuron_IDs[is_good_aclus]

        shuffled_aclus, shuffle_IDXs = build_shuffled_ids(shared_aclus_only_neuron_IDs, num_shuffles=num_shuffles, seed=1337)

        # shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs 

        ## 2023-10-11 - Get the long/short peak locations
        decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in decoder_args]


        ## Compute the ranks:
        decoder_pf_peak_ranks_list = [scipy.stats.rankdata(a_peaks_com, method='dense') for a_peaks_com in decoder_peak_coms_list]
        
        # return shared_aclus_only_neuron_IDs, is_good_aclus, long_pf_peak_ranks, short_pf_peak_ranks, shuffled_aclus, shuffle_IDXs
        return cls(shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, decoder_pf_peak_ranks_list=decoder_pf_peak_ranks_list)



    @classmethod
    def init_from_long_short_shared_aclus_only_decoders(cls, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles: int = 100, bimodal_exclude_aclus = [5, 14, 25, 46, 61, 66, 86, 88, 95]) -> "ShuffleHelper":
        """ two (long/short) decoders only version."""
        return cls.init_from_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=num_shuffles, bimodal_exclude_aclus=bimodal_exclude_aclus)
    




@define()
class Zscorer:
    original_values: np.array
    mean: float
    std_dev: float
    n_values: int

    real_value: float = None
    z_score_value: float = None # z-score values


    @classmethod
    def init_from_values(cls, stats_corr_values: np.array, real_value=None):
        _obj = cls(original_values=stats_corr_values, mean=np.mean(stats_corr_values), std_dev=np.std(stats_corr_values), n_values=len(stats_corr_values), real_value=real_value, z_score_value=None)
        _obj.z_score_value = _obj.Zscore(real_value)
        return _obj


    def Zscore(self, xcritical):
        self.z_score_value = (xcritical - self.mean)/self.std_dev
        return self.z_score_value


    # def Zscore(self, xcritical: np.array) -> np.array:
    #     return (xcritical - self.mean)/self.std_dev

# def Zscore(xcritical, mean, stdev):
#     return (xcritical - mean)/stdev

def build_track_templates_for_shuffle(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles: int = 100, bimodal_exclude_aclus = [5, 14, 25, 46, 61, 66, 86, 88, 95]) -> ShuffleHelper:
    return ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=num_shuffles, bimodal_exclude_aclus=bimodal_exclude_aclus)
    

@function_attributes(short_name=None, tags=['shuffle', 'rank_order'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-21 00:23', related_items=[])
def compute_shuffled_rankorder_analyses(active_spikes_df, active_epochs, shuffle_helper, rank_alignment: str = 'first', disable_re_ranking:bool=True, debug_print=True):
    """ 

        

    """
    shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, (long_pf_peak_ranks, short_pf_peak_ranks) = astuple(shuffle_helper)


    active_spikes_df = deepcopy(active_spikes_df).spikes.sliced_by_neuron_id(shared_aclus_only_neuron_IDs)
    active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # NOTE: `active_aclu_to_fragile_linear_neuron_IDX_dict` is actually pretty important here. It's an ordered dict that maps each aclu to a flat neuronIDX!
    # unique_neuron_identities = active_spikes_df.spikes.extract_unique_neuron_identities()
    # [['t_rel_seconds', 'shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap', 'maze_relative_lap', 'flat_spike_idx', 'maze_id', 'fragile_linear_neuron_IDX', 'neuron_type', 'PBE_id']]
    # add the active_epoch's id to each spike in active_spikes_df to make filtering and grouping easier and more efficient:
    active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=-1)[['t_rel_seconds', 'shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap', 'maze_relative_lap', 'maze_id', 'fragile_linear_neuron_IDX', 'neuron_type', 'flat_spike_idx', 'PBE_id', 'Probe_Epoch_id']]
    # uses new add_epochs_id_identity

    # Sort by columns: 't_rel_seconds' (ascending), 'aclu' (ascending)
    active_spikes_df = active_spikes_df.sort_values(['t_rel_seconds', 'aclu'])

    # Get all aclus and epoch_idxs used throughout the entire spikes_df:
    all_aclus = active_spikes_df['aclu'].unique()
    all_probe_epoch_ids = active_spikes_df['Probe_Epoch_id'].unique()

    ## Determine which spikes to use to represent the order:
    selected_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[active_spikes_df.spikes.time_variable_name]

    if rank_alignment == 'first':
        selected_spikes = selected_spikes.first() # first spike times only
    elif rank_alignment == 'median':
        selected_spikes = selected_spikes.median() # median spike times only
    elif rank_alignment == 'center_of_mass':
        selected_spikes = compute_placefield_center_of_masses(selected_spikes)
    else:
        raise NotImplementedError(f'invalid rank_alignment specified : {rank_alignment}. valid options are [first, median, ...]')

    
    # rank the aclu values by their first t value in each Probe_Epoch_id
    ranked_aclus = selected_spikes.groupby('Probe_Epoch_id').rank(method='dense') # resolve ties in ranking by assigning the same rank to each and then incrimenting for the next item

    # create a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
    epoch_ranked_aclus_dict = {} # this one isn't needed anymore probably, the `epoch_ranked_fragile_linear_neuron_IDX_dict` is easier.
    epoch_ranked_fragile_linear_neuron_IDX_dict = {} # structure is different 

    epoch_selected_spikes_fragile_linear_neuron_IDX_dict = {}

    for (epoch_id, aclu), rank in zip(ranked_aclus.index, ranked_aclus):
        if epoch_id not in epoch_ranked_aclus_dict:
            # Initialize new dicts/arrays for the epoch if needed:
            epoch_ranked_aclus_dict[epoch_id] = {}
            epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id] = []
            epoch_selected_spikes_fragile_linear_neuron_IDX_dict[epoch_id] = []
            
        ## Add the rank to the dict/array
        epoch_ranked_aclus_dict[epoch_id][aclu] = int(rank)        
        neuron_IDX = active_aclu_to_fragile_linear_neuron_IDX_dict[aclu] # ordered dict that maps each aclu to a flat neuronIDX!
        epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id].append((neuron_IDX, int(rank))) # note we are adding indicies, not aclus
        
        a_value: float = selected_spikes.groupby(['Probe_Epoch_id', 'aclu']).get_group((epoch_id, aclu)).values[0] # extracts the single float item
        epoch_selected_spikes_fragile_linear_neuron_IDX_dict[epoch_id].append((neuron_IDX, a_value))
        
    # Convert all to np.ndarrays post-hoc:
    epoch_ranked_fragile_linear_neuron_IDX_dict = {epoch_id:np.array(epoch_ranked_fragile_linear_neuron_IDXs) for epoch_id, epoch_ranked_fragile_linear_neuron_IDXs in epoch_ranked_fragile_linear_neuron_IDX_dict.items()}
    epoch_selected_spikes_fragile_linear_neuron_IDX_dict = {epoch_id:np.array(epoch_ranked_fragile_linear_neuron_IDXs) for epoch_id, epoch_ranked_fragile_linear_neuron_IDXs in epoch_selected_spikes_fragile_linear_neuron_IDX_dict.items()}

    ## Loop over the results now to do the actual stats:
    epoch_ranked_aclus_stats_dict = {}


    # epoch_neuron_IDX_selected_spikes = np.squeeze(epoch_selected_spikes_fragile_linear_neuron_IDX_dict[:,1])
    

    for epoch_id in list(epoch_ranked_aclus_dict.keys()):
        # rank_dict = epoch_ranked_aclus_dict[epoch_id]
        # epoch_aclus = np.array(list(rank_dict.keys()))
        # epoch_aclu_ranks = np.array(list(rank_dict.values()))

        epoch_ranked_fragile_linear_neuron_IDXs_array = epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id]
        epoch_neuron_IDXs = np.squeeze(epoch_ranked_fragile_linear_neuron_IDXs_array[:,0])
        epoch_neuron_IDX_ranks = np.squeeze(epoch_ranked_fragile_linear_neuron_IDXs_array[:,1])
        # epoch_neuron_IDX_selected_spikes = np.squeeze(epoch_selected_spikes_fragile_linear_neuron_IDX_dict[:,1])
        
        
        if debug_print:
            print(f'epoch_id: {epoch_id}')
            # print(f'\tepoch_ranked_fragile_linear_neuron_IDXs_array:\n{epoch_ranked_fragile_linear_neuron_IDXs_array}')
            print(f'\tepoch_neuron_IDXs: {print_array(epoch_neuron_IDXs)}')
            print(f'\tepoch_neuron_IDX_ranks: {print_array(epoch_neuron_IDX_ranks)}')
            # print(f'\tepoch_neuron_IDX_selected_spikes: {print_array(epoch_neuron_IDX_selected_spikes)}')

        long_spearmanr_rank_stats_results = []
        short_spearmanr_rank_stats_results = []

        # The "real" result for this epoch:
        # active_epoch_aclu_long_ranks = long_pf_peak_ranks[epoch_neuron_IDXs]
        active_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, epoch_neuron_IDXs, disable_re_ranking=disable_re_ranking)
        real_long_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
        real_long_result_corr_value = (np.abs(real_long_rank_stats.statistic), real_long_rank_stats.pvalue)[0]
        
        # active_epoch_aclu_short_ranks = short_pf_peak_ranks[epoch_neuron_IDXs]
        active_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, epoch_neuron_IDXs, disable_re_ranking=disable_re_ranking)
        real_short_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
        real_short_result_corr_value = (np.abs(real_short_rank_stats.statistic), real_short_rank_stats.pvalue)[0]
        
        if debug_print:
            print(f'\tactive_epoch_aclu_long_ranks[{epoch_id}]: {print_array(active_epoch_aclu_long_ranks)}')
            print(f'\tactive_epoch_aclu_short_ranks[{epoch_id}]: {print_array(active_epoch_aclu_short_ranks)}')
            
        ## PERFORM SHUFFLE HERE:
        for i, (a_shuffled_aclus, a_shuffled_IDXs) in enumerate(zip(shuffled_aclus, shuffle_IDXs)):
            # long_shared_aclus_only_decoder.pf.ratemap.get_by_id(a_shuffled_aclus)
            epoch_specific_shuffled_indicies = a_shuffled_IDXs[epoch_neuron_IDXs] # get only the subset that is active during this epoch
            # long_pf_peak_ranks[epoch_specific_shuffled_indicies] # get the shuffled entries from the
            # short_pf_peak_ranks[epoch_specific_shuffled_indicies]

            ## Get the matching components of the long/short pf ranks using epoch_ranked_fragile_linear_neuron_IDXs's first column which are the relevant indicies:
            active_shuffle_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
            long_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
            long_result = (np.abs(long_rank_stats.statistic), long_rank_stats.pvalue)
            long_spearmanr_rank_stats_results.append(long_result)
            
            active_shuffle_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
            short_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
            short_result = (np.abs(short_rank_stats.statistic), short_rank_stats.pvalue)
            short_spearmanr_rank_stats_results.append(short_result)
            
        long_spearmanr_rank_stats_results = np.array(long_spearmanr_rank_stats_results)
        short_spearmanr_rank_stats_results = np.array(short_spearmanr_rank_stats_results)

        long_stats_corr_values = long_spearmanr_rank_stats_results[:,0]
        short_stats_corr_values = short_spearmanr_rank_stats_results[:,0]

        long_stats_z_scorer = Zscorer.init_from_values(long_stats_corr_values, real_long_result_corr_value)
        short_stats_z_scorer = Zscorer.init_from_values(short_stats_corr_values, real_short_result_corr_value)

        long_short_z_diff: float = long_stats_z_scorer.z_score_value - short_stats_z_scorer.z_score_value

        # epoch_ranked_aclus_stats_dict[epoch_id] = (np.array(long_spearmanr_rank_stats_results),  np.array(short_spearmanr_rank_stats_results))
        epoch_ranked_aclus_stats_dict[epoch_id] = (long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff)

    # 16.9s
    # for epoch_id, epoch_stats in epoch_ranked_aclus_stats_dict.items():
    #     long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff = epoch_stats
    #     paired_test = pho_stats_paired_t_test(long_stats_z_scorer.z_score_values, short_stats_z_scorer.z_score_values) # this doesn't seem to work well

    # Extract the results:
    long_z_score_values = []
    short_z_score_values = []
    long_short_z_score_diff_values = []

    for epoch_id, epoch_stats in epoch_ranked_aclus_stats_dict.items():
        long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff = epoch_stats
        # paired_test = pho_stats_paired_t_test(long_stats_z_scorer.z_score_values, short_stats_z_scorer.z_score_values) # this doesn't seem to work well
        long_z_score_values.append(long_stats_z_scorer.z_score_value)
        short_z_score_values.append(short_stats_z_scorer.z_score_value)
        long_short_z_score_diff_values.append(long_short_z_diff)

    long_z_score_values = np.array(long_z_score_values)
    short_z_score_values = np.array(short_z_score_values)
    long_short_z_score_diff_values = np.array(long_short_z_score_diff_values)
    
    return epoch_ranked_aclus_stats_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (long_z_score_values, short_z_score_values, long_short_z_score_diff_values)


class RankOrderAnalyses:
    """ 

    Potential Speedups:
        Multiprocessing could be used to parallelize:
            - Basically the four templates:
                Direction (Odd/Even)
                    Track (Long/Short)


    """
    ## Plot Laps Analysis
    def _plot_laps_shuffle_analysis(laps_long_short_z_score_diff_values, suffix_str=''):
        laps_fig, laps_ax = plt.subplots()
        laps_ax.scatter(np.arange(len(laps_long_short_z_score_diff_values)), laps_long_short_z_score_diff_values, label=f'laps{suffix_str}')
        plt.title(f'Rank-Order Long-Short ZScore Diff for Laps over time ({suffix_str})')
        plt.ylabel(f'Long-Short Z-Score Diff ({suffix_str})')
        plt.xlabel('Lap Index')
        return laps_fig, laps_ax

    ## Plot Ripple-Events Analysis
    def _plot_ripple_events_shuffle_analysis(ripple_evts_long_short_z_score_diff_values, global_replays, suffix_str=''):
        replay_fig, replay_ax = plt.subplots()
        # replay_ax.scatter(np.arange(len(ripple_evts_long_z_score_values)), ripple_evts_long_z_score_values, label='ripple-events')

        # Plot with actual times:
        replays_midpoint_times = global_replays.starts + ((global_replays.stops - global_replays.starts)/2.0)
        replay_ax.scatter(replays_midpoint_times, ripple_evts_long_short_z_score_diff_values[1:], label=f'ripple-events{suffix_str}')
        plt.title(f'Rank-Order Long-Short ZScore Diff for Ripple-Events over time ({suffix_str})')
        plt.ylabel(f'Long-Short Z-Score Diff ({suffix_str})')
        # plt.xlabel('Ripple Event Index')
        plt.xlabel('Ripple Event Mid-time (t)')
        return replay_fig, replay_ax


    def _perform_plot_z_score_raw(epoch_idx_list, odd_laps_long_z_score_values, odd_laps_short_z_score_values, even_laps_long_z_score_values, even_laps_short_z_score_values, variable_name='Lap', x_axis_name_suffix='Index'):
        """ plots the raw z-scores for each of the four templates 

        Usage:
            app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D) = _perform_plot_z_score_raw(deepcopy(global_laps).lap_id, odd_laps_long_z_score_values, odd_laps_short_z_score_values, even_laps_long_z_score_values, even_laps_short_z_score_values)

        """
        app = pg.mkQApp(f"Rank Order {variable_name}s Epoch Debugger")
        win = pg.GraphicsLayoutWidget(show=True, title=f"Rank-Order (Raw) {variable_name} Epoch Debugger")
        win.setWindowTitle(f'Rank Order (Raw) {variable_name} Epoch Debugger')
        label = pg.LabelItem(justify='right')
        win.addItem(label)
        p1 = win.addPlot(row=1, col=0, title=f'Rank-Order Long-Short ZScore (Raw) for {variable_name}s over time', left='Z-Score (Raw)', bottom=f'{variable_name} {x_axis_name_suffix}')
        p1.addLegend()
        p1.showGrid(x=False, y=True, alpha=1.0) # p1 is a new_ax

        # epoch_idx_list = np.arange(len(even_laps_long_short_z_score_diff_values))
        # epoch_idx_list = deepcopy(global_laps).lap_id # np.arange(len(even_laps_long_short_z_score_diff_values))
        n_x_points = len(epoch_idx_list)
        n_y_points = np.shape(even_laps_long_z_score_values)[0]
        if n_y_points > n_x_points:
            num_missing_points: int = n_y_points - n_x_points
            print(f'WARNING: trimming y-data to [{num_missing_points}:]')
            even_laps_long_z_score_values = even_laps_long_z_score_values[num_missing_points:]
            odd_laps_long_z_score_values = odd_laps_long_z_score_values[num_missing_points:]
            even_laps_short_z_score_values = even_laps_short_z_score_values[num_missing_points:]
            odd_laps_short_z_score_values = odd_laps_short_z_score_values[num_missing_points:]


        # Need to modify the symbol for each one, to emphasize the correct one?

        ## Build indicators for the right index
        # symbolPen = pg.mkPen('#FFFFFF')


        symbolPens = [pg.mkPen('#FFFFFF11') for idx in epoch_idx_list]
        # determine the "correct" items


        long_even_out_plot_1D = p1.plot(epoch_idx_list, even_laps_long_z_score_values, pen=None, symbolBrush='orange', symbolPen=symbolPens, symbol='t2', name='long_even') ## setting pen=None disables line drawing
        long_odd_out_plot_1D = p1.plot(epoch_idx_list, odd_laps_long_z_score_values, pen=None, symbolBrush='red', symbolPen=symbolPens, symbol='t3', name='long_odd') ## setting pen=None disables line drawing
        short_even_out_plot_1D = p1.plot(epoch_idx_list, even_laps_short_z_score_values, pen=None, symbolBrush='blue', symbolPen=symbolPens, symbol='t2', name='short_even') ## setting pen=None disables line drawing
        short_odd_out_plot_1D = p1.plot(epoch_idx_list, odd_laps_short_z_score_values, pen=None, symbolBrush='teal', symbolPen=symbolPens, symbol='t3', name='short_odd') ## setting pen=None disables line drawing
        return app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D)

    def _perform_plot_z_score_diff(epoch_idx_list, even_laps_long_short_z_score_diff_values, odd_laps_long_short_z_score_diff_values, variable_name='Lap', x_axis_name_suffix='Index'):
        """ plots the z-score differences 
        Usage:
            app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _perform_plot_z_score_diff(deepcopy(global_laps).lap_id, even_laps_long_short_z_score_diff_values, odd_laps_long_short_z_score_diff_values)
        """
        app = pg.mkQApp(f"Rank Order {variable_name}s Epoch Debugger")
        win = pg.GraphicsLayoutWidget(show=True, title=f"Rank Order {variable_name} Epoch Debugger")
        win.setWindowTitle(f'Rank Order {variable_name}s Epoch Debugger')
        label = pg.LabelItem(justify='right')
        win.addItem(label)
        p1 = win.addPlot(row=1, col=0, title=f'Rank-Order Long-Short ZScore Diff for {variable_name}s over time', left='Long-Short Z-Score Diff', bottom=f'{variable_name} {x_axis_name_suffix}')
        p1.addLegend()
        p1.showGrid(x=False, y=True, alpha=1.0) # p1 is a new_ax

        n_x_points = len(epoch_idx_list)
        n_y_points = np.shape(even_laps_long_short_z_score_diff_values)[0]
        if n_y_points > n_x_points:
            num_missing_points: int = n_y_points - n_x_points
            print(f'WARNING: trimming y-data to [{num_missing_points}:]')
            even_laps_long_short_z_score_diff_values = even_laps_long_short_z_score_diff_values[num_missing_points:]
            odd_laps_long_short_z_score_diff_values = odd_laps_long_short_z_score_diff_values[num_missing_points:]

        # laps_fig, laps_ax = plt.subplots()
        # laps_ax.scatter(np.arange(len(laps_long_short_z_score_diff_values)), laps_long_short_z_score_diff_values, label=f'laps{suffix_str}')
        # plt.title(f'Rank-Order Long-Short ZScore Diff for Laps over time ({suffix_str})')
        # plt.ylabel(f'Long-Short Z-Score Diff ({suffix_str})')
        # plt.xlabel('Lap Index')

        # epoch_idx_list = np.arange(len(even_laps_long_short_z_score_diff_values))
        # epoch_idx_list = deepcopy(global_laps).lap_id # np.arange(len(even_laps_long_short_z_score_diff_values))
        # out_plot_1D = pg.plot(epoch_idx_list, even_laps_long_short_z_score_diff_values[1:], pen=None, symbol='o', title='Rank-Order Long-Short ZScore Diff for Laps over time', left='Long-Short Z-Score Diff', bottom='Lap Index') ## setting pen=None disables line drawing

        even_out_plot_1D = p1.plot(epoch_idx_list, even_laps_long_short_z_score_diff_values, pen=None, symbolBrush='orange', symbolPen='w', symbol='o', name='even') ## setting pen=None disables line drawing
        odd_out_plot_1D = p1.plot(epoch_idx_list, odd_laps_long_short_z_score_diff_values, pen=None, symbolBrush='blue', symbolPen='w', symbol='p', name='odd') ## setting pen=None disables line drawing

        return app, win, p1, (even_out_plot_1D, odd_out_plot_1D)


    @classmethod
    def common_analysis_helper(cls, curr_active_pipeline, num_shuffles:int=300):
        ## Shared:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        
        # Recover from the saved global result:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D = [directional_laps_results.__dict__[k] for k in ['long_odd_shared_aclus_only_one_step_decoder_1D', 'long_even_shared_aclus_only_one_step_decoder_1D', 'short_odd_shared_aclus_only_one_step_decoder_1D', 'short_even_shared_aclus_only_one_step_decoder_1D']]
        # track_templates: TrackTemplates = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_odd_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D), RL_decoder_pair=(long_even_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D))

        ## 2023-10-24 - Simple long/short (2-template, direction independent) analysis:
        # shuffle_helper = build_track_templates_for_shuffle(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=1000, bimodal_exclude_aclus=[5, 14, 25, 46, 61, 66, 86, 88, 95])

        # 2023-10-26 - Direction Dependent (4 template) analysis: long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D
        odd_shuffle_helper = build_track_templates_for_shuffle(long_odd_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        even_shuffle_helper = build_track_templates_for_shuffle(long_even_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        
        return global_spikes_df, (odd_shuffle_helper, even_shuffle_helper)

    @function_attributes(short_name=None, tags=['rank-order', 'ripples', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_ripples_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='first'):
        
        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles)

        ## Ripple Rank-Order Analysis: needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

        global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
        if isinstance(global_replays, pd.DataFrame):
            global_replays = Epoch(global_replays.epochs.get_valid_df())

        ## Replay Epochs:
        odd_outputs = compute_shuffled_rankorder_analyses(spikes_df, deepcopy(global_replays), odd_shuffle_helper, rank_alignment=rank_alignment, debug_print=False)
        even_outputs = compute_shuffled_rankorder_analyses(spikes_df, deepcopy(global_replays), even_shuffle_helper, rank_alignment=rank_alignment, debug_print=False)

        # Unwrap
        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values) = odd_outputs
        even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values) = even_outputs

        ripple_evts_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values), (even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values))]
        print(f'ripple_evts_paired_tests: {ripple_evts_paired_tests}')
        # [TtestResult(statistic=3.5572800536164495, pvalue=0.0004179523066872734, df=415),
        #  TtestResult(statistic=3.809779392137816, pvalue=0.0001601254566506359, df=415)]

        # All plots
        replay_fig_odd, replay_ax_odd = RankOrderAnalyses._plot_ripple_events_shuffle_analysis(odd_ripple_evts_long_short_z_score_diff_values, global_replays, suffix_str='_odd')
        replay_fig_even, replay_ax_even = RankOrderAnalyses._plot_ripple_events_shuffle_analysis(even_ripple_evts_long_short_z_score_diff_values, global_replays, suffix_str='_even')
        _display_replay_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(global_replays.labels.astype(float), even_ripple_evts_long_short_z_score_diff_values[1:], odd_ripple_evts_long_short_z_score_diff_values[1:], variable_name='Ripple')
        _display_replay_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(global_replays.labels.astype(float), odd_ripple_evts_long_z_score_values[1:], even_ripple_evts_long_z_score_values[1:], odd_ripple_evts_short_z_score_values[1:], even_ripple_evts_short_z_score_values[1:], variable_name='Ripple')
        # ["replay_fig_odd", "replay_ax_odd", "replay_fig_even", "replay_ax_even", "_display_replay_z_score_diff_outputs", "_display_replay_z_score_raw_outputs"]
        return (odd_outputs, even_outputs, ripple_evts_paired_tests), (replay_fig_odd, replay_ax_odd,
                replay_fig_even, replay_ax_even,
                _display_replay_z_score_diff_outputs,
                _display_replay_z_score_raw_outputs)
    

    @function_attributes(short_name=None, tags=['rank-order', 'laps', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_laps_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='median'):
        """
        
        _laps_outputs = RankOrderAnalyses.main_laps_analysis(curr_active_pipeline, num_shuffles=1000, rank_alignment='median')
        
        # Unwrap
        (odd_outputs, even_outputs, laps_paired_tests), (laps_fig_odd, laps_ax_odd,
                        laps_fig_even, laps_ax_even,
                        _display_laps_z_score_raw_outputs,
                        _display_laps_z_score_diff_outputs) = _laps_outputs

        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values) = odd_outputs
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values) = even_outputs

        
        """
        ## Shared:
        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles)

        ## Laps Epochs: Needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()

        # TODO: CenterOfMass for Laps instead of median spike
        # laps_rank_alignment = 'center_of_mass'
        odd_outputs = compute_shuffled_rankorder_analyses(global_spikes_df, deepcopy(global_laps), odd_shuffle_helper, rank_alignment=rank_alignment, debug_print=False)
        even_outputs = compute_shuffled_rankorder_analyses(global_spikes_df, deepcopy(global_laps), even_shuffle_helper, rank_alignment=rank_alignment, debug_print=False)

        # Unwrap
        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values) = odd_outputs
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values) = even_outputs
        
        laps_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((odd_laps_long_z_score_values, odd_laps_short_z_score_values), (even_laps_long_z_score_values, even_laps_short_z_score_values))]
        print(f'laps_paired_tests: {laps_paired_tests}')

        ## All Plots:
        laps_fig_odd, laps_ax_odd = RankOrderAnalyses._plot_laps_shuffle_analysis(odd_laps_long_short_z_score_diff_values, suffix_str='_odd')
        laps_fig_even, laps_ax_even = RankOrderAnalyses._plot_laps_shuffle_analysis(even_laps_long_short_z_score_diff_values, suffix_str='_even')
        _display_laps_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(global_laps.lap_id.astype(float), odd_laps_long_z_score_values, odd_laps_short_z_score_values, even_laps_long_z_score_values, even_laps_short_z_score_values, variable_name='Lap')
        # app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D) = _display_z_score_raw_outputs
        _display_laps_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(global_laps.lap_id.astype(float), even_laps_long_short_z_score_diff_values, odd_laps_long_short_z_score_diff_values, variable_name='Lap')
        # app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _display_z_score_diff_outputs

        return (odd_outputs, even_outputs, laps_paired_tests), (laps_fig_odd, laps_ax_odd,
                laps_fig_even, laps_ax_even,
                _display_laps_z_score_raw_outputs,
                _display_laps_z_score_diff_outputs)