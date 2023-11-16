
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import concurrent.futures
from functools import partial
from itertools import repeat
import multiprocessing
from multiprocessing import Pool, freeze_support

# from matplotlib.colors import ListedColormap
from pathlib import Path
from neuropy.core import Epoch
from neuropy.analyses.placefields import PfND
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt


from nptyping import NDArray
import attrs
from attrs import asdict, define, field, Factory, astuple

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder

from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_paired_t_test
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
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult # used in TrackTemplates

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol




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

@define(slots=False, repr=False, eq=False)
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


@define(slots=False, repr=False, eq=False)
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
    

@define(slots=False, repr=False, eq=False)
class Zscorer:
    original_values: np.array
    mean: float
    std_dev: float
    n_values: int

    real_value: float = field(default=None)
    real_p_value: float = field(default=None)
    z_score_value: float = field(default=None) # z-score values


    @classmethod
    def init_from_values(cls, stats_corr_values: np.array, real_value=None, real_p_value=None):
        _obj = cls(original_values=stats_corr_values, mean=np.mean(stats_corr_values), std_dev=np.std(stats_corr_values), n_values=len(stats_corr_values), real_value=real_value, real_p_value=real_p_value, z_score_value=None)
        _obj.z_score_value = _obj.Zscore(real_value)
        return _obj

    def Zscore(self, xcritical):
        self.z_score_value = (xcritical - self.mean)/self.std_dev
        return self.z_score_value


    def plot_distribution(self):
        """ plots a standalone figure showing the distribution of the original values and their fisher_z_transformed version in a histogram. """
        win = pg.GraphicsLayoutWidget(show=True)
        win.resize(800,350)
        win.setWindowTitle('Z-Scorer: Histogram')
        plt1 = win.addPlot()
        vals = self.original_values
        fisher_z_transformed_vals = np.arctanh(vals)

        ## compute standard histogram
        y, x = np.histogram(vals) # , bins=np.linspace(-3, 8, 40)
        fisher_z_transformed_y, x = np.histogram(fisher_z_transformed_vals, bins=x)

        ## Using stepMode="center" causes the plot to draw two lines for each sample.
        ## notice that len(x) == len(y)+1
        plt1.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,50), name='original_values')
        plt1.plot(x, fisher_z_transformed_y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,255,100,50), name='fisher_z_values')

        # ## Now draw all points as a nicely-spaced scatter plot
        # y = pg.pseudoScatter(vals, spacing=0.15)
        # #plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5)
        # plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(0,0,255,150))

        return win, plt1
    

    # def Zscore(self, xcritical: np.array) -> np.array:
    #     return (xcritical - self.mean)/self.std_dev

# def Zscore(xcritical, mean, stdev):
#     return (xcritical - mean)/stdev



@define(slots=False, repr=False, eq=False)
class RankOrderResult(HDFMixin, AttrsBasedClassHelperMixin, ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even
    
    TODO: add spikes_df, epochs_df?
    
    """
    ranked_aclus_stats_dict: Dict[int, Tuple[Zscorer, Zscorer, float]] = serialized_field()
    selected_spikes_fragile_linear_neuron_IDX_dict: Dict = serialized_field()
    
    long_z_score: NDArray = serialized_field()
    short_z_score: NDArray = serialized_field()
    long_short_z_score_diff: NDArray = serialized_field()
    
    @classmethod
    def init_from_analysis_output_tuple(cls, a_tuple):
        """
        ## Ripple Rank-Order Analysis:
        _ripples_outputs = RankOrderAnalyses.main_ripples_analysis(curr_active_pipeline, num_shuffles=1000, rank_alignment='first')

        # Unwrap:
        (odd_ripple_outputs, even_ripple_outputs, ripple_evts_paired_tests), ripple_plots_outputs = _ripples_outputs


        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values) = odd_ripple_outputs
        
        """
        ranked_aclus_stats_dict, selected_spikes_fragile_linear_neuron_IDX_dict, (long_z_score_values, short_z_score_values, long_short_z_score_diff_values) = a_tuple
        return cls(is_global=True, ranked_aclus_stats_dict=ranked_aclus_stats_dict, selected_spikes_fragile_linear_neuron_IDX_dict=selected_spikes_fragile_linear_neuron_IDX_dict, long_z_score=long_z_score_values, short_z_score=short_z_score_values, long_short_z_score_diff=long_short_z_score_diff_values)

    def __iter__(self):
        """ allows unpacking. See https://stackoverflow.com/questions/37837520/implement-packing-unpacking-in-an-object """
        return iter(astuple(self, filter=attrs.filters.exclude(self.__attrs_attrs__.is_global))) #  'is_global'
    

@define(slots=False, repr=False, eq=False)
class RankOrderComputationsContainer(HDFMixin, AttrsBasedClassHelperMixin, ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even
    

    Usage:    
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderComputationsContainer, RankOrderResult
    
        odd_ripple_rank_order_result = RankOrderResult.init_from_analysis_output_tuple(odd_ripple_outputs)
        even_ripple_rank_order_result = RankOrderResult.init_from_analysis_output_tuple(even_ripple_outputs)
        curr_active_pipeline.global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(odd_ripple=odd_ripple_rank_order_result, even_ripple=even_ripple_rank_order_result, odd_laps=odd_laps_rank_order_result, even_laps=even_laps_rank_order_result)

    """
    odd_ripple: RankOrderResult = serialized_field()
    even_ripple: RankOrderResult = serialized_field()
    odd_laps: RankOrderResult = serialized_field()
    even_laps: RankOrderResult = serialized_field()
    

    def __iter__(self):
        """ allows unpacking. See https://stackoverflow.com/questions/37837520/implement-packing-unpacking-in-an-object """
        # return iter(astuple(self)) # deep unpacking causes problems
        return iter(astuple(self, filter=attrs.filters.exclude(self.__attrs_attrs__.is_global))) #  'is_global'
        # return iter(self.__dict__.values())





def build_track_templates_for_shuffle(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles: int = 100, bimodal_exclude_aclus = [5, 14, 25, 46, 61, 66, 86, 88, 95]) -> ShuffleHelper:
    return ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=num_shuffles, bimodal_exclude_aclus=bimodal_exclude_aclus)
    

def _subfn_rank_order_shuffle(epoch_specific_shuffled_indicies, epoch_neuron_IDX_ranks, long_pf_peak_ranks, short_pf_peak_ranks, disable_re_ranking:bool):
    """ attempts to parallelize the rank-order computations """
    ## Get the matching components of the long/short pf ranks using epoch_ranked_fragile_linear_neuron_IDXs's first column which are the relevant indicies:
    active_shuffle_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
    long_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
    # long_result = (np.abs(long_rank_stats.statistic), long_rank_stats.pvalue)
    long_result = (long_rank_stats.statistic, long_rank_stats.pvalue)
    
    active_shuffle_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
    short_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
    # short_result = (np.abs(short_rank_stats.statistic), short_rank_stats.pvalue)
    short_result = (short_rank_stats.statistic, short_rank_stats.pvalue)
    return long_result, short_result



@function_attributes(short_name=None, tags=['shuffle', 'rank_order', 'main'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-21 00:23', related_items=[])
def compute_shuffled_rankorder_analyses(active_spikes_df, active_epochs, shuffle_helper, rank_alignment: str = 'first', disable_re_ranking:bool=True, debug_print=True) -> RankOrderResult:
    """ Extracts the two templates (long/short) from the shuffle_helper in addition to the shuffled_aclus, shuffle_IDXs.

    """
    shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, (long_pf_peak_ranks, short_pf_peak_ranks) = astuple(shuffle_helper)

    # post_process_statistic_value_fn = lambda x: np.abs(x)
    post_process_statistic_value_fn = lambda x: float(x) # basically NO-OP

    active_spikes_df = deepcopy(active_spikes_df).spikes.sliced_by_neuron_id(shared_aclus_only_neuron_IDs)
    active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # NOTE: `active_aclu_to_fragile_linear_neuron_IDX_dict` is actually pretty important here. It's an ordered dict that maps each aclu to a flat neuronIDX!
    # unique_neuron_identities = active_spikes_df.spikes.extract_unique_neuron_identities()
    # [['t_rel_seconds', 'shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap', 'maze_relative_lap', 'flat_spike_idx', 'maze_id', 'fragile_linear_neuron_IDX', 'neuron_type', 'PBE_id']]
    no_interval_fill_value = -1
    # add the active_epoch's id to each spike in active_spikes_df to make filtering and grouping easier and more efficient:
    active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=no_interval_fill_value)[['t_rel_seconds', 'shank', 'cluster', 'aclu', 'qclu', 'traj', 'lap', 'maze_relative_lap', 'maze_id', 'fragile_linear_neuron_IDX', 'neuron_type', 'flat_spike_idx', 'PBE_id', 'Probe_Epoch_id']]
    # uses new add_epochs_id_identity
    active_spikes_df.drop(active_spikes_df.loc[active_spikes_df['Probe_Epoch_id']==no_interval_fill_value].index, inplace=True)

    # Sort by columns: 't_rel_seconds' (ascending), 'aclu' (ascending)
    active_spikes_df = active_spikes_df.sort_values(['t_rel_seconds', 'aclu'])

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

    ## OUTPUT DICTS:
    # create a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
    epoch_ranked_aclus_dict = {} # this one isn't needed anymore probably, the `epoch_ranked_fragile_linear_neuron_IDX_dict` is easier.
    epoch_ranked_fragile_linear_neuron_IDX_dict = {} # structure is different 
    epoch_selected_spikes_fragile_linear_neuron_IDX_dict = {}

    for (epoch_id, aclu), rank in zip(ranked_aclus.index, ranked_aclus):
        assert (epoch_id != -1), f"should have no -1 entries since we tried to drop them before grouping."
        # skip the sentinal value (no epoch) entry
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

    for epoch_id in list(epoch_ranked_aclus_dict.keys()):

        epoch_ranked_fragile_linear_neuron_IDXs_array = epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id]
        epoch_neuron_IDXs = np.squeeze(epoch_ranked_fragile_linear_neuron_IDXs_array[:,0])
        epoch_neuron_IDX_ranks = np.squeeze(epoch_ranked_fragile_linear_neuron_IDXs_array[:,1])
        
        if debug_print:
            print(f'epoch_id: {epoch_id}')
            # print(f'\tepoch_ranked_fragile_linear_neuron_IDXs_array:\n{epoch_ranked_fragile_linear_neuron_IDXs_array}')
            print(f'\tepoch_neuron_IDXs: {print_array(epoch_neuron_IDXs)}')
            print(f'\tepoch_neuron_IDX_ranks: {print_array(epoch_neuron_IDX_ranks)}')
            # print(f'\tepoch_neuron_IDX_selected_spikes: {print_array(epoch_neuron_IDX_selected_spikes)}')

        ## EPOCH SPECIFIC:
        long_spearmanr_rank_stats_results = []
        short_spearmanr_rank_stats_results = []

        # The "real" result for this epoch:
        active_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, epoch_neuron_IDXs, disable_re_ranking=disable_re_ranking)
        real_long_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
        real_long_result_corr_value = (post_process_statistic_value_fn(real_long_rank_stats.statistic), real_long_rank_stats.pvalue)[0]
        
        active_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, epoch_neuron_IDXs, disable_re_ranking=disable_re_ranking)
        real_short_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
        real_short_result_corr_value = (post_process_statistic_value_fn(real_short_rank_stats.statistic), real_short_rank_stats.pvalue)[0]
        
        if debug_print:
            print(f'\tactive_epoch_aclu_long_ranks[{epoch_id}]: {print_array(active_epoch_aclu_long_ranks)}')
            print(f'\tactive_epoch_aclu_short_ranks[{epoch_id}]: {print_array(active_epoch_aclu_short_ranks)}')
            
        ## PERFORM SHUFFLE HERE:
        # epoch_specific_shuffled_indicies_list = [a_shuffled_IDXs[epoch_neuron_IDXs] for i, (a_shuffled_aclus, a_shuffled_IDXs) in enumerate(zip(shuffled_aclus, shuffle_IDXs))]
        # # _active_rank_order_shuffle_fn = lambda x: _subfn_rank_order_shuffle(epoch_neuron_IDX_ranks=epoch_neuron_IDX_ranks, long_pf_peak_ranks=long_pf_peak_ranks, short_pf_peak_ranks=short_pf_peak_ranks, epoch_specific_shuffled_indicies=x, disable_re_ranking=disable_re_ranking)

        # with multiprocessing.Pool(processes=6) as pool:
        #     # results = pool.starmap(_subfn_rank_order_shuffle, product(names, repeat=2))
        #     # results = pool.starmap(_subfn_rank_order_shuffle, zip(epoch_neuron_IDX_ranks=epoch_neuron_IDX_ranks, long_pf_peak_ranks=long_pf_peak_ranks, short_pf_peak_ranks=short_pf_peak_ranks, epoch_specific_shuffled_indicies=x, disable_re_ranking=disable_re_ranking), repeat(second_arg)))
        #     results = pool.map(partial(_subfn_rank_order_shuffle, epoch_neuron_IDX_ranks=epoch_neuron_IDX_ranks, long_pf_peak_ranks=long_pf_peak_ranks, short_pf_peak_ranks=short_pf_peak_ranks, disable_re_ranking=disable_re_ranking), epoch_specific_shuffled_indicies_list)
        # print(f'done with multiprocessing.')
        
        # for a_result in results:
        #     long_result, short_result = a_result
        #     long_spearmanr_rank_stats_results.append(long_result)
        #     short_spearmanr_rank_stats_results.append(short_result)
                    
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for number, a_result in zip(epoch_specific_shuffled_indicies_list, executor.map(_active_rank_order_shuffle_fn, epoch_specific_shuffled_indicies_list)):
        #         # print('%d is prime: %s' % (number, prime))
        #         long_result, short_result = a_result
        #         long_spearmanr_rank_stats_results.append(long_result)
        #         short_spearmanr_rank_stats_results.append(short_result)
                    

        for i, (a_shuffled_aclus, a_shuffled_IDXs) in enumerate(zip(shuffled_aclus, shuffle_IDXs)):
            # long_shared_aclus_only_decoder.pf.ratemap.get_by_id(a_shuffled_aclus)
            epoch_specific_shuffled_indicies = a_shuffled_IDXs[epoch_neuron_IDXs] # get only the subset that is active during this epoch
            
            ## Get the matching components of the long/short pf ranks using epoch_ranked_fragile_linear_neuron_IDXs's first column which are the relevant indicies:
            active_shuffle_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
            long_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
            long_result = (post_process_statistic_value_fn(long_rank_stats.statistic), long_rank_stats.pvalue)
            long_spearmanr_rank_stats_results.append(long_result)
            
            active_shuffle_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
            short_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
            short_result = (post_process_statistic_value_fn(short_rank_stats.statistic), short_rank_stats.pvalue)
            short_spearmanr_rank_stats_results.append(short_result)
        ## END for shuffle


        long_spearmanr_rank_stats_results = np.array(long_spearmanr_rank_stats_results)
        short_spearmanr_rank_stats_results = np.array(short_spearmanr_rank_stats_results)

        long_stats_corr_values = long_spearmanr_rank_stats_results[:,0]
        short_stats_corr_values = short_spearmanr_rank_stats_results[:,0]

        long_stats_z_scorer = Zscorer.init_from_values(long_stats_corr_values, real_long_result_corr_value, real_long_rank_stats.pvalue)
        short_stats_z_scorer = Zscorer.init_from_values(short_stats_corr_values, real_short_result_corr_value, real_short_rank_stats.pvalue)

        long_short_z_diff: float = long_stats_z_scorer.z_score_value - short_stats_z_scorer.z_score_value

        # epoch_ranked_aclus_stats_dict[epoch_id] = (np.array(long_spearmanr_rank_stats_results),  np.array(short_spearmanr_rank_stats_results))
        epoch_ranked_aclus_stats_dict[epoch_id] = (long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff)


    ## END for epoch_id


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
    
    # return epoch_ranked_aclus_stats_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (long_z_score_values, short_z_score_values, long_short_z_score_diff_values)
    
    return RankOrderResult.init_from_analysis_output_tuple((epoch_ranked_aclus_stats_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (long_z_score_values, short_z_score_values, long_short_z_score_diff_values)))


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


    def _perform_plot_z_score_raw(epoch_idx_list, odd_long_z_score_values, odd_short_z_score_values, even_long_z_score_values, even_short_z_score_values, variable_name='Lap', x_axis_name_suffix='Index', point_data_values=None):
        """ plots the raw z-scores for each of the four templates 

        Usage:
            app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D) = _perform_plot_z_score_raw(deepcopy(global_laps).lap_id, odd_laps_long_z_score_values, odd_laps_short_z_score_values, even_laps_long_z_score_values, even_laps_short_z_score_values)

        """
        app = pg.mkQApp(f"Rank Order {variable_name}s Epoch Debugger")
        win = pg.GraphicsLayoutWidget(show=True, title=f"Rank-Order (Raw) {variable_name} Epoch Debugger")
        win.setWindowTitle(f'Rank Order (Raw) {variable_name} Epoch Debugger')
        label = pg.LabelItem(justify='right')
        win.addItem(label)
        p1: pg.PlotItem = win.addPlot(row=1, col=0, title=f'Rank-Order Long-Short ZScore (Raw) for {variable_name}s over time', left='Z-Score (Raw)', bottom=f'{variable_name} {x_axis_name_suffix}')
        p1.addLegend()
        p1.showGrid(x=False, y=True, alpha=1.0) # p1 is a new_ax

        # epoch_idx_list = np.arange(len(even_laps_long_short_z_score_diff_values))
        # epoch_idx_list = deepcopy(global_laps).lap_id # np.arange(len(even_laps_long_short_z_score_diff_values))
        n_x_points = len(epoch_idx_list)
        n_y_points = np.shape(even_long_z_score_values)[0]
        if n_y_points > n_x_points:
            num_missing_points: int = n_y_points - n_x_points
            print(f'WARNING: trimming y-data to [{num_missing_points}:]')
            even_long_z_score_values = even_long_z_score_values[num_missing_points:]
            odd_long_z_score_values = odd_long_z_score_values[num_missing_points:]
            even_short_z_score_values = even_short_z_score_values[num_missing_points:]
            odd_short_z_score_values = odd_short_z_score_values[num_missing_points:]


        # Need to modify the symbol for each one, to emphasize the correct one?

        ## Build indicators for the right index
        # symbolPen = pg.mkPen('#FFFFFF')

        symbolPens = [pg.mkPen('#FFFFFF11') for idx in epoch_idx_list]
        # determine the "correct" items

        # # symbol='t2' is a left-facing arrow and 't3' is a right-facing one:
        # long_even_out_plot_1D = p1.plot(epoch_idx_list, even_laps_long_z_score_values, pen=None, symbolBrush='orange', symbolPen=symbolPens, symbol='t2', name='long_even') ## setting pen=None disables line drawing
        # long_odd_out_plot_1D = p1.plot(epoch_idx_list, odd_laps_long_z_score_values, pen=None, symbolBrush='red', symbolPen=symbolPens, symbol='t3', name='long_odd') ## setting pen=None disables line drawing
        # short_even_out_plot_1D = p1.plot(epoch_idx_list, even_laps_short_z_score_values, pen=None, symbolBrush='blue', symbolPen=symbolPens, symbol='t2', name='short_even') ## setting pen=None disables line drawing
        # short_odd_out_plot_1D = p1.plot(epoch_idx_list, odd_laps_short_z_score_values, pen=None, symbolBrush='teal', symbolPen=symbolPens, symbol='t3', name='short_odd') ## setting pen=None disables line drawing
        
        # for masked arrays:
        # when using pg.ScatterPlotItem(...) compared to p1.plot(...), you must use the non-'symbol' prefixed argument names: {'symbolBrush':'brush', 'symbolPen':'pen'} 
        long_even_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~even_long_z_score_values.mask], even_long_z_score_values[~even_long_z_score_values.mask], brush=pg.mkBrush('orange'), pen=pg.mkPen('#FFFFFF11'), symbol='t2', name='long_even', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~even_long_z_score_values.mask]) ## setting pen=None disables line drawing
        long_odd_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~odd_long_z_score_values.mask], odd_long_z_score_values[~odd_long_z_score_values.mask], brush=pg.mkBrush('red'), pen=pg.mkPen('#FFFFFF11'), symbol='t3', name='long_odd', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~odd_long_z_score_values.mask]) ## setting pen=None disables line drawing
        short_even_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~even_short_z_score_values.mask], even_short_z_score_values[~even_short_z_score_values.mask], brush=pg.mkBrush('blue'), pen=pg.mkPen('#FFFFFF11'), symbol='t2', name='short_even', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~even_short_z_score_values.mask]) ## setting pen=None disables line drawing
        short_odd_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~odd_short_z_score_values.mask], odd_short_z_score_values[~odd_short_z_score_values.mask], brush=pg.mkBrush('teal'), pen=pg.mkPen('#FFFFFF11'), symbol='t3', name='short_odd', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~odd_short_z_score_values.mask]) ## setting pen=None disables line drawing

        p1.addItem(long_even_out_plot_1D)
        p1.addItem(long_odd_out_plot_1D)
        p1.addItem(short_even_out_plot_1D)
        p1.addItem(short_odd_out_plot_1D)

        return app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D)

    def _perform_plot_z_score_diff(epoch_idx_list, even_laps_long_short_z_score_diff_values, odd_laps_long_short_z_score_diff_values, variable_name='Lap', x_axis_name_suffix='Index', point_data_values=None):
        """ plots the z-score differences 
        Usage:
            app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _perform_plot_z_score_diff(deepcopy(global_laps).lap_id, even_laps_long_short_z_score_diff_values, odd_laps_long_short_z_score_diff_values)
        """
        app = pg.mkQApp(f"Rank Order {variable_name}s Epoch Debugger")
        win = pg.GraphicsLayoutWidget(show=True, title=f"Rank Order {variable_name} Epoch Debugger")
        win.setWindowTitle(f'Rank Order {variable_name}s Epoch Debugger')
        label = pg.LabelItem(justify='right')
        win.addItem(label)
        p1: pg.PlotItem = win.addPlot(row=1, col=0, title=f'Rank-Order Long-Short ZScore Diff for {variable_name}s over time', left='Long-Short Z-Score Diff', bottom=f'{variable_name} {x_axis_name_suffix}', hoverable=True) # PlotItem
        p1.addLegend()
        p1.showGrid(x=False, y=True, alpha=1.0) # p1 is a new_ax

        n_x_points = len(epoch_idx_list)
        n_y_points = np.shape(even_laps_long_short_z_score_diff_values)[0]
        if n_y_points > n_x_points:
            num_missing_points: int = n_y_points - n_x_points
            print(f'WARNING: trimming y-data to [{num_missing_points}:]')
            even_laps_long_short_z_score_diff_values = even_laps_long_short_z_score_diff_values[num_missing_points:]
            if odd_laps_long_short_z_score_diff_values is not None:
                odd_laps_long_short_z_score_diff_values = odd_laps_long_short_z_score_diff_values[num_missing_points:]

        # laps_fig, laps_ax = plt.subplots()
        # laps_ax.scatter(np.arange(len(laps_long_short_z_score_diff_values)), laps_long_short_z_score_diff_values, label=f'laps{suffix_str}')
        # plt.title(f'Rank-Order Long-Short ZScore Diff for Laps over time ({suffix_str})')
        # plt.ylabel(f'Long-Short Z-Score Diff ({suffix_str})')
        # plt.xlabel('Lap Index')

        # epoch_idx_list = np.arange(len(even_laps_long_short_z_score_diff_values))
        # epoch_idx_list = deepcopy(global_laps).lap_id # np.arange(len(even_laps_long_short_z_score_diff_values))
        # out_plot_1D = pg.plot(epoch_idx_list, even_laps_long_short_z_score_diff_values[1:], pen=None, symbol='o', title='Rank-Order Long-Short ZScore Diff for Laps over time', left='Long-Short Z-Score Diff', bottom='Lap Index') ## setting pen=None disables line drawing

        # 
        
        # 'orange'
        # symbolPen = 'w'
        symbolPen = None
        
        # def _tip_fn(x, y, data):
        #     """ the function required by pg.ScatterPlotItem's `tip` argument to print the tooltip for each spike. """
        #     # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in zip(active_datapoint_column_names, data)])
        #     data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in asdict(data).items()])
        #     print(f'_tip_fn(...): data_string: {data_string}')
        #     return f"spike: (x={x}, y={y})\n{data_string}"

        # # hover_kwargs = {}
        # hover_kwargs = dict(hoverable=True, hoverPen=pg.mkPen('w', width=2), tip=_tip_fn)
        
        # symbol='t2' is a left-facing arrow and 't3' is a right-facing one:
        # even_out_plot_1D: pg.PlotDataItem = p1.plot(epoch_idx_list, even_laps_long_short_z_score_diff_values, pen=None, symbolBrush=pg.mkBrush(DisplayColorsEnum.Laps.even), symbolPen=symbolPen, symbol='t2', name='even', hoverable=True, hoverPen=pg.mkPen('w', width=2)) ## setting pen=None disables line drawing
        # odd_out_plot_1D: pg.PlotDataItem = p1.plot(epoch_idx_list, odd_laps_long_short_z_score_diff_values, pen=None, symbolBrush=pg.mkBrush(DisplayColorsEnum.Laps.odd), symbolPen=symbolPen, symbol='t3', name='odd', hoverable=True, hoverPen=pg.mkPen('w', width=2)) ## setting pen=None disables line drawing

        # when using pg.ScatterPlotItem(...) compared to p1.plot(...), you must use the non-'symbol' prefixed argument names: {'symbolBrush':'brush', 'symbolPen':'pen'} 

        if odd_laps_long_short_z_score_diff_values is not None:
            first_plot_name = 'even'
        else:
            first_plot_name = 'best'

        even_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list, even_laps_long_short_z_score_diff_values, brush=pg.mkBrush(DisplayColorsEnum.Laps.even), pen=symbolPen, symbol='t2', name=first_plot_name, hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()) ## setting pen=None disables line drawing
        if odd_laps_long_short_z_score_diff_values is not None:
            odd_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list, odd_laps_long_short_z_score_diff_values, brush=pg.mkBrush(DisplayColorsEnum.Laps.odd), pen=symbolPen, symbol='t3', name='odd', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()) ## setting pen=None disables line drawing
        else:
            odd_out_plot_1D = None

        p1.addItem(even_out_plot_1D)
        if odd_laps_long_short_z_score_diff_values is not None:
            p1.addItem(odd_out_plot_1D)
        
        # even_out_plot_1D: pg.PlotDataItem = p1.scatterPlot(epoch_idx_list, even_laps_long_short_z_score_diff_values, pen=None, symbolBrush=pg.mkBrush(DisplayColorsEnum.Laps.even), symbolPen=symbolPen, symbol='t2', name='even', hoverable=True, hoverPen=pg.mkPen('w', width=2)) ## setting pen=None disables line drawing
        # odd_out_plot_1D: pg.PlotDataItem = p1.scatterPlot(epoch_idx_list, odd_laps_long_short_z_score_diff_values, pen=None, symbolBrush=pg.mkBrush(DisplayColorsEnum.Laps.odd), symbolPen=symbolPen, symbol='t3', name='odd', hoverable=True, hoverPen=pg.mkPen('w', width=2)) ## setting pen=None disables line drawing
        
        return app, win, p1, (even_out_plot_1D, odd_out_plot_1D)


    @classmethod
    def common_analysis_helper(cls, curr_active_pipeline, num_shuffles:int=300):
        ## Shared:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)

        # Recover from the saved global result:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D = [directional_laps_results.__dict__[k] for k in ['long_odd_shared_aclus_only_one_step_decoder_1D', 'long_even_shared_aclus_only_one_step_decoder_1D', 'short_odd_shared_aclus_only_one_step_decoder_1D', 'short_even_shared_aclus_only_one_step_decoder_1D']]
        # track_templates: TrackTemplates = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_odd_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D), RL_decoder_pair=(long_even_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D))

        ## 2023-10-24 - Simple long/short (2-template, direction independent) analysis:
        # shuffle_helper = build_track_templates_for_shuffle(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=1000, bimodal_exclude_aclus=[5, 14, 25, 46, 61, 66, 86, 88, 95])

        # 2023-10-26 - Direction Dependent (4 template) analysis: long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D
        odd_shuffle_helper: ShuffleHelper = build_track_templates_for_shuffle(long_odd_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        even_shuffle_helper: ShuffleHelper = build_track_templates_for_shuffle(long_even_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        
        return global_spikes_df, (odd_shuffle_helper, even_shuffle_helper)

    @function_attributes(short_name=None, tags=['rank-order', 'ripples', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_ripples_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='first', enable_plots=False):
        
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
        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values = odd_outputs
        even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values = even_outputs

        ripple_evts_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values), (even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values))]
        print(f'ripple_evts_paired_tests: {ripple_evts_paired_tests}')
        # [TtestResult(statistic=3.5572800536164495, pvalue=0.0004179523066872734, df=415),
        #  TtestResult(statistic=3.809779392137816, pvalue=0.0001601254566506359, df=415)]

        if enable_plots:
            # All plots
            replay_fig_odd, replay_ax_odd = RankOrderAnalyses._plot_ripple_events_shuffle_analysis(odd_ripple_evts_long_short_z_score_diff_values, global_replays, suffix_str='_odd')
            replay_fig_even, replay_ax_even = RankOrderAnalyses._plot_ripple_events_shuffle_analysis(even_ripple_evts_long_short_z_score_diff_values, global_replays, suffix_str='_even')
            _display_replay_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(global_replays.labels.astype(float), even_ripple_evts_long_short_z_score_diff_values[1:], odd_ripple_evts_long_short_z_score_diff_values[1:], variable_name='Ripple')
            _display_replay_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(global_replays.labels.astype(float), odd_ripple_evts_long_z_score_values[1:], even_ripple_evts_long_z_score_values[1:], odd_ripple_evts_short_z_score_values[1:], even_ripple_evts_short_z_score_values[1:], variable_name='Ripple')
            # ["replay_fig_odd", "replay_ax_odd", "replay_fig_even", "replay_ax_even", "_display_replay_z_score_diff_outputs", "_display_replay_z_score_raw_outputs"]
            _plots_outputs = (replay_fig_odd, replay_ax_odd,
                replay_fig_even, replay_ax_even,
                _display_replay_z_score_diff_outputs,
                _display_replay_z_score_raw_outputs)
        else:
            _plots_outputs = None

        return (odd_outputs, even_outputs, ripple_evts_paired_tests), _plots_outputs
    

    @function_attributes(short_name=None, tags=['rank-order', 'laps', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_laps_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='median', enable_plots=False):
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
        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values = odd_outputs
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values = even_outputs
        
        laps_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((odd_laps_long_z_score_values, odd_laps_short_z_score_values), (even_laps_long_z_score_values, even_laps_short_z_score_values))]
        print(f'laps_paired_tests: {laps_paired_tests}')

        ## All Plots:
        if enable_plots:
            laps_fig_odd, laps_ax_odd = RankOrderAnalyses._plot_laps_shuffle_analysis(odd_laps_long_short_z_score_diff_values, suffix_str='_odd')
            laps_fig_even, laps_ax_even = RankOrderAnalyses._plot_laps_shuffle_analysis(even_laps_long_short_z_score_diff_values, suffix_str='_even')
            _display_laps_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(global_laps.lap_id.astype(float), odd_laps_long_z_score_values, odd_laps_short_z_score_values, even_laps_long_z_score_values, even_laps_short_z_score_values, variable_name='Lap')
            # app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D) = _display_z_score_raw_outputs
            _display_laps_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(global_laps.lap_id.astype(float), even_laps_long_short_z_score_diff_values, odd_laps_long_short_z_score_diff_values, variable_name='Lap')
            # app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _display_z_score_diff_outputs
            _plots_outputs = (laps_fig_odd, laps_ax_odd,
                laps_fig_even, laps_ax_even,
                _display_laps_z_score_raw_outputs,
                _display_laps_z_score_diff_outputs)
        else:
            _plots_outputs = None

        return (odd_outputs, even_outputs, laps_paired_tests), _plots_outputs


    @classmethod
    def validate_has_rank_order_results(cls, curr_active_pipeline, computation_filter_name='maze'):
        # Unpacking:
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values = rank_order_results.odd_laps
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values = rank_order_results.even_laps
        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values = rank_order_results.odd_ripple
        even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values = rank_order_results.even_ripple
        return True



class RankOrderGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'rank_order'
    _computationPrecidence = 1001
    _is_global = True

    @function_attributes(short_name='rank_order_shuffle_analysis', tags=['directional_pf', 'laps', 'rank_order', 'session', 'pf1D', 'pf2D'], input_requires=['DirectionalLaps'], output_provides=['RankOrder'], uses=['RankOrderAnalyses'], used_by=[], creation_date='2023-11-08 17:27', related_items=[],
        validate_computation_test=RankOrderAnalyses.validate_has_rank_order_results, is_global=True)
    def perform_rank_order_shuffle_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, num_shuffles:int=1000):
        """ 
        
        Requires:
            ['sess']
            
        Provides:
            global_computation_results.computed_data['RankOrder']
                ['RankOrder'].odd_ripple
                ['RankOrder'].even_ripple
                ['RankOrder'].odd_laps
                ['RankOrder'].even_laps

        
        """
        if include_includelist is not None:
            print(f'WARN: perform_rank_order_shuffle_analysis(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        print(f'perform_rank_order_shuffle_analysis(..., num_shuffles={num_shuffles})')
        ## Laps Rank-Order Analysis:
        print(f'\tcomputing Laps rank-order shuffles:')
        # _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='center_of_mass')
        _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='median')
        # _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='first')
        (odd_laps_outputs, even_laps_outputs, laps_paired_tests), laps_plots_outputs  = _laps_outputs

        ## Ripple Rank-Order Analysis:
        print(f'\tcomputing Ripple rank-order shuffles:')
        _ripples_outputs = RankOrderAnalyses.main_ripples_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='first')
        (odd_ripple_outputs, even_ripple_outputs, ripple_evts_paired_tests), ripple_plots_outputs = _ripples_outputs

        # Set the global result:
        print(f'\tdone. building global result.')
        global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(odd_ripple=odd_ripple_outputs, even_ripple=even_ripple_outputs,
                                                                                                            odd_laps=odd_laps_outputs, even_laps=even_laps_outputs)

        """ Usage:
        
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']

        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values = rank_order_results.odd_laps
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values = rank_order_results.even_laps

        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values = rank_order_results.odd_ripple
        even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values = rank_order_results.even_ripple

        """
        return global_computation_results




from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import _debug_plot_directional_template_rasters
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import build_selected_spikes_df
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import add_selected_spikes_df_points_to_scatter_plot
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper


class RankOrderGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='rank_order_debugger', tags=['rank-order','debugger','shuffle'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-09 01:12', related_items=[], is_global=True)
    def _display_rank_order_debugger(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ 
            
            """
            reuse_axs_tuple = kwargs.pop('reuse_axs_tuple', None)
            # reuse_axs_tuple = None # plot fresh
            # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
            # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
            single_figure = kwargs.pop('single_figure', True)
            debug_print = kwargs.pop('debug_print', False)

            active_config_name = kwargs.pop('active_config_name', None)
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')
            

            defer_render = kwargs.pop('defer_render', False) 


            # Plot 1D Keywoard args:
            shared_kwargs = kwargs.pop('shared_kwargs', {})
            long_kwargs = kwargs.pop('long_kwargs', {})
            short_kwargs = kwargs.pop('short_kwargs', {})

            shared_kwargs['active_context'] = active_context

            ## Inputs: track_templates, global_replays, owning_pipeline_reference
            global_spikes_df = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].spikes_df)

            global_ripples_epochs_df = global_replays.to_dataframe()
            active_epochs_df = global_ripples_epochs_df.copy()

            odd_display_outputs, even_display_outputs = _debug_plot_directional_template_rasters(global_spikes_df, active_epochs_df, track_templates)
            odd_app, odd_win, odd_plots, odd_plots_data, odd_on_update_active_epoch, odd_on_update_active_scatterplot_kwargs = odd_display_outputs
            even_app, even_win, even_plots, even_plots_data, even_on_update_active_epoch, even_on_update_active_scatterplot_kwargs = even_display_outputs

            # Build the wrapping window using `DockAreaWrapper`:
            active_root_main_widget = odd_win.window()
            root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(active_root_main_widget, even_win, title='Pho Rank-Order Epochs Debugger')

            def on_update_active_epoch(an_epoch_idx, an_epoch):
                """ captures: odd_on_update_active_epoch, even_on_update_active_epoch, root_dockAreaWindow """
                odd_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
                even_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
                root_dockAreaWindow.setWindowTitle(f'Pho Rank-Order Epochs Debugger: Epoch[{an_epoch_idx}]')


            def on_update_epoch_IDX(an_epoch_idx):
                """ captures on_update_active_epoch, active_epochs_df to extract the epoch time range and call `on_update_active_epoch` """
                # curr_epoch_spikes = spikes_df[(spikes_df.new_lap_IDX == an_epoch_idx)]
                # curr_epoch_df = active_epochs_df[(active_epochs_df.lap_id == (an_epoch_idx+1))]
                # curr_epoch_df = active_epochs_df[(active_epochs_df.label.astype(float) == float(an_epoch_idx))]
                curr_epoch_df = active_epochs_df[(active_epochs_df.index.astype(float) == float(an_epoch_idx))]
                curr_epoch = list(curr_epoch_df.itertuples())[0]
                on_update_active_epoch(an_epoch_idx, curr_epoch)

            ## Build the selected spikes df:

            ## Laps: even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict
            ## Ripples: even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict

            (even_selected_spike_df, even_neuron_id_to_new_IDX_map), (odd_selected_spike_df, odd_neuron_id_to_new_IDX_map) = build_selected_spikes_df(track_templates, active_epochs_df, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

            ## Add the spikes
            add_selected_spikes_df_points_to_scatter_plot(plots_data=odd_plots_data, plots=odd_plots, selected_spikes_df=deepcopy(odd_selected_spike_df), _active_plot_identifier = 'long_odd')
            add_selected_spikes_df_points_to_scatter_plot(plots_data=odd_plots_data, plots=odd_plots, selected_spikes_df=deepcopy(odd_selected_spike_df), _active_plot_identifier = 'short_odd')
            add_selected_spikes_df_points_to_scatter_plot(plots_data=even_plots_data, plots=even_plots, selected_spikes_df=deepcopy(even_selected_spike_df), _active_plot_identifier = 'long_even')
            add_selected_spikes_df_points_to_scatter_plot(plots_data=even_plots_data, plots=even_plots, selected_spikes_df=deepcopy(even_selected_spike_df), _active_plot_identifier = 'short_even')


            return graphics_output_dict


# ==================================================================================================================== #
# 2023-11-16 - Long/Short Most-likely LR/RL decoder                                                                    #
# ==================================================================================================================== #
from collections import namedtuple

# Define the namedtuple
DirectionalRankOrderLikelihoods = namedtuple('DirectionalRankOrderLikelihoods', ['long_relative_direction_likelihoods', 
                                                           'short_relative_direction_likelihoods', 
                                                           'long_best_direction_indices', 
                                                           'short_best_direction_indices'])


DirectionalRankOrderResult = namedtuple('DirectionalRankOrderResult', ['active_epochs', 
                                                                       'long_best_dir_z_score_values', 
                                                           'short_best_dir_z_score_values', 
                                                           'long_short_best_dir_z_score_diff_values', 
                                                           'directional_likelihoods_tuple'])


def most_likely_directional_rank_order_shuffling(curr_active_pipeline, decoding_time_bin_size=0.003):
    """ 
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import most_likely_directional_rank_order_shuffling
    
    """
    # Unpack all directional variables:
    ## {"even": "RL", "odd": "LR"}
    long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
    global_epoch_name = global_any_name

    # Most popular
    # long_LR_name, short_LR_name, long_RL_name, short_RL_name, global_any_name

    # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
    # (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
    # long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'].pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
    # (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
    (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
    # (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
    (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
    # (long_LR_pf2D, long_RL_pf2D, short_LR_pf2D, short_RL_pf2D) = (long_LR_results.pf2D, long_RL_results.pf2D, short_LR_results.pf2D, short_RL_results.pf2D)
    # (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

    rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']


    # # Use the four epochs to make to a pseudo-y:
    # all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
    # all_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D), debug_print=False)
    # all_directional_pf1D_Decoder = BasePositionDecoder(all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

    ## Combine the non-directional PDFs and renormalize to get the directional PDF:
    # Inputs: long_LR_pf1D, long_RL_pf1D
    long_directional_decoder_names = ['long_LR', 'long_RL']
    long_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), debug_print=False)
    long_directional_pf1D_Decoder = BasePositionDecoder(long_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

    # Inputs: short_LR_pf1D, short_RL_pf1D
    short_directional_decoder_names = ['short_LR', 'short_RL']
    short_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D), debug_print=False)
    short_directional_pf1D_Decoder = BasePositionDecoder(short_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
    # takes 6.3 seconds

    # Decode using specific directional_decoders:
    global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=1000)
    spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

    def _compute_best(active_epochs):
        """ captures: long_directional_pf1D_Decoder, short_directional_pf1D_Decoder, spikes_df, decoding_time_bin_size """
        long_directional_decoding_result: DecodedFilterEpochsResult = long_directional_pf1D_Decoder.decode_specific_epochs(spikes_df, active_epochs, decoding_time_bin_size=decoding_time_bin_size)
        short_directional_decoding_result: DecodedFilterEpochsResult = short_directional_pf1D_Decoder.decode_specific_epochs(spikes_df, active_epochs, decoding_time_bin_size=decoding_time_bin_size)
        # all_directional_decoding_result: DecodedFilterEpochsResult = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df, active_epochs, decoding_time_bin_size=decoding_time_bin_size)

        # sum across timebins to get total likelihood for each of the two directions
        long_relative_direction_likelihoods = np.vstack([(np.sum(long_directional_decoding_result.marginal_y_list[epoch_idx].p_x_given_n, axis=1)/long_directional_decoding_result.time_bin_containers[epoch_idx].num_bins) for epoch_idx in np.arange(long_directional_decoding_result.num_filter_epochs)]) # should get 2 values
        short_relative_direction_likelihoods = np.vstack([(np.sum(short_directional_decoding_result.marginal_y_list[epoch_idx].p_x_given_n, axis=1)/short_directional_decoding_result.time_bin_containers[epoch_idx].num_bins) for epoch_idx in np.arange(short_directional_decoding_result.num_filter_epochs)]) # should get 2 values
        # display(long_relative_direction_likelihoods.shape) # (n_epochs, 2)

        # np.all(np.sum(long_relative_direction_likelihoods, axis=1) == 1)
        # np.sum(long_relative_direction_likelihoods, axis=1) # not sure why some NaN values are getting in there -- actually I do, it's because there aren't spikes in that epoch
        long_is_good_epoch = np.isfinite(np.sum(long_relative_direction_likelihoods, axis=1))
        short_is_good_epoch = np.isfinite(np.sum(short_relative_direction_likelihoods, axis=1))

        # Use the relative likelihoods to determine which points to use:
        long_best_direction_indicies = np.argmax(long_relative_direction_likelihoods, axis=1)
        short_best_direction_indicies = np.argmax(short_relative_direction_likelihoods, axis=1)

        # Creating an instance of the namedtuple
        return DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=long_relative_direction_likelihoods,
                                            short_relative_direction_likelihoods=short_relative_direction_likelihoods,
                                            long_best_direction_indices=long_best_direction_indicies,
                                            short_best_direction_indices=short_best_direction_indicies)

        # return long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies


    ## Replays:
    global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay))
    active_epochs = global_replays.copy()
    ripple_directional_likelihoods_tuple = _compute_best(active_epochs)
    long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = ripple_directional_likelihoods_tuple
    # now do the shuffle:
    # Old-style (Odd/Even) naming:
    odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values = rank_order_results.odd_ripple # LR_ripple_rank_order_result # rank_order_results.odd_ripple
    even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values = rank_order_results.even_ripple # RL_ripple_rank_order_result # rank_order_results.even_ripple

    ## 2023-11-16 - Finally, get the raw z-score values for the best direction at each epoch and then take the long - short difference of those to get `ripple_evts_long_short_best_dir_z_score_diff_values`:
    # Using NumPy advanced indexing to select from array_a or array_b:
    ripple_evts_long_best_dir_z_score_values = np.where(long_best_direction_indicies, odd_ripple_evts_long_z_score_values, even_ripple_evts_long_z_score_values)
    ripple_evts_short_best_dir_z_score_values = np.where(short_best_direction_indicies, odd_ripple_evts_short_z_score_values, even_ripple_evts_short_z_score_values)
    # print(f'np.shape(ripple_evts_long_best_dir_z_score_values): {np.shape(ripple_evts_long_best_dir_z_score_values)}')
    ripple_evts_long_short_best_dir_z_score_diff_values = ripple_evts_long_best_dir_z_score_values - ripple_evts_short_best_dir_z_score_values
    # print(f'np.shape(ripple_evts_long_short_best_dir_z_score_diff_values): {np.shape(ripple_evts_long_short_best_dir_z_score_diff_values)}')

    # outputs: ripple_evts_long_short_best_dir_z_score_diff_values
    ripple_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(active_epochs, long_best_dir_z_score_values=ripple_evts_long_best_dir_z_score_values, short_best_dir_z_score_values=ripple_evts_short_best_dir_z_score_values,
                                                                                long_short_best_dir_z_score_diff_values=ripple_evts_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=ripple_directional_likelihoods_tuple)


    ## Laps:
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
    laps_directional_likelihoods_tuple = _compute_best(global_laps)
    long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = laps_directional_likelihoods_tuple
    odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values = rank_order_results.odd_laps # LR_laps_rank_order_result
    even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values = rank_order_results.even_laps

    # Using NumPy advanced indexing to select from array_a or array_b:
    laps_long_best_dir_z_score_values = np.where(long_best_direction_indicies, odd_laps_long_z_score_values, even_laps_long_z_score_values)
    laps_short_best_dir_z_score_values = np.where(short_best_direction_indicies, odd_laps_short_z_score_values, even_laps_short_z_score_values)
    # print(f'np.shape(laps_long_best_dir_z_score_values): {np.shape(laps_long_best_dir_z_score_values)}')
    laps_long_short_best_dir_z_score_diff_values = laps_long_best_dir_z_score_values - laps_short_best_dir_z_score_values
    # print(f'np.shape(laps_long_short_best_dir_z_score_diff_values): {np.shape(laps_long_short_best_dir_z_score_diff_values)}')

    laps_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(global_laps, long_best_dir_z_score_values=laps_long_best_dir_z_score_values, short_best_dir_z_score_values=laps_short_best_dir_z_score_values,
                                                                                long_short_best_dir_z_score_diff_values=laps_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=laps_directional_likelihoods_tuple)

    # return long_relative_direction_likelihoods, short_relative_direction_likelihoods, (ripple_evts_long_best_dir_z_score_values, ripple_evts_short_best_dir_z_score_values, ripple_evts_long_short_best_dir_z_score_diff_values)

    return ripple_result_tuple, laps_result_tuple
