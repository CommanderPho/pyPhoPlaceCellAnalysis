
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
from functools import partial
from itertools import repeat
from collections import namedtuple
import multiprocessing
from multiprocessing import Pool, freeze_support

# from matplotlib.colors import ListedColormap
from pathlib import Path
import numpy as np
import numpy.ma as ma # used in `most_likely_directional_rank_order_shuffling`
from neuropy.core import Epoch
from neuropy.analyses.placefields import PfND
from neuropy.utils.misc import safe_pandas_get_group
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

from pyphocorehelpers.programming_helpers import metadata_attributes
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
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multi_sort_raster_browser
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

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


LongShortStatsTuple = namedtuple('LongShortStatsTuple', ['long_stats_z_scorer', 'short_stats_z_scorer', 'long_short_z_diff', 'long_short_naive_z_diff', 'is_forward_replay']) # used in `compute_shuffled_rankorder_analyses`
# LongShortStatsTuple: Tuple[Zscorer, Zscorer, float, float, bool]


@function_attributes(short_name=None, tags=['rank_order', 'shuffle', 'renormalize'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-23 13:05', related_items=[])
def relative_re_ranking(rank_array: NDArray, filter_indicies: Optional[NDArray], debug_checking=False, disable_re_ranking: bool=False) -> NDArray:
    """ Re-index the rank_array once filtered flat to extract the global ranks. 
        
    Idea: During each ripple event epoch, only a subset of all cells are active. As a result, we need to extract valid ranks from the epoch's subset so they can be compared directly to the ranks within that epoch.

    """
    if filter_indicies is None:
        filter_indicies = np.arange(len(rank_array)) # all entries in rank_array
        
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


def determine_good_aclus_by_qclu(curr_active_pipeline, included_qclu_values=[1,2,4,9]):
	""" 
	From all neuron_IDs in the session, get the ones that meet the new qclu criteria (their value is in) `included_qclu_values`
	
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import determine_good_aclus_by_qclu

	allowed_aclus = determine_good_aclus_by_qclu(curr_active_pipeline, included_qclu_values=[1,2,4,9])
	allowed_aclus

	"""
	from neuropy.core.neuron_identities import NeuronType
	
	neuron_identities: pd.DataFrame = curr_active_pipeline.get_session_unique_aclu_information()
	print(f"original {len(neuron_identities)}")
	filtered_neuron_identities: pd.DataFrame = neuron_identities[neuron_identities.neuron_type == NeuronType.PYRAMIDAL]
	print(f"post PYRAMIDAL filtering {len(filtered_neuron_identities)}")
	filtered_neuron_identities = filtered_neuron_identities[['aclu', 'shank', 'cluster', 'qclu']]
	filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.qclu, included_qclu_values)] # drop [6, 7], which are said to have double fields - 80 remain
	print(f"post (qclu != [6, 7]) filtering {len(filtered_neuron_identities)}")
	return filtered_neuron_identities.aclu.to_numpy()




@define(slots=False, repr=False, eq=False)
class ShuffleHelper:
    """ holds the result of shuffling templates. Used for rank-order analyses """
    shared_aclus_only_neuron_IDs: NDArray = field()
    is_good_aclus: NDArray = field(repr=False)
    
    num_shuffles: int = field(repr=True) # default=1000
    shuffled_aclus = field(repr=False)
    shuffle_IDX = field(repr=False)
    
    decoder_pf_peak_ranks_list = field(repr=True)
    

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
        shared_aclus_only_neuron_IDs, is_good_aclus, num_shuffles, shuffled_aclus, shuffle_IDXs, (long_pf_peak_ranks, short_pf_peak_ranks) = a_shuffle_helper.to_tuple()
        # exclude num_shuffles?
        """
        
        return astuple(self)
    
    @classmethod
    def _compute_ranks_template(cls, a_decoder):
        """ computes the rank template from a decoder such as `long_shared_aclus_only_decoder` """
        return scipy.stats.rankdata(compute_placefield_center_of_masses(a_decoder.pf.ratemap.pdf_normalized_tuning_curves), method='dense')

    def generate_shuffle(self, shared_aclus_only_neuron_IDs: NDArray, num_shuffles: Optional[int]=None, seed:int=1337) -> Tuple[NDArray, NDArray]:
        """ 
        shuffled_aclus, shuffle_IDXs = shuffle_helper.generate_shuffle(shared_aclus_only_neuron_IDs)
        """
        if num_shuffles is None:
            num_shuffles = self.num_shuffles
        return build_shuffled_ids(shared_aclus_only_neuron_IDs, num_shuffles=num_shuffles, seed=seed)
        

    @classmethod
    def init_from_shared_aclus_only_decoders(cls, *decoder_args, num_shuffles: int = 100, bimodal_exclude_aclus=[]) -> "ShuffleHelper":
        assert len(decoder_args) > 0
        
        shared_aclus_only_neuron_IDs = deepcopy(decoder_args[0].neuron_IDs)
        ## TODO: NOTE: this assumes that the first decoder has the same aclus as all the others, which is only true if they're shared.
        assert np.all([np.all(a_decoder.neuron_IDs == shared_aclus_only_neuron_IDs) for a_decoder in decoder_args]), f"all neuron_ids should be the same but neuron_IDs: {[a_decoder.neuron_IDs for a_decoder in decoder_args]}"
        
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
        return cls(shared_aclus_only_neuron_IDs, is_good_aclus, num_shuffles, shuffled_aclus, shuffle_IDXs, decoder_pf_peak_ranks_list=decoder_pf_peak_ranks_list)

    @classmethod
    def init_from_long_short_shared_aclus_only_decoders(cls, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles: int = 100, bimodal_exclude_aclus = [5, 14, 25, 46, 61, 66, 86, 88, 95]) -> "ShuffleHelper":
        """ two (long/short) decoders only version."""
        return cls.init_from_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=num_shuffles, bimodal_exclude_aclus=bimodal_exclude_aclus)
    

@define(slots=False, repr=False, eq=False)
class Zscorer:
    original_values: NDArray = field(repr=False)
    mean: float = field(repr=True)
    std_dev: float = field(repr=True)
    n_values: int = field(repr=True)

    real_value: float = field(default=None, repr=True)
    real_p_value: float = field(default=None, repr=True)
    z_score_value: float = field(default=None, repr=True) # z-score values


    @classmethod
    def init_from_values(cls, stats_corr_values: NDArray, real_value=None, real_p_value=None):
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





@define(slots=False, repr=False, eq=False)
class RankOrderResult(HDFMixin, AttrsBasedClassHelperMixin, ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even
    
    
    """
    ranked_aclus_stats_dict: Dict[int, LongShortStatsTuple] = serialized_field(repr=False)
    selected_spikes_fragile_linear_neuron_IDX_dict: Dict = serialized_field(repr=False)
    
    long_z_score: NDArray = serialized_field()
    short_z_score: NDArray = serialized_field()
    long_short_z_score_diff: NDArray = serialized_field()
    
    spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    epochs_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)

    selected_spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    extra_info_dict: Dict = serialized_field(default=Factory(dict), repr=False)
    
    @classmethod
    def init_from_analysis_output_tuple(cls, a_tuple):
        """
        ## Ripple Rank-Order Analysis:
        _ripples_outputs = RankOrderAnalyses.main_ripples_analysis(curr_active_pipeline, num_shuffles=1000, rank_alignment='first')

        # Unwrap:
        (odd_ripple_outputs, even_ripple_outputs, ripple_evts_paired_tests) = _ripples_outputs


        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values) = odd_ripple_outputs
        
        """
        ranked_aclus_stats_dict, selected_spikes_fragile_linear_neuron_IDX_dict, (long_z_score_values, short_z_score_values, long_short_z_score_diff_values) = a_tuple
        return cls(is_global=True, ranked_aclus_stats_dict=ranked_aclus_stats_dict, selected_spikes_fragile_linear_neuron_IDX_dict=selected_spikes_fragile_linear_neuron_IDX_dict, long_z_score=long_z_score_values, short_z_score=short_z_score_values, long_short_z_score_diff=long_short_z_score_diff_values)

    def __iter__(self):
        """ allows unpacking. See https://stackoverflow.com/questions/37837520/implement-packing-unpacking-in-an-object """
        return iter(astuple(self, filter=attrs.filters.exclude((self.__attrs_attrs__.is_global, self.__attrs_attrs__.spikes_df, self.__attrs_attrs__.epochs_df, self.__attrs_attrs__.selected_spikes_df)))) #  'is_global'
    

# Define the namedtuples for most-likely computations:
DirectionalRankOrderLikelihoods = namedtuple('DirectionalRankOrderLikelihoods', ['long_relative_direction_likelihoods', 
                                                           'short_relative_direction_likelihoods', 
                                                           'long_best_direction_indices', 
                                                           'short_best_direction_indices'])


DirectionalRankOrderResultBase = namedtuple('DirectionalRankOrderResultBase', ['active_epochs', 
                                                                       'long_best_dir_z_score_values', 
                                                           'short_best_dir_z_score_values', 
                                                           'long_short_best_dir_z_score_diff_values', 
                                                           'directional_likelihoods_tuple', "masked_z_score_values_list"])

class DirectionalRankOrderResult(DirectionalRankOrderResultBase):
    def plot_histogram(self):
        return pd.DataFrame({'long_z_scores': self.long_best_dir_z_score_values, 'short_z_scores': self.short_best_dir_z_score_values}).hist()
    


@define(slots=False, repr=False, eq=False)
class RankOrderComputationsContainer(HDFMixin, AttrsBasedClassHelperMixin, ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even
    

    Usage:    
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderComputationsContainer, RankOrderResult
    
        odd_ripple_rank_order_result = RankOrderResult.init_from_analysis_output_tuple(odd_ripple_outputs)
        even_ripple_rank_order_result = RankOrderResult.init_from_analysis_output_tuple(even_ripple_outputs)
        curr_active_pipeline.global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(odd_ripple=odd_ripple_rank_order_result, even_ripple=even_ripple_rank_order_result, odd_laps=odd_laps_rank_order_result, even_laps=even_laps_rank_order_result)

    """
    LR_ripple: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    RL_ripple: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    LR_laps: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    RL_laps: Optional[RankOrderResult] = serialized_field(default=None, repr=False)
    
    ripple_most_likely_result_tuple: Optional[DirectionalRankOrderResult] = serialized_field(default=None, repr=False)
    laps_most_likely_result_tuple: Optional[DirectionalRankOrderResult] = serialized_field(default=None, repr=False)

    minimum_inclusion_fr_Hz: float = serialized_attribute_field(default=2.0, repr=True)
    

    def __iter__(self):
        """ allows unpacking. See https://stackoverflow.com/questions/37837520/implement-packing-unpacking-in-an-object """
        # return iter(astuple(self)) # deep unpacking causes problems
        return iter(astuple(self, filter=attrs.filters.exclude(self.__attrs_attrs__.is_global, self.__attrs_attrs__.ripple_most_likely_result_tuple, self.__attrs_attrs__.laps_most_likely_result_tuple, self.__attrs_attrs__.minimum_inclusion_fr_Hz))) #  'is_global'




# ==================================================================================================================== #
# 2023-11-16 - Long/Short Most-likely LR/RL decoder                                                                    #
# ==================================================================================================================== #




class RankOrderAnalyses:
    """ 
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses
    
    odd_outputs, even_outputs, ripple_evts_paired_tests = RankOrderAnalyses.main_ripples_analysis(curr_active_pipeline, num_shuffles:int=300, rank_alignment='first')
    
    
    Potential Speedups:
        Multiprocessing could be used to parallelize:
            - Basically the four templates:
                Direction (Odd/Even)
                    Track (Long/Short)


    """
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

    def _perform_plot_z_score_diff(epoch_idx_list, RL_laps_long_short_z_score_diff_values, LR_laps_long_short_z_score_diff_values, variable_name='Lap', x_axis_name_suffix='Index', point_data_values=None):
        """ plots the z-score differences 
        Usage:
            app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _perform_plot_z_score_diff(deepcopy(global_laps).lap_id, RL_laps_long_short_z_score_diff_values, LR_laps_long_short_z_score_diff_values)
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
        n_y_points = np.shape(RL_laps_long_short_z_score_diff_values)[0]
        if n_y_points > n_x_points:
            num_missing_points: int = n_y_points - n_x_points
            print(f'WARNING: trimming y-data to [{num_missing_points}:]')
            RL_laps_long_short_z_score_diff_values = RL_laps_long_short_z_score_diff_values[num_missing_points:]
            if LR_laps_long_short_z_score_diff_values is not None:
                LR_laps_long_short_z_score_diff_values = LR_laps_long_short_z_score_diff_values[num_missing_points:]

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

        if LR_laps_long_short_z_score_diff_values is not None:
            first_plot_name = 'even'
        else:
            first_plot_name = 'best'

        even_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list, RL_laps_long_short_z_score_diff_values, brush=pg.mkBrush(DisplayColorsEnum.Laps.RL), pen=symbolPen, symbol='t2', name=first_plot_name, hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()) ## setting pen=None disables line drawing
        if LR_laps_long_short_z_score_diff_values is not None:
            odd_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list, LR_laps_long_short_z_score_diff_values, brush=pg.mkBrush(DisplayColorsEnum.Laps.LR), pen=symbolPen, symbol='t3', name='odd', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()) ## setting pen=None disables line drawing
        else:
            odd_out_plot_1D = None

        p1.addItem(even_out_plot_1D)
        if LR_laps_long_short_z_score_diff_values is not None:
            p1.addItem(odd_out_plot_1D)
        
        # even_out_plot_1D: pg.PlotDataItem = p1.scatterPlot(epoch_idx_list, even_laps_long_short_z_score_diff_values, pen=None, symbolBrush=pg.mkBrush(DisplayColorsEnum.Laps.even), symbolPen=symbolPen, symbol='t2', name='even', hoverable=True, hoverPen=pg.mkPen('w', width=2)) ## setting pen=None disables line drawing
        # odd_out_plot_1D: pg.PlotDataItem = p1.scatterPlot(epoch_idx_list, odd_laps_long_short_z_score_diff_values, pen=None, symbolBrush=pg.mkBrush(DisplayColorsEnum.Laps.odd), symbolPen=symbolPen, symbol='t3', name='odd', hoverable=True, hoverPen=pg.mkPen('w', width=2)) ## setting pen=None disables line drawing


        # good_RL_laps_long_short_z_score_diff_values = RL_laps_long_short_z_score_diff_values[np.isfinite(RL_laps_long_short_z_score_diff_values)]


        ## Add marginal histogram to the right of the main plot here:
        py: pg.PlotItem = win.addPlot(row=1, col=1, right='Marginal Long-Short Z-Score Diff', hoverable=True) # , bottom=f'{variable_name} {x_axis_name_suffix}', title=f'Marginal Rank-Order Long-Short ZScore Diff for {variable_name}'
        ## compute standard histogram
        number_of_bins: int = 21
        vals = deepcopy(RL_laps_long_short_z_score_diff_values)
        y,x = np.histogram(vals, bins=number_of_bins)
        # Plot histogram along y-axis:
        py.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,150), orientation='horizontal')

        # x = x[:-1] + np.diff(x) / 2 # Adjust x values for stepMode="center"
        # py.plot(y, x, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,150), orientation='horizontal')

        # ## Using stepMode="center" causes the plot to draw two lines for each sample. notice that len(x) == len(y)+1
        # py.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,150)) # swapping x, y should make it horizontal? , stepMode="center"

        if LR_laps_long_short_z_score_diff_values is not None:
            print('TODO: add the LR scatter')
            # good_LR_laps_long_short_z_score_diff_values = LR_laps_long_short_z_score_diff_values[np.isfinite(LR_laps_long_short_z_score_diff_values)]
            vals = deepcopy(LR_laps_long_short_z_score_diff_values)
            y,x = np.histogram(vals, bins=number_of_bins)
            # Plot histogram along y-axis:
            py.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,150), orientation='horizontal', name='LR')


        return app, win, p1, (even_out_plot_1D, odd_out_plot_1D), (py, )






    @classmethod
    def common_analysis_helper(cls, curr_active_pipeline, num_shuffles:int=300, minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,9]):
        ## Shared:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)

        # Recover from the saved global result:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']

        # non-shared templates:
        non_shared_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) #.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = non_shared_templates.get_decoders()
        any_list_neuron_IDs = non_shared_templates.any_decoder_neuron_IDs # neuron_IDs as they appear in any list   
        # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        global_spikes_df = global_spikes_df.spikes.sliced_by_neuron_id(any_list_neuron_IDs)
        
        # ## OLD method, directly get the decoders from `directional_laps_results` using `.get_decoders(...)` or `.get_shared_aclus_only_decoders(...)`:
        # long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()

        # NEW 2023-11-22 method: Get the templates (which can be filtered by frate first) and the from those get the decoders):        
        # shared_aclus_only_templates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = shared_aclus_only_templates.get_decoders()

        ## 2023-10-24 - Simple long/short (2-template, direction independent) analysis:
        # shuffle_helper = ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=1000, bimodal_exclude_aclus=[5, 14, 25, 46, 61, 66, 86, 88, 95])

        # 2023-10-26 - Direction Dependent (4 template) analysis: long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D
        # odd_shuffle_helper: ShuffleHelper = ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        # even_shuffle_helper: ShuffleHelper = ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])

        ### non-shared_aclus_only
        odd_shuffle_helper: ShuffleHelper = ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_LR_one_step_decoder_1D, short_LR_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        even_shuffle_helper: ShuffleHelper = ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_RL_one_step_decoder_1D, short_RL_one_step_decoder_1D, num_shuffles=num_shuffles, bimodal_exclude_aclus=[])
        

        # ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=num_shuffles, bimodal_exclude_aclus=bimodal_exclude_aclus)
        # ShuffleHelper.init_from_long_short_shared_aclus_only_decoders(long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, num_shuffles=num_shuffles, bimodal_exclude_aclus=bimodal_exclude_aclus)
        return global_spikes_df, (odd_shuffle_helper, even_shuffle_helper)



    @classmethod
    @function_attributes(short_name=None, tags=['subfn', 'preprocess', 'spikes_df'], input_requires=[], output_provides=[], uses=[], used_by=['compute_shuffled_rankorder_analyses'], creation_date='2023-11-22 11:04', related_items=[])
    def preprocess_spikes_df(cls, active_spikes_df: pd.DataFrame, active_epochs, shuffle_helper: ShuffleHelper, no_interval_fill_value=-1, min_num_unique_aclu_inclusions: int=5):
        """Preprocesses the active spikes DataFrame and the active_epochs dataframe by extracting shuffle helper data, deep copying, and adding epoch IDs and dropping epochs with fewer than minimum aclus.
        
        # 2023-12-08 12:53: - [X] Drop epochs with fewer than the minimum active aclus
        
        """
        # Extract data from shuffle helper
        shared_aclus_only_neuron_IDs, is_good_aclus, num_shuffles, shuffled_aclus, shuffle_IDXs, (long_pf_peak_ranks, short_pf_peak_ranks) = shuffle_helper.to_tuple()
        # Deep copy and preprocess active spikes DataFrame
        active_spikes_df = deepcopy(active_spikes_df)

        # Drop spikes outside of `shared_aclus_only_neuron_IDs`
        #TODO 2023-12-08 13:30: - [ ] This might be limiting to only the shared
         
        # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        # active_spikes_df = active_spikes_df.spikes.sliced_by_neuron_id(shared_aclus_only_neuron_IDs)

        active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        # Add epoch IDs to the spikes DataFrame
        active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=no_interval_fill_value)
        active_spikes_df.drop(active_spikes_df.loc[active_spikes_df['Probe_Epoch_id'] == no_interval_fill_value].index, inplace=True)
        # Sort by columns: 't_rel_seconds' (ascending), 'aclu' (ascending)
        active_spikes_df = active_spikes_df.sort_values(['t_rel_seconds', 'aclu'])

        # now filter epochs based on the number of included aclus:        
        filtered_active_epochs = Epoch.filter_epochs(active_epochs, pos_df=None, spikes_df=active_spikes_df, min_epoch_included_duration=None, max_epoch_included_duration=None, maximum_speed_thresh=None,
                                                     min_inclusion_fr_active_thresh=None, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions)


        # Now that we have `filtered_active_epochs`, we need to update the 'Probe_Epoch_id' because the epoch id's might have changed
        active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        # Add epoch IDs to the spikes DataFrame
        active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=filtered_active_epochs.to_dataframe(), epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name=None, override_time_variable_name='t_rel_seconds', no_interval_fill_value=no_interval_fill_value)
        active_spikes_df.drop(active_spikes_df.loc[active_spikes_df['Probe_Epoch_id'] == no_interval_fill_value].index, inplace=True)
        # Sort by columns: 't_rel_seconds' (ascending), 'aclu' (ascending)
        active_spikes_df = active_spikes_df.sort_values(['t_rel_seconds', 'aclu'])
        active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()

        return shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, long_pf_peak_ranks, short_pf_peak_ranks, active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, filtered_active_epochs


    @classmethod
    def select_and_rank_spikes(cls, active_spikes_df: pd.DataFrame, active_aclu_to_fragile_linear_neuron_IDX_dict, rank_alignment: str, time_variable_name_override: Optional[str]=None):
        """Selects and ranks spikes based on rank_alignment, and organizes them into structured dictionaries.
        
        epoch_ranked_aclus_dict: Dict[int, Dict[int, float]]: a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
        epoch_ranked_fragile_linear_neuron_IDX_dict: Dict[int, NDArray]: 
        epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict[int, NDArray]: 
        selected_spikes_only_df: pd.DataFrame: an output dataframe containing only the select (e.g. .first(), .median(), etc) spike from each template.
        
        
        """
        if time_variable_name_override is None:
            time_variable_name_override = active_spikes_df.spikes.time_variable_name
        # Determine which spikes to use to represent the order
        selected_spikes = active_spikes_df.groupby(['Probe_Epoch_id', 'aclu'])[time_variable_name_override]

        if rank_alignment == 'first':
            selected_spikes = selected_spikes.first()  # first spike times only
        elif rank_alignment == 'median':
            selected_spikes = selected_spikes.median()  # median spike times only
        elif rank_alignment == 'center_of_mass':
            selected_spikes = compute_placefield_center_of_masses(selected_spikes) # NOTE: I'm not sure if this returns spikes from spikes_df or just times.
        else:
            raise NotImplementedError(f'Invalid rank_alignment specified: {rank_alignment}. Valid options are [first, median, center_of_mass].')
        
        # Build a `selected_spikes_only_df` containing only the selected spikes for use in plotting:
        # selected_spikes_only_df = deepcopy(active_spikes_df)[active_spikes_df.index == selected_spikes.index] # this won't work because it has a multi-index
        # Reset index on selected_spikes to make 'Probe_Epoch_id' and 'aclu' as columns
        _selected_spikes_reset = deepcopy(selected_spikes).reset_index()
        # Merge with original DataFrame to filter only the selected spikes
        selected_spikes_only_df: pd.DataFrame = pd.merge(active_spikes_df, _selected_spikes_reset, on=['Probe_Epoch_id', 'aclu', time_variable_name_override])


        # Rank the aclu values by their first t value in each Probe_Epoch_id
        ranked_aclus = selected_spikes.groupby('Probe_Epoch_id').rank(method='dense')  # Resolve ties in ranking

        # Create structured dictionaries
        epoch_ranked_aclus_dict: Dict[int, Dict[int, float]] = {} # create a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
        epoch_ranked_fragile_linear_neuron_IDX_dict: Dict[int, NDArray] = {}
        epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict[int, NDArray] = {}

        grouped = selected_spikes.groupby(['Probe_Epoch_id', 'aclu'])
        for (epoch_id, aclu), rank in zip(ranked_aclus.index, ranked_aclus):
            if epoch_id not in epoch_ranked_aclus_dict:
                # Initialize new dicts/arrays for the epoch if needed:
                epoch_ranked_aclus_dict[epoch_id] = {}
                epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id] = []
                epoch_selected_spikes_fragile_linear_neuron_IDX_dict[epoch_id] = []

            ## Add the rank to the dict/array
            epoch_ranked_aclus_dict[epoch_id][aclu] = float(rank)
            neuron_IDX = active_aclu_to_fragile_linear_neuron_IDX_dict[aclu] # ordered dict that maps each aclu to a flat neuronIDX!
            epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id].append((neuron_IDX, float(rank)))
            a_value: float = float(grouped.get_group((epoch_id, aclu)).values[0]) # extracts the single float item
            epoch_selected_spikes_fragile_linear_neuron_IDX_dict[epoch_id].append((neuron_IDX, a_value)) # note we are adding indicies, not aclus
        
        # Convert to np.ndarrays
        epoch_ranked_fragile_linear_neuron_IDX_dict = {epoch_id: np.array(vals) for epoch_id, vals in epoch_ranked_fragile_linear_neuron_IDX_dict.items()}
        epoch_selected_spikes_fragile_linear_neuron_IDX_dict = {epoch_id: np.array(vals) for epoch_id, vals in epoch_selected_spikes_fragile_linear_neuron_IDX_dict.items()} # selected:

        return epoch_ranked_aclus_dict, epoch_ranked_fragile_linear_neuron_IDX_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, selected_spikes_only_df



    @classmethod
    @function_attributes(short_name=None, tags=['shuffle', 'rank_order', 'main'], input_requires=[], output_provides=['LongShortStatsTuple'], uses=['cls.preprocess_spikes_df', 'cls.select_and_rank_spikes', 'LongShortStatsTuple'], used_by=[], creation_date='2023-10-21 00:23', related_items=[])
    def compute_shuffled_rankorder_analyses(cls, active_spikes_df: pd.DataFrame, active_epochs, shuffle_helper: ShuffleHelper, rank_alignment: str = 'first', disable_re_ranking:bool=True, min_num_unique_aclu_inclusions: int=5, debug_print=True) -> RankOrderResult:
        """ Extracts the two templates (long/short) from the shuffle_helper in addition to the shuffled_aclus, shuffle_IDXs.

        
        min_num_unique_aclu_inclusions: int
        
        """
        # post_process_statistic_value_fn = lambda x: np.abs(x)
        post_process_statistic_value_fn = lambda x: float(x) # basically NO-OP
        
        # Preprocess the spikes DataFrame
        (shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, long_pf_peak_ranks, short_pf_peak_ranks, active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, filtered_active_epochs) = cls.preprocess_spikes_df(active_spikes_df, active_epochs, shuffle_helper, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions)
        
        # TODO 2023-11-21 05:42: - [ ] todo_description want the aclus as well, not just the `long_pf_peak_ranks`

        #TODO 2023-12-08 12:53: - [ ] Drop epochs with fewer than the minimum active aclus
        
        # Select and rank spikes
        epoch_ranked_aclus_dict, epoch_ranked_fragile_linear_neuron_IDX_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, selected_spikes_only_df = cls.select_and_rank_spikes(active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, rank_alignment)

        ## OUTPUT DICTS:
        # create a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
        output_dict = {}

        ## Loop over the results now to do the actual stats:
        epoch_ranked_aclus_stats_dict = {}

        for epoch_id in list(epoch_ranked_aclus_dict.keys()):

            ## TODO: might need to get the specific aclus that are active in the epoch and limit to the intersection of those and the current decoder:
            epoch_spikes_active_aclus = np.array(list(epoch_ranked_aclus_dict[epoch_id].keys())) # get the actual aclus instead of the indicies here.
            epoch_spikes_active_ranks = np.array(list(epoch_ranked_aclus_dict[epoch_id].values()))
            ## 2. Now get the template aclus to filter the epoch_active_aclus by (note there are way more `epoch_active_aclus` (like 81) than template ones.
            # shared_aclus_only_neuron_IDs # (for now). In the future the `template_aclus` might be template-specific instead of shared:
            template_aclus: NDArray = shared_aclus_only_neuron_IDs
            is_epoch_aclu_included_in_template: NDArray[np.bool_] = np.isin(epoch_spikes_active_aclus, template_aclus) # a bool array indicating whether each aclu active in the epoch (spikes_df) is included in the template.


            # BEGIN 2023-11-22 NEW Implementation: _______________________________________________________________________________ #

            # Chop the template down to the active spikes AND chop the active spikes down to the template:
            actually_included_epoch_aclus = epoch_spikes_active_aclus[is_epoch_aclu_included_in_template] # note this must be strictly smaller than the template aclus, AND strictly less than the epoch_active_aclus.
            actually_included_epoch_ranks = epoch_spikes_active_ranks[is_epoch_aclu_included_in_template]

            #TODO 2023-11-22 11:30: - [ ] Does chopping the template down vs. leaving those entries in there change the spearman?

            # long_pf_peak_ranks, short_pf_peak_ranks
            assert np.shape(long_pf_peak_ranks) == np.shape(shared_aclus_only_neuron_IDs)
            assert np.shape(short_pf_peak_ranks) == np.shape(shared_aclus_only_neuron_IDs)
            
            # Chop the other direction:
            is_template_aclu_actually_active_in_epoch: NDArray = np.isin(template_aclus, actually_included_epoch_aclus) # a bool array indicating whether each aclu in the template is active in  in the epoch (spikes_df). Used for indexing into the template peak_ranks (`long_pf_peak_ranks`, `short_pf_peak_ranks`)
            template_epoch_actually_included_aclus: NDArray = np.array(template_aclus)[is_template_aclu_actually_active_in_epoch] ## `actually_included_template_aclus`: the final aclus for this template actually active in this epoch

            epoch_active_long_pf_peak_ranks = np.array(long_pf_peak_ranks)[is_template_aclu_actually_active_in_epoch]
            epoch_active_short_pf_peak_ranks = np.array(short_pf_peak_ranks)[is_template_aclu_actually_active_in_epoch]
            #TODO 2023-11-22 11:35: - [ ] Is there the possibility that the template doesn't have spikes that are present in the epoch? I think so in general.
            assert np.shape(epoch_active_short_pf_peak_ranks) == np.shape(actually_included_epoch_ranks), f"np.shape(epoch_active_short_pf_peak_ranks): {np.shape(epoch_active_short_pf_peak_ranks)}, np.shape(actually_included_epoch_ranks): {np.shape(actually_included_epoch_ranks)}\n\tTODO 2023-11-22 11:35: - [ ] Is there the possibility that the template doesn't have spikes that are present in the epoch? I think so in general." # 
            assert np.shape(epoch_active_short_pf_peak_ranks) == np.shape(epoch_active_long_pf_peak_ranks)
            # NEW 2023-11-22 - So now have: actually_included_epoch_aclus, actually_included_epoch_ranks, (actually_included_template_aclus, epoch_active_long_pf_peak_ranks, epoch_active_short_pf_peak_ranks)

            # END NEW:

            # 4. Final step is getting the actual indicies into the template aclus (the template-relative neuronIDXs):
            _template_aclu_list = list(template_aclus) # convert to a temporary basic python list so that `.index(aclu)` works in the next line.
            template_epoch_neuron_IDXs: NDArray[int] = np.array([_template_aclu_list.index(aclu) for aclu in actually_included_epoch_aclus]) # should be the appropriate neuronIDXs in the template-relative array

            epoch_ranked_fragile_linear_neuron_IDXs_array = epoch_ranked_fragile_linear_neuron_IDX_dict[epoch_id]
            epoch_neuron_IDX_ranks = np.squeeze(epoch_ranked_fragile_linear_neuron_IDXs_array[is_epoch_aclu_included_in_template,1]) # the ranks just for this epoch, just for this template
            

            # FINAL NOTE: `actually_included_template_aclus`, `template_epoch_neuron_IDXs` contain the actual IDX and aclus for this template active during this epoch
  
            # Note that now (after boolean slicing), both `epoch_neuron_IDXs` and `epoch_neuron_IDX_ranks` can be LESS than the `shared_aclus_only_neuron_IDs`. They are indexed?
            # Instead of `epoch_neuron_IDXs`, use `template_epoch_neuron_IDXs` to the get neuron_IDXs relative to this template:`
            assert np.size(template_epoch_neuron_IDXs) == np.size(epoch_neuron_IDX_ranks), f"{np.size(epoch_neuron_IDX_ranks)} and len(template_epoch_neuron_IDXs): {np.size(template_epoch_neuron_IDXs)}"
            #TODO 2023-11-21 20:49: - [ ] HERE IS WHERE I LEFT OFF. I now have filtered neuron_IDXs corresponding to the ranks for this epoch, but now I DON'T think they correspond to the template neuron_IDXs!!
            if debug_print:
                print(f'epoch_id: {epoch_id}')
                # print(f'\tepoch_neuron_IDXs: {print_array(epoch_neuron_IDXs)}')
                print(f'\ttemplate_epoch_neuron_IDXs: {print_array(template_epoch_neuron_IDXs)}')
                print(f'\tepoch_neuron_IDX_ranks: {print_array(epoch_neuron_IDX_ranks)}')
    
            #TODO 2023-11-22 08:35: - [ ] keep da' indicies we actually use for this template/epoch. They're needed in the RankOrderRastersDebugger.
            output_dict[epoch_id] = (template_epoch_neuron_IDXs, template_epoch_actually_included_aclus, epoch_neuron_IDX_ranks) # might need multiple for each templates if they aren't clipped to shared.

            ## EPOCH SPECIFIC:
            long_spearmanr_rank_stats_results = []
            short_spearmanr_rank_stats_results = []

            # The "real" result for this epoch:
            # active_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, template_epoch_neuron_IDXs, disable_re_ranking=disable_re_ranking) # encountering np.shape(epoch_neuron_IDXs): (41,) but np.shape(long_pf_peak_ranks): (34,)
            # real_long_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
            # NEW 2023-11-22: epoch_active_long_pf_peak_ranks mode:
            active_epoch_aclu_long_ranks = relative_re_ranking(epoch_active_long_pf_peak_ranks, None, disable_re_ranking=disable_re_ranking) 
            real_long_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_long_ranks, actually_included_epoch_ranks) # active_epoch_aclu_long_ranks: np.array([3]), actually_included_epoch_ranks: np.actually_included_epoch_ranks
            real_long_result_corr_value = post_process_statistic_value_fn(real_long_rank_stats.statistic)
            
            # active_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, template_epoch_neuron_IDXs, disable_re_ranking=disable_re_ranking)
            # real_short_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
            # NEW 2023-11-22: epoch_active_long_pf_peak_ranks mode:
            active_epoch_aclu_short_ranks = relative_re_ranking(epoch_active_short_pf_peak_ranks, None, disable_re_ranking=disable_re_ranking)
            real_short_rank_stats = scipy.stats.spearmanr(active_epoch_aclu_short_ranks, actually_included_epoch_ranks)
            real_short_result_corr_value = post_process_statistic_value_fn(real_short_rank_stats.statistic)
            
            if debug_print:
                print(f'\tactive_epoch_aclu_long_ranks[{epoch_id}]: {print_array(active_epoch_aclu_long_ranks)}')
                print(f'\tactive_epoch_aclu_short_ranks[{epoch_id}]: {print_array(active_epoch_aclu_short_ranks)}')
                
            ## PERFORM SHUFFLE HERE:
            # On-the-fly shuffling mode using shuffle_helper:
            epoch_specific_shuffled_aclus, epoch_specific_shuffled_indicies = shuffle_helper.generate_shuffle(template_epoch_actually_included_aclus) # TODO: peformance, might be slower than pre-shuffling method. Wait, with a fixed seed are all the shuffles the same????
                
            for i, (epoch_specific_shuffled_aclus, epoch_specific_shuffled_indicies) in enumerate(zip(epoch_specific_shuffled_aclus, epoch_specific_shuffled_indicies)):
                #TODO 2023-11-22 12:50: - [X] Are the sizes correct since `a_shuffled_IDXs` doesn't know the size of the template?

                ## Get the matching components of the long/short pf ranks using epoch_ranked_fragile_linear_neuron_IDXs's first column which are the relevant indicies:
                # active_shuffle_epoch_aclu_long_ranks = relative_re_ranking(long_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
                # long_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_long_ranks, epoch_neuron_IDX_ranks)
                # NEW 2023-11-22: epoch_active_long_pf_peak_ranks mode:
                active_shuffle_epoch_aclu_long_ranks = relative_re_ranking(epoch_active_long_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
                long_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_long_ranks, actually_included_epoch_ranks)
                assert np.shape(active_shuffle_epoch_aclu_long_ranks) == np.shape(actually_included_epoch_ranks)
                long_result = (post_process_statistic_value_fn(long_rank_stats.statistic), long_rank_stats.pvalue)
                long_spearmanr_rank_stats_results.append(long_result)
                
                # active_shuffle_epoch_aclu_short_ranks = relative_re_ranking(short_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
                # short_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_short_ranks, epoch_neuron_IDX_ranks)
                # NEW 2023-11-22: epoch_active_long_pf_peak_ranks mode:
                active_shuffle_epoch_aclu_short_ranks = relative_re_ranking(epoch_active_short_pf_peak_ranks, epoch_specific_shuffled_indicies, disable_re_ranking=disable_re_ranking)
                short_rank_stats = scipy.stats.spearmanr(active_shuffle_epoch_aclu_short_ranks, actually_included_epoch_ranks)
                assert np.shape(active_shuffle_epoch_aclu_short_ranks) == np.shape(actually_included_epoch_ranks)
                short_result = (post_process_statistic_value_fn(short_rank_stats.statistic), short_rank_stats.pvalue)
                short_spearmanr_rank_stats_results.append(short_result)
            ## END for shuffle

            long_spearmanr_rank_stats_results = np.array(long_spearmanr_rank_stats_results)
            short_spearmanr_rank_stats_results = np.array(short_spearmanr_rank_stats_results)

            long_stats_corr_values = long_spearmanr_rank_stats_results[:,0]
            short_stats_corr_values = short_spearmanr_rank_stats_results[:,0]
            
            long_stats_z_scorer = Zscorer.init_from_values(long_stats_corr_values, real_long_result_corr_value, real_long_rank_stats.pvalue)
            short_stats_z_scorer = Zscorer.init_from_values(short_stats_corr_values, real_short_result_corr_value, real_short_rank_stats.pvalue)


            is_forward_replay: bool = ((np.mean([long_stats_z_scorer.z_score_value, short_stats_z_scorer.z_score_value])) > 0.0)

            # long_short_z_diff: float = np.sign(np.abs(long_stats_z_scorer.z_score_value) - np.abs(short_stats_z_scorer.z_score_value))

            
            always_positive_long_short_magnitude_diff: float = np.max([np.abs(long_stats_z_scorer.z_score_value), np.abs(short_stats_z_scorer.z_score_value)]) - np.min([np.abs(long_stats_z_scorer.z_score_value), np.abs(short_stats_z_scorer.z_score_value)])
            assert (always_positive_long_short_magnitude_diff >= 0.0), f"always_positive_long_short_magnitude_diff: {always_positive_long_short_magnitude_diff}"
            long_or_short_polarity_multiplier: float = np.sign(np.abs(long_stats_z_scorer.z_score_value) - np.abs(short_stats_z_scorer.z_score_value)) # -1 if short is bigger, +1 if long is bigger
            if (always_positive_long_short_magnitude_diff > 0.0):
                assert (np.isclose(long_or_short_polarity_multiplier, -1.0) or np.isclose(long_or_short_polarity_multiplier, 1.0)), f"long_or_short_polarity_multiplier: {long_or_short_polarity_multiplier} should equal -1 or +1"            
                long_short_z_diff: float = long_or_short_polarity_multiplier * always_positive_long_short_magnitude_diff
            else:
                long_short_z_diff: float = 0.0 # the value is exactly zero. Surprising.
                
            long_short_naive_z_diff: float = long_stats_z_scorer.z_score_value - short_stats_z_scorer.z_score_value # `long_short_naive_z_diff` was the old pre-2023-12-07 way of calculating the z-score diff.
            epoch_ranked_aclus_stats_dict[epoch_id] = LongShortStatsTuple(long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay)


        ## END for epoch_id

        # Extract the results:
        long_z_score_values = []
        short_z_score_values = []
        long_short_z_score_diff_values = []
        long_short_z_score_diff_values = []

        for epoch_id, epoch_stats in epoch_ranked_aclus_stats_dict.items():
            long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay = epoch_stats
            # paired_test = pho_stats_paired_t_test(long_stats_z_scorer.z_score_values, short_stats_z_scorer.z_score_values) # this doesn't seem to work well
            long_z_score_values.append(long_stats_z_scorer.z_score_value)
            short_z_score_values.append(short_stats_z_scorer.z_score_value)
            long_short_z_score_diff_values.append(long_short_z_diff)

        long_z_score_values = np.array(long_z_score_values)
        short_z_score_values = np.array(short_z_score_values)
        long_short_z_score_diff_values = np.array(long_short_z_score_diff_values)
        
        return RankOrderResult(is_global=True, ranked_aclus_stats_dict=epoch_ranked_aclus_stats_dict, selected_spikes_fragile_linear_neuron_IDX_dict=epoch_selected_spikes_fragile_linear_neuron_IDX_dict,
                               long_z_score=long_z_score_values, short_z_score=short_z_score_values, long_short_z_score_diff=long_short_z_score_diff_values,
                               spikes_df=active_spikes_df, epochs_df=filtered_active_epochs, selected_spikes_df=selected_spikes_only_df, extra_info_dict=output_dict)
    

    


    @classmethod
    @function_attributes(short_name=None, tags=['rank-order', 'shuffle', 'inst_fr', 'epoch', 'lap', 'replay', 'computation'], input_requires=[], output_provides=[], uses=['DirectionalRankOrderLikelihoods', 'DirectionalRankOrderResult'], used_by=[], creation_date='2023-11-16 18:43', related_items=['plot_rank_order_epoch_inst_fr_result_tuples'])
    def most_likely_directional_rank_order_shuffling(cls, curr_active_pipeline, decoding_time_bin_size=0.003) -> Tuple[DirectionalRankOrderResult, DirectionalRankOrderResult]:
        """ A version of the rank-order shufffling for a set of epochs that tries to use the most-likely direction as the one to decode with.

        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses

            ## Main
            ripple_result_tuple, laps_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(curr_active_pipeline, decoding_time_bin_size=0.003)
        
            
        Reference:
            {"even": "RL", "odd": "LR"}
            [LR, RL], {'LR': 0, 'RL': 1}
            odd (LR) = 0, even (RL) = 1
        """
        # ODD: 0, EVEN: 1
        _ODD_INDEX = 0
        _EVEN_INDEX = 1
        
        # Unpack all directional variables:
        ## {"even": "RL", "odd": "LR"}
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        global_epoch_name = global_any_name

        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)


        ## Extract the rank_order_results:
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']

        # # Use the four epochs to make to a pseudo-y:
        # all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        # all_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D), debug_print=False)
        # all_directional_pf1D_Decoder = BasePositionDecoder(all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        ## Combine the non-directional PDFs and renormalize to get the directional PDF:
        # Inputs: long_LR_pf1D, long_RL_pf1D
        long_directional_decoder_names = ['long_LR', 'long_RL']
        long_directional_pf1D: PfND = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), debug_print=False)
        long_directional_pf1D_Decoder = BasePositionDecoder(long_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        # Inputs: short_LR_pf1D, short_RL_pf1D
        short_directional_decoder_names = ['short_LR', 'short_RL']
        short_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D), debug_print=False) # [LR, RL], {'LR': 0, 'RL': 1}
        short_directional_pf1D_Decoder = BasePositionDecoder(short_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        # takes 6.3 seconds

        ## validation of PfNDs:
        # short_directional_pf1D.plot_ratemaps_2D()

        # Decode using specific directional_decoders:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)
        # spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

        def _compute_best(active_epochs):
            """ captures: long_directional_pf1D_Decoder, short_directional_pf1D_Decoder, spikes_df, decoding_time_bin_size """
            long_directional_decoding_result: DecodedFilterEpochsResult = long_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(spikes_df), active_epochs, decoding_time_bin_size=decoding_time_bin_size)
            short_directional_decoding_result: DecodedFilterEpochsResult = short_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(spikes_df), active_epochs, decoding_time_bin_size=decoding_time_bin_size)
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




        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        ## Replays:
        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay))
        active_epochs = global_replays.copy()
        ripple_directional_likelihoods_tuple = _compute_best(active_epochs)
        long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = ripple_directional_likelihoods_tuple
        # now do the shuffle:

        ## 2023-11-16 - Finally, get the raw z-score values for the best direction at each epoch and then take the long - short difference of those to get `ripple_evts_long_short_best_dir_z_score_diff_values`:
        # Using NumPy advanced indexing to select from array_a or array_b:
        ripple_evts_long_best_dir_z_score_values = np.where(long_best_direction_indicies, rank_order_results.LR_ripple.long_z_score, rank_order_results.RL_ripple.long_z_score)
        ripple_evts_short_best_dir_z_score_values = np.where(short_best_direction_indicies, rank_order_results.LR_ripple.short_z_score, rank_order_results.RL_ripple.short_z_score)
        # print(f'np.shape(ripple_evts_long_best_dir_z_score_values): {np.shape(ripple_evts_long_best_dir_z_score_values)}')
        ripple_evts_long_short_best_dir_z_score_diff_values = ripple_evts_long_best_dir_z_score_values - ripple_evts_short_best_dir_z_score_values
        # print(f'np.shape(ripple_evts_long_short_best_dir_z_score_diff_values): {np.shape(ripple_evts_long_short_best_dir_z_score_diff_values)}')

        # preferred order, but not the current standard: (long_odd_mask, long_even_mask, short_odd_mask, short_even_mask)
        # current standard order: (long_odd_mask, short_odd_mask, long_even_mask, short_even_mask)
        #TODO 2023-11-20 22:02: - [ ] ERROR: CORRECTNESS FAULT: I think the two lists zipped over below are out of order.
        ripple_masked_z_score_values_list: List[ma.masked_array] = [ma.masked_array(x, mask=np.logical_not(a_mask)) for x, a_mask in zip((rank_order_results.LR_ripple.long_z_score, rank_order_results.LR_ripple.short_z_score, rank_order_results.RL_ripple.long_z_score, rank_order_results.RL_ripple.short_z_score),
                                                                                                            ((long_best_direction_indicies == _ODD_INDEX), (long_best_direction_indicies == _EVEN_INDEX), (short_best_direction_indicies == _ODD_INDEX), (short_best_direction_indicies == _EVEN_INDEX)))]
        
        # outputs: ripple_evts_long_short_best_dir_z_score_diff_values
        ripple_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(active_epochs, long_best_dir_z_score_values=ripple_evts_long_best_dir_z_score_values, short_best_dir_z_score_values=ripple_evts_short_best_dir_z_score_values,
                                                                                    long_short_best_dir_z_score_diff_values=ripple_evts_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=ripple_directional_likelihoods_tuple,
                                                                                    masked_z_score_values_list=ripple_masked_z_score_values_list)


        ## Laps:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
        laps_directional_likelihoods_tuple = _compute_best(global_laps)
        long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = laps_directional_likelihoods_tuple
        # odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, rank_order_results.LR_laps.short_z_score, odd_laps_long_short_z_score_diff_values = rank_order_results.LR_laps # LR_laps_rank_order_result
        # even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, rank_order_results.RL_laps.long_z_score, rank_order_results.RL_laps.short_z_score, even_laps_long_short_z_score_diff_values = rank_order_results.RL_laps

        # Using NumPy advanced indexing to select from array_a or array_b:
        laps_long_best_dir_z_score_values = np.where(long_best_direction_indicies, rank_order_results.LR_laps.long_z_score, rank_order_results.RL_laps.long_z_score)
        laps_short_best_dir_z_score_values = np.where(short_best_direction_indicies, rank_order_results.LR_laps.short_z_score, rank_order_results.RL_laps.short_z_score)
        # print(f'np.shape(laps_long_best_dir_z_score_values): {np.shape(laps_long_best_dir_z_score_values)}')
        laps_long_short_best_dir_z_score_diff_values = laps_long_best_dir_z_score_values - laps_short_best_dir_z_score_values
        # print(f'np.shape(laps_long_short_best_dir_z_score_diff_values): {np.shape(laps_long_short_best_dir_z_score_diff_values)}')
        #TODO 2023-11-20 22:02: - [ ] ERROR: CORRECTNESS FAULT: I think the two lists zipped over below are out of order.
        laps_masked_z_score_values_list: List[ma.masked_array] = [ma.masked_array(x, mask=np.logical_not(a_mask)) for x, a_mask in zip((rank_order_results.LR_laps.long_z_score, rank_order_results.LR_laps.short_z_score, rank_order_results.RL_laps.long_z_score, rank_order_results.RL_laps.short_z_score),
                                                                                                            ((long_best_direction_indicies == _ODD_INDEX), (long_best_direction_indicies == _EVEN_INDEX), (short_best_direction_indicies == _ODD_INDEX), (short_best_direction_indicies == _EVEN_INDEX)))]

        laps_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(global_laps, long_best_dir_z_score_values=laps_long_best_dir_z_score_values, short_best_dir_z_score_values=laps_short_best_dir_z_score_values,
                                                                                    long_short_best_dir_z_score_diff_values=laps_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=laps_directional_likelihoods_tuple, 
                                                                                    masked_z_score_values_list=laps_masked_z_score_values_list)

        return ripple_result_tuple, laps_result_tuple


    @function_attributes(short_name=None, tags=['rank-order', 'ripples', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_ripples_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='first', minimum_inclusion_fr_Hz:float=5.0):
        
        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)

        ## Ripple Rank-Order Analysis: needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)


        # curr_active_pipeline.sess.config.preprocessing_parameters
        min_num_unique_aclu_inclusions: int = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions

        global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
        if isinstance(global_replays, pd.DataFrame):
            global_replays = Epoch(global_replays.epochs.get_valid_df())

        ## Replay Epochs:
        LR_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(spikes_df), deepcopy(global_replays), odd_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, debug_print=False)
        RL_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(spikes_df), deepcopy(global_replays), even_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, debug_print=False)

        ripple_evts_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((LR_outputs.long_z_score, LR_outputs.short_z_score), (RL_outputs.long_z_score, RL_outputs.short_z_score))]
        print(f'ripple_evts_paired_tests: {ripple_evts_paired_tests}')
        # [TtestResult(statistic=3.5572800536164495, pvalue=0.0004179523066872734, df=415),
        #  TtestResult(statistic=3.809779392137816, pvalue=0.0001601254566506359, df=415)]
        
        return (LR_outputs, RL_outputs, ripple_evts_paired_tests)
    

    @function_attributes(short_name=None, tags=['rank-order', 'laps', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_laps_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='median', minimum_inclusion_fr_Hz:float=5.0):
        """
        
        _laps_outputs = RankOrderAnalyses.main_laps_analysis(curr_active_pipeline, num_shuffles=1000, rank_alignment='median')
        
        # Unwrap
        (LR_outputs, RL_outputs, laps_paired_tests) = _laps_outputs

        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values) = odd_outputs
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values) = even_outputs

        
        """
        ## Shared:
        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)

        ## Laps Epochs: Needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()

        # curr_active_pipeline.sess.config.preprocessing_parameters
        min_num_unique_aclu_inclusions: int = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions
        
        # TODO: CenterOfMass for Laps instead of median spike
        # laps_rank_alignment = 'center_of_mass'
        LR_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(global_spikes_df), deepcopy(global_laps), odd_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, debug_print=False)
        RL_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(global_spikes_df), deepcopy(global_laps), even_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, debug_print=False)        
        laps_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((LR_outputs.long_z_score, LR_outputs.short_z_score), (RL_outputs.long_z_score, RL_outputs.short_z_score))]
        print(f'laps_paired_tests: {laps_paired_tests}')

        return (LR_outputs, RL_outputs, laps_paired_tests)


    @classmethod
    def validate_has_rank_order_results(cls, curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
        """ 
        TODO: make sure minimum can be passed. Actually, can get it from the pipeline.
        
        """
        # Unpacking:
        rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        oripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple

        # Extract the real spearman-values/p-values:
        LR_long_relative_real_p_values = np.array([x[0].real_p_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
        LR_long_relative_real_values = np.array([x[0].real_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])

        LR_short_relative_real_p_values = np.array([x[1].real_p_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
        LR_short_relative_real_values = np.array([x[1].real_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])

        LR_template_epoch_actually_included_aclus = [v[1] for v in rank_order_results.LR_ripple.extra_info_dict.values()] # (template_epoch_neuron_IDXs, template_epoch_actually_included_aclus, epoch_neuron_IDX_ranks)
        LR_relative_num_cells = np.array([len(v[1]) for v in rank_order_results.LR_ripple.extra_info_dict.values()])

        RL_long_relative_real_p_values = np.array([x[0].real_p_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])
        RL_long_relative_real_values = np.array([x[0].real_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])

        RL_short_relative_real_p_values = np.array([x[1].real_p_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])
        RL_short_relative_real_values = np.array([x[1].real_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])

        RL_template_epoch_actually_included_aclus = [v[1] for v in rank_order_results.RL_ripple.extra_info_dict.values()] # (template_epoch_neuron_IDXs, template_epoch_actually_included_aclus, epoch_neuron_IDX_ranks)
        RL_relative_num_cells = np.array([len(v[1]) for v in rank_order_results.RL_ripple.extra_info_dict.values()])

        ## z-diffs:
        LR_long_short_z_diff = np.array([x.long_short_z_diff for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
        LR_long_short_naive_z_diff = np.array([x.long_short_naive_z_diff for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
        RL_long_short_z_diff = np.array([x.long_short_z_diff for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])
        RL_long_short_naive_z_diff = np.array([x.long_short_naive_z_diff for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])

        # make sure result is for the current minimimum:
        results_minimum_inclusion_fr_Hz = rank_order_results.minimum_inclusion_fr_Hz

        if minimum_inclusion_fr_Hz is not None:
            return (minimum_inclusion_fr_Hz == results_minimum_inclusion_fr_Hz) # makes sure same
        else:
            #TODO 2023-11-29 08:42: - [ ] cannot validate minimum because none was passed, eventually reformulate to use parameters
            return True    

    @classmethod
    def _validate_can_display_RankOrderRastersDebugger(cls, curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
        spikes_df = curr_active_pipeline.sess.spikes_df
        rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        results_minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=results_minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        # make sure result is for the current minimimum:
        if results_minimum_inclusion_fr_Hz is not None:
            return (results_minimum_inclusion_fr_Hz == minimum_inclusion_fr_Hz) # makes sure same
        else:
            return True #TODO 2023-11-29 08:42: - [ ] cannot validate minimum because none was passed, eventually reformulate to use parameters
        

    @classmethod
    def find_only_significant_events(cls, rank_order_results, high_z_criteria: float = 1.96):
        # Find only the significant events (|z| > 1.96):
        _out_z_score = pd.DataFrame({'LR_long_z_scores': rank_order_results.LR_ripple.long_z_score, 'LR_short_z_scores': rank_order_results.LR_ripple.short_z_score,
                    'RL_long_z_scores': rank_order_results.RL_ripple.long_z_score, 'RL_short_z_scores': rank_order_results.RL_ripple.short_z_score})

        n_events: int = len(_out_z_score)
        print(f'n_events: {n_events}')
        
        # Filter rows based on columns: 'LR_long_z_scores', 'LR_short_z_scores' and 2 other columns
        # filtered_z_score_df: pd.DataFrame = _out_z_score[(_out_z_score['LR_long_z_scores'].abs() > high_z_criteria) | (_out_z_score['LR_short_z_scores'].abs() > high_z_criteria) | (_out_z_score['RL_long_z_scores'].abs() > high_z_criteria) | (_out_z_score['RL_short_z_scores'].abs() > high_z_criteria)] # any z-score at all > 1.96
        filtered_z_score_df: pd.DataFrame = _out_z_score[((_out_z_score['LR_long_z_scores'].abs() > high_z_criteria) | (_out_z_score['LR_short_z_scores'].abs() > high_z_criteria)) & ((_out_z_score['RL_long_z_scores'].abs() > high_z_criteria) | (_out_z_score['RL_short_z_scores'].abs() > high_z_criteria))] # at least one direction (both short and long) > 1.96
        # filtered_z_score_df: pd.DataFrame = _out_z_score[((_out_z_score['LR_long_z_scores'].abs() > high_z_criteria) & (_out_z_score['LR_short_z_scores'].abs() > high_z_criteria)) & ((_out_z_score['RL_long_z_scores'].abs() > high_z_criteria) & (_out_z_score['RL_short_z_scores'].abs() > high_z_criteria))] # all required to be > 1.96
        n_significant_events: int = len(filtered_z_score_df)
        print(f'n_significant_events: {n_significant_events}')

        percent_significant_events = float(n_significant_events) / float(n_events)
        print(f'percent_significant_events: {percent_significant_events}')
        
        return filtered_z_score_df, (n_events, n_significant_events, percent_significant_events)



class RankOrderGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'rank_order'
    _computationPrecidence = 1001
    _is_global = True

    @function_attributes(short_name='rank_order_shuffle_analysis', tags=['directional_pf', 'laps', 'rank_order', 'session', 'pf1D', 'pf2D'], input_requires=['DirectionalLaps'], output_provides=['RankOrder'], uses=['RankOrderAnalyses'], used_by=[], creation_date='2023-11-08 17:27', related_items=[],
        validate_computation_test=RankOrderAnalyses.validate_has_rank_order_results, is_global=True)
    def perform_rank_order_shuffle_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, num_shuffles:int=1000, minimum_inclusion_fr_Hz:float=2.0, skip_laps=False):
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
        
        # Needs to store the parameters
        # num_shuffles:int=1000
        # minimum_inclusion_fr_Hz:float=12.0
        
        
        if ('RankOrder' not in global_computation_results.computed_data) or (not hasattr(global_computation_results.computed_data, 'RankOrder')):
            # initialize
            global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(LR_ripple=None, RL_ripple=None, LR_laps=None, RL_laps=None, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, is_global=True)
        
        ## Laps Rank-Order Analysis:
        if not skip_laps:
            print(f'\tcomputing Laps rank-order shuffles:')
            print(f'\t\tnum_shuffles: {num_shuffles}, minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz} Hz')
            # _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='center_of_mass')
            _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='median', minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
            # _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='first')
            (LR_laps_outputs, RL_laps_outputs, laps_paired_tests)  = _laps_outputs
            global_computation_results.computed_data['RankOrder'].LR_laps = LR_laps_outputs
            global_computation_results.computed_data['RankOrder'].RL_laps = RL_laps_outputs

        ## Ripple Rank-Order Analysis:
        print(f'\tcomputing Ripple rank-order shuffles:')
        _ripples_outputs = RankOrderAnalyses.main_ripples_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='first', minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        (LR_ripple_outputs, RL_ripple_outputs, ripple_evts_paired_tests) = _ripples_outputs
        global_computation_results.computed_data['RankOrder'].LR_ripple = LR_ripple_outputs
        global_computation_results.computed_data['RankOrder'].RL_ripple = RL_ripple_outputs

        # Set the global result:
        print(f'\tdone. building global result.')
        global_computation_results.computed_data['RankOrder'].ripple_most_likely_result_tuple, global_computation_results.computed_data['RankOrder'].laps_most_likely_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(owning_pipeline_reference, decoding_time_bin_size=0.01) # 10ms bins

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
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _helper_add_long_short_session_indicator_regions # used in `plot_z_score_diff_and_raw`


class RankOrderGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """
    @function_attributes(short_name='rank_order_debugger', tags=['rank-order','debugger','shuffle', 'interactive', 'slider'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-09 01:12', related_items=[],
                         validate_computation_test=RankOrderAnalyses._validate_can_display_RankOrderRastersDebugger, is_global=True)
    def _display_rank_order_debugger(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ 
            
            """
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results: RankOrderComputationsContainer = global_computation_results.computed_data['RankOrder']
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz

            # track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # shared-only
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            global_spikes_df = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].spikes_df) # #TODO 2023-12-08 12:44: - [ ] does ripple_result_tuple contain a spikes_df?

            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
            print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')

            ## RankOrderRastersDebugger: 
            _out_rank_order_event_raster_debugger = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, ripple_result_tuple.active_epochs, track_templates, rank_order_results.RL_ripple.selected_spikes_fragile_linear_neuron_IDX_dict, rank_order_results.LR_ripple.selected_spikes_fragile_linear_neuron_IDX_dict)

            return _out_rank_order_event_raster_debugger


# def plot_z_score_diff_and_raw(x_values: np.ndarray, long_short_best_dir_z_score_diff_values: np.ndarray, masked_z_score_values_list: List[np.ma.core.MaskedArray], variable_name: str, x_axis_name_suffix: str, point_data_values: np.ndarray, long_epoch=None, short_epoch=None) -> Tuple:
# 	"""
# 	Plots the z-score diff and raw plots for either ripple or lap data.

# 	Args:
# 	- x_values (np.ndarray): array of x-axis values
# 	- long_short_best_dir_z_score_diff_values (np.ndarray): array of z-score diff values
# 	- masked_z_score_values_list (List[np.ma.core.MaskedArray]): list of masked arrays of z-score values
# 	- variable_name (str): name of the variable being plotted (e.g. "Ripple" or "Lap")
# 	- x_axis_name_suffix (str): suffix for the x-axis name (e.g. "Index" or "Mid-time (Sec)")
# 	- point_data_values (np.ndarray): array of point data values
# 	- long_epoch (optional): epoch object for the long epoch
# 	- short_epoch (optional): epoch object for the short epoch

# 	Returns:
# 	- Tuple: tuple containing the plot objects
# 	"""
#     result_tuple
# 	# Plot z-score diff
# 	display_replay_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(x_values, result_tuple.long_short_best_dir_z_score_diff_values, None, variable_name=variable_name, x_axis_name_suffix=x_axis_name_suffix, point_data_values=point_data_values)
# 	app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = display_replay_z_score_diff_outputs # unwrap

# 	# Plot z-score raw
# 	display_replay_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(x_values, *result_tuple.masked_z_score_values_list, variable_name=variable_name, x_axis_name_suffix=x_axis_name_suffix, point_data_values=point_data_values)
# 	raw_app, raw_win, raw_p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D) = display_replay_z_score_raw_outputs

# 	# Add long/short epoch indicator regions
# 	if long_epoch is not None and short_epoch is not None:
# 		long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(p1, long_epoch, short_epoch)
# 		long_epoch_indicator_region_items_raw, short_epoch_indicator_region_items_raw = _helper_add_long_short_session_indicator_regions(raw_p1, long_epoch, short_epoch)
# 	else:
# 		long_epoch_indicator_region_items, short_epoch_indicator_region_items = None, None
# 		long_epoch_indicator_region_items_raw, short_epoch_indicator_region_items_raw = None, None

# 	# Return plot objects
# 	return (app, win, p1, (even_out_plot_1D, odd_out_plot_1D), long_epoch_indicator_region_items, short_epoch_indicator_region_items), (raw_app, raw_win, raw_p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D), long_epoch_indicator_region_items_raw, short_epoch_indicator_region_items_raw)


@function_attributes(short_name=None, tags=['rank-order', 'inst_fr', 'epoch', 'lap', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-16 18:42', related_items=['most_likely_directional_rank_order_shuffling'])
def plot_rank_order_epoch_inst_fr_result_tuples(curr_active_pipeline, result_tuple, analysis_type, included_epoch_idxs=None):
    """
    Generalized function to perform analysis and plot for either ripples or laps.

    Args:
    - curr_active_pipeline: The current active pipeline object.
    - result_tuple: The result tuple specific to ripples or laps.
    - analysis_type: A string, either 'Ripple' or 'Lap', to specify the type of analysis.
    

    Usage:    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import most_likely_directional_rank_order_shuffling
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import plot_rank_order_epoch_inst_fr_result_tuples
        
        ## Main Compute inst-fr-based rank-order shuffling:
        ripple_result_tuple, laps_result_tuple = most_likely_directional_rank_order_shuffling(curr_active_pipeline, decoding_time_bin_size=0.003)
        
        # Plot the Ripple results:
        ripple_outputs = plot_rank_order_epoch_inst_fr_result_tuples(curr_active_pipeline, ripple_result_tuple, 'Ripple')

        # Plot the Lap results:
        lap_outputs = plot_rank_order_epoch_inst_fr_result_tuples(curr_active_pipeline, laps_result_tuple, 'Lap')

    """

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch = curr_active_pipeline.filtered_epochs[long_epoch_name]
    short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]

    # global_spikes_df, _ = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=1000)
    # spikes_df = deepcopy(global_spikes_df)

    if analysis_type == 'Ripple':
        global_events = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
    elif analysis_type == 'Lap':
        global_events = deepcopy(result_tuple.active_epochs)
    else:
        raise ValueError("Invalid analysis type. Choose 'Ripple' or 'Lap'.")

    if isinstance(global_events, pd.DataFrame):
        global_events = Epoch(global_events.epochs.get_valid_df())

    if included_epoch_idxs is not None:
        print(f'filtering global epochs:')
        is_epoch_significant = np.isin(global_events.to_dataframe().index, included_epoch_idxs)
        global_events = deepcopy(global_events).boolean_indicies_slice(is_epoch_significant)
        
        # significant_ripple_epochs: pd.DataFrame = deepcopy(global_events.epochs.get_valid_df())[is_epoch_significant]
    else:
        is_epoch_significant = np.arange(global_events.n_epochs)


    epoch_identifiers = np.arange(global_events.n_epochs)
    x_values = global_events.midtimes
    x_axis_name_suffix = 'Mid-time (Sec)'

    _display_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(
        x_values, result_tuple.long_short_best_dir_z_score_diff_values[is_epoch_significant], None, 
        variable_name=analysis_type, x_axis_name_suffix=x_axis_name_suffix, 
        point_data_values=epoch_identifiers
    )
    _display_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(
        x_values, *[x[is_epoch_significant] for x in result_tuple.masked_z_score_values_list], 
        variable_name=analysis_type, x_axis_name_suffix=x_axis_name_suffix, 
        point_data_values=epoch_identifiers
    )

    app, win, diff_p1, out_plot_1D, *out_hist_stuff = _display_z_score_diff_outputs
    long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(diff_p1, long_epoch, short_epoch)
    raw_app, raw_win, raw_p1, raw_out_plot_1D = _display_z_score_raw_outputs
    long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(raw_p1, long_epoch, short_epoch)

    active_connections_dict = {}  # for holding connections
    return app, win, diff_p1, out_plot_1D, raw_app, raw_win, raw_p1, raw_out_plot_1D



