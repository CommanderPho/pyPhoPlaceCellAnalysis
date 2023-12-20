
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
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
# from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange

from scipy import stats # _recover_samples_per_sec_from_laps_df
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates

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



class SaveStringGenerator:
    """ 
    # 2023-11-27 - I'd like to be able to save/load single results a time, (meaning specific to their parameters):
	day_date_str: str = '2023-12-11-minimum_inclusion_fr_Hz_2_included_qclu_values_1-2_'

    """
    _minimal_decimals_float_formatter = lambda x: f"{x:.1f}".rstrip('0').rstrip('.')
    
    @classmethod
    def generate_save_suffix(cls, minimum_inclusion_fr_Hz: float, included_qclu_values: List[int], day_date: str='2023-12-11') -> str:
        # day_date_str: str = '2023-12-11-minimum_inclusion_fr_Hz_2_included_qclu_values_1-2_'
        print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
        print(f'included_qclu_values: {included_qclu_values}')
        out_filename_str: str = '-'.join([day_date, f'minimum_inclusion_fr', cls._minimal_decimals_float_formatter(minimum_inclusion_fr_Hz), f'included_qclu_values', f'{included_qclu_values}'])
        return out_filename_str

# list = ['2Hz', '12Hz']

def save_rank_order_results(curr_active_pipeline, day_date: str='2023-12-19_729pm'):
    """ saves out the rnak-order and directional laps results to disk.
    
    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
    ## Uses `SaveStringGenerator.generate_save_suffix` and the current rank_order_result's parameters to build a reasonable save name:
    assert curr_active_pipeline.global_computation_results.computed_data['RankOrder'] is not None
    minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computed_data['RankOrder'].minimum_inclusion_fr_Hz
    included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computed_data['RankOrder'].included_qclu_values
    out_filename_str = SaveStringGenerator.generate_save_suffix(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values, day_date=day_date)
    print(f'out_filename_str: "{out_filename_str}"')
    directional_laps_output_path = curr_active_pipeline.get_output_path().joinpath(f'{out_filename_str}DirectionalLaps.pkl').resolve()
    saveData(directional_laps_output_path, (curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']))
    rank_order_output_path = curr_active_pipeline.get_output_path().joinpath(f'{out_filename_str}RankOrder.pkl').resolve()
    saveData(rank_order_output_path, (curr_active_pipeline.global_computation_results.computed_data['RankOrder']))
    # saveData(rank_order_output_path, (asdict(curr_active_pipeline.global_computation_results.computed_data['RankOrder'], recurse=True)))

    # saveData(directional_laps_output_path, (curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'], asdict(curr_active_pipeline.global_computation_results.computed_data['RankOrder'], recurse=False))) 
    # saveData(directional_laps_output_path, (curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']))
    return rank_order_output_path, directional_laps_output_path, out_filename_str
    


@define(slots=False, repr=False, eq=False)
class ShuffleHelper(HDFMixin):
    """ holds the result of shuffling templates. Used for rank-order analyses """
    shared_aclus_only_neuron_IDs: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    is_good_aclus: NDArray = serialized_field(repr=False, eq=attrs.cmp_using(eq=np.array_equal))

    num_shuffles: int = serialized_attribute_field(repr=True) # default=1000
    shuffled_aclus = non_serialized_field(repr=False, is_computable=True)
    shuffle_IDX = non_serialized_field(repr=False, is_computable=True)

    decoder_pf_peak_ranks_list = serialized_field(repr=True)


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

    def generate_shuffle(self, shared_aclus_only_neuron_IDs: NDArray, num_shuffles: Optional[int]=None, seed:Optional[int]=None) -> Tuple[NDArray, NDArray]:
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

        shuffled_aclus, shuffle_IDXs = build_shuffled_ids(shared_aclus_only_neuron_IDs, num_shuffles=num_shuffles, seed=None)
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
class Zscorer(HDFMixin):
    """
    Zscorer recieves the list of raw metric values, one for each shuffle, which is stores in .original_values

    It also receives the "real" non-shuffled value, which is stores in .real_value



    """
    original_values: NDArray = serialized_field(repr=False, is_computable=False, eq=attrs.cmp_using(eq=np.array_equal))
    mean: float = serialized_attribute_field(repr=True, is_computable=True)
    std_dev: float = serialized_attribute_field(repr=True, is_computable=True)
    n_values: int = serialized_attribute_field(repr=True, is_computable=True)

    real_value: float = serialized_attribute_field(default=None, repr=True, is_computable=False)
    real_p_value: float = serialized_attribute_field(default=None, repr=True, is_computable=False)
    z_score_value: float = serialized_attribute_field(default=None, repr=True, is_computable=False) # z-score values


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


@define(slots=False, repr=False, eq=False)
class RankOrderResult(ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even


    """
    # is_global: bool = non_serialized_field(default=True, repr=False)
    ranked_aclus_stats_dict: Dict[int, LongShortStatsTuple] = serialized_field(repr=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v))) # , serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v))
    selected_spikes_fragile_linear_neuron_IDX_dict: Dict[int, NDArray] = serialized_field(repr=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v)))

    long_z_score: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    short_z_score: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    long_short_z_score_diff: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))

    spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    epochs_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)

    selected_spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    extra_info_dict: Dict = non_serialized_field(default=Factory(dict), repr=False)

    @property
    def epoch_template_active_aclus(self) -> Dict[int, NDArray]:
        """

        Usage:
            label_column_type = 'int'
            active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.LR_ripple.epoch_template_active_aclus[x])

        """
        return {k:v[1] for k, v in self.extra_info_dict.items()} # [1] corresponds to `template_epoch_actually_included_aclus`





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
                                                           'directional_likelihoods_tuple', "masked_z_score_values_list", "rank_order_z_score_df"])

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram


class DirectionalRankOrderResult(DirectionalRankOrderResultBase):

    def plot_histograms(self) -> MatplotlibRenderPlots:
        fig = plt.figure(layout="constrained", num='RipplesRankOrderZscore')
        ax_dict = fig.subplot_mosaic(
            [
                ["long_short_best_z_score_diff", "long_short_best_z_score_diff"],
                ["long_best_z_scores", "short_best_z_scores"],
            ],
            # set the height ratios between the rows
            # height_ratios=[8, 1],
            # height_ratios=[1, 1],
            # set the width ratios between the columns
            # width_ratios=[1, 8, 8, 1],
            # sharey=True,
            # gridspec_kw=dict(wspace=0, hspace=0.15) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
        )
        # pd.DataFrame({'long_best_z_scores': ripple_result_tuple.long_best_dir_z_score_values, 'short_best_z_scores': ripple_result_tuple.short_best_dir_z_score_values}).hist(ax=(ax_dict['long_best_z_scores'], ax_dict['short_best_z_scores']))

        # MatplotLibResultContainer(
        plots = (pd.DataFrame({'long_best_z_scores': self.long_best_dir_z_score_values}).hist(ax=ax_dict['long_best_z_scores'], bins=21, alpha=0.8),
            pd.DataFrame({'short_best_z_scores': self.short_best_dir_z_score_values}).hist(ax=ax_dict['short_best_z_scores'], bins=21, alpha=0.8),
            pd.DataFrame({'long_short_best_z_score_diff': self.long_short_best_dir_z_score_diff_values}).hist(ax=ax_dict['long_short_best_z_score_diff'], bins=21, alpha=0.8),
        )
        # return pd.DataFrame({'long_z_scores': self.long_best_dir_z_score_values, 'short_z_scores': self.short_best_dir_z_score_values}).hist()
        return MatplotlibRenderPlots(name='plot_histogram_figure', figures=[fig], axes=ax_dict)




@define(slots=False, repr=False, eq=False)
class RankOrderComputationsContainer(ComputedResult):
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

    ripple_combined_epoch_stats_df: Optional[pd.DataFrame] = serialized_field(default=None, repr=False)
    ripple_new_output_tuple: Optional[Tuple] = non_serialized_field(default=None, repr=False)
    # ripple_n_valid_shuffles: Optional[int] = serialized_attribute_field(default=None, repr=False)

    laps_combined_epoch_stats_df: Optional[pd.DataFrame] = serialized_field(default=None, repr=False)
    laps_new_output_tuple: Optional[Tuple] = non_serialized_field(default=None, repr=False)

    minimum_inclusion_fr_Hz: float = serialized_attribute_field(default=2.0, repr=True)
    included_qclu_values: Optional[List] = serialized_attribute_field(default=None, repr=True)

    def __iter__(self):
        """ allows unpacking. See https://stackoverflow.com/questions/37837520/implement-packing-unpacking-in-an-object """
        # return iter(astuple(self)) # deep unpacking causes problems
        return iter(astuple(self, filter=attrs.filters.exclude(self.__attrs_attrs__.is_global, self.__attrs_attrs__.ripple_most_likely_result_tuple, self.__attrs_attrs__.laps_most_likely_result_tuple, self.__attrs_attrs__.minimum_inclusion_fr_Hz))) #  'is_global'


    # @function_attributes(short_name=None, tags=['aligned', 'epochs', 'filtering'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-13 12:16', related_items=[])
    # def get_aligned_events(self, epochs_df: pd.DataFrame, is_laps: bool = False):
    #     """ gets the values for both directions alligned to the same epochs for the given epoch_name and direction.

    #     Pure, does not modify self.

    #     Usage:
    #     ## Replays:
    #         long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    #         global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay))
    #         if isinstance(global_replays, pd.DataFrame):
    #             global_replays = Epoch(global_replays.epochs.get_valid_df())
    #         # get the aligned epochs and the z-scores aligned to them:
    #         active_replay_epochs, ripple_rank_order_z_score_df, (active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score) = rank_order_results.get_aligned_events(global_replays.to_dataframe().copy(), is_laps=False)

    #     ## Laps:
    #         long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    #         global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
    #         active_laps_epochs, laps_rank_order_z_score_df, (active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score) = rank_order_results.get_aligned_events(global_laps.copy(), is_laps=True)

    #     """
    #     def _subfn_compute(LR_values, RL_values, all_epochs_df):
    #         """ gets the values for either the laps or the ripples

    #         is_included_epoch = np.logical_or(is_included_epoch_LR, is_included_epoch_RL)

    #         """
    #         # all_epochs_df: (611,)
    #         # bak_active_epochs_df = deepcopy(active_epochs_df)
    #         is_included_epoch_LR = np.isin(all_epochs_df.label.to_numpy(), LR_values.epochs_df.label.to_numpy()) # (611,)
    #         is_included_epoch_RL = np.isin(all_epochs_df.label.to_numpy(), RL_values.epochs_df.label.to_numpy()) # (611,)
    #         is_included_epoch = np.logical_or(is_included_epoch_LR, is_included_epoch_RL) # (611,)
    #         active_epochs_df = all_epochs_df[is_included_epoch] # cuts down to only those included by one or the other. Should this be AND? # (586, )

    #         # Convert to same epochs:
    #         LR_values_long_z_score = pd.Series(LR_values.long_z_score, index=LR_values.epochs_df.label.to_numpy()) # (608,)
    #         RL_values_long_z_score = pd.Series(RL_values.long_z_score, index=RL_values.epochs_df.label.to_numpy()) # (608,)

    #         LR_values_short_z_score = pd.Series(LR_values.short_z_score, index=LR_values.epochs_df.label.to_numpy()) # (608,)
    #         RL_values_short_z_score = pd.Series(RL_values.short_z_score, index=RL_values.epochs_df.label.to_numpy()) # (608,)

    #         active_LR_ripple_long_z_score = LR_values_long_z_score[active_epochs_df.label.to_numpy()] # 586
    #         active_RL_ripple_long_z_score = RL_values_long_z_score[active_epochs_df.label.to_numpy()] # 586
    #         active_LR_ripple_short_z_score = LR_values_short_z_score[active_epochs_df.label.to_numpy()] # 586
    #         active_RL_ripple_short_z_score = RL_values_short_z_score[active_epochs_df.label.to_numpy()] # 586

    #         # Creating a new DataFrame with the reset series
    #         rank_order_z_score_df: pd.DataFrame = deepcopy(all_epochs_df)
    #         # Adding four new float columns filled with np.nans in a single line
    #         rank_order_z_score_df[['LR_Long_Z', 'RL_Long_Z', 'LR_Short_Z', 'RL_Short_Z']] = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], index=rank_order_z_score_df.index)

    #         # Assert statements to ensure shapes match
    #         assert (np.shape(LR_values.epochs_df.label.to_numpy()) == np.shape(LR_values.long_z_score))
    #         assert (np.shape(RL_values.epochs_df.label.to_numpy()) == np.shape(RL_values.long_z_score))
    #         assert (np.shape(LR_values.epochs_df.label.to_numpy()) == np.shape(LR_values.short_z_score))
    #         assert (np.shape(RL_values.epochs_df.label.to_numpy()) == np.shape(RL_values.short_z_score))

    #         # Creating mapping dictionaries for each set of values
    #         label_to_LR_long_z_score_mapping = dict(zip(LR_values.epochs_df.label.to_numpy(), LR_values.long_z_score)) # Creating a mapping dictionary
    #         label_to_RL_long_z_score_mapping = dict(zip(RL_values.epochs_df.label.to_numpy(), RL_values.long_z_score))
    #         label_to_LR_short_z_score_mapping = dict(zip(LR_values.epochs_df.label.to_numpy(), LR_values.short_z_score))
    #         label_to_RL_short_z_score_mapping = dict(zip(RL_values.epochs_df.label.to_numpy(), RL_values.short_z_score))

    #         # Updating the DataFrame using the mapping
    #         rank_order_z_score_df['LR_Long_Z'] = rank_order_z_score_df['label'].map(label_to_LR_long_z_score_mapping) # Updating the DataFrame
    #         rank_order_z_score_df['RL_Long_Z'] = rank_order_z_score_df['label'].map(label_to_RL_long_z_score_mapping)
    #         rank_order_z_score_df['LR_Short_Z'] = rank_order_z_score_df['label'].map(label_to_LR_short_z_score_mapping)
    #         rank_order_z_score_df['RL_Short_Z'] = rank_order_z_score_df['label'].map(label_to_RL_short_z_score_mapping)

    #         return active_epochs_df, rank_order_z_score_df, (active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score)

    #     # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    #     if is_laps:
    #         assert ((self.LR_laps is not None) and (self.RL_laps is not None)), f"self.LR_laps or self.RL_laps is None - Rank-order was not computed for Laps?"
    #         active_epochs_df, rank_order_z_score_df, _out_tuple = _subfn_compute(self.LR_laps, self.RL_laps, epochs_df)
    #     else:
    #         assert ((self.LR_ripple is not None) and (self.RL_ripple is not None)), f"self.LR_ripple or self.RL_ripple is None - Rank-order was not computed for Laps?"
    #         active_epochs_df, rank_order_z_score_df, _out_tuple = _subfn_compute(self.LR_ripple, self.RL_ripple, epochs_df)
    #     return active_epochs_df, rank_order_z_score_df, _out_tuple

    def adding_active_aclus_info(self):
        """
        Updates the `epochs_df` of each of its members.
        
        # ['LR_Long_ActuallyIncludedAclus', 'RL_Long_ActuallyIncludedAclus', 'LR_Short_ActuallyIncludedAclus', 'RL_Short_ActuallyIncludedAclus']
        """
        self.LR_laps.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.LR_laps.epochs_df, is_laps=True)
        self.RL_laps.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.RL_laps.epochs_df, is_laps=True)

        self.LR_ripple.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.LR_ripple.epochs_df, is_laps=False)
        self.RL_ripple.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.RL_ripple.epochs_df, is_laps=False)


    @classmethod
    def add_active_aclus_info(cls, rank_order_results, active_epochs_df: pd.DataFrame, is_laps: bool = False):
        """ adds the columns about the number of cells in each epoch to the epochs_df """
        label_column_type = 'int'
    
        if is_laps:
            # Laps:
            ## LARGE columns (lists of actually active number of cells, etc):
            active_epochs_df['LR_Long_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.LR_laps.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['LR_Long_rel_num_cells'] = 0
            active_epochs_df['LR_Long_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.LR_laps.extra_info_dict[x][1]))

            active_epochs_df['RL_Long_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.RL_laps.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['RL_Long_rel_num_cells'] = 0
            active_epochs_df['RL_Long_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.RL_laps.extra_info_dict[x][1]))
            ## Short
            active_epochs_df['LR_Short_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.LR_laps.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['LR_Short_rel_num_cells'] = 0
            active_epochs_df['LR_Short_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.LR_laps.extra_info_dict[x][1]))

            active_epochs_df['RL_Short_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.RL_laps.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['RL_Short_rel_num_cells'] = 0
            active_epochs_df['RL_Short_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.RL_laps.extra_info_dict[x][1]))

        else:
            # Ripples:
            ## LARGE columns (lists of actually active number of cells, etc):
            active_epochs_df['LR_Long_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.LR_ripple.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['LR_Long_rel_num_cells'] = 0
            active_epochs_df['LR_Long_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.LR_ripple.extra_info_dict[x][1]))

            active_epochs_df['RL_Long_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.RL_ripple.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['RL_Long_rel_num_cells'] = 0
            active_epochs_df['RL_Long_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.RL_ripple.extra_info_dict[x][1]))
            ## Short
            active_epochs_df['LR_Short_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.LR_ripple.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['LR_Short_rel_num_cells'] = 0
            active_epochs_df['LR_Short_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.LR_ripple.extra_info_dict[x][1]))

            active_epochs_df['RL_Short_ActuallyIncludedAclus'] = active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.RL_ripple.extra_info_dict[x][1]) # corresponds to `template_epoch_actually_included_aclus`
            active_epochs_df['RL_Short_rel_num_cells'] = 0
            active_epochs_df['RL_Short_rel_num_cells'] = active_epochs_df.label.astype(label_column_type).map(lambda x: len(rank_order_results.RL_ripple.extra_info_dict[x][1]))

        return active_epochs_df


    def to_dict(self) -> Dict:
        return asdict(self, filter=attrs.filters.exclude((self.__attrs_attrs__.is_global))) #  'is_global'


    def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        enable_hdf_testing_mode: bool - default False - if True, errors are not thrown for the first field that cannot be serialized, and instead all are attempted to see which ones work.


        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, **kwargs)
        # handle custom properties here



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
    # Plotting/Figure Helper Functions ___________________________________________________________________________________ #
    def _perform_plot_z_score_raw(epoch_idx_list, LR_long_z_score_values, RL_long_z_score_values, LR_short_z_score_values, RL_short_z_score_values, variable_name='Lap', x_axis_name_suffix='Index', point_data_values=None):
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

        ## Add table:
        # layoutWidget = pg.LayoutWidget()
        # win.addItem(layoutWidget)
        # layoutWidget.addWidget(

        # epoch_idx_list = np.arange(len(even_laps_long_short_z_score_diff_values))
        # epoch_idx_list = deepcopy(global_laps).lap_id # np.arange(len(even_laps_long_short_z_score_diff_values))
        n_x_points = len(epoch_idx_list)
        n_y_points = np.shape(RL_long_z_score_values)[0]
        if n_y_points > n_x_points:
            num_missing_points: int = n_y_points - n_x_points
            print(f'WARNING: trimming y-data to [{num_missing_points}:]')
            RL_long_z_score_values = RL_long_z_score_values[num_missing_points:]
            LR_long_z_score_values = LR_long_z_score_values[num_missing_points:]
            RL_short_z_score_values = RL_short_z_score_values[num_missing_points:]
            LR_short_z_score_values = LR_short_z_score_values[num_missing_points:]


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
        long_LR_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~LR_long_z_score_values.mask], LR_long_z_score_values[~LR_long_z_score_values.mask], brush=pg.mkBrush('red'), pen=pg.mkPen('#FFFFFF11'), symbol='t3', name='long_LR', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~LR_long_z_score_values.mask]) ## setting pen=None disables line drawing
        long_RL_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~RL_long_z_score_values.mask], RL_long_z_score_values[~RL_long_z_score_values.mask], brush=pg.mkBrush('orange'), pen=pg.mkPen('#FFFFFF11'), symbol='t2', name='long_RL', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~RL_long_z_score_values.mask]) ## setting pen=None disables line drawing
        short_LR_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~LR_short_z_score_values.mask], LR_short_z_score_values[~LR_short_z_score_values.mask], brush=pg.mkBrush('teal'), pen=pg.mkPen('#FFFFFF11'), symbol='t3', name='short_LR', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~LR_short_z_score_values.mask]) ## setting pen=None disables line drawing
        short_RL_out_plot_1D: pg.ScatterPlotItem = pg.ScatterPlotItem(epoch_idx_list[~RL_short_z_score_values.mask], RL_short_z_score_values[~RL_short_z_score_values.mask], brush=pg.mkBrush('blue'), pen=pg.mkPen('#FFFFFF11'), symbol='t2', name='short_RL', hoverable=True, hoverPen=pg.mkPen('w', width=2), hoverBrush=pg.mkBrush('#FFFFFF'), data=point_data_values.copy()[~RL_short_z_score_values.mask]) ## setting pen=None disables line drawing

        p1.addItem(long_LR_out_plot_1D)
        p1.addItem(long_RL_out_plot_1D)
        p1.addItem(short_LR_out_plot_1D)
        p1.addItem(short_RL_out_plot_1D)

        def find_closest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        # Function to connect points vertically
        def connect_vertical_points(x, y_values):
            for y in y_values:
                if not np.ma.is_masked(y):  # Check if the value is not masked
                    line = pg.PlotDataItem([x, x], [y, 0], pen=pg.mkPen('#FFFFFF33', width=0.5), hoverable=True, hoverPen=pg.mkPen('#FFFFFF55', width=0.75))
                    p1.addItem(line)

        # Connecting points
        for x in epoch_idx_list:
            idx = find_closest(epoch_idx_list, x)
            y_values = [
                LR_long_z_score_values[idx] if not LR_long_z_score_values.mask[idx] else np.ma.masked,
                RL_long_z_score_values[idx] if not RL_long_z_score_values.mask[idx] else np.ma.masked,
                LR_short_z_score_values[idx] if not LR_short_z_score_values.mask[idx] else np.ma.masked,
                RL_short_z_score_values[idx] if not RL_short_z_score_values.mask[idx] else np.ma.masked,
            ]
            connect_vertical_points(x, y_values)


        return app, win, p1, (long_LR_out_plot_1D, long_RL_out_plot_1D, short_LR_out_plot_1D, short_RL_out_plot_1D)



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




    # Computation Helpers ________________________________________________________________________________________________ #

    @classmethod
    def common_analysis_helper(cls, curr_active_pipeline, num_shuffles:int=300, minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,9]):
        ## Shared:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

        directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']

        if included_qclu_values is not None:
            qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=included_qclu_values)
            modified_directional_laps_results: DirectionalLapsResult = directional_laps_results.filtered_by_included_aclus(qclu_included_aclus)
            active_directional_laps_results = modified_directional_laps_results
        else:
            active_directional_laps_results = directional_laps_results


        # non-shared templates:
        non_shared_templates: TrackTemplates = active_directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) #.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = non_shared_templates.get_decoders()
        any_list_neuron_IDs = non_shared_templates.any_decoder_neuron_IDs # neuron_IDs as they appear in any list


        # global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)
        # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:
        global_spikes_df = global_spikes_df.spikes.sliced_by_neuron_id(any_list_neuron_IDs)

        # ## OLD method, directly get the decoders from `active_directional_laps_results` using `.get_decoders(...)` or `.get_shared_aclus_only_decoders(...)`:
        # long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = active_directional_laps_results.get_decoders()
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = active_directional_laps_results.get_shared_aclus_only_decoders()

        # NEW 2023-11-22 method: Get the templates (which can be filtered by frate first) and the from those get the decoders):
        # shared_aclus_only_templates = active_directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
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
    @function_attributes(short_name=None, tags=['subfn', 'preprocess', 'spikes_df'], input_requires=[], output_provides=[], uses=['add_epochs_id_identity'], used_by=['compute_shuffled_rankorder_analyses'], creation_date='2023-11-22 11:04', related_items=[])
    def preprocess_spikes_df(cls, active_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, shuffle_helper: ShuffleHelper, no_interval_fill_value=-1, min_num_unique_aclu_inclusions: int=5):
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
        active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=active_epochs_df, epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name='label', override_time_variable_name='t_rel_seconds', no_interval_fill_value=no_interval_fill_value)
        active_spikes_df.drop(active_spikes_df.loc[active_spikes_df['Probe_Epoch_id'] == no_interval_fill_value].index, inplace=True)
        # Sort by columns: 't_rel_seconds' (ascending), 'aclu' (ascending)
        active_spikes_df = active_spikes_df.sort_values(['t_rel_seconds', 'aclu'])

        # now filter epochs based on the number of included aclus:
        filtered_active_epochs_df = Epoch.filter_epochs(active_epochs_df, pos_df=None, spikes_df=active_spikes_df, min_epoch_included_duration=None, max_epoch_included_duration=None, maximum_speed_thresh=None,
                                                     min_inclusion_fr_active_thresh=None, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions).to_dataframe() # convert back to dataframe when done.
        filtered_active_epochs_df['label'] = filtered_active_epochs_df['label'].astype('int')

        # Now that we have `filtered_active_epochs`, we need to update the 'Probe_Epoch_id' because the epoch id's might have changed
        active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
        # Add epoch IDs to the spikes DataFrame
        active_spikes_df = add_epochs_id_identity(active_spikes_df, epochs_df=filtered_active_epochs_df, epoch_id_key_name='Probe_Epoch_id', epoch_label_column_name='label', override_time_variable_name='t_rel_seconds', no_interval_fill_value=no_interval_fill_value)
        active_spikes_df.drop(active_spikes_df.loc[active_spikes_df['Probe_Epoch_id'] == no_interval_fill_value].index, inplace=True)
        # Sort by columns: 't_rel_seconds' (ascending), 'aclu' (ascending)
        active_spikes_df = active_spikes_df.sort_values(['t_rel_seconds', 'aclu'])
        active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict = active_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()

        return shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, long_pf_peak_ranks, short_pf_peak_ranks, active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, filtered_active_epochs_df


    @classmethod
    def select_chosen_spikes(cls, active_spikes_df: pd.DataFrame, rank_alignment: str, time_variable_name_override: Optional[str]=None):
        """Selects and ranks spikes based on rank_alignment, and organizes them into structured dictionaries.

        Usage:

                # Determine which spikes to use to represent the order
                selected_spikes, selected_spikes_only_df = cls.select_chosen_spikes(active_spikes_df=active_spikes_df, rank_alignment=rank_alignment, time_variable_name_override=time_variable_name_override)

        Returns:
        selected_spikes: pandas groupby object?:
        selected_spikes_only_df: pd.DataFrame: an output dataframe containing only the select (e.g. .first(), .median(), etc) spike from each template.

        """
        assert (rank_alignment in ['first', 'median', 'center_of_mass']), f"rank_alignment must be either ['first', 'median', 'center_of_mass'], but it is: {rank_alignment}"
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
        return selected_spikes, selected_spikes_only_df


    @classmethod
    def select_and_rank_spikes(cls, active_spikes_df: pd.DataFrame, active_aclu_to_fragile_linear_neuron_IDX_dict, rank_alignment: str, time_variable_name_override: Optional[str]=None):
        """Selects and ranks spikes based on rank_alignment, and organizes them into structured dictionaries.


        Returns:
        epoch_ranked_aclus_dict: Dict[int, Dict[int, float]]: a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
        epoch_ranked_fragile_linear_neuron_IDX_dict: Dict[int, NDArray]:
        epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Dict[int, NDArray]:
        selected_spikes_only_df: pd.DataFrame: an output dataframe containing only the select (e.g. .first(), .median(), etc) spike from each template.

        """
        if time_variable_name_override is None:
            time_variable_name_override = active_spikes_df.spikes.time_variable_name
        # Determine which spikes to use to represent the order
        selected_spikes, selected_spikes_only_df = cls.select_chosen_spikes(active_spikes_df=active_spikes_df, rank_alignment=rank_alignment, time_variable_name_override=time_variable_name_override)

        # Rank the aclu values by their first t value in each Probe_Epoch_id
        ranked_aclus = selected_spikes.groupby('Probe_Epoch_id').rank(method='dense')  # Resolve ties in ranking

        # Create structured OUTPUT dictionaries
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
    def compute_shuffled_rankorder_analyses(cls, active_spikes_df: pd.DataFrame, active_epochs: pd.DataFrame, shuffle_helper: ShuffleHelper, rank_alignment: str = 'first', disable_re_ranking:bool=True, min_num_unique_aclu_inclusions: int=5, debug_print=True) -> RankOrderResult:
        """ Main rank-order analyses function called for a single shuffle_helper object.

        Extracts the two templates (long/short) from the shuffle_helper in addition to the shuffled_aclus, shuffle_IDXs.


        min_num_unique_aclu_inclusions: int

        """
        # post_process_statistic_value_fn = lambda x: np.abs(x)
        post_process_statistic_value_fn = lambda x: float(x) # basically NO-OP

        if not isinstance(active_epochs, pd.DataFrame):
            active_epochs = active_epochs.to_dataframe()

        assert isinstance(active_epochs, pd.DataFrame), f"active_epochs should be a dataframe but it is: {type(active_epochs)}"
        # CAST the labels into the correct format
        active_epochs['label'] = active_epochs['label'].astype('int')

        # Preprocess the spikes DataFrame
        (shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, long_pf_peak_ranks, short_pf_peak_ranks, active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, filtered_active_epochs) = cls.preprocess_spikes_df(active_spikes_df, active_epochs, shuffle_helper, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions)

        # TODO 2023-11-21 05:42: - [ ] todo_description want the aclus as well, not just the `long_pf_peak_ranks`

        #TODO 2023-12-08 12:53: - [ ] Drop epochs with fewer than the minimum active aclus

        #TODO 2023-12-10 19:40: - [ ] Need to save the epochs that were used to compute.

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



    # ==================================================================================================================== #
    # Directional Determination                                                                                            #
    # ==================================================================================================================== #

    @classmethod
    @function_attributes(short_name=None, tags=['subfn', 'rank-order', 'active_set', 'directional'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-14 13:40', related_items=[])
    def epoch_directionality_active_set_evidence(cls, decoders_dict, epochs_df: pd.DataFrame):
        """ 2023-12-14 - Replay Direction Active Set FR Classification - A method I came up with Kamran as a super quick way of using the active set (of cells) to determine the likelihood that a given epoch belongs to a certain direction.
        Used to classify replays as LR/RL
        
        Returns:
            epoch_rate_dfs
            epoch_accumulated_evidence

        Usage:    
        
        from PendingNotebookCode import epoch_directionality_active_set_evidence

        # recieves lists of identities (such as cell aclus) and a function that returns a sortable value for each identity:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=None) # non-shared-only
        decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_
        # LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay))
        active_replay_epochs, active_epochs_df, active_selected_spikes_df = combine_rank_order_results(rank_order_results, global_replays, track_templates=track_templates)

        # 
        # ['start', 'stop', 'label', 'duration', 'LR_Long_spearman', 'RL_Long_spearman', 'LR_Short_spearman', 'RL_Short_spearman', 'LR_Long_pearson', 'RL_Long_pearson', 'LR_Short_pearson', 'RL_Short_pearson', 'LR_Long_Old_Spearman', 'RL_Long_Old_Spearman', 'LR_Short_Old_Spearman', 'RL_Short_Old_Spearman', 'LR_Long_ActuallyIncludedAclus', 'LR_Long_rel_num_cells', 'RL_Long_ActuallyIncludedAclus', 'RL_Long_rel_num_cells', 'LR_Short_ActuallyIncludedAclus', 'LR_Short_rel_num_cells', 'RL_Short_ActuallyIncludedAclus', 'RL_Short_rel_num_cells', 'LR_Long_Z', 'RL_Long_Z', 'LR_Short_Z', 'RL_Short_Z']
        active_epochs_df.columns
        # accumulated_evidence_df = pd.DataFrame({'LR_evidence': accumulated_evidence['Normed_LR_rate'], 'RL_evidence': accumulated_evidence['Normed_LR_rate']}) epoch_accumulated_evidence.items()
        epoch_accumulated_evidence, epoch_rate_dfs, epochs_df_L = epoch_directionality_active_set_evidence(decoders_dict, active_epochs_df)
        epochs_df_L


        """
        def _subfn_compute_evidence_for_epoch(epoch_rate_df):
            """ 
            
            """
            epoch_rate_df['norm_term'] = epoch_rate_df.sum(axis=1)


            # epoch_rate_df.update({'Normed_LR_rate': epoch_rate_df['LR_rate']/epoch_rate_df['norm_term'], 'Normed_RL_rate': epoch_rate_df['RL_rate']/epoch_rate_df['norm_term']})

            # Update DataFrame with new columns
            epoch_rate_df = epoch_rate_df.assign(
                Normed_LR_rate=epoch_rate_df['LR_rate'] / epoch_rate_df['norm_term'],
                Normed_RL_rate=epoch_rate_df['RL_rate'] / epoch_rate_df['norm_term']
            )


            # Drop missing rows, not sure why these emerge (where there are cells with 0.0 fr for both directions.
            epoch_rate_df = epoch_rate_df[~((epoch_rate_df['Normed_LR_rate'].isna()) | (epoch_rate_df['Normed_RL_rate'].isna()))]

            ## Accumulate over all cells in the event:
            accumulated_evidence_by_sum: pd.Series = epoch_rate_df.sum(axis=0)
                    
            # Transpose the DataFrame to have columns as rows
            # accumulated_evidence_df = accumulated_evidence_df.transpose()

            # Rename the columns if necessary
            # accumulated_evidence_df.columns = ['Accumulated_LR_rate', 'Accumulated_RL_rate'] # , 'Product_Accumulated_LR_rate', 'Product_Accumulated_RL_rate'

            accumulated_evidence_by_product: pd.Series = epoch_rate_df.product(axis=0)
            accumulated_evidence = {
                'Sum_Accumulated_LR_rate': accumulated_evidence_by_sum['Normed_LR_rate'],
                'Sum_Accumulated_RL_rate': accumulated_evidence_by_sum['Normed_RL_rate'],
                'Product_Accumulated_LR_rate': accumulated_evidence_by_product['Normed_LR_rate'],
                'Product_Accumulated_RL_rate': accumulated_evidence_by_product['Normed_RL_rate']
            }

            return accumulated_evidence, epoch_rate_df


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        decoders_aclu_peak_fr_dict = {a_decoder_name:dict(zip(np.array(a_decoder.pf.ratemap.neuron_ids), a_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates)) for a_decoder_name, a_decoder in decoders_dict.items()}

        long_LR_aclu_peak_fr_map = decoders_aclu_peak_fr_dict['long_LR']
        long_RL_aclu_peak_fr_map = decoders_aclu_peak_fr_dict['long_RL']

        epoch_rate_dfs = {}
        epoch_accumulated_evidence = {}

        for row in epochs_df.itertuples(name="EpochRow"):
            try:
                ## This never seems to do anything anymore.
                active_unique_aclus = row.active_unique_aclus

            except (KeyError, AttributeError):    
                ## Make map exhaustive
                either_direction_aclus = np.sort(np.union1d(row.LR_Long_ActuallyIncludedAclus, row.RL_Long_ActuallyIncludedAclus)) #TODO 2023-12-18 16:56: - [ ] Note only uses 'LONG' to make decisions. I think SHORT are constrained to be equal, but this test should be explicitly performed.
                active_unique_aclus = either_direction_aclus
                # LR_Long_ActuallyIncludedAclus
                # RL_Long_ActuallyIncludedAclus
                pass

            epoch_LR_rates = []
            epoch_RL_rates = []

            for an_aclu in active_unique_aclus:
                LR_rate = long_LR_aclu_peak_fr_map.get(an_aclu, 0.0)
                RL_rate = long_RL_aclu_peak_fr_map.get(an_aclu, 0.0)
                epoch_LR_rates.append(LR_rate)
                epoch_RL_rates.append(RL_rate)
                
                # _norm_term = (LR_rate + RL_rate)
                
                
            epoch_LR_rates = np.array(epoch_LR_rates)
            epoch_RL_rates = np.array(epoch_RL_rates)
            
            epoch_rate_df = pd.DataFrame({'LR_rate': epoch_LR_rates, 'RL_rate': epoch_RL_rates})
            accumulated_evidence, epoch_rate_df = _subfn_compute_evidence_for_epoch(epoch_rate_df)

            # Update the LR_evidence and RL_evidence columns in epochs_df_L
            epochs_df.at[row.Index, 'LR_evidence'] = accumulated_evidence['Sum_Accumulated_LR_rate']
            epochs_df.at[row.Index, 'RL_evidence'] = accumulated_evidence['Sum_Accumulated_RL_rate']
            epochs_df.at[row.Index, 'LR_product_evidence'] = accumulated_evidence['Product_Accumulated_LR_rate']
            epochs_df.at[row.Index, 'RL_product_evidence'] = accumulated_evidence['Product_Accumulated_RL_rate']

            ## add to the output dicts:
            epoch_rate_dfs[int(row.label)] = epoch_rate_df
            epoch_accumulated_evidence[int(row.label)] = accumulated_evidence

        
        epochs_df['normed_LR_evidence'] = epochs_df['LR_evidence']/epochs_df[['LR_evidence', 'RL_evidence']].sum(axis=1)
        epochs_df['normed_RL_evidence'] = epochs_df['RL_evidence']/epochs_df[['LR_evidence', 'RL_evidence']].sum(axis=1)
        
        epochs_df['normed_product_LR_evidence'] = epochs_df['LR_product_evidence']/epochs_df[['LR_product_evidence', 'RL_product_evidence']].sum(axis=1)
        epochs_df['normed_product_RL_evidence'] = epochs_df['RL_product_evidence']/epochs_df[['LR_product_evidence', 'RL_product_evidence']].sum(axis=1)

        return epoch_accumulated_evidence, epoch_rate_dfs, epochs_df



    @classmethod
    @function_attributes(short_name=None, tags=['subfn', 'rank-order'], input_requires=[], output_provides=[], uses=[], used_by=['cls.most_likely_directional_rank_order_shuffling'], creation_date='2023-12-12 10:49', related_items=[])
    def _compute_best_direction_likelihoods(cls, active_epochs, long_LR_results, long_RL_results, short_LR_results, short_RL_results, spikes_df, decoding_time_bin_size) -> DirectionalRankOrderLikelihoods:
        """ Computes the likelihoods for each direction (LR/RL) for both long and short.
        History: Factored out of `most_likely_directional_rank_order_shuffling`
        """
        (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)

        # # Use the four epochs to make to a pseudo-y:
        # all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        # all_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D), debug_print=False)
        # all_directional_pf1D_Decoder = BasePositionDecoder(all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        ## Combine the non-directional PDFs and renormalize to get the directional PDF:
        # Inputs: long_LR_pf1D, long_RL_pf1D
        long_directional_decoder_names = ['long_LR', 'long_RL']
        long_directional_pf1D: PfND = PfND.build_merged_directional_placefields(deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), debug_print=False) # long_LR: y=1.0, long_RL: y=2.0
        long_directional_pf1D_Decoder = BasePositionDecoder(long_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        # Inputs: short_LR_pf1D, short_RL_pf1D
        short_directional_decoder_names = ['short_LR', 'short_RL']
        short_directional_pf1D = PfND.build_merged_directional_placefields(deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D), debug_print=False) # [LR, RL], {'LR': 0, 'RL': 1}
        short_directional_pf1D_Decoder = BasePositionDecoder(short_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        # takes 6.3 seconds

        ## validation of PfNDs:
        # short_directional_pf1D.plot_ratemaps_2D()

        # Decode using specific directional_decoders:
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

        # Use the relative likelihoods to determine which direction to use for each point:
        long_best_direction_indicies = np.argmax(long_relative_direction_likelihoods, axis=1)
        short_best_direction_indicies = np.argmax(short_relative_direction_likelihoods, axis=1)

        # Creating an instance of the namedtuple
        return DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=long_relative_direction_likelihoods,
                                            short_relative_direction_likelihoods=short_relative_direction_likelihoods,
                                            long_best_direction_indices=long_best_direction_indicies,
                                            short_best_direction_indices=short_best_direction_indicies)


    @classmethod
    @function_attributes(short_name=None, tags=['rank-order', 'shuffle', 'inst_fr', 'epoch', 'lap', 'replay', 'computation'], input_requires=[], output_provides=[], uses=['DirectionalRankOrderLikelihoods', 'DirectionalRankOrderResult', 'cls._compute_best'], used_by=[], creation_date='2023-11-16 18:43', related_items=['plot_rank_order_epoch_inst_fr_result_tuples'])
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
        _LR_INDEX = 0
        _RL_INDEX = 1

        # Unpack all directional variables:
        ## {"even": "RL", "odd": "LR"}
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any'] # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_epoch_name = global_any_name

        ## Extract the rank_order_results:
        rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        rank_order_results.adding_active_aclus_info()
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=rank_order_results.minimum_inclusion_fr_Hz) # non-shared-only
        decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_
        # LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # def _perform_compute_directional_likelihoods_tuple_methods(active_epochs, active_LR_long_z_score=None, active_RL_long_z_score=None, active_LR_short_z_score=None, active_RL_short_z_score=None) -> DirectionalRankOrderLikelihoods:
        #     """ Used to switch between three different temporary methods of determining the proper direction.
        #     captures curr_active_pipeline, global_epoch_name """
        #     # ALT_METHOD : main `_compute_best_direction_likelihoods`
        #     # #TODO 2023-12-12 11:56: - [ ] Should these be saved to the ripple result?
        #     # spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)
        #     # # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        #     # (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        #     # return cls._compute_best_direction_likelihoods(active_epochs, long_LR_results, long_RL_results, short_LR_results, short_RL_results, spikes_df, decoding_time_bin_size)

        #     ## ALT 2023-12-12 - APPROXIMATE best-direction without using the decoder as a workaround since `cls._compute_best_direction_likelihoods` isn't working yet:
        #     print(f'WARNING: approximating best-direction without using the decoder as a workaround.')

        #     # ALT_METHOD: mean_determined_epoch_dir: `mean_determined_epoch_dir`
        #     # mean_LR_z_scores = ((np.abs(rank_order_results.LR_ripple.long_z_score) + np.abs(rank_order_results.LR_ripple.short_z_score)) / 2.0)
        #     # mean_RL_z_scores = ((np.abs(rank_order_results.RL_ripple.long_z_score) + np.abs(rank_order_results.RL_ripple.short_z_score)) / 2.0)
        #     # mean_determined_epoch_dir = (mean_LR_z_scores >= mean_RL_z_scores).astype(int)
        #     # long_best_direction_indicies = mean_determined_epoch_dir.copy()
        #     # short_best_direction_indicies = mean_determined_epoch_dir.copy()

        #     # # ALT_METHOD: best max method:
        #     # # filtered_active_epochs, (active_LR_long_z_score, active_RL_long_z_score, active_LR_short_z_score, active_RL_short_z_score) = rank_order_results.get_aligned_events(active_epochs)
        #     # long_best_direction_indicies = np.argmax(np.vstack([np.abs(active_LR_long_z_score), np.abs(active_RL_long_z_score)]), axis=0).astype(int)
        #     # short_best_direction_indicies = np.argmax(np.vstack([np.abs(active_LR_short_z_score), np.abs(active_RL_short_z_score)]), axis=0).astype(int)

        #     # ## build the new directional_likelihods_tuple:
        #     # #TODO 2023-12-12 13:50: - [ ] Add `filtered_active_epochs` to the DirectionalRankOrderLikelihoods?
        #     # return DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=None, short_relative_direction_likelihoods=None, long_best_direction_indices=long_best_direction_indicies, short_best_direction_indices=short_best_direction_indicies)


        #     # ALT_METHOD: best max method:
        #     # filtered_active_epochs, (active_LR_long_z_score, active_RL_long_z_score, active_LR_short_z_score, active_RL_short_z_score) = rank_order_results.get_aligned_events(active_epochs)
        #     long_best_direction_indicies = np.argmax(np.vstack([np.abs(active_LR_long_z_score), np.abs(active_RL_long_z_score)]), axis=0).astype(int)
        #     short_best_direction_indicies = np.argmax(np.vstack([np.abs(active_LR_short_z_score), np.abs(active_RL_short_z_score)]), axis=0).astype(int)


        #     LR_laps_epoch_accumulated_evidence, LR_laps_epoch_rate_dfs, LR_laps_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.LR_laps.epochs_df)
        #     RL_laps_epoch_accumulated_evidence, RL_laps_epoch_rate_dfs, RL_laps_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.RL_laps.epochs_df)

        #     rank_order_results.laps_most_likely_result_tuple.directional_likelihoods_tuple = DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=LR_laps_epochs_df['normed_LR_evidence'], short_relative_direction_likelihoods=LR_laps_epochs_df['normed_RL_evidence'],
        #                                     long_best_direction_indices=(LR_laps_epochs_df['normed_LR_evidence']>=LR_laps_epochs_df['normed_RL_evidence']), 
        #                                     short_best_direction_indices=(LR_laps_epochs_df['normed_LR_evidence']>=LR_laps_epochs_df['normed_RL_evidence']))


        #     ## build the new directional_likelihods_tuple:
        #     #TODO 2023-12-12 13:50: - [ ] Add `filtered_active_epochs` to the DirectionalRankOrderLikelihoods?
        #     return DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=None, short_relative_direction_likelihoods=None, long_best_direction_indices=long_best_direction_indicies, short_best_direction_indices=short_best_direction_indicies)



        ## Replays:
        try:
            ## Post-process Z-scores with their most likely directions:
            # rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']

            ripple_combined_epoch_stats_df = deepcopy(rank_order_results.ripple_combined_epoch_stats_df)
            active_replay_epochs = deepcopy(rank_order_results.LR_ripple.epochs_df)

            # _traditional_arg_dict_key_names = ('active_LR_long_z_score', 'active_RL_long_z_score', 'active_LR_short_z_score', 'active_RL_short_z_score')
            # _traditional_arg_dict_values_tuple = (ripple_combined_epoch_stats_df.LR_Long_spearman_Z, ripple_combined_epoch_stats_df.RL_Long_spearman_Z, ripple_combined_epoch_stats_df.LR_Short_spearman_Z, ripple_combined_epoch_stats_df.LR_Short_spearman_Z)
            active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score = ripple_combined_epoch_stats_df.LR_Long_spearman_Z, ripple_combined_epoch_stats_df.RL_Long_spearman_Z, ripple_combined_epoch_stats_df.LR_Short_spearman_Z, ripple_combined_epoch_stats_df.LR_Short_spearman_Z

            # _arg_dict_from_ripple_combined_epoch_stats_df = dict(zip(_traditional_arg_dict_key_names, _traditional_arg_dict_values_tuple))
            # ripple_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = _perform_compute_directional_likelihoods_tuple_methods(active_replay_epochs, **_arg_dict_from_ripple_combined_epoch_stats_df)
            
            LR_ripple_epoch_accumulated_evidence, LR_ripple_epoch_rate_dfs, LR_ripple_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, active_replay_epochs)
            # RL_ripple_epoch_accumulated_evidence, RL_ripple_epoch_rate_dfs, RL_ripple_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.RL_ripple.epochs_df)

            long_best_direction_indicies = np.argmax(np.vstack([np.abs(LR_ripple_epochs_df['normed_LR_evidence'].to_numpy()), np.abs(LR_ripple_epochs_df['normed_RL_evidence'].to_numpy())]), axis=0).astype(int)
            short_best_direction_indicies = np.argmax(np.vstack([np.abs(LR_ripple_epochs_df['normed_LR_evidence'].to_numpy()), np.abs(LR_ripple_epochs_df['normed_RL_evidence'].to_numpy())]), axis=0).astype(int)

            ripple_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=LR_ripple_epochs_df['normed_LR_evidence'].to_numpy(),
                                                                                                                   short_relative_direction_likelihoods=LR_ripple_epochs_df['normed_RL_evidence'].to_numpy(),
                                            long_best_direction_indices=long_best_direction_indicies, #(LR_ripple_epochs_df['normed_LR_evidence'].to_numpy()>=LR_ripple_epochs_df['normed_RL_evidence'].to_numpy()).astype(int), 
                                            short_best_direction_indices=short_best_direction_indicies, #(LR_ripple_epochs_df['normed_LR_evidence'].to_numpy()>=LR_ripple_epochs_df['normed_RL_evidence'].to_numpy()).astype(int)
                                            )

            long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = ripple_directional_likelihoods_tuple

            ripple_evts_long_best_dir_z_score_values = np.where(long_best_direction_indicies, active_LR_ripple_long_z_score, active_RL_ripple_long_z_score)
            ripple_evts_short_best_dir_z_score_values = np.where(short_best_direction_indicies, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score)
            ripple_evts_long_short_best_dir_z_score_diff_values = ripple_evts_long_best_dir_z_score_values - ripple_evts_short_best_dir_z_score_values
            ripple_masked_z_score_values_list: List[ma.masked_array] = [ma.masked_array(x, mask=np.logical_not(a_mask)) for x, a_mask in zip((active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score),
                                                                                                                ((long_best_direction_indicies == _LR_INDEX), (long_best_direction_indicies == _RL_INDEX), (short_best_direction_indicies == _LR_INDEX), (short_best_direction_indicies == _RL_INDEX)))]

            # outputs: ripple_evts_long_short_best_dir_z_score_diff_values
            ripple_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(active_replay_epochs, long_best_dir_z_score_values=ripple_evts_long_best_dir_z_score_values, short_best_dir_z_score_values=ripple_evts_short_best_dir_z_score_values,
                                                                                        long_short_best_dir_z_score_diff_values=ripple_evts_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=ripple_directional_likelihoods_tuple,
                                                                                        masked_z_score_values_list=ripple_masked_z_score_values_list, rank_order_z_score_df=None)


            active_laps_epochs = deepcopy(rank_order_results.LR_laps.epochs_df)

            LR_laps_epoch_accumulated_evidence, LR_laps_epoch_rate_dfs, LR_laps_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.LR_laps.epochs_df)
            # RL_laps_epoch_accumulated_evidence, RL_laps_epoch_rate_dfs, RL_laps_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.RL_laps.epochs_df)

            laps_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=LR_laps_epochs_df['normed_LR_evidence'].to_numpy(),
                                                                                                                   short_relative_direction_likelihoods=LR_laps_epochs_df['normed_RL_evidence'].to_numpy(),
                                            long_best_direction_indices=(LR_laps_epochs_df['normed_LR_evidence'].to_numpy()>=LR_laps_epochs_df['normed_RL_evidence'].to_numpy()).astype(int), 
                                            short_best_direction_indices=(LR_laps_epochs_df['normed_LR_evidence'].to_numpy()>=LR_laps_epochs_df['normed_RL_evidence'].to_numpy()).astype(int))





        except (AttributeError, KeyError, IndexError, ValueError):
            raise # fail for ripples, but not for laps currently
            ripple_result_tuple = None


        ## Laps:
        try:
            # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            # global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
            # active_laps_epochs = global_laps
            laps_combined_epoch_stats_df = deepcopy(rank_order_results.laps_combined_epoch_stats_df)

            # _traditional_arg_dict_key_names = ('active_LR_long_z_score', 'active_RL_long_z_score', 'active_LR_short_z_score', 'active_RL_short_z_score')
            # _traditional_arg_dict_values_tuple = (laps_combined_epoch_stats_df.LR_Long_spearman_Z, laps_combined_epoch_stats_df.RL_Long_spearman_Z, laps_combined_epoch_stats_df.LR_Short_spearman_Z, laps_combined_epoch_stats_df.LR_Short_spearman_Z)
            active_LR_laps_long_z_score, active_RL_laps_long_z_score, active_LR_laps_short_z_score, active_RL_laps_short_z_score = laps_combined_epoch_stats_df.LR_Long_spearman_Z, laps_combined_epoch_stats_df.RL_Long_spearman_Z, laps_combined_epoch_stats_df.LR_Short_spearman_Z, laps_combined_epoch_stats_df.LR_Short_spearman_Z

            # _arg_dict_from_laps_combined_epoch_stats_df = dict(zip(_traditional_arg_dict_key_names, _traditional_arg_dict_values_tuple))
            # laps_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = _perform_compute_directional_likelihoods_tuple_methods(active_laps_epochs, **_arg_dict_from_laps_combined_epoch_stats_df)
            active_laps_epochs = deepcopy(rank_order_results.LR_laps.epochs_df)

            LR_laps_epoch_accumulated_evidence, LR_laps_epoch_rate_dfs, LR_laps_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.LR_laps.epochs_df)
            RL_laps_epoch_accumulated_evidence, RL_laps_epoch_rate_dfs, RL_laps_epochs_df = cls.epoch_directionality_active_set_evidence(decoders_dict, rank_order_results.RL_laps.epochs_df)

            laps_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=LR_laps_epochs_df['normed_LR_evidence'].to_numpy(),
                                                                                                                   short_relative_direction_likelihoods=LR_laps_epochs_df['normed_RL_evidence'].to_numpy(),
                                            long_best_direction_indices=(LR_laps_epochs_df['normed_LR_evidence'].to_numpy()>=LR_laps_epochs_df['normed_RL_evidence'].to_numpy()), 
                                            short_best_direction_indices=(LR_laps_epochs_df['normed_LR_evidence'].to_numpy()>=LR_laps_epochs_df['normed_RL_evidence'].to_numpy()))
            


            ## OLD METHOD:
            # active_laps_epochs, laps_rank_order_z_score_df, (active_LR_laps_long_z_score, active_RL_laps_long_z_score, active_LR_laps_short_z_score, active_RL_laps_short_z_score) = rank_order_results.get_aligned_events(global_laps, is_laps=True)
            # laps_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = _perform_compute_directional_likelihoods_tuple_methods(active_epochs=active_laps_epochs, active_LR_long_z_score=active_LR_laps_long_z_score, active_RL_long_z_score=active_RL_laps_long_z_score,
            #                                                                                                                             active_LR_short_z_score=active_LR_laps_short_z_score, active_RL_short_z_score=active_RL_laps_short_z_score)

            ## Common between both:
            long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = laps_directional_likelihoods_tuple.long_relative_direction_likelihoods, laps_directional_likelihoods_tuple.short_relative_direction_likelihoods, laps_directional_likelihoods_tuple.long_best_direction_indices, laps_directional_likelihoods_tuple.short_best_direction_indices

            # Using NumPy advanced indexing to select from array_a or array_b:
            laps_long_best_dir_z_score_values = np.where(long_best_direction_indicies, rank_order_results.LR_laps.long_z_score, rank_order_results.RL_laps.long_z_score)
            laps_short_best_dir_z_score_values = np.where(short_best_direction_indicies, rank_order_results.LR_laps.short_z_score, rank_order_results.RL_laps.short_z_score)
            # print(f'np.shape(laps_long_best_dir_z_score_values): {np.shape(laps_long_best_dir_z_score_values)}')
            laps_long_short_best_dir_z_score_diff_values = laps_long_best_dir_z_score_values - laps_short_best_dir_z_score_values
            # print(f'np.shape(laps_long_short_best_dir_z_score_diff_values): {np.shape(laps_long_short_best_dir_z_score_diff_values)}')
            #TODO 2023-11-20 22:02: - [ ] ERROR: CORRECTNESS FAULT: I think the two lists zipped over below are out of order.
            laps_masked_z_score_values_list: List[ma.masked_array] = [ma.masked_array(x, mask=np.logical_not(a_mask)) for x, a_mask in zip((active_LR_laps_long_z_score, active_RL_laps_long_z_score, active_LR_laps_short_z_score, active_RL_laps_short_z_score),
                                                                                                                ((long_best_direction_indicies == _LR_INDEX), (long_best_direction_indicies == _RL_INDEX), (short_best_direction_indicies == _LR_INDEX), (short_best_direction_indicies == _RL_INDEX)))]

            laps_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(active_laps_epochs, long_best_dir_z_score_values=laps_long_best_dir_z_score_values, short_best_dir_z_score_values=laps_short_best_dir_z_score_values,
                                                                                        long_short_best_dir_z_score_diff_values=laps_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=laps_directional_likelihoods_tuple,
                                                                                        masked_z_score_values_list=laps_masked_z_score_values_list, rank_order_z_score_df=None)

        except (AttributeError, KeyError, IndexError, ValueError):
            laps_result_tuple = None


        return ripple_result_tuple, laps_result_tuple


    @function_attributes(short_name=None, tags=['rank-order', 'ripples', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_ripples_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='first', minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,9]):

        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

        ## Ripple Rank-Order Analysis: needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

        # curr_active_pipeline.sess.config.preprocessing_parameters
        min_num_unique_aclu_inclusions: int = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions

        global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
        if isinstance(global_replays, pd.DataFrame):
            ## why do we convert to Epoch before passing in? this doesn't make sense to me. Just to use the .get_valid_df()?
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
    def main_laps_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='median', minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,9]):
        """

        _laps_outputs = RankOrderAnalyses.main_laps_analysis(curr_active_pipeline, num_shuffles=1000, rank_alignment='median')

        # Unwrap
        (LR_outputs, RL_outputs, laps_paired_tests) = _laps_outputs

        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values) = odd_outputs
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, (even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values) = even_outputs


        """
        ## Shared:
        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper) = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

        ## Laps Epochs: Needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()

        # curr_active_pipeline.sess.config.preprocessing_parameters
        min_num_unique_aclu_inclusions: int = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions


        if not isinstance(global_laps, pd.DataFrame):
            global_laps_df = deepcopy(global_laps).to_dataframe()
            global_laps_df['label'] = global_laps_df['label'].astype('int')

        # TODO: CenterOfMass for Laps instead of median spike
        # laps_rank_alignment = 'center_of_mass'
        LR_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(global_spikes_df), deepcopy(global_laps), odd_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, debug_print=False)
        RL_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(global_spikes_df), deepcopy(global_laps), even_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions, debug_print=False)
        laps_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((LR_outputs.long_z_score, LR_outputs.short_z_score), (RL_outputs.long_z_score, RL_outputs.short_z_score))]
        print(f'laps_paired_tests: {laps_paired_tests}')

        return (LR_outputs, RL_outputs, laps_paired_tests)


    @classmethod
    def validate_has_rank_order_results(cls, curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
        """ Returns True if the pipeline has a valid RankOrder results set of the latest version

        TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

        """
        # Unpacking:
        rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple

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
        included_qclu_values = rank_order_results.included_qclu_values

        ## TODO: require same `included_qclu_values` values
        rank_order_z_score_df = ripple_result_tuple.rank_order_z_score_df
        if rank_order_z_score_df is None:
            return False

        # 2023-12-15 - Newest method:
        ripple_combined_epoch_stats_df = rank_order_results.ripple_combined_epoch_stats_df
        if ripple_combined_epoch_stats_df is None:
            return False

        if np.isnan(rank_order_results.ripple_combined_epoch_stats_df.index).any():
            return False # can't have dataframe index that is missing values.


        laps_combined_epoch_stats_df = rank_order_results.laps_combined_epoch_stats_df
        if laps_combined_epoch_stats_df is None:
            return False

        # rank_order_results.included_qclu_values

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
        """
        #TODO 2023-12-10 19:01: - [ ] Only works for ripples, ignores laps


        """
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

    # 2023-12-13 - New pd.DataFrame simplified correlations: Spearman and Pearson ________________________________________ #

    @classmethod
    def pho_compute_rank_order(cls, track_templates, curr_epoch_spikes_df: pd.DataFrame, rank_method="average", stats_nan_policy='omit') -> Dict[str, Tuple]:
        """ 2023-12-20 - Actually working spearman rank-ordering!! 

        # rank_method: str = "dense"
        # rank_method: str = "average"
        
        
        Usage:
            curr_epoch_spikes_df = deepcopy(active_plotter.get_active_epoch_spikes_df())[['t_rel_seconds', 'aclu', 'shank', 'cluster', 'qclu', 'maze_id', 'flat_spike_idx', 'Probe_Epoch_id']]
            curr_epoch_spikes_df["spike_rank"] = curr_epoch_spikes_df["t_rel_seconds"].rank(method="average")
            # Sort by column: 'aclu' (ascending)
            curr_epoch_spikes_df = curr_epoch_spikes_df.sort_values(['aclu'])
            curr_epoch_spikes_df

        """
        curr_epoch_spikes_df["spike_rank"] = curr_epoch_spikes_df["t_rel_seconds"].rank(method=rank_method)
        # curr_epoch_spikes_df = curr_epoch_spikes_df.sort_values(['aclu'], inplace=False) # Sort by column: 'aclu' (ascending)

        n_spikes = np.shape(curr_epoch_spikes_df)[0]
        curr_epoch_spikes_aclus = deepcopy(curr_epoch_spikes_df.aclu.to_numpy())
        curr_epoch_spikes_aclu_ranks = deepcopy(curr_epoch_spikes_df.spike_rank.to_numpy())
        # curr_epoch_spikes_aclu_rank_map = dict(zip(curr_epoch_spikes_aclus, curr_epoch_spikes_aclu_ranks)) # could build a map equiv to template versions
        n_unique_aclus = np.shape(curr_epoch_spikes_df.aclu.unique())[0]
        assert n_spikes == n_unique_aclus, f"there is more than one spike in curr_epoch_spikes_df for an aclu! n_spikes: {n_spikes}, n_unique_aclus: {n_unique_aclus}"

        track_templates.rank_method = rank_method
        decoder_aclu_peak_rank_dict_dict = track_templates.decoder_aclu_peak_rank_dict_dict

        template_spearman_real_results = {}
        for a_decoder_name, a_decoder_aclu_peak_rank_dict in decoder_aclu_peak_rank_dict_dict.items():
            # template_corresponding_aclu_rank_list: the list of template ranks for each aclu present in the `curr_epoch_spikes_aclus`
            template_corresponding_aclu_rank_list = np.array([a_decoder_aclu_peak_rank_dict.get(key, np.nan) for key in curr_epoch_spikes_aclus]) #  if key in decoder_aclu_peak_rank_dict_dict['long_LR']
            # curr_epoch_spikes_aclu_rank_list = np.array([curr_epoch_spikes_aclu_rank_map.get(key, np.nan) for key in curr_epoch_spikes_aclus])
            curr_epoch_spikes_aclu_rank_list = curr_epoch_spikes_aclu_ranks
            n_missing_aclus = np.isnan(template_corresponding_aclu_rank_list).sum()
            real_long_rank_stats = scipy.stats.spearmanr(curr_epoch_spikes_aclu_rank_list, template_corresponding_aclu_rank_list, nan_policy=stats_nan_policy)
            # real_long_rank_stats = calculate_spearman_rank_correlation(curr_epoch_spikes_aclu_rank_list, template_corresponding_aclu_rank_list)
            template_spearman_real_results[a_decoder_name] = (*real_long_rank_stats, n_missing_aclus)
        
        return template_spearman_real_results
    


    @classmethod
    @function_attributes(short_name=None, tags=['subfn', 'correlation', 'spearman', 'rank-order', 'pearson'], input_requires=[], output_provides=[], uses=[], used_by=['pandas_df_based_correlation_computations'], creation_date='2023-12-13 03:47', related_items=[])
    def _subfn_calculate_correlations(cls, group, method='spearman', enable_shuffle: bool=False) -> pd.Series:
        """ computes the pearson correlations between the spiketimes during a specific epoch (identified by each 'Probe_Epoch_id' group) and that spike's pf_peak_x location in the template.

        correlations = active_selected_spikes_df.groupby('Probe_Epoch_id').apply(lambda group: calculate_correlations(group, method='spearman'))
        correlations = active_selected_spikes_df.groupby('Probe_Epoch_id').apply(lambda group: calculate_correlations(group, method='pearson'))

        """
        if enable_shuffle:
            rng = np.random.default_rng()
            shuffle_fn = lambda x: pd.Series(rng.permuted(x.values), index=x.index)  # x.values
        else:
            shuffle_fn = lambda x: x # no-op

        # correlations = {
        #     f'LR_Long_{method}': shuffle_fn(group['t_rel_seconds']).corr(group['LR_Long_pf_peak_x'], method=method),
        #     f'RL_Long_{method}': shuffle_fn(group['t_rel_seconds']).corr(group['RL_Long_pf_peak_x'], method=method),
        #     f'LR_Short_{method}': shuffle_fn(group['t_rel_seconds']).corr(group['LR_Short_pf_peak_x'], method=method),
        #     f'RL_Short_{method}': shuffle_fn(group['t_rel_seconds']).corr(group['RL_Short_pf_peak_x'], method=method)
        # }
        correlations = {
            f'LR_Long_{method}': shuffle_fn(group['t_rel_seconds']).rank(method="dense").corr(group['LR_Long_pf_peak_x'], method=method),
            f'RL_Long_{method}': shuffle_fn(group['t_rel_seconds']).rank(method="dense").corr(group['RL_Long_pf_peak_x'], method=method),
            f'LR_Short_{method}': shuffle_fn(group['t_rel_seconds']).rank(method="dense").corr(group['LR_Short_pf_peak_x'], method=method),
            f'RL_Short_{method}': shuffle_fn(group['t_rel_seconds']).rank(method="dense").corr(group['RL_Short_pf_peak_x'], method=method)
        }

        return pd.Series(correlations)

    @classmethod
    def build_stacked_arrays(cls, output_active_epoch_computed_values):
        # Convert each DataFrame to a NumPy array and stack them # shape of this array will be (n, m, p) where n is the number of DataFrames, m is the number of rows, and p is the number of columns in each DataFrame.
        names0 = ['LR_Long_spearman', 'RL_Long_spearman', 'LR_Short_spearman', 'RL_Short_spearman']
        dfs0 = [a_tuple[0] for a_tuple in output_active_epoch_computed_values] # [0] is just spearman_correlations
        stacked_arrays0 = np.array([df[names0].values for df in dfs0])
        names1 = ['LR_Long_pearson', 'RL_Long_pearson', 'LR_Short_pearson', 'RL_Short_pearson']
        dfs1 = [a_tuple[1] for a_tuple in output_active_epoch_computed_values] # [1] is pearson_correlations
        stacked_arrays1 = np.array([df[names1].values for df in dfs1])
        stacked_arrays = np.concatenate((stacked_arrays0, stacked_arrays1), axis=-1) # .shape: (100, 412, 8) # (n_shuffles, n_epochs, n_columns)
        return stacked_arrays
    
    
    @classmethod
    @function_attributes(short_name=None, tags=['active', 'shuffle', 'rank_order', 'main'], input_requires=[], output_provides=[], uses=['_subfn_calculate_correlations', 'build_stacked_arrays'], used_by=[], creation_date='2023-12-15 14:17', related_items=[])
    def pandas_df_based_correlation_computations(cls, selected_spikes_df: pd.DataFrame, active_epochs_df: Optional[pd.DataFrame], track_templates: TrackTemplates, num_shuffles:int=1000, debug_print=True):
        """ 2023-12-15 - Absolute newest complete Rank-Order shuffle implementation. Does both Pearson and Spearman.

        selected_spikes_df: pd.DataFrame - spikes dataframe containing only the first spike (the "selected one") for each cell within the periods of interest.


        Outputs:

            combined_variable_names: ['LR_Long_spearman', 'RL_Long_spearman', 'LR_Short_spearman', 'RL_Short_spearman', 'LR_Long_pearson', 'RL_Long_pearson', 'LR_Short_pearson', 'RL_Short_pearson']
            combined_variable_z_score_column_names: ['LR_Long_spearman_Z', 'RL_Long_spearman_Z', 'LR_Short_spearman_Z', 'RL_Short_spearman_Z', 'LR_Long_pearson_Z', 'RL_Long_pearson_Z', 'LR_Short_pearson_Z', 'RL_Short_pearson_Z']

        Usage:

            from PendingNotebookCode import pandas_df_based_correlation_computations

            combined_epoch_stats_df, (output_active_epoch_computed_values, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles) = pandas_df_based_correlation_computations(selected_spikes_df, track_templates, num_shuffles=1000)


        """
        ## Shuffle each map's aclus, takes `selected_spikes_df`

        # LongShortStatsTuple: Tuple[Zscorer, Zscorer, float, float, bool]

        _NaN_Type = np.nan
        _NaN_Type = pd.NA

        rng = np.random.default_rng() # seed=13378 #TODO 2023-12-13 05:13: - [ ] DO NOT SET THE SEED! This makes the random permutation/shuffle the same every time!!!
        long_LR_aclu_peak_map, long_RL_aclu_peak_map, short_LR_aclu_peak_map, short_RL_aclu_peak_map = track_templates.get_decoder_aclu_peak_maps()

        track_templates.get_decoder_aclu_peak_map_dict()

        ## Restrict to only the relevant columns, and Initialize the dataframe columns to np.nan:
        active_selected_spikes_df: pd.DataFrame = deepcopy(selected_spikes_df[['t_rel_seconds', 'aclu', 'Probe_Epoch_id']]).sort_values(['Probe_Epoch_id', 't_rel_seconds', 'aclu']).astype({'Probe_Epoch_id': 'int'}) # Sort by columns: 'Probe_Epoch_id' (ascending), 't_rel_seconds' (ascending), 'aclu' (ascending)
        _pf_peak_x_column_names = ['LR_Long_pf_peak_x', 'RL_Long_pf_peak_x', 'LR_Short_pf_peak_x', 'RL_Short_pf_peak_x']
        active_selected_spikes_df[_pf_peak_x_column_names] = pd.DataFrame([[_NaN_Type, _NaN_Type, _NaN_Type, _NaN_Type]], index=active_selected_spikes_df.index)

        ## Normal:
        active_selected_spikes_df['LR_Long_pf_peak_x'] = active_selected_spikes_df.aclu.map(long_LR_aclu_peak_map)
        active_selected_spikes_df['RL_Long_pf_peak_x'] = active_selected_spikes_df.aclu.map(long_RL_aclu_peak_map)
        active_selected_spikes_df['LR_Short_pf_peak_x'] = active_selected_spikes_df.aclu.map(short_LR_aclu_peak_map)
        active_selected_spikes_df['RL_Short_pf_peak_x'] = active_selected_spikes_df.aclu.map(short_RL_aclu_peak_map)

        #TODO 2023-12-18 13:20: - [ ] This assumes that `'Probe_Epoch_id'` is correct and consistent for both directions, yeah?

        ## Compute real values here:
        epoch_id_grouped_selected_spikes_df =  active_selected_spikes_df.groupby('Probe_Epoch_id') # I can even compute this outside the loop?
        spearman_correlations = epoch_id_grouped_selected_spikes_df.apply(lambda group: RankOrderAnalyses._subfn_calculate_correlations(group, method='spearman', enable_shuffle=False)).reset_index() # Reset index to make 'Probe_Epoch_id' a column
        pearson_correlations = epoch_id_grouped_selected_spikes_df.apply(lambda group: RankOrderAnalyses._subfn_calculate_correlations(group, method='pearson', enable_shuffle=False)).reset_index() # Reset index to make 'Probe_Epoch_id' a column

        real_stats_tuple = (spearman_correlations, pearson_correlations)

        names0 = ['LR_Long_spearman', 'RL_Long_spearman', 'LR_Short_spearman', 'RL_Short_spearman']
        dfs0 = [real_stats_tuple[0]] # [0] is just spearman_correlations
        stacked_arrays0 = np.array([df[names0].values for df in dfs0])
        names1 = ['LR_Long_pearson', 'RL_Long_pearson', 'LR_Short_pearson', 'RL_Short_pearson']
        dfs1 = [real_stats_tuple[1]] # [1] is pearson_correlations
        stacked_arrays1 = np.array([df[names1].values for df in dfs1])
        real_stacked_arrays = np.squeeze(np.concatenate((stacked_arrays0, stacked_arrays1), axis=-1)[0,:,:]) # .shape: (412, 8) # (n_epochs, n_columns)

        ## Combined column names:
        combined_variable_names = names0 + names1

        # ==================================================================================================================== #
        # PERFORM SHUFFLE HERE:                                                                                                #
        # ==================================================================================================================== #
        # On-the-fly shuffling mode using shuffle_helper:
        # epoch_specific_shuffled_aclus, epoch_specific_shuffled_indicies = build_shuffled_ids(list(long_LR_aclu_peak_map.keys()), num_shuffles=num_shuffles, seed=None) # .shape: ((num_shuffles, n_neurons), (num_shuffles, n_neurons))

        ## USES selected_spikes_df
        all_decoder_aclus_map_keys_list = [np.array(list(a_map.keys())) for a_map in (long_LR_aclu_peak_map, long_RL_aclu_peak_map, short_LR_aclu_peak_map, short_RL_aclu_peak_map)] # list of four elements
        all_shuffled_decoder_aclus_map_keys_list = [build_shuffled_ids(a_map_keys, num_shuffles=num_shuffles, seed=None)[0] for a_map_keys in all_decoder_aclus_map_keys_list] # [0] only gets the shuffled_aclus themselves, which are of shape .shape: ((num_shuffles, n_neurons[i]) where i is the decoder_index

        long_LR_epoch_specific_shuffled_aclus, long_RL_epoch_specific_shuffled_aclus, short_LR_epoch_specific_shuffled_aclus, short_RL_epoch_specific_shuffled_aclus = all_shuffled_decoder_aclus_map_keys_list

        ## Shuffle a single map, but will eventually need one for each of the four decoders::
        # epoch_specific_shuffled_aclus, epoch_specific_shuffled_indicies = build_shuffled_ids(list(long_LR_aclu_peak_map.keys()), num_shuffles=num_shuffles, seed=None) # .shape: ((num_shuffles, n_neurons), (num_shuffles, n_neurons))

        output_active_epoch_computed_values = []

        for shuffle_IDX in np.arange(num_shuffles):
            """ within the loop we modify: 
                active_selected_spikes_df, active_epochs 
                
                From active_selected_spikes_df I only need: ['t_rel_seconds', 'aclu', 'Probe_Epoch_id', 'label']  ## 'Probe_Epoch_id' goes up to 610 for some reason?!?!? It does NOT seem to be 'label'
                
            """
            active_selected_spikes_df[_pf_peak_x_column_names] = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]], index=active_selected_spikes_df.index)
            # assert len(epoch_specific_shuffled_aclus_tuple) == 4, f"len(epoch_specific_shuffled_aclus_tuple): {len(epoch_specific_shuffled_aclus_tuple)}"

            #TODO 2023-11-22 12:50: - [X] Are the sizes correct since `a_shuffled_IDXs` doesn't know the size of the template?
            a_name = 'LR_Long_pf_peak_x'
            active_selected_spikes_df[a_name] = active_selected_spikes_df.aclu.map(dict(zip(long_LR_epoch_specific_shuffled_aclus[shuffle_IDX], long_LR_aclu_peak_map.values())))

            a_name = 'RL_Long_pf_peak_x'
            active_selected_spikes_df[a_name] = active_selected_spikes_df.aclu.map(dict(zip(long_RL_epoch_specific_shuffled_aclus[shuffle_IDX], long_RL_aclu_peak_map.values())))

            a_name = 'LR_Short_pf_peak_x'
            active_selected_spikes_df[a_name] = active_selected_spikes_df.aclu.map(dict(zip(short_LR_epoch_specific_shuffled_aclus[shuffle_IDX], short_LR_aclu_peak_map.values())))

            a_name = 'RL_Short_pf_peak_x'
            active_selected_spikes_df[a_name] = active_selected_spikes_df.aclu.map(dict(zip(short_RL_epoch_specific_shuffled_aclus[shuffle_IDX], short_RL_aclu_peak_map.values())))

            epoch_id_grouped_selected_spikes_df =  active_selected_spikes_df.groupby('Probe_Epoch_id') # I can even compute this outside the loop?
            spearman_correlations = epoch_id_grouped_selected_spikes_df.apply(lambda group: RankOrderAnalyses._subfn_calculate_correlations(group, method='spearman', enable_shuffle=False)).reset_index() # Reset index to make 'Probe_Epoch_id' a column
            pearson_correlations = epoch_id_grouped_selected_spikes_df.apply(lambda group: RankOrderAnalyses._subfn_calculate_correlations(group, method='pearson', enable_shuffle=False)).reset_index() # Reset index to make 'Probe_Epoch_id' a column

            output_active_epoch_computed_values.append((spearman_correlations, pearson_correlations))

        # Build the output `stacked_arrays`: _________________________________________________________________________________ #
        stacked_arrays = cls.build_stacked_arrays(output_active_epoch_computed_values)

        ## Drop any shuffle indicies where NaNs are returned for any of the stats values.
        is_valid_row = np.logical_not(np.isnan(stacked_arrays)).all(axis=(1,2))
        n_valid_shuffles = np.sum(is_valid_row)
        if debug_print:
            print(f'n_valid_shuffles: {n_valid_shuffles}')
        valid_stacked_arrays = stacked_arrays[is_valid_row] ## Get only the rows where all elements along both axis (1, 2) are True

        # Need: valid_stacked_arrays, real_stacked_arrays, combined_variable_names
        combined_epoch_stats_df: pd.DataFrame = pd.DataFrame(real_stacked_arrays, columns=combined_variable_names)
        combined_variable_z_score_column_names = [f"{a_name}_Z" for a_name in combined_variable_names] # combined_variable_z_score_column_names: ['LR_Long_spearman_Z', 'RL_Long_spearman_Z', 'LR_Short_spearman_Z', 'RL_Short_spearman_Z', 'LR_Long_pearson_Z', 'RL_Long_pearson_Z', 'LR_Short_pearson_Z', 'RL_Short_pearson_Z']

        ## Extract the stats values for each shuffle from `valid_stacked_arrays`:
        n_epochs = np.shape(real_stacked_arrays)[0]
        n_variables = np.shape(real_stacked_arrays)[1]

        assert n_epochs == np.shape(valid_stacked_arrays)[-2]
        assert n_variables == np.shape(valid_stacked_arrays)[-1]

        for variable_IDX, a_column_name in enumerate(combined_variable_z_score_column_names):
            z_scorer_list = [Zscorer.init_from_values(stats_corr_values=np.squeeze(valid_stacked_arrays[:, :, variable_IDX]), real_value=real_stacked_arrays[epoch_IDX, variable_IDX]) for epoch_IDX in np.arange(n_epochs)]
            z_score_values = np.array([a_zscorer.z_score_value for a_zscorer in z_scorer_list])
            combined_epoch_stats_df[a_column_name] = z_score_values

        if debug_print:
            print(f'combined_variable_names: {combined_variable_names}')
            print(f'combined_variable_z_score_column_names: {combined_variable_z_score_column_names}')

        # Try to add epoch labels. Only used at the very end: active_epochs_df
        try:
            if (active_epochs_df is not None) and ('label' in active_epochs_df.columns):
                active_epochs_df['label'] = active_epochs_df['label'].astype('int')
                if (np.shape(active_epochs_df)[0] == np.shape(combined_epoch_stats_df)[0]):
                    combined_epoch_stats_df['label'] = active_epochs_df['label'].copy()
                else:
                    print(f'failed to add label column, shapes differ! np.shape(active_epochs_df)[0] : {np.shape(active_epochs_df)[0] }, np.shape(combined_epoch_stats_df)[0]): {np.shape(combined_epoch_stats_df)[0]}')

                # combined_epoch_stats_df = combined_epoch_stats_df.set_index('label')
            else:
                print('invalid active_epochs_df. skipping adding labels')
        except BaseException as e:
            print(f'Not giving up: e: {e}')
            pass

        return combined_epoch_stats_df, (output_active_epoch_computed_values, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles)
    





class RankOrderGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'rank_order'
    _computationPrecidence = 1001
    _is_global = True

    @function_attributes(short_name='rank_order_shuffle_analysis', tags=['directional_pf', 'laps', 'rank_order', 'session', 'pf1D', 'pf2D'], input_requires=['DirectionalLaps'], output_provides=['RankOrder'], uses=['RankOrderAnalyses'], used_by=[], creation_date='2023-11-08 17:27', related_items=[],
        validate_computation_test=RankOrderAnalyses.validate_has_rank_order_results, is_global=True)
    def perform_rank_order_shuffle_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, num_shuffles:int=500, minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2], skip_laps=False):
        """ Performs the computation of the spearman and pearson correlations for the ripple and lap epochs.

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
        # included_qclu_values=[1,2]

        if ('RankOrder' not in global_computation_results.computed_data) or (not hasattr(global_computation_results.computed_data, 'RankOrder')):
            # initialize
            global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(LR_ripple=None, RL_ripple=None, LR_laps=None, RL_laps=None,
                                                                                                   minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz,
                                                                                                   included_qclu_values=included_qclu_values,
                                                                                                   is_global=True)

        global_computation_results.computed_data['RankOrder'].included_qclu_values = included_qclu_values

        ## Laps Rank-Order Analysis:
        if not skip_laps:
            print(f'\tcomputing Laps rank-order shuffles:')
            print(f'\t\tnum_shuffles: {num_shuffles}, minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz} Hz')
            # _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='center_of_mass')
            _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='median', minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
            # _laps_outputs = RankOrderAnalyses.main_laps_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='first')
            (LR_laps_outputs, RL_laps_outputs, laps_paired_tests)  = _laps_outputs
            global_computation_results.computed_data['RankOrder'].LR_laps = LR_laps_outputs
            global_computation_results.computed_data['RankOrder'].RL_laps = RL_laps_outputs

            try:
                print(f'\tdone. building global result.')
                directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
                selected_spikes_df = deepcopy(global_computation_results.computed_data['RankOrder'].LR_laps.selected_spikes_df) # WARNING: this is only using the `selected_spikes_df` from LR_laps!! This would miss spikes of any RL-specific cells?
                # active_epochs = global_computation_results.computed_data['RankOrder'].laps_most_likely_result_tuple.active_epochs
                active_epochs = deepcopy(LR_laps_outputs.epochs_df)
                track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
                laps_combined_epoch_stats_df, laps_new_output_tuple = RankOrderAnalyses.pandas_df_based_correlation_computations(selected_spikes_df=selected_spikes_df, active_epochs_df=active_epochs, track_templates=track_templates, num_shuffles=num_shuffles)
                # new_output_tuple (output_active_epoch_computed_values, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles) = laps_new_output_tuple
                global_computation_results.computed_data['RankOrder'].laps_combined_epoch_stats_df, global_computation_results.computed_data['RankOrder'].laps_new_output_tuple = laps_combined_epoch_stats_df, laps_new_output_tuple
                print(f'done!')

            except (AssertionError, BaseException) as e:
                print(f'Issue with Laps computation in new method 2023-12-15: e: {e}')
                raise

        ## END `if not skip_laps`


        ## Ripple Rank-Order Analysis:
        print(f'\tcomputing Ripple rank-order shuffles:')
        _ripples_outputs = RankOrderAnalyses.main_ripples_analysis(owning_pipeline_reference, num_shuffles=num_shuffles, rank_alignment='first', minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # rank_alignment='first'
        (LR_ripple_outputs, RL_ripple_outputs, ripple_evts_paired_tests) = _ripples_outputs
        global_computation_results.computed_data['RankOrder'].LR_ripple = LR_ripple_outputs
        global_computation_results.computed_data['RankOrder'].RL_ripple = RL_ripple_outputs

        # New method 2023-12-15:
        try:
            print(f'\tdone. building global result.')
            directional_laps_results: DirectionalLapsResult = global_computation_results.computed_data['DirectionalLaps']
            selected_spikes_df = deepcopy(global_computation_results.computed_data['RankOrder'].LR_ripple.selected_spikes_df)
            # active_epochs = global_computation_results.computed_data['RankOrder'].ripple_most_likely_result_tuple.active_epochs
            active_epochs = deepcopy(LR_ripple_outputs.epochs_df)
            track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
            ripple_combined_epoch_stats_df, ripple_new_output_tuple = RankOrderAnalyses.pandas_df_based_correlation_computations(selected_spikes_df=selected_spikes_df, active_epochs_df=active_epochs, track_templates=track_templates, num_shuffles=num_shuffles)
            # new_output_tuple (output_active_epoch_computed_values, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles) = ripple_new_output_tuple
            global_computation_results.computed_data['RankOrder'].ripple_combined_epoch_stats_df, global_computation_results.computed_data['RankOrder'].ripple_new_output_tuple = ripple_combined_epoch_stats_df, ripple_new_output_tuple
            print(f'done!')

        except (AssertionError, BaseException) as e:
            print(f'New method 2023-12-15: e: {e}')
            raise


        ## Requires "New method 2023-12-15" result
        # Set the global result:
        try:
            print(f'\tdone. building global result.')
            global_computation_results.computed_data['RankOrder'].adding_active_aclus_info()
            global_computation_results.computed_data['RankOrder'].ripple_most_likely_result_tuple, global_computation_results.computed_data['RankOrder'].laps_most_likely_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(owning_pipeline_reference, decoding_time_bin_size=0.006) # 6ms bins
        
        except (AssertionError, BaseException) as e:
            print(f'Issue with `RankOrderAnalyses.most_likely_directional_rank_order_shuffling(...)` e: {e}')
            raise


        """ Usage:
        
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']

        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values = rank_order_results.odd_laps
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values = rank_order_results.even_laps

        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values = rank_order_results.odd_ripple
        even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values = rank_order_results.even_ripple

        """
        return global_computation_results




from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

# from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
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


    @function_attributes(short_name='rank_order_z_stats', tags=['rank-order','debugger','shuffle'], input_requires=[], output_provides=[], uses=['plot_rank_order_epoch_inst_fr_result_tuples'], used_by=[], creation_date='2023-12-15 21:46', related_items=[],
        validate_computation_test=RankOrderAnalyses._validate_can_display_RankOrderRastersDebugger, is_global=True)
    def _display_rank_order_z_stats_results(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ Plots the z-scores differences and their raw-values

            """
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())



            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results: RankOrderComputationsContainer = global_computation_results.computed_data['RankOrder']
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz

            ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
            
            ripple_outputs = plot_rank_order_epoch_inst_fr_result_tuples(owning_pipeline_reference, ripple_result_tuple, 'Ripple')
            lap_outputs = plot_rank_order_epoch_inst_fr_result_tuples(owning_pipeline_reference, laps_result_tuple, 'Lap')

            ripple_result_tuple.plot_histograms()
            laps_result_tuple.plot_histograms()

            return ripple_outputs, lap_outputs




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


def plot_new(ripple_result_tuple: DirectionalRankOrderResult):
    # ripple_result_tuple: DirectionalRankOrderResult
    fig = plt.figure(layout="constrained", num='RipplesRankOrderZscore')
    ax_dict = fig.subplot_mosaic(
        [
            ["long_short_best_z_score_diff", "long_short_best_z_score_diff"],
            ["long_best_z_scores", "short_best_z_scores"],
        ],
        # set the height ratios between the rows
        # height_ratios=[8, 1],
        # height_ratios=[1, 1],
        # set the width ratios between the columns
        # width_ratios=[1, 8, 8, 1],
        # sharey=True,
        # gridspec_kw=dict(wspace=0, hspace=0.15) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
    )
    # pd.DataFrame({'long_best_z_scores': ripple_result_tuple.long_best_dir_z_score_values, 'short_best_z_scores': ripple_result_tuple.short_best_dir_z_score_values}).hist(ax=(ax_dict['long_best_z_scores'], ax_dict['short_best_z_scores']))
    pd.DataFrame({'long_best_z_scores': ripple_result_tuple.long_best_dir_z_score_values}).hist(ax=ax_dict['long_best_z_scores'], bins=21, alpha=0.8)
    pd.DataFrame({'short_best_z_scores': ripple_result_tuple.short_best_dir_z_score_values}).hist(ax=ax_dict['short_best_z_scores'], bins=21, alpha=0.8)
    pd.DataFrame({'long_short_best_z_score_diff': ripple_result_tuple.long_short_best_dir_z_score_diff_values}).hist(ax=ax_dict['long_short_best_z_score_diff'], bins=21, alpha=0.8)

    # plt.suptitle('Ripple Rank-Order')


@function_attributes(short_name=None, tags=['histogram', '1D', 'rank-order'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-12 09:20', related_items=[])
def plot_rank_order_histograms(rank_order_results: RankOrderComputationsContainer, number_of_bins: int = 21, post_title_info: str = '') -> Tuple:
    """ plots 1D histograms from the rank-order shuffled data during the ripples.

    https://pandas.pydata.org/pandas-docs/version/0.24.1/user_guide/visualization.html

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import plot_rank_order_histograms

        # Plot histograms:
        post_title_info: str = f'{minimum_inclusion_fr_Hz} Hz\n{curr_active_pipeline.get_session_context().get_description()}'
        _out_z_score, _out_real, _out_most_likely_z = plot_rank_order_histograms(rank_order_results, post_title_info=post_title_info)

    """

    # fig = build_or_reuse_figure(fignum=f'1D Histograms')
    # ax1 = fig.add_subplot(3, 1, 1)
    # ax2 = fig.add_subplot(3, 1, 2)
    # ax3 = fig.add_subplot(3, 1, 3)

    LR_results_real_values = np.array([(long_stats_z_scorer.real_value, short_stats_z_scorer.real_value) for epoch_id, (long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay) in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
    RL_results_real_values = np.array([(long_stats_z_scorer.real_value, short_stats_z_scorer.real_value) for epoch_id, (long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay) in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])

    LR_results_long_short_z_diffs = np.array([long_short_z_diff for epoch_id, (long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay) in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
    RL_results_long_short_z_diff = np.array([long_short_z_diff for epoch_id, (long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay) in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])

    ax1, ax2, ax3 = None, None, None

    _out_z_score = pd.DataFrame({'LR_long_z_scores': rank_order_results.LR_ripple.long_z_score, 'LR_short_z_scores': rank_order_results.LR_ripple.short_z_score,
              'RL_long_z_scores': rank_order_results.RL_ripple.long_z_score, 'RL_short_z_scores': rank_order_results.RL_ripple.short_z_score}).hist(bins=number_of_bins, ax=ax1, sharex=True, sharey=True)
    plt.suptitle(': '.join([f'Ripple Z-scores', post_title_info]))

    _out_real = pd.DataFrame({'LR_long_real_corr': np.squeeze(LR_results_real_values[:,0]), 'LR_short_real_corr': np.squeeze(LR_results_real_values[:,1]),
              'RL_long_real_corr': np.squeeze(RL_results_real_values[:,0]), 'RL_short_real_corr': np.squeeze(RL_results_real_values[:,1])}).hist(bins=number_of_bins, ax=ax2, sharex=True, sharey=True)
    plt.suptitle(': '.join([f'Ripple real correlations', post_title_info]))

    _out_most_likely_z = pd.DataFrame({'most_likely_long_z_scores': rank_order_results.ripple_most_likely_result_tuple.long_best_dir_z_score_values, 'most_likely_short_z_scores': rank_order_results.ripple_most_likely_result_tuple.short_best_dir_z_score_values}).hist(bins=number_of_bins, ax=ax3, sharex=True, sharey=True)
    plt.suptitle(': '.join([f'Ripple Most-likely z-scores', post_title_info]))

    return _out_z_score, _out_real, _out_most_likely_z



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
        # global_events = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
        global_events = deepcopy(result_tuple.active_epochs)
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


    # epoch_identifiers = np.arange(global_events.n_epochs) # these should be labels!
    epoch_identifiers = global_events._df.label.astype({'label': 'int'}).values #.labels
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



