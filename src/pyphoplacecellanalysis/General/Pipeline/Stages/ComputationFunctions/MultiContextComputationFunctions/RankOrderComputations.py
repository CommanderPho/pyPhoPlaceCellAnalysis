from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
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
import nptyping as ND
from nptyping import NDArray
import attrs
from attrs import asdict, define, field, Factory, astuple

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


from neuropy.utils.mixins.time_slicing import add_epochs_id_identity
import scipy.stats
from scipy import ndimage
from pyphocorehelpers.indexing_helpers import NumpyHelpers
from neuropy.utils.misc import build_shuffled_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.print_helpers import print_array
import matplotlib.pyplot as plt

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences
# from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange

from scipy import stats # _recover_samples_per_sec_from_laps_df
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, DirectionalPseudo2DDecodersResult

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult # used in TrackTemplates

from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

if TYPE_CHECKING:
    ## typehinting only imports here
    # import pyphoplacecellanalysis.External.pyqtgraph as pg
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?
    
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


@define(slots=False, repr=False, eq=False)
class Zscorer(HDFMixin, AttrsBasedClassHelperMixin):
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

    # def plot_distribution(self):
    #     """ plots a standalone figure showing the distribution of the original values and their fisher_z_transformed version in a histogram. """
    #     win = pg.GraphicsLayoutWidget(show=True)
    #     win.resize(800,350)
    #     win.setWindowTitle('Z-Scorer: Histogram')
    #     plt1 = win.addPlot()
    #     vals = self.original_values
    #     fisher_z_transformed_vals = np.arctanh(vals)

    #     ## compute standard histogram
    #     y, x = np.histogram(vals) # , bins=np.linspace(-3, 8, 40)
    #     fisher_z_transformed_y, x = np.histogram(fisher_z_transformed_vals, bins=x)

    #     ## Using stepMode="center" causes the plot to draw two lines for each sample.
    #     ## notice that len(x) == len(y)+1
    #     plt1.plot(x, y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,0,255,50), name='original_values')
    #     plt1.plot(x, fisher_z_transformed_y, stepMode="center", fillLevel=0, fillOutline=True, brush=(0,255,100,50), name='fisher_z_values')

    #     # ## Now draw all points as a nicely-spaced scatter plot
    #     # y = pg.pseudoScatter(vals, spacing=0.15)
    #     # #plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5)
    #     # plt2.plot(vals, y, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(0,0,255,150))

    #     return win, plt1

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # if 'identity' not in state:
        #     print(f'unpickling from old NeuropyPipelineStage')
        #     state['identity'] = None
        #     state['identity'] = type(self).get_stage_identity()
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(Zscorer, self).__init__() # from
        


LongShortStatsTuple = namedtuple('LongShortStatsTuple', ['long_stats_z_scorer', 'short_stats_z_scorer', 'long_short_z_diff', 'long_short_naive_z_diff', 'is_forward_replay']) # used in `compute_shuffled_rankorder_analyses`
# LongShortStatsTuple: Tuple[Zscorer, Zscorer, float, float, bool]

@define(slots=False)
class LongShortStatsItem(object):
    """ built using # CodeConversion.convert_dictionary_to_class_defn
    from namedtuple LongShortStatsTuple
    """
    long_stats_z_scorer: Zscorer = field()
    short_stats_z_scorer: Zscorer = field()
    long_short_z_diff: np.float64 = field()
    long_short_naive_z_diff: np.float64 = field()
    is_forward_replay: np.bool_ = field()
    
    @classmethod
    def init_from_LongShortStatsTuple(cls, a_tuple) -> "LongShortStatsItem":
        return cls(**a_tuple._asdict())


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # del state['long_stats_z_scorer']
        # del state['short_stats_z_scorer']

        try:
            state['long_stats_z_scorer'] = state['long_stats_z_scorer'].__getstate__()
            state['short_stats_z_scorer'] = state['short_stats_z_scorer'].__getstate__()
        except AttributeError:
            # state['long_stats_z_scorer'] is already a dict
            pass
        
        return state

    def fixup_types_if_needed(self):
        """ 2024-01-06 - Fix ZScorer types being dicts after loading:
        # this was resulting in `AttributeError: 'dict' object has no attribute 'real_value'`
        """
        try:
            self.long_stats_z_scorer.real_value
            self.short_stats_z_scorer.real_value
        except AttributeError:
            # fix the values, they weren't loaded correctly
            if isinstance(self.long_stats_z_scorer, dict):
                self.long_stats_z_scorer = Zscorer(**self.long_stats_z_scorer)
            if isinstance(self.short_stats_z_scorer, dict):
                self.short_stats_z_scorer = Zscorer(**self.short_stats_z_scorer)
        


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # if 'identity' not in state:
        #     print(f'unpickling from old NeuropyPipelineStage')
        #     state['identity'] = None
        #     state['identity'] = type(self).get_stage_identity()
        try:
            state['long_stats_z_scorer'] = Zscorer(**state['long_stats_z_scorer']) # __setstate__ or something instead?
            state['short_stats_z_scorer'] = Zscorer(**state['short_stats_z_scorer'])
        except AttributeError:
            # state['long_stats_z_scorer'] is already a dict
            pass

        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(LongShortStatsItem, self).__init__() # from


        self.fixup_types_if_needed()



    



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


# class SaveStringGenerator:
#     """ 
#     # 2023-11-27 - I'd like to be able to save/load single results a time, (meaning specific to their parameters):
#     day_date_str: str = '2023-12-11-minimum_inclusion_fr_Hz_2_included_qclu_values_1-2_'

#     """
#     _minimal_decimals_float_formatter = lambda x: f"{x:.1f}".rstrip('0').rstrip('.')
    
#     @classmethod
#     def generate_save_suffix(cls, day_date: str='2023-12-11', session_identifier: Optional[str]=None, minimum_inclusion_fr_Hz: Optional[float]=None, included_qclu_values: Optional[List[int]]=None) -> str:
#         """ 
        
#         """
#         # day_date_str: str = '2023-12-11-minimum_inclusion_fr_Hz_2_included_qclu_values_1-2_'
#         print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
#         print(f'included_qclu_values: {included_qclu_values}')
#         # _format_arr = [day_date, f'minimum_inclusion_fr', cls._minimal_decimals_float_formatter(minimum_inclusion_fr_Hz), f'included_qclu_values', f'{included_qclu_values}']
#         _format_arr = [day_date]
#         if ((session_identifier is not None) and (len(session_identifier) > 0)):
#             _format_arr.append(session_identifier)
#         if ((minimum_inclusion_fr_Hz is not None) and (len(str(minimum_inclusion_fr_Hz)) > 0)):
#             _format_arr.append(cls._minimal_decimals_float_formatter(minimum_inclusion_fr_Hz))
#         if ((included_qclu_values is not None) and (len(included_qclu_values) > 0)):
#             _format_arr.append(f'{included_qclu_values}')
#         # if ((session_identifier is not None) and (len(session_identifier) > 0)):
#         #     _format_arr.append(session_identifier)

#         #  f'minimum_inclusion_fr', cls._minimal_decimals_float_formatter(minimum_inclusion_fr_Hz), f'included_qclu_values', f'{included_qclu_values}']

#         out_filename_str: str = '-'.join(_format_arr)
#         return out_filename_str

# list = ['2Hz', '12Hz']

def save_rank_order_results(curr_active_pipeline, day_date: str='2023-12-19_729pm', override_output_parent_path: Optional[Path]=None):
    """ saves out the rnak-order and directional laps results to disk.
    
    rank_order_output_path, directional_laps_output_path, directional_merged_decoders_output_path, out_filename_str

    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import save_rank_order_results
    
     rank_order_output_path, directional_laps_output_path, directional_merged_decoders_output_path, out_filename_str = save_rank_order_results(curr_active_pipeline, day_date: str='2024-09-26_503pm')
    
     "2024-11-15_Lab-kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]-(DirectionalLaps).pkl"
     "2024-11-15_Lab-kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]-(RankOrder).pkl"
     "2024-11-15_Lab-kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]-(DirectionalMergedDecoders).pkl"
     
    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
    ## Uses `SaveStringGenerator.generate_save_suffix` and the current rank_order_result's parameters to build a reasonable save name:
    assert curr_active_pipeline.global_computation_results.computed_data['RankOrder'] is not None
    # minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computed_data['RankOrder'].minimum_inclusion_fr_Hz
    # included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computed_data['RankOrder'].included_qclu_values
    # curr_session_name: str = curr_active_pipeline.session_name # '2006-6-08_14-26-15'
    complete_session_identifier_string: str = curr_active_pipeline.get_complete_session_identifier_string() # 'kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]'
    
    # CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}" # self.BATCH_DATE_TO_USE: '2024-11-15_Lab'
    CURR_BATCH_OUTPUT_PREFIX: str = f"{day_date}-{complete_session_identifier_string}" # self.BATCH_DATE_TO_USE: '2024-11-15_Lab'
    print(f'CURR_BATCH_OUTPUT_PREFIX: "{CURR_BATCH_OUTPUT_PREFIX}"') # CURR_BATCH_OUTPUT_PREFIX: "2024-11-15_Lab-kdiba-gor01-one-2006-6-09_1-22-43__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]"
    
    # out_filename_str = SaveStringGenerator.generate_save_suffix(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values, day_date=day_date)
    out_filename_str: str = CURR_BATCH_OUTPUT_PREFIX

    print(f'save_rank_order_results(...): out_filename_str: "{out_filename_str}"')
    output_parent_path: Path = (override_output_parent_path or curr_active_pipeline.get_output_path()).resolve()

    try:
        directional_laps_output_path = output_parent_path.joinpath(f'{out_filename_str}-(DirectionalLaps).pkl').resolve()
        saveData(directional_laps_output_path, (curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']))
    except Exception as e:
        print(f'issue saving "{directional_laps_output_path}": error: {e}')
        pass
    
    try:
        rank_order_output_path = output_parent_path.joinpath(f'{out_filename_str}-(RankOrder).pkl').resolve()
        saveData(rank_order_output_path, (curr_active_pipeline.global_computation_results.computed_data['RankOrder']))
    except Exception as e:
        print(f'issue saving "{directional_laps_output_path}": error: {e}')
        pass

    try:
        directional_merged_decoders_output_path = output_parent_path.joinpath(f'{out_filename_str}-(DirectionalMergedDecoders).pkl').resolve()
        saveData(directional_merged_decoders_output_path, (curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']))
    except Exception as e:
        print(f'issue saving "{directional_laps_output_path}": error: {e}')
        pass
    
    return rank_order_output_path, directional_laps_output_path, directional_merged_decoders_output_path, out_filename_str
    


@define(slots=False, repr=False, eq=False)
class ShuffleHelper(HDFMixin, AttrsBasedClassHelperMixin):
    """ holds the result of shuffling templates. Used for rank-order analyses """
    shared_aclus_only_neuron_IDs: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal)) # #TODO 2024-08-07 16:55: - [ ] is the inline definition in `eq=` the lambda that's preventing serialization?
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
class RankOrderResult(ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even


    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    # is_global: bool = non_serialized_field(default=True, repr=False)
    ranked_aclus_stats_dict: Dict[int, LongShortStatsItem] = serialized_field(repr=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v))) # , serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v))
    selected_spikes_fragile_linear_neuron_IDX_dict: Dict[int, NDArray] = serialized_field(repr=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v)))

    long_z_score: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    short_z_score: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    long_short_z_score_diff: NDArray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))

    spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    epochs_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)

    selected_spikes_df: pd.DataFrame = serialized_field(default=Factory(pd.DataFrame), repr=False)
    extra_info_dict: Dict = non_serialized_field(default=Factory(dict), repr=False)

    result_version: str = serialized_attribute_field(default='2024.01.11_0', is_computable=False, repr=False) # this field specfies the version of the result. 


    @property
    def epoch_template_active_aclus(self) -> Dict[int, NDArray]:
        """

        Usage:
            label_column_type = 'int'
            active_epochs_df.label.astype(label_column_type).map(lambda x: rank_order_results.LR_ripple.epoch_template_active_aclus[x])

        """
        return {k:v[1] for k, v in self.extra_info_dict.items()} # [1] corresponds to `template_epoch_actually_included_aclus`

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # del state['extra_info_dict'] # drop 'extra_info_dict'
        if len(state['ranked_aclus_stats_dict']) > 0:
            example_item = list(state['ranked_aclus_stats_dict'].values())[0] # first item
            if not isinstance(example_item, LongShortStatsItem):
                # convert all to stats items:
                try:
                    state['ranked_aclus_stats_dict'] = {k:LongShortStatsItem(**v._asdict()) for k,v in state['ranked_aclus_stats_dict'].items()} # AttributeError: 'LongShortStatsItem' object has no attribute '_asdict'
                except AttributeError:
                    state['ranked_aclus_stats_dict'] = {k:v for k,v in state['ranked_aclus_stats_dict'].items()}
                    

            state['ranked_aclus_stats_dict'] = {k:v.__getstate__() for k,v in state['ranked_aclus_stats_dict'].items()} # call the .__getstate__() for each dict entry

        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).

        result_version: str = state.get('result_version', None)
        if result_version is None:
            result_version = "2024.01.10_0"
            state['result_version'] = result_version # set result version


        # convert from old named-tuple based items to LongShortStatsItem (2024-01-02):
        if len(state['ranked_aclus_stats_dict']) > 0:
            example_item = list(state['ranked_aclus_stats_dict'].values())[0] # first item
            if not isinstance(example_item, LongShortStatsItem):
                # convert all to stats items:
                if isinstance(example_item, dict):
                    state['ranked_aclus_stats_dict'] = {k:LongShortStatsItem(**v) for k,v in state['ranked_aclus_stats_dict'].items()}
                else:
                    try:
                        state['ranked_aclus_stats_dict'] = {k:LongShortStatsItem(**v._asdict()) for k,v in state['ranked_aclus_stats_dict'].items()}
                    except AttributeError:
                        # AttributeError: 'tuple' object has no attribute '_asdict' 
                        fixed_ranked_aclus_stats_dict = {}
                        assert NumpyHelpers.all_array_equal([len(v) for k,v in state['ranked_aclus_stats_dict'].items()])
                        for k,v in state['ranked_aclus_stats_dict'].items():
                            num_missing_items = (5 - len(v))
                            completed_position_args_list = [*v, *(num_missing_items * [None])]
                            # if (len(v) == 3)
                            assert (isinstance(completed_position_args_list[0], Zscorer) and isinstance(completed_position_args_list[1], Zscorer))
                            assert (isinstance(completed_position_args_list[2], float))
                            # fixed_ranked_aclus_stats_dict[k] = LongShortStatsItem(long_stats_z_scorer=v[0], short_stats_z_scorer=v[1], long_short_z_diff=v[2], long_short_naive_z_diff=None, is_forward_replay=None)
                            fixed_ranked_aclus_stats_dict[k] = LongShortStatsItem(*completed_position_args_list)
                        state['ranked_aclus_stats_dict'] = fixed_ranked_aclus_stats_dict # replace it with the new dict
                        
                        

        self.__dict__.update(state)

        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        super(RankOrderResult, self).__init__()
        




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



# @define(slots=False, order=True)
class DirectionalRankOrderResult(DirectionalRankOrderResultBase):

    @property
    def directional_likelihoods_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.directional_likelihoods_tuple._asdict()).astype({'long_best_direction_indices': 'int8', 'short_best_direction_indices': 'int8'})


    def plot_histograms(self, **kwargs): #  -> "MatplotlibRenderPlots"
        """ 
        num='RipplesRankOrderZscore'
        """
        from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots # plot_histogram #TODO 2024-01-02 12:41: - [ ] Is this where the Qt5 Import dependency Pickle complains about is coming from?

        print(f'.plot_histograms(..., kwargs: {kwargs})')
        layout = kwargs.pop('layout', 'none')
        fig = plt.figure(layout=layout, **kwargs) # layout="constrained", 
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

    # def get_best_direction_raw_stats_values(self):
    #     """ 
    #     rank_order_results.ripple_most_likely_result_tuple
        
    #     """
    #     active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score = self.masked_z_score_values_list # unpack z_score_values
    #     long_best_direction_indicies = deepcopy(self.directional_likelihoods_tuple.long_best_direction_indices)
    #     short_best_direction_indicies = deepcopy(self.directional_likelihoods_tuple.short_best_direction_indices)

    #     np.shape(active_LR_ripple_long_z_score)
    #     np.shape(active_RL_ripple_long_z_score)

    #     np.shape(active_LR_ripple_short_z_score)
    #     np.shape(active_RL_ripple_short_z_score)

    #     np.shape(long_best_direction_indicies)
    #     np.shape(short_best_direction_indicies)

    #     ripple_evts_long_best_dir_raw_stats_values = np.where(long_best_direction_indicies, active_LR_ripple_long_z_score, active_RL_ripple_long_z_score)
    #     ripple_evts_short_best_dir_raw_stats_values = np.where(short_best_direction_indicies, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score)

    #     ripple_evts_long_best_dir_raw_stats_values

    #     np.shape(ripple_evts_long_best_dir_raw_stats_values)
    #     np.shape(ripple_evts_short_best_dir_raw_stats_values)


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes (_mapping and _keys_at_init). Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self._asdict().copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self = DirectionalRankOrderResult(**state)
        # self.__dict__.update(state)


@define(slots=False, repr=False, eq=False)
class RankOrderComputationsContainer(ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even


    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderComputationsContainer, RankOrderResult

        odd_ripple_rank_order_result = RankOrderResult.init_from_analysis_output_tuple(odd_ripple_outputs)
        even_ripple_rank_order_result = RankOrderResult.init_from_analysis_output_tuple(even_ripple_outputs)
        curr_active_pipeline.global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(odd_ripple=odd_ripple_rank_order_result, even_ripple=even_ripple_rank_order_result, odd_laps=odd_laps_rank_order_result, even_laps=even_laps_rank_order_result)

    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
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


    def adding_active_aclus_info(self):
        """
        Updates the `epochs_df` of each of its members.
        
        # ['LR_Long_ActuallyIncludedAclus', 'RL_Long_ActuallyIncludedAclus', 'LR_Short_ActuallyIncludedAclus', 'RL_Short_ActuallyIncludedAclus']
        """
        if self.LR_laps is not None:
            self.LR_laps.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.LR_laps.epochs_df, is_laps=True) # self.LR_laps is none
        if self.RL_laps is not None:
            self.RL_laps.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.RL_laps.epochs_df, is_laps=True)
        if self.LR_ripple is not None:
            self.LR_ripple.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.LR_ripple.epochs_df, is_laps=False)
        if self.RL_ripple is not None:
            self.RL_ripple.epochs_df = self.add_active_aclus_info(self, active_epochs_df=self.RL_ripple.epochs_df, is_laps=False)


    @classmethod
    def add_active_aclus_info(cls, rank_order_results, active_epochs_df: pd.DataFrame, is_laps: bool = False):
        """ adds the columns about the number of cells in each epoch to the epochs_df """
        label_column_type = RankOrderAnalyses._label_column_type
    
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


    @property
    def ripple_merged_complete_epoch_stats_df(self) -> pd.DataFrame:
        """ builds a single complete combined DataFrame for the ripples epochs, with all of the stats columns computed in various places. 
        
        # NOTE: `active_replay_epochs_df` has the correct label column
        
        Combines: [active_replay_epochs_df, directional_likelihoods_df, ripple_combined_epoch_stats_df]
        """
        ## All three DataFrames are the same number of rows, each with one row corresponding to an Epoch:
        active_replay_epochs_df = deepcopy(self.LR_ripple.epochs_df)
        # Change column type to int8 for columns: 'long_best_direction_indices', 'short_best_direction_indices'
        # directional_likelihoods_df = pd.DataFrame.from_dict(ripple_result_tuple.directional_likelihoods_tuple._asdict()).astype({'long_best_direction_indices': 'int8', 'short_best_direction_indices': 'int8'})
        directional_likelihoods_df = deepcopy(self.ripple_most_likely_result_tuple.directional_likelihoods_df)
        # 2023-12-15 - Newest method:
        ripple_combined_epoch_stats_df = deepcopy(self.ripple_combined_epoch_stats_df)
        # Concatenate the three DataFrames along the columns axis:
        # Assert that all DataFrames have the same number of rows:
        assert len(active_replay_epochs_df) == len(directional_likelihoods_df) == len(ripple_combined_epoch_stats_df), "DataFrames have different numbers of rows."
        # Assert that all DataFrames have at least one row:
        assert len(active_replay_epochs_df) > 0, "active_replay_epochs_df is empty."
        assert len(directional_likelihoods_df) > 0, "directional_likelihoods_df is empty."
        assert len(ripple_combined_epoch_stats_df) > 0, "ripple_combined_epoch_stats_df is empty."
        merged_complete_epoch_stats_df: pd.DataFrame = pd.concat([active_replay_epochs_df.reset_index(drop=True, inplace=False), directional_likelihoods_df.reset_index(drop=True, inplace=False), ripple_combined_epoch_stats_df.reset_index(drop=True, inplace=False)], axis=1)
        merged_complete_epoch_stats_df = merged_complete_epoch_stats_df.set_index(active_replay_epochs_df.index, inplace=False)
        merged_complete_epoch_stats_df = merged_complete_epoch_stats_df.loc[:, ~merged_complete_epoch_stats_df.columns.duplicated()] # drop duplicated 'label' column
        return merged_complete_epoch_stats_df
    

    @property
    def laps_merged_complete_epoch_stats_df(self) -> pd.DataFrame:
        """ builds a single complete combined DataFrame for the laps epochs, with all of the stats columns computed in various places. 
        
        Combines: [active_laps_epochs_df, directional_likelihoods_df, laps_combined_epoch_stats_df]
        """
        ## All three DataFrames are the same number of rows, each with one row corresponding to an Epoch:
        active_laps_epochs_df = deepcopy(self.LR_laps.epochs_df)
        # Change column type to int8 for columns: 'long_best_direction_indices', 'short_best_direction_indices'
        # directional_likelihoods_df = pd.DataFrame.from_dict(laps_result_tuple.directional_likelihoods_tuple._asdict()).astype({'long_best_direction_indices': 'int8', 'short_best_direction_indices': 'int8'})
        directional_likelihoods_df = deepcopy(self.laps_most_likely_result_tuple.directional_likelihoods_df)
        # 2023-12-15 - Newest method:
        laps_combined_epoch_stats_df = deepcopy(self.laps_combined_epoch_stats_df)
        # Concatenate the three DataFrames along the columns axis:
        # Assert that all DataFrames have the same number of rows:
        assert len(active_laps_epochs_df) == len(directional_likelihoods_df) == len(laps_combined_epoch_stats_df), "DataFrames have different numbers of rows."
        # Assert that all DataFrames have at least one row:
        assert len(active_laps_epochs_df) > 0, "active_laps_epochs_df is empty."
        assert len(directional_likelihoods_df) > 0, "directional_likelihoods_df is empty."
        assert len(laps_combined_epoch_stats_df) > 0, "laps_combined_epoch_stats_df is empty."
        merged_complete_epoch_stats_df: pd.DataFrame = pd.concat([active_laps_epochs_df.reset_index(drop=True, inplace=False), directional_likelihoods_df.reset_index(drop=True, inplace=False), laps_combined_epoch_stats_df.reset_index(drop=True, inplace=False)], axis=1)
        merged_complete_epoch_stats_df = merged_complete_epoch_stats_df.set_index(active_laps_epochs_df.index, inplace=False)
        merged_complete_epoch_stats_df = merged_complete_epoch_stats_df.loc[:, ~merged_complete_epoch_stats_df.columns.duplicated()] # drop duplicated 'label' column
        return merged_complete_epoch_stats_df
    

    def get_significant_ripple_merged_complete_epoch_stats_df(self, quantile_significance_threshold: float = 0.95) -> pd.DataFrame:
        """ 
        Gets the events above a certain significance threshold
        
        """
        ripple_combined_epoch_stats_df = deepcopy(self.merged_complete_epoch_stats_df)

        # Filter rows based on columns: 'Long_BestDir_quantile', 'Short_BestDir_quantile'
        quantile_significance_threshold: float = 0.95
        significant_ripple_combined_epoch_stats_df = ripple_combined_epoch_stats_df[(ripple_combined_epoch_stats_df['Long_BestDir_quantile'] > quantile_significance_threshold) | (ripple_combined_epoch_stats_df['Short_BestDir_quantile'] > quantile_significance_threshold)]
        # significant_ripple_combined_epoch_stats_df
        is_epoch_significant = np.isin(ripple_combined_epoch_stats_df.index, significant_ripple_combined_epoch_stats_df.index)
        active_replay_epochs_df = self.LR_ripple.epochs_df
        significant_ripple_epochs: Epoch = Epoch(deepcopy(active_replay_epochs_df).epochs.get_valid_df()).boolean_indicies_slice(is_epoch_significant)
        epoch_identifiers = significant_ripple_epochs._df.label.astype({'label': RankOrderAnalyses._label_column_type}).values #.labels
        x_values = significant_ripple_epochs.midtimes
        x_axis_name_suffix = 'Mid-time (Sec)'

        significant_ripple_epochs_df = significant_ripple_epochs.to_dataframe()
        significant_ripple_epochs_df

        significant_ripple_combined_epoch_stats_df['midtimes'] = significant_ripple_epochs.midtimes
        return significant_ripple_combined_epoch_stats_df






    # Utility Methods ____________________________________________________________________________________________________ #

    def to_dict(self) -> Dict:
        # return asdict(self, filter=attrs.filters.exclude((self.__attrs_attrs__.is_global))) #  'is_global'
        return {k:v for k, v in self.__dict__.items() if k not in ['is_global']}
    


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
    # _NaN_Type = np.nan
    _NaN_Type = pd.NA

    # _label_column_type: str = 'int'
    _label_column_type: str = 'int64'


    # Plotting/Figure Helper Functions ___________________________________________________________________________________ #
    def _subfn_perform_common_build_plot(title_str: str, plot_title: str='', left='Long-Short Z-Score Diff', variable_name='Lap', x_axis_name_suffix='Index', active_display_context=None, show=True):
        """ plots the z-score differences
                
        
        title_str: str = f"Rank Order {variable_name}s Epoch Debugger"
                
        Usage:
            app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _subfn_perform_common_build_plot(deepcopy(global_laps).lap_id, RL_laps_long_short_z_score_diff_values, LR_laps_long_short_z_score_diff_values)

        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from neuropy.utils.matplotlib_helpers import FormattedFigureText
        
        if active_display_context is not None:
            raw_sub_context = active_display_context.adding_context('subplot', subplot_name='raw')
            active_display_sub_context = raw_sub_context
            text_formatter = FormattedFigureText()
            active_footer_string = text_formatter._build_footer_string(active_context=active_display_sub_context)
        else:
            active_display_sub_context = None
            active_footer_string = None
            
        app = pg.mkQApp(title_str)
        win = pg.GraphicsLayoutWidget(show=show, title=title_str)
        win.setWindowTitle(title_str)

        header_label = pg.LabelItem(justify='left')
        header_label.setText('')
        win.addItem(header_label, row=1, col=0, colspan=2)
        
        p1: pg.PlotItem = win.addPlot(row=2, col=0, colspan=2, title=plot_title, left=left, bottom=f'{variable_name} {x_axis_name_suffix}', hoverable=True) # PlotItem

        # Add footer label at the bottom of the window
        footer_label = pg.LabelItem(justify='left')
        footer_label.setText(active_footer_string)
        win.addItem(footer_label, row=3, col=0, colspan=2)
        
        return app, win, p1, (header_label, footer_label)


    def _perform_plot_z_score_raw(epoch_idx_list, LR_long_z_score_values, RL_long_z_score_values, LR_short_z_score_values, RL_short_z_score_values, variable_name='Lap', x_axis_name_suffix='Index', point_data_values=None, active_display_context=None, show=True):
        """ plots the raw z-scores for each of the four templates

        Usage:
            app, win, p1, (long_even_out_plot_1D, long_odd_out_plot_1D, short_even_out_plot_1D, short_odd_out_plot_1D) = _perform_plot_z_score_raw(deepcopy(global_laps).lap_id, odd_laps_long_z_score_values, odd_laps_short_z_score_values, even_laps_long_z_score_values, even_laps_short_z_score_values)

        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from neuropy.utils.matplotlib_helpers import FormattedFigureText
        
        if active_display_context is not None:
            raw_sub_context = active_display_context.adding_context('subplot', subplot_name='raw')
            active_display_sub_context = raw_sub_context
            text_formatter = FormattedFigureText()
            active_footer_string = text_formatter._build_footer_string(active_context=active_display_sub_context)

        else:
            active_display_sub_context = None
            active_footer_string = None
            
        app, win, p1, (header_label, footer_label) = RankOrderAnalyses._subfn_perform_common_build_plot(title_str=f"Rank Order {variable_name}s Long-Short ZScore (Raw)",
                                                                                plot_title=f'Rank-Order Long-Short ZScore (Raw) for {variable_name}s over time', left='Z-Score (Raw)', variable_name=variable_name, x_axis_name_suffix=x_axis_name_suffix,
                                                                                active_display_context=active_display_context, show=show)
        
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


        if active_footer_string is not None:
            header_label.setText('')
            footer_label.setText(active_footer_string)

        return app, win, p1, (long_LR_out_plot_1D, long_RL_out_plot_1D, short_LR_out_plot_1D, short_RL_out_plot_1D), (header_label, footer_label), active_display_sub_context



    def _perform_plot_z_score_diff(epoch_idx_list, RL_laps_long_short_z_score_diff_values, LR_laps_long_short_z_score_diff_values, variable_name='Lap', x_axis_name_suffix='Index', point_data_values=None, include_marginal_histogram:bool=False, active_display_context=None, show=True):
        """ plots the z-score differences
        Usage:
            app, win, p1, (even_out_plot_1D, odd_out_plot_1D) = _perform_plot_z_score_diff(deepcopy(global_laps).lap_id, RL_laps_long_short_z_score_diff_values, LR_laps_long_short_z_score_diff_values)
        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg
        from neuropy.utils.matplotlib_helpers import FormattedFigureText
        
        if active_display_context is not None:
            z_score_diff_sub_context = active_display_context.adding_context('subplot', subplot_name='z_score_diff')
            active_display_sub_context = z_score_diff_sub_context
            text_formatter = FormattedFigureText()
            active_footer_string = text_formatter._build_footer_string(active_context=active_display_sub_context)

        else:
            active_display_sub_context = None
            active_footer_string = None
            
        # app, win, p1, (header_label, footer_label) = RankOrderAnalyses._subfn_perform_common_build_plot(title_str=f"Rank Order {variable_name}s Long-Short ZScore (Raw)",
        #                                                                         plot_title=f'Rank-Order Long-Short ZScore Diff for {variable_name}s over time', left='Long-Short Z-Score Diff', variable_name=variable_name, x_axis_name_suffix=x_axis_name_suffix)

        title_str = f"Rank Order {variable_name}s Long-Short ZScore (Raw)"
        plot_title = f'Rank-Order Long-Short ZScore Diff for {variable_name}s over time'
        left = 'Long-Short Z-Score Diff'
        
        app = pg.mkQApp(title_str)
        win = pg.GraphicsLayoutWidget(show=show, title=title_str)
        win.setWindowTitle(title_str)

        header_label = pg.LabelItem(justify='left')
        header_label.setText('')
        win.addItem(header_label, row=1, col=0, colspan=2)
        
        p1: pg.PlotItem = win.addPlot(row=2, col=0, colspan=2, title=plot_title, left=left, bottom=f'{variable_name} {x_axis_name_suffix}', hoverable=True) # PlotItem
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

        
        # 'orange'
        # symbolPen = 'w'
        symbolPen = None

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


        ## Add marginal histogram to the right of the main plot here:
        if include_marginal_histogram:
            ## Add as another row:
            py: pg.PlotItem = win.addPlot(row=3, col=0, colspan=2, right='Marginal Long-Short Z-Score Diff', hoverable=True) # , bottom=f'{variable_name} {x_axis_name_suffix}', title=f'Marginal Rank-Order Long-Short ZScore Diff for {variable_name}'
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

            footer_label_row = 4
        else:
            py = None
            footer_label_row = 3

        # Add footer label at the bottom of the window
        footer_label = pg.LabelItem(justify='left')
        footer_label.setText(active_footer_string or 'Your Footer Label Text')
        win.addItem(footer_label, row=footer_label_row, col=0, colspan=2)

        if active_footer_string is not None:
            header_label.setText('')
            footer_label.setText(active_footer_string)


        return app, win, p1, (even_out_plot_1D, odd_out_plot_1D), (py, ), (header_label, footer_label), active_display_sub_context




    # Computation Helpers ________________________________________________________________________________________________ #

    @classmethod
    def common_analysis_helper(cls, curr_active_pipeline, num_shuffles:int=300, minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,9]):
        ## Shared:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

        directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']

        if included_qclu_values is not None:
            qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=included_qclu_values)
            modified_directional_laps_results: DirectionalLapsResult = directional_laps_results.filtered_by_included_aclus(qclu_included_aclus) 
            # 2023-12-21 - Need to update the core directional_
            print(f"ERROR! BUG?! 2023-12-21 - Need to update the previous 'DirectionalLaps' object with the qclu-restricted one right? on filter")
            # curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] = modified_directional_laps_results ## Need to update the previous 'DirectionalLaps' object with the qclu-restricted one right?
            active_directional_laps_results = modified_directional_laps_results
        else:
            active_directional_laps_results = directional_laps_results


        # non-shared templates:
        non_shared_templates: TrackTemplates = active_directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) #.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
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
        return global_spikes_df, (odd_shuffle_helper, even_shuffle_helper), active_directional_laps_results



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
        filtered_active_epochs_df['label'] = filtered_active_epochs_df['label'].astype(cls._label_column_type)

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
    def select_and_rank_spikes(cls, active_spikes_df: pd.DataFrame, active_aclu_to_fragile_linear_neuron_IDX_dict, rank_alignment: str, time_variable_name_override: Optional[str]=None, min_num_unique_aclu_inclusions: int=5):
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

        ## After selection, drop any epochs that are less than min_num_unique_aclu_inclusions
        initial_selected_spikes = np.shape(selected_spikes)[0]
        initial_selected_spikes_only_df = np.shape(selected_spikes_only_df)[0]
        
        ## #TODO 2024-01-09 04:40: - [ ] Remove under-powered laps
        active_selected_spikes_df: pd.DataFrame = deepcopy(selected_spikes_only_df[['t_rel_seconds', 'aclu', 'Probe_Epoch_id']]).sort_values(['Probe_Epoch_id', 't_rel_seconds', 'aclu']).astype({'Probe_Epoch_id': 'int'}) # Sort by columns: 'Probe_Epoch_id' (ascending), 't_rel_seconds' (ascending), 'aclu' (ascending)
        active_num_unique_aclus_df = active_selected_spikes_df.groupby(['Probe_Epoch_id']).agg(aclu_count=('aclu', 'count')).reset_index()
        active_num_unique_aclus_df = active_num_unique_aclus_df[active_num_unique_aclus_df['aclu_count'] >= min_num_unique_aclu_inclusions] # Filter rows based on column: 'aclu_count'
        final_good_Probe_Epoch_ids: NDArray = active_num_unique_aclus_df.Probe_Epoch_id.unique()
        

        ## Drop the entries in active_selected_spikes_df that have Probe_Epoch_id correspodnding to the dropped epochs
        selected_spikes_only_df = selected_spikes_only_df[np.isin(selected_spikes_only_df['Probe_Epoch_id'], final_good_Probe_Epoch_ids)]
        selected_spikes = selected_spikes[np.isin(deepcopy(selected_spikes).reset_index()['Probe_Epoch_id'], final_good_Probe_Epoch_ids)]

        # Drop the bad ones.
        final_num_selected_spikes = np.shape(selected_spikes)[0]
        final_num_selected_spikes_only_df = np.shape(selected_spikes_only_df)[0]
        
        # num_dropped = final_num_selected_spikes - initial_selected_spikes
        num_dropped = final_num_selected_spikes_only_df - initial_selected_spikes_only_df
        if num_dropped > 0:
            # print(f'num_dropped: {num_dropped} = final_num_selected_spikes: {final_num_selected_spikes} - initial_selected_spikes: {initial_selected_spikes}') 
            print(f'num_dropped: {num_dropped} = final_num_selected_spikes_only_df: {final_num_selected_spikes_only_df} - initial_selected_spikes_only_df: {initial_selected_spikes_only_df}') 


       

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

        return epoch_ranked_aclus_dict, epoch_ranked_fragile_linear_neuron_IDX_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, selected_spikes_only_df, final_good_Probe_Epoch_ids



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
        active_epochs['label'] = active_epochs['label'].astype(cls._label_column_type)

        # Preprocess the spikes DataFrame
        (shared_aclus_only_neuron_IDs, is_good_aclus, shuffled_aclus, shuffle_IDXs, long_pf_peak_ranks, short_pf_peak_ranks, active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, filtered_active_epochs) = cls.preprocess_spikes_df(active_spikes_df, active_epochs, shuffle_helper, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions)

        # TODO 2023-11-21 05:42: - [ ] todo_description want the aclus as well, not just the `long_pf_peak_ranks`

        #TODO 2023-12-08 12:53: - [X] Drop epochs with fewer than the minimum active aclus

        #TODO 2023-12-10 19:40: - [ ] Need to save the epochs that were used to compute.

        # Select and rank spikes
        epoch_ranked_aclus_dict, epoch_ranked_fragile_linear_neuron_IDX_dict, epoch_selected_spikes_fragile_linear_neuron_IDX_dict, selected_spikes_only_df, final_good_Probe_Epoch_ids = cls.select_and_rank_spikes(active_spikes_df, active_aclu_to_fragile_linear_neuron_IDX_dict, rank_alignment, min_num_unique_aclu_inclusions=min_num_unique_aclu_inclusions)

        ## Drop the entries in active_selected_spikes_df that have Probe_Epoch_id correspodnding to the dropped epochs
        # active_spikes_df: pd.DataFrame = active_spikes_df.copy()
        active_spikes_df = active_spikes_df[np.isin(active_spikes_df['Probe_Epoch_id'], final_good_Probe_Epoch_ids)]

        # active_epochs_df: pd.DataFrame = filtered_active_epochs.copy()
        filtered_active_epochs = filtered_active_epochs[np.isin(filtered_active_epochs['label'], final_good_Probe_Epoch_ids)]

        
        ## OUTPUT DICTS:
        # create a nested dictionary of {Probe_Epoch_id: {aclu: rank}} from the ranked_aclu values
        output_dict = {}

        ## Loop over the results now to do the actual stats:
        epoch_ranked_aclus_stats_dict = {}

        _omitted_epoch_ids = []
        

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


            ## ONLY one aclu remaining after chopping results in erronious spearman

            #TODO 2023-11-22 11:30: - [ ] Does chopping the template down vs. leaving those entries in there change the spearman?

            # long_pf_peak_ranks, short_pf_peak_ranks
            assert np.shape(long_pf_peak_ranks) == np.shape(shared_aclus_only_neuron_IDs)
            assert np.shape(short_pf_peak_ranks) == np.shape(shared_aclus_only_neuron_IDs)

            # Chop the other direction:
            is_template_aclu_actually_active_in_epoch: NDArray = np.isin(template_aclus, actually_included_epoch_aclus) # a bool array indicating whether each aclu in the template is active in  in the epoch (spikes_df). Used for indexing into the template peak_ranks (`long_pf_peak_ranks`, `short_pf_peak_ranks`)
            template_epoch_actually_included_aclus: NDArray = np.array(template_aclus)[is_template_aclu_actually_active_in_epoch] ## `actually_included_template_aclus`: the final aclus for this template actually active in this epoch

            epoch_active_long_pf_peak_ranks = np.array(long_pf_peak_ranks)[is_template_aclu_actually_active_in_epoch] ## occurs when `epoch_active_long_pf_peak_ranks` only has one entry
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


            # detect invalid epochs and add them to `_omitted_epoch_ids`
            is_invalid_epoch: bool = np.isnan(real_short_result_corr_value) or np.isnan(real_long_result_corr_value)
            
            if (not is_invalid_epoch):
                
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

                long_stats_z_scorer = Zscorer.init_from_values(long_stats_corr_values, real_long_result_corr_value, real_long_rank_stats.pvalue) # long_stats_corr_values are all NaN, real_long_result_corr_value is NaN, 
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
                # epoch_ranked_aclus_stats_dict[epoch_id] = LongShortStatsItem(long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff, long_short_naive_z_diff, is_forward_replay)
                epoch_ranked_aclus_stats_dict[epoch_id] = LongShortStatsItem(long_stats_z_scorer=long_stats_z_scorer, short_stats_z_scorer=short_stats_z_scorer, long_short_z_diff=long_short_z_diff, long_short_naive_z_diff=long_short_naive_z_diff, is_forward_replay=is_forward_replay)

            else:
                ## invalid epoch
                print(f'WARNING: invalid epoch {epoch_id}')
                _omitted_epoch_ids.append(epoch_id)
                
        ## END for epoch_id

        # Extract the results:
        long_z_score_values = []
        short_z_score_values = []
        long_short_z_score_diff_values = []
        long_short_z_score_diff_values = []

        for epoch_id, epoch_stats in epoch_ranked_aclus_stats_dict.items():
            long_stats_z_scorer, short_stats_z_scorer, long_short_z_diff = epoch_stats.long_stats_z_scorer, epoch_stats.short_stats_z_scorer, epoch_stats.long_short_z_diff 
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
    required_min_percentage_of_active_cells: float = 0.2 # 20% of active cells


    @classmethod
    @function_attributes(short_name=None, tags=['subfn', 'rank-order', 'merged_decoder', 'merged_pseduo2D_decoder', 'active', 'directional'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-04 12:47', related_items=[])
    def epoch_directionality_merged_pseduo2D_decoder_evidence(cls, decoders_dict, ripple_marginals, ripple_directional_likelihoods_tuple, combined_best_direction_indicies, epochs_df: pd.DataFrame):
        """ 2024-01-04 - Replay Direction Decoder-based Classification
        Used to classify replays as LR/RL
        
        """
        ## Not needed, but could add more columns in future:
        # ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir = ripple_marginals
        # long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = ripple_directional_likelihoods_tuple
        
        assert np.shape(epochs_df)[0] == np.shape(combined_best_direction_indicies)[0]
        epochs_df['combined_best_direction_indicies'] = combined_best_direction_indicies.astype('int8')

        return epochs_df



    @classmethod
    def percentiles_computations(cls, rank_order_results):
        """ 2023-12-21 - Computes Quantiles/Percentiles - Computing Spearman Percentiles as an alternative to the Z-score from shuffling, which does not seem to work for small numbers of active cells in an event:
        
        Uses:
        
        rank_order_results.ripple_combined_epoch_stats_df


        Modifies:
            `ripple_combined_epoch_stats_df`:
                ripple_combined_epoch_stats_df['LongShort_BestDir_quantile_diff']
        
        
        Usage:
        
        _perform_compute_quantiles(rank_order_results)
        rank_order_results.ripple_combined_epoch_stats_df

        """

        def compute_percentile(real_value, original_shuffle_values):
            return (1.0 - float(np.sum((np.abs(real_value) < original_shuffle_values)))/float(len(original_shuffle_values)))
            # return (1.0 - float(np.sum((real_value < original_shuffle_values)))/float(len(original_shuffle_values)))

        def compute_percentiles_from_shuffle_results(combined_variable_names, valid_stacked_arrays, real_stacked_arrays):
            """ computes all of the percentiles from the columns of the datafrrame

            Usage:	
                # # From new tuple:
                output_active_epoch_computed_values, combined_variable_names, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles = rank_order_results.laps_new_output_tuple      
                results_quantile_value_laps = compute_percentiles_from_shuffle_results(combined_variable_names, valid_stacked_arrays, real_stacked_arrays)
                results_quantile_value_laps
                
            Returns:


            """
            decoder_name_to_column_name_prefix_map = RankOrderAnalyses._subfn_build_pandas_df_based_correlation_computations_column_rename_dict(combined_variable_names)
            # print(f'decoder_name_to_column_name_prefix_map: {decoder_name_to_column_name_prefix_map}')

            ## Extract the stats values for each shuffle from `valid_stacked_arrays`:
            n_epochs = np.shape(real_stacked_arrays)[0]
            n_variables = np.shape(real_stacked_arrays)[1]
            assert n_variables == len(combined_variable_names)


            n_valid_shuffles = np.shape(valid_stacked_arrays)[0]

            quantile_result_column_suffix: str = 'percentile'
            quantile_result_column_names = [f'{decoder_name_to_column_name_prefix_map[a_column_name]}_{quantile_result_column_suffix}' for a_column_name in combined_variable_names]
            # print(f'quantile_result_column_names: {quantile_result_column_names}') # quantile_result_column_names: ['LR_Short_spearman_percentile', 'LR_Short_pearson_percentile', 'RL_Short_spearman_percentile', 'LR_Long_spearman_percentile', 'RL_Long_spearman_percentile', 'LR_Long_pearson_percentile', 'RL_Long_pearson_percentile', 'RL_Short_pearson_percentile']
            

            # recover from the valid stacked rarys valid_stacked_arrays
            results_quantile_value = {}
            for variable_IDX, a_column_name in enumerate(combined_variable_names):
                # Do one variable at a time, there's approximately 8, 
                # print(f'valid_stacked_arrays.shape: {valid_stacked_arrays.shape}') # valid_stacked_arrays.shape: (n_shuffles, n_epochs, n_variables)	
                ## Extract the stats values for each shuffle from `valid_stacked_arrays`:
                # n_epochs: int = np.shape(real_values)[0]
                assert n_epochs == np.shape(valid_stacked_arrays)[-2] # penultimate element
                assert n_variables == np.shape(valid_stacked_arrays)[-1]		
                a_result_column_name: str = quantile_result_column_names[variable_IDX] # column name with the suffix '_percentile' added to it
                if n_valid_shuffles > 0:
                    results_quantile_value[a_result_column_name] = np.array([compute_percentile(real_stacked_arrays[epoch_IDX, variable_IDX], np.squeeze(valid_stacked_arrays[:, epoch_IDX, variable_IDX])) for epoch_IDX in np.arange(n_epochs)]) # real_stacked_arrays based version
                # results_quantile_value[a_column_name] = np.array([compute_percentile(real_values[epoch_IDX], np.squeeze(valid_stacked_arrays[:, epoch_IDX, variable_IDX])) for epoch_IDX in np.arange(n_epochs)]) # working df-based version
                else:
                    print(f'WARNING: .percentiles_computations(...): no valid shuffles detected (valid_stacked_arrays is empty) so results_quantile_value will be an array of all NaNs')
                    results_quantile_value[a_result_column_name] = np.array([np.nan for epoch_IDX in np.arange(n_epochs)]) # ALL NANs

            # Add old columns for compatibility:
            for old_col_name, new_col_name in zip(['LR_Long_percentile', 'RL_Long_percentile', 'LR_Short_percentile', 'RL_Short_percentile'], ['LR_Long_pearson_percentile', 'RL_Long_pearson_percentile', 'LR_Short_pearson_percentile', 'RL_Short_pearson_percentile']):
                results_quantile_value[old_col_name] = results_quantile_value[new_col_name].copy()

            return results_quantile_value # Dict[str, NDArray]


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #


        # Ripples: ___________________________________________________________________________________________________________ #
        ripple_combined_epoch_stats_df = rank_order_results.ripple_combined_epoch_stats_df
        # prev method:
        # # `LongShortStatsItem` form (2024-01-02):        
        # new_LR_results_quantile_values = np.array([(compute_percentile(a_result_item.long_stats_z_scorer.real_value, a_result_item.long_stats_z_scorer.original_values), compute_percentile(a_result_item.short_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.original_values)) for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
        # new_RL_results_quantile_values = np.array([(compute_percentile(a_result_item.long_stats_z_scorer.real_value, a_result_item.long_stats_z_scorer.original_values), compute_percentile(a_result_item.short_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.original_values)) for epoch_id, a_result_item in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])
    
        ## 2023-12-23 Method:        
        output_active_epoch_computed_values, combined_variable_names, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles = rank_order_results.ripple_new_output_tuple        
        # recover from the valid stacked arrays: `valid_stacked_arrays`
        quantile_results_dict_ripple = compute_percentiles_from_shuffle_results(combined_variable_names, valid_stacked_arrays, real_stacked_arrays) # valid_stacked_arrays is empty throwing a division by zero error
        
        # new_LR_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(shuffled_results_output_dict['long_LR_spearman_Z'][0], shuffled_results_output_dict['short_LR_spearman_Z'][0])])
        # new_RL_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(shuffled_results_output_dict['short_LR_spearman_Z'][0], shuffled_results_output_dict['short_RL_spearman_Z'][0])])
        
        
        # new_LR_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(recovered_shuffle_results_dict['long_LR_pearson_Z'], shuffled_results_output_dict['short_LR_pearson_Z'])])
        # new_RL_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(recovered_shuffle_results_dict['short_LR_pearson_Z'], shuffled_results_output_dict['short_RL_pearson_Z'])])
        # quantile_results_dict = dict(zip(['LR_Long_percentile', 'LR_Short_percentile', 'RL_Long_percentile', 'RL_Short_percentile'], np.hstack((new_LR_results_quantile_values, new_RL_results_quantile_values)).T))
 
        ## Add the new columns into the `ripple_combined_epoch_stats_df`
        for a_col_name, col_vals in quantile_results_dict_ripple.items():
            ripple_combined_epoch_stats_df[a_col_name] = col_vals
            

        # `combined_best_direction_indicies` method:
        active_replay_epochs_df = deepcopy(rank_order_results.LR_ripple.epochs_df)
        assert 'combined_best_direction_indicies' in active_replay_epochs_df, f"active_replay_epochs_df needs 'combined_best_direction_indicies'"
        combined_best_direction_indicies = deepcopy(active_replay_epochs_df['combined_best_direction_indicies'])
        assert np.shape(combined_best_direction_indicies)[0] == np.shape(rank_order_results.ripple_combined_epoch_stats_df)[0]
        long_best_direction_indicies = combined_best_direction_indicies # use same (globally best) indicies for Long/Short
        short_best_direction_indicies = combined_best_direction_indicies # use same (globally best) indicies for Long/Short

        ripple_evts_long_best_dir_quantile_stats_values = np.where(long_best_direction_indicies, rank_order_results.ripple_combined_epoch_stats_df['LR_Long_percentile'].to_numpy(), rank_order_results.ripple_combined_epoch_stats_df['RL_Long_percentile'].to_numpy())
        ripple_evts_short_best_dir_quantile_stats_values = np.where(short_best_direction_indicies, rank_order_results.ripple_combined_epoch_stats_df['LR_Short_percentile'].to_numpy(), rank_order_results.ripple_combined_epoch_stats_df['RL_Short_percentile'].to_numpy())
        assert np.shape(ripple_evts_long_best_dir_quantile_stats_values) == np.shape(ripple_evts_short_best_dir_quantile_stats_values)
        rank_order_results.ripple_combined_epoch_stats_df['Long_BestDir_quantile'] = ripple_evts_long_best_dir_quantile_stats_values
        rank_order_results.ripple_combined_epoch_stats_df['Short_BestDir_quantile'] = ripple_evts_short_best_dir_quantile_stats_values

        ripple_combined_epoch_stats_df['LongShort_BestDir_quantile_diff'] = ripple_combined_epoch_stats_df['Long_BestDir_quantile'] - ripple_combined_epoch_stats_df['Short_BestDir_quantile']

        ## 2023-12-22 - Add the LR-LR, RL-RL differences
        ripple_combined_epoch_stats_df['LongShort_LR_quantile_diff'] = ripple_combined_epoch_stats_df['LR_Long_percentile'] - ripple_combined_epoch_stats_df['LR_Short_percentile']
        ripple_combined_epoch_stats_df['LongShort_RL_quantile_diff'] = ripple_combined_epoch_stats_df['RL_Long_percentile'] - ripple_combined_epoch_stats_df['RL_Short_percentile']


        # Laps: ______________________________________________________________________________________________________________ #
        laps_combined_epoch_stats_df = rank_order_results.laps_combined_epoch_stats_df
        
        # # `LongShortStatsItem` form (2024-01-02):        
        # new_LR_results_quantile_values = np.array([(compute_percentile(a_result_item.long_stats_z_scorer.real_value, a_result_item.long_stats_z_scorer.original_values), compute_percentile(a_result_item.short_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.original_values)) for epoch_id, a_result_item in rank_order_results.LR_laps.ranked_aclus_stats_dict.items()])
        # new_RL_results_quantile_values = np.array([(compute_percentile(a_result_item.long_stats_z_scorer.real_value, a_result_item.long_stats_z_scorer.original_values), compute_percentile(a_result_item.short_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.original_values)) for epoch_id, a_result_item in rank_order_results.RL_laps.ranked_aclus_stats_dict.items()])

        ## 2023-12-23 Method:        
        # recover from the valid stacked arrays: `valid_stacked_arrays`
        output_active_epoch_computed_values, combined_variable_names, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles = rank_order_results.laps_new_output_tuple
        assert (n_valid_shuffles > 0), f'ERR: n_valid_shuffles: {n_valid_shuffles} == 0!'

        quantile_results_dict_laps = compute_percentiles_from_shuffle_results(combined_variable_names, valid_stacked_arrays, real_stacked_arrays)
        
        # new_LR_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(shuffled_results_output_dict['long_LR_pearson_Z'][0], shuffled_results_output_dict['short_LR_pearson_Z'][0])])
        # new_RL_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(shuffled_results_output_dict['short_LR_pearson_Z'][0], shuffled_results_output_dict['short_RL_pearson_Z'][0])])
        # quantile_results_dict = dict(zip(['LR_Long_percentile', 'LR_Short_percentile', 'RL_Long_percentile', 'RL_Short_percentile'], np.hstack((new_LR_results_quantile_values, new_RL_results_quantile_values)).T))
        # quantile_results_df = pd.DataFrame(np.hstack((new_LR_results_real_values, new_RL_results_real_values)), columns=['LR_Long_percentile', 'LR_Short_percentile', 'RL_Long_percentile', 'RL_Short_percentile'])

        ## Add the new columns into the `laps_combined_epoch_stats_df`
        for a_col_name, col_vals in quantile_results_dict_laps.items():
            laps_combined_epoch_stats_df[a_col_name] = col_vals

        # `combined_best_direction_indicies` method:
        active_laps_epochs_df = deepcopy(rank_order_results.LR_laps.epochs_df)
        assert 'combined_best_direction_indicies' in active_laps_epochs_df, f"active_laps_epochs_df needs combined_best_direction_indicies"
        combined_best_direction_indicies = deepcopy(active_laps_epochs_df['combined_best_direction_indicies'])
        assert np.shape(combined_best_direction_indicies)[0] == np.shape(rank_order_results.laps_combined_epoch_stats_df)[0]
        long_best_direction_indicies = combined_best_direction_indicies # use same (globally best) indicies for Long/Short
        short_best_direction_indicies = combined_best_direction_indicies # use same (globally best) indicies for Long/Short

        laps_evts_long_best_dir_quantile_stats_values = np.where(long_best_direction_indicies, rank_order_results.laps_combined_epoch_stats_df['LR_Long_percentile'].to_numpy(), rank_order_results.laps_combined_epoch_stats_df['RL_Long_percentile'].to_numpy())
        laps_evts_short_best_dir_quantile_stats_values = np.where(short_best_direction_indicies, rank_order_results.laps_combined_epoch_stats_df['LR_Short_percentile'].to_numpy(), rank_order_results.laps_combined_epoch_stats_df['RL_Short_percentile'].to_numpy())
        assert np.shape(laps_evts_long_best_dir_quantile_stats_values) == np.shape(laps_evts_short_best_dir_quantile_stats_values)
        rank_order_results.laps_combined_epoch_stats_df['Long_BestDir_quantile'] = laps_evts_long_best_dir_quantile_stats_values
        rank_order_results.laps_combined_epoch_stats_df['Short_BestDir_quantile'] = laps_evts_short_best_dir_quantile_stats_values

        laps_combined_epoch_stats_df['LongShort_BestDir_quantile_diff'] = laps_combined_epoch_stats_df['Long_BestDir_quantile'] - laps_combined_epoch_stats_df['Short_BestDir_quantile']

        ## 2023-12-22 - Add the LR-LR, RL-RL differences
        laps_combined_epoch_stats_df['LongShort_LR_quantile_diff'] = laps_combined_epoch_stats_df['LR_Long_percentile'] - laps_combined_epoch_stats_df['LR_Short_percentile']
        laps_combined_epoch_stats_df['LongShort_RL_quantile_diff'] = laps_combined_epoch_stats_df['RL_Long_percentile'] - laps_combined_epoch_stats_df['RL_Short_percentile']

        # return ripple_combined_epoch_stats_df


    @classmethod
    @function_attributes(short_name=None, tags=['rank-order', 'shuffle', 'inst_fr', 'epoch', 'lap', 'replay', 'computation'], input_requires=[], output_provides=[], uses=['DirectionalRankOrderLikelihoods', 'DirectionalRankOrderResult', 'cls._compute_best'], used_by=[], creation_date='2023-11-16 18:43', related_items=['plot_rank_order_epoch_inst_fr_result_tuples'])
    def most_likely_directional_rank_order_shuffling(cls, curr_active_pipeline) -> Tuple[DirectionalRankOrderResult, DirectionalRankOrderResult]:
        """ A version of the rank-order shufffling for a set of epochs that tries to use the most-likely direction (independently (e.g. long might be LR and short could be RL)) as the one to decode with.

        Called once for the laps + ripples
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses

            ## Main
            ripple_result_tuple, laps_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(curr_active_pipeline)


        Reference:
            {"even": "RL", "odd": "LR"}
            [LR, RL], {'LR': 0, 'RL': 1}
            odd (LR) = 0, even (RL) = 1
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult

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

        # 2024-01-04 - Get the `directional_merged_decoders_result` to determining most-likely direction from the merged pseudo-2D decoder:
        directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        ## Replays:
        try:
            ## Post-process Z-scores with their most likely directions:

            ripple_combined_epoch_stats_df = rank_order_results.ripple_combined_epoch_stats_df
            active_replay_epochs_df = rank_order_results.LR_ripple.epochs_df

            active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score = ripple_combined_epoch_stats_df.LR_Long_spearman_Z, ripple_combined_epoch_stats_df.RL_Long_spearman_Z, ripple_combined_epoch_stats_df.LR_Short_spearman_Z, ripple_combined_epoch_stats_df.RL_Short_spearman_Z

            ## 2024-01-04 - DirectionalMergedDecoders version:
            # NOTE: ripple_most_likely_direction_from_decoder comes with with more epochs than the already filtered `rank_order_results.ripple_combined_epoch_stats_df` version. We'll get only the active indicies from `rank_order_results.ripple_combined_epoch_stats_df.index`
            # needs: rank_order_results, ripple_most_likely_direction_from_decoder, ripple_directional_all_epoch_bins_marginal, 
            ripple_marginals = DirectionalPseudo2DDecodersResult.determine_directional_likelihoods(directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result)
            ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir = ripple_marginals

            combined_best_direction_indicies = deepcopy(ripple_most_likely_direction_from_decoder) # .shape (611,)
            combined_best_direction_indicies = combined_best_direction_indicies[rank_order_results.ripple_combined_epoch_stats_df['label'].to_numpy()] # get only the indicies for the active epochs
            assert np.shape(combined_best_direction_indicies)[0] == np.shape(rank_order_results.ripple_combined_epoch_stats_df)[0]
            long_best_direction_indicies = combined_best_direction_indicies.copy() # use same (globally best) indicies for Long/Short
            short_best_direction_indicies = combined_best_direction_indicies.copy() # use same (globally best) indicies for Long/Short

            # gets the LR likelihood for each of these (long/short)
            long_relative_direction_likelihoods = ripple_directional_all_epoch_bins_marginal[rank_order_results.ripple_combined_epoch_stats_df['label'].to_numpy(), 0] # (n_epochs, 2)
            short_relative_direction_likelihoods = ripple_directional_all_epoch_bins_marginal[rank_order_results.ripple_combined_epoch_stats_df['label'].to_numpy(), 0] # (n_epochs, 2)

            ripple_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=long_relative_direction_likelihoods,
                                                                                            short_relative_direction_likelihoods=short_relative_direction_likelihoods,
                                                                                            long_best_direction_indices=long_best_direction_indicies, 
                                                                                            short_best_direction_indices=short_best_direction_indicies,
                                                                                            )
            long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = ripple_directional_likelihoods_tuple
            active_replay_epochs_df = cls.epoch_directionality_merged_pseduo2D_decoder_evidence(decoders_dict, ripple_marginals, ripple_directional_likelihoods_tuple, combined_best_direction_indicies, active_replay_epochs_df)


            ripple_evts_long_best_dir_z_score_values = np.where(long_best_direction_indicies, active_LR_ripple_long_z_score, active_RL_ripple_long_z_score)
            ripple_evts_short_best_dir_z_score_values = np.where(short_best_direction_indicies, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score)
            ripple_evts_long_short_best_dir_z_score_diff_values = ripple_evts_long_best_dir_z_score_values - ripple_evts_short_best_dir_z_score_values
            ripple_masked_z_score_values_list: List[ma.masked_array] = [ma.masked_array(x, mask=np.logical_not(a_mask)) for x, a_mask in zip((active_LR_ripple_long_z_score, active_RL_ripple_long_z_score, active_LR_ripple_short_z_score, active_RL_ripple_short_z_score),
                                                                                                                ((long_best_direction_indicies == _LR_INDEX), (long_best_direction_indicies == _RL_INDEX), (short_best_direction_indicies == _LR_INDEX), (short_best_direction_indicies == _RL_INDEX)))]

            # outputs: ripple_evts_long_short_best_dir_z_score_diff_values
            ripple_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(active_replay_epochs_df, long_best_dir_z_score_values=ripple_evts_long_best_dir_z_score_values, short_best_dir_z_score_values=ripple_evts_short_best_dir_z_score_values,
                                                                                        long_short_best_dir_z_score_diff_values=ripple_evts_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=ripple_directional_likelihoods_tuple,
                                                                                        masked_z_score_values_list=ripple_masked_z_score_values_list, rank_order_z_score_df=None)

            # re-assign:
            rank_order_results.LR_ripple.epochs_df = active_replay_epochs_df

            ## Get the raw spearman_rho values for the best-direction for both Long/Short:
            # Adds ['Long_BestDir_spearman', 'Short_BestDir_spearman']

            long_best_direction_indicies = deepcopy(ripple_result_tuple.directional_likelihoods_tuple.long_best_direction_indices)
            short_best_direction_indicies = deepcopy(ripple_result_tuple.directional_likelihoods_tuple.short_best_direction_indices)

            assert np.shape(active_LR_ripple_long_z_score) == np.shape(active_RL_ripple_long_z_score)
            assert np.shape(active_LR_ripple_short_z_score) == np.shape(active_RL_ripple_short_z_score)
            assert np.shape(long_best_direction_indicies) == np.shape(short_best_direction_indicies)

            ripple_evts_long_best_dir_raw_stats_values = np.where(long_best_direction_indicies, rank_order_results.ripple_combined_epoch_stats_df['LR_Long_spearman'].to_numpy(), rank_order_results.ripple_combined_epoch_stats_df['RL_Long_spearman'].to_numpy())
            ripple_evts_short_best_dir_raw_stats_values = np.where(short_best_direction_indicies, rank_order_results.ripple_combined_epoch_stats_df['LR_Short_spearman'].to_numpy(), rank_order_results.ripple_combined_epoch_stats_df['RL_Short_spearman'].to_numpy())
            assert np.shape(ripple_evts_long_best_dir_raw_stats_values) == np.shape(ripple_evts_short_best_dir_raw_stats_values)
            rank_order_results.ripple_combined_epoch_stats_df['Long_BestDir_spearman'] = ripple_evts_long_best_dir_raw_stats_values
            rank_order_results.ripple_combined_epoch_stats_df['Short_BestDir_spearman'] = ripple_evts_short_best_dir_raw_stats_values


        except (AttributeError, KeyError, IndexError, ValueError, ZeroDivisionError):
            raise # fail for ripples, but not for laps currently
            ripple_result_tuple = None


        ## Laps:
        try:
            # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            # global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
            # active_laps_epochs = global_laps
            # if ank_order_results.laps_combined_epoch_stats_df is not None:
            laps_combined_epoch_stats_df = deepcopy(rank_order_results.laps_combined_epoch_stats_df)
            active_laps_epochs_df = deepcopy(rank_order_results.LR_laps.epochs_df)
            
            active_LR_laps_long_z_score, active_RL_laps_long_z_score, active_LR_laps_short_z_score, active_RL_laps_short_z_score = laps_combined_epoch_stats_df.LR_Long_spearman_Z, laps_combined_epoch_stats_df.RL_Long_spearman_Z, laps_combined_epoch_stats_df.LR_Short_spearman_Z, laps_combined_epoch_stats_df.RL_Short_spearman_Z

            ## 2024-01-04 - DirectionalMergedDecoders version:
            # NOTE: laps_most_likely_direction_from_decoder comes with with more epochs than the already filtered `rank_order_results.laps_combined_epoch_stats_df` version. We'll get only the active indicies from `rank_order_results.ripple_combined_epoch_stats_df.index`
            # needs: rank_order_results, laps_most_likely_direction_from_decoder, laps_directional_all_epoch_bins_marginal, 
            laps_marginals = DirectionalPseudo2DDecodersResult.determine_directional_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)
            laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = laps_marginals

            combined_best_direction_indicies = deepcopy(laps_most_likely_direction_from_decoder) # .shape (611,)
            # np.shape(combined_best_direction_indicies)
            combined_best_direction_indicies = combined_best_direction_indicies[rank_order_results.laps_combined_epoch_stats_df['label'].to_numpy()] # get only the indicies for the active epochs
            # np.shape(combined_best_direction_indicies)
            assert np.shape(combined_best_direction_indicies)[0] == np.shape(rank_order_results.laps_combined_epoch_stats_df)[0]
            long_best_direction_indicies = combined_best_direction_indicies.copy() # use same (globally best) indicies for Long/Short
            short_best_direction_indicies = combined_best_direction_indicies.copy() # use same (globally best) indicies for Long/Short

            # gets the LR likelihood for each of these (long/short)
            long_relative_direction_likelihoods = laps_directional_all_epoch_bins_marginal[rank_order_results.laps_combined_epoch_stats_df['label'].to_numpy(), 0] # (n_epochs, 2)
            short_relative_direction_likelihoods = laps_directional_all_epoch_bins_marginal[rank_order_results.laps_combined_epoch_stats_df['label'].to_numpy(), 0] # (n_epochs, 2)

            laps_directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = DirectionalRankOrderLikelihoods(long_relative_direction_likelihoods=long_relative_direction_likelihoods,
                                                                                            short_relative_direction_likelihoods=short_relative_direction_likelihoods,
                                                                                            long_best_direction_indices=long_best_direction_indicies, 
                                                                                            short_best_direction_indices=short_best_direction_indicies,
                                                                                            )
            long_relative_direction_likelihoods, short_relative_direction_likelihoods, long_best_direction_indicies, short_best_direction_indicies = laps_directional_likelihoods_tuple.long_relative_direction_likelihoods, laps_directional_likelihoods_tuple.short_relative_direction_likelihoods, laps_directional_likelihoods_tuple.long_best_direction_indices, laps_directional_likelihoods_tuple.short_best_direction_indices
            # epoch_directionality_merged_pseduo2D_decoder_evidence update active_laps_epochs_df
            active_laps_epochs_df = cls.epoch_directionality_merged_pseduo2D_decoder_evidence(decoders_dict, laps_marginals, laps_directional_likelihoods_tuple, combined_best_direction_indicies, active_laps_epochs_df)

            # Using NumPy advanced indexing to select from array_a or array_b:
            laps_long_best_dir_z_score_values = np.where(long_best_direction_indicies, active_LR_laps_long_z_score, active_RL_laps_long_z_score)
            laps_short_best_dir_z_score_values = np.where(short_best_direction_indicies, active_LR_laps_short_z_score, active_RL_laps_short_z_score)
            # print(f'np.shape(laps_long_best_dir_z_score_values): {np.shape(laps_long_best_dir_z_score_values)}')
            laps_long_short_best_dir_z_score_diff_values = laps_long_best_dir_z_score_values - laps_short_best_dir_z_score_values
            # print(f'np.shape(laps_long_short_best_dir_z_score_diff_values): {np.shape(laps_long_short_best_dir_z_score_diff_values)}')
            #TODO 2023-11-20 22:02: - [ ] ERROR: CORRECTNESS FAULT: I think the two lists zipped over below are out of order.
            laps_masked_z_score_values_list: List[ma.masked_array] = [ma.masked_array(x, mask=np.logical_not(a_mask)) for x, a_mask in zip((active_LR_laps_long_z_score, active_RL_laps_long_z_score, active_LR_laps_short_z_score, active_RL_laps_short_z_score),
                                                                                                                ((long_best_direction_indicies == _LR_INDEX), (long_best_direction_indicies == _RL_INDEX), (short_best_direction_indicies == _LR_INDEX), (short_best_direction_indicies == _RL_INDEX)))]

            laps_result_tuple: DirectionalRankOrderResult = DirectionalRankOrderResult(active_laps_epochs_df, long_best_dir_z_score_values=laps_long_best_dir_z_score_values, short_best_dir_z_score_values=laps_short_best_dir_z_score_values,
                                                                                        long_short_best_dir_z_score_diff_values=laps_long_short_best_dir_z_score_diff_values, directional_likelihoods_tuple=laps_directional_likelihoods_tuple,
                                                                                        masked_z_score_values_list=laps_masked_z_score_values_list, rank_order_z_score_df=None)

            # Re-assign the updated dataframe:
            rank_order_results.LR_laps.epochs_df = active_laps_epochs_df
            

            assert np.shape(active_LR_laps_long_z_score) == np.shape(active_RL_laps_long_z_score)
            assert np.shape(active_LR_laps_short_z_score) == np.shape(active_RL_laps_short_z_score)
            assert np.shape(long_best_direction_indicies) == np.shape(short_best_direction_indicies)

            laps_evts_long_best_dir_raw_stats_values = np.where(long_best_direction_indicies, rank_order_results.laps_combined_epoch_stats_df['LR_Long_spearman'].to_numpy(), rank_order_results.laps_combined_epoch_stats_df['RL_Long_spearman'].to_numpy())
            laps_evts_short_best_dir_raw_stats_values = np.where(short_best_direction_indicies, rank_order_results.laps_combined_epoch_stats_df['LR_Short_spearman'].to_numpy(), rank_order_results.laps_combined_epoch_stats_df['RL_Short_spearman'].to_numpy())
            assert np.shape(laps_evts_long_best_dir_raw_stats_values) == np.shape(laps_evts_short_best_dir_raw_stats_values)
            rank_order_results.laps_combined_epoch_stats_df['Long_BestDir_spearman'] = laps_evts_long_best_dir_raw_stats_values
            rank_order_results.laps_combined_epoch_stats_df['Short_BestDir_spearman'] = laps_evts_short_best_dir_raw_stats_values
            
        except (AttributeError, KeyError, IndexError, ValueError, ZeroDivisionError) as e:
            # raise
            print(f"failed for laps with error e: {e} but skipping laps is allowed, so just passing")
            laps_result_tuple = None


        # Compute the quantiles:
        cls.percentiles_computations(rank_order_results=rank_order_results)
        
        return ripple_result_tuple, laps_result_tuple


    @function_attributes(short_name=None, tags=['rank-order', 'ripples', 'shuffle'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-01 20:20', related_items=[])
    @classmethod
    def main_ripples_analysis(cls, curr_active_pipeline, num_shuffles:int=300, rank_alignment='first', minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,9]):
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_paired_t_test

        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper), active_directional_laps_results = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

        ## Ripple Rank-Order Analysis: needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

        # track templates:
        track_templates: TrackTemplates = active_directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        ## Compute the dynamic minimum number of active cells from current num total cells and the `curr_active_pipeline.sess.config.preprocessing_parameters` values:`
        active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=cls.required_min_percentage_of_active_cells)
            
        global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
        if isinstance(global_replays, pd.DataFrame):
            ## why do we convert to Epoch before passing in? this doesn't make sense to me. Just to use the .get_valid_df()?
            global_replays = Epoch(global_replays.epochs.get_valid_df())

        ## Replay Epochs:
        LR_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(spikes_df), deepcopy(global_replays), odd_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, debug_print=False)
        RL_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(spikes_df), deepcopy(global_replays), even_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, debug_print=False)

        try:
            ripple_evts_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((LR_outputs.long_z_score, LR_outputs.short_z_score), (RL_outputs.long_z_score, RL_outputs.short_z_score))]
            print(f'ripple_evts_paired_tests: {ripple_evts_paired_tests}')
            # [TtestResult(statistic=3.5572800536164495, pvalue=0.0004179523066872734, df=415),
            #  TtestResult(statistic=3.809779392137816, pvalue=0.0001601254566506359, df=415)]
        except Exception as e:
            print(f'error in ripples paired t-test: {e}. Skipping.')
            # raise e
            ripple_evts_paired_tests = None

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
        from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_paired_t_test
        
        ## Shared:
        global_spikes_df, (odd_shuffle_helper, even_shuffle_helper), active_directional_laps_results = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=num_shuffles, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

        ## Laps Epochs: Needs `global_spikes_df`
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()

        # track templates:
        track_templates: TrackTemplates = active_directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        ## Compute the dynamic minimum number of active cells from current num total cells and the `curr_active_pipeline.sess.config.preprocessing_parameters` values:`
        active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=cls.required_min_percentage_of_active_cells)

        if not isinstance(global_laps, pd.DataFrame):
            global_laps_df = deepcopy(global_laps).to_dataframe()
            global_laps_df['label'] = global_laps_df['label'].astype(cls._label_column_type)

        # TODO: CenterOfMass for Laps instead of median spike
        # laps_rank_alignment = 'center_of_mass'
        LR_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(global_spikes_df), deepcopy(global_laps), odd_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, debug_print=False)
        RL_outputs = cls.compute_shuffled_rankorder_analyses(deepcopy(global_spikes_df), deepcopy(global_laps), even_shuffle_helper, rank_alignment=rank_alignment, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement, debug_print=False)
        laps_paired_tests = [pho_stats_paired_t_test(long_z_score_values, short_z_score_values) for long_z_score_values, short_z_score_values in zip((LR_outputs.long_z_score, LR_outputs.short_z_score), (RL_outputs.long_z_score, RL_outputs.short_z_score))]
        print(f'laps_paired_tests: {laps_paired_tests}')

        return (LR_outputs, RL_outputs, laps_paired_tests)

    @classmethod
    def validate_has_rank_order_results_quantiles(cls, curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
        """ Returns True if the pipeline has a valid RankOrder results set of the latest version

        TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

        """
        from neuropy.utils.indexing_helpers import PandasHelpers

        # Unpacking:
        rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
        # ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple

        # 2023-12-15 - Newest method:
        ripple_combined_epoch_stats_df = rank_order_results.ripple_combined_epoch_stats_df
        if ripple_combined_epoch_stats_df is None:
            return False

        if np.isnan(rank_order_results.ripple_combined_epoch_stats_df.index).any():
            return False # can't have dataframe index that is missing values.

        if ('LongShort_BestDir_quantile_diff' not in ripple_combined_epoch_stats_df):
            return False

        laps_combined_epoch_stats_df = rank_order_results.laps_combined_epoch_stats_df
        if laps_combined_epoch_stats_df is None:
            return False

        # missing: 'combined_best_direction_indicies'
        # rank_order_results.c
        # global_computation_results.computed_data['RankOrder'].ripple_most_likely_result_tuple, global_computation_results.computed_data['RankOrder'].laps_most_likely_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(owning_pipeline_reference)

        shared_required_cols = ['Long_BestDir_quantile', 'Short_BestDir_quantile', 'LR_Long_pearson_percentile', 'LR_Short_percentile', 'RL_Long_percentile', 'RL_Short_percentile']
        ripple_required_cols = shared_required_cols # + ['combined_best_direction_indicies'] #TODO 2024-05-08 08:20: - [ ] Disabled checking for 'combined_best_direction_indicies', which is missing from both the laps and ripple dataframes after fresh computation for some reason.
        print(f"WARNING: FIXME TODO 2024-05-08 08:20: - [ ] Disabled checking for 'combined_best_direction_indicies', which is missing from both the laps and ripple dataframes after fresh computation for some reason.")
        required_cols_dict = dict(laps=shared_required_cols, ripple=ripple_required_cols)    

        # for a_combined_epoch_stats_df_name, a_combined_epoch_stats_df in {'laps_combined_epoch_stats_df': laps_combined_epoch_stats_df, 'ripple_combined_epoch_stats_df': ripple_combined_epoch_stats_df}.items():
        for a_combined_epoch_stats_df_name, a_combined_epoch_stats_df in {'laps': laps_combined_epoch_stats_df, 'ripple': ripple_combined_epoch_stats_df}.items():
            has_required_columns = PandasHelpers.require_columns(a_combined_epoch_stats_df, required_cols_dict[a_combined_epoch_stats_df_name], print_missing_columns=True)
            if (not has_required_columns):
                return False

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
        """ 2023-12-20 - Actually working spearman rank-ordering!! Independent computation, useful for debugging a period displayed in the DebugTemplateRasters GUI or w/e

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
    @function_attributes(short_name=None, tags=['subfn', 'correlation', 'spearman', 'rank-order', 'pearson'], input_requires=[], output_provides=[], uses=[], used_by=['_compute_single_rank_order_shuffle', 'pandas_df_based_correlation_computations'], creation_date='2023-12-13 03:47', related_items=[])
    def _subfn_calculate_correlations(cls, group, method='spearman', decoder_names: List[str]=None) -> pd.Series:
        """ computes the pearson correlations between the spiketimes during a specific epoch (identified by each 'Probe_Epoch_id' group) and that spike's pf_peak_x location in the template.

        correlations = active_selected_spikes_df.groupby('Probe_Epoch_id').apply(lambda group: calculate_correlations(group, method='spearman'))
        correlations = active_selected_spikes_df.groupby('Probe_Epoch_id').apply(lambda group: calculate_correlations(group, method='pearson'))

        """
        assert decoder_names is not None, f"2023-12-20 - decoder_names must be provided. Usually: `decoder_names = track_templates.get_decoder_names()`"    
        _pf_peak_x_column_names = [f'{a_decoder_name}_pf_peak_x' for a_decoder_name in decoder_names]
        _output_column_names = [f'{a_decoder_name}_{method}' for a_decoder_name in decoder_names]
        # correlations = {f'{a_decoder_name}_{method}': shuffle_fn(group['t_rel_seconds']).rank(method="dense").corr(group[f'{a_decoder_name}_pf_peak_x'], method=method) for a_decoder_name in _decoder_names}
        # Encountered `AttributeError: 'float' object has no attribute 'shape'` and fixed by using .astype(float) as suggested here: `https://stackoverflow.com/questions/53200129/attributeerror-float-object-has-no-attribute-shape-when-using-linregress`
        # TypeError: float() argument must be a string or a number, not 'NAType'
        correlations = {an_output_col_name:group['t_rel_seconds'].rank(method="dense").astype(float).corr(group[a_pf_peak_x_column_name].astype(float), method=method) for a_decoder_name, a_pf_peak_x_column_name, an_output_col_name in zip(decoder_names, _pf_peak_x_column_names, _output_column_names)}

        return pd.Series(correlations)


    @classmethod
    def _subfn_build_all_pf_peak_x_columns(cls, track_templates, selected_spikes_df: pd.DataFrame, override_decoder_aclu_peak_map_dict=None, _MODERN_INTERNAL_SHUFFLE: bool = True):
        """ 2023-12-20 - Returns `active_selected_spikes_df` but with its `f'{a_decoder_name}_pf_peak_x'` columns all shuffled according to `override_decoder_aclu_peak_map_dict` (which was previously shuffled)
        
        _MODERN_INTERNAL_SHUFFLE: bool - A parameter added on 2024-01-09 to allow bypassing the old shuffling method, which was strangely failing for certain sessions and it turned out to be from periods with too few unique active cells in them.
            _MODERN_INTERNAL_SHUFFLE: False - the pre-2024-01-09 method of shuffling where the provided `override_decoder_aclu_peak_map_dict` are used.
            _MODERN_INTERNAL_SHUFFLE: True - a temporary workaround on 2024-01-09 that shuffles only amongst the aclus within each Probe_Epoch_id instead of pre-shuffling the  `override_decoder_aclu_peak_map_dict`. I thought this was what was leading to NaNs in the calculation, because some of these peaks don't occur on one of the tracks.
            
        
        """
        # long_LR_aclu_peak_map, long_RL_aclu_peak_map, short_LR_aclu_peak_map, short_RL_aclu_peak_map = track_templates.get_decoder_aclu_peak_maps()
        is_shuffle: bool = False
        if override_decoder_aclu_peak_map_dict is not None:
            if _MODERN_INTERNAL_SHUFFLE:
                #TODO 2024-01-09 03:15: - [ ] Prevents the overriden maps from being used. Insetad just shuffles each epoch independently.
                # use the default from track_templates
                decoder_aclu_peak_map_dict = track_templates.get_decoder_aclu_peak_map_dict()
            else:
                # use the provided one, for example during a shuffle:
                decoder_aclu_peak_map_dict = override_decoder_aclu_peak_map_dict
            is_shuffle = True # if the dict is passed, it is a shuffle
        else:
            # use the default from track_templates
            decoder_aclu_peak_map_dict = track_templates.get_decoder_aclu_peak_map_dict()

        ## Restrict to only the relevant columns, and Initialize the dataframe columns to np.nan:
        active_selected_spikes_df: pd.DataFrame = deepcopy(selected_spikes_df[['t_rel_seconds', 'aclu', 'Probe_Epoch_id']]).sort_values(['Probe_Epoch_id', 't_rel_seconds', 'aclu']).astype({'Probe_Epoch_id': cls._label_column_type}) # Sort by columns: 'Probe_Epoch_id' (ascending), 't_rel_seconds' (ascending), 'aclu' (ascending)
        
        # _pf_peak_x_column_names = ['LR_Long_pf_peak_x', 'RL_Long_pf_peak_x', 'LR_Short_pf_peak_x', 'RL_Short_pf_peak_x']
        _pf_peak_x_column_names = [f'{a_decoder_name}_pf_peak_x' for a_decoder_name in track_templates.get_decoder_names()]
        active_selected_spikes_df[_pf_peak_x_column_names] = pd.DataFrame([[cls._NaN_Type, cls._NaN_Type, cls._NaN_Type, cls._NaN_Type]], index=active_selected_spikes_df.index)

        # 2023-01-09 Shuffle Amongst only Probe_Epoch_ID
        if is_shuffle:
            if _MODERN_INTERNAL_SHUFFLE:
                unique_Probe_Epoch_IDs = active_selected_spikes_df['Probe_Epoch_id'].unique()
                for a_probe_epoch_ID in unique_Probe_Epoch_IDs:
                    # probe_epoch_df = active_selected_spikes_df[a_probe_epoch_ID == active_selected_spikes_df['Probe_Epoch_id']]
                    # epoch_unique_aclus = probe_epoch_df.aclu.unique()
                    mask = (a_probe_epoch_ID == active_selected_spikes_df['Probe_Epoch_id'])
                    # epoch_unique_aclus = active_selected_spikes_df.loc[mask, 'aclu'].unique()
                    for a_decoder_name, a_aclu_peak_map in decoder_aclu_peak_map_dict.items():
                        # Shuffle aclus here:
                        active_selected_spikes_df.loc[mask, 'aclu'] = active_selected_spikes_df.loc[mask, 'aclu'].sample(frac=1).values
                        active_selected_spikes_df.loc[mask, f'{a_decoder_name}_pf_peak_x'] = active_selected_spikes_df.loc[mask, 'aclu'].map(a_aclu_peak_map)

            else:            
                # Pre 2023-01-09 - Shuffle Amongst all: ______________________________________________________________________________ #
                for a_decoder_name, a_aclu_peak_map in decoder_aclu_peak_map_dict.items():
                    active_selected_spikes_df[f'{a_decoder_name}_pf_peak_x'] = active_selected_spikes_df.aclu.map(a_aclu_peak_map)

            # end if is_shuffle
        else:
            # Non-shuffle:
            for a_decoder_name, a_aclu_peak_map in decoder_aclu_peak_map_dict.items():
                active_selected_spikes_df[f'{a_decoder_name}_pf_peak_x'] = active_selected_spikes_df.aclu.map(a_aclu_peak_map)
        
        return active_selected_spikes_df
    

    @classmethod
    def _compute_single_rank_order_shuffle(cls, track_templates, selected_spikes_df: pd.DataFrame, override_decoder_aclu_peak_map_dict=None):
        """ 2023-12-20 - Candidate for moving into RankOrderComputations 
        
        """
        # active_selected_spikes_df = cls._subfn_build_all_pf_peak_x_columns(track_templates, selected_spikes_df=selected_spikes_df, override_decoder_aclu_peak_map_dict=override_decoder_aclu_peak_map_dict)
        
        #TODO 2023-12-18 13:20: - [ ] This assumes that `'Probe_Epoch_id'` is correct and consistent for both directions, yeah?

        ## Compute real values here:
        decoder_names = track_templates.get_decoder_names()

            
        epoch_id_grouped_selected_spikes_df =  selected_spikes_df.groupby('Probe_Epoch_id') # I can even compute this outside the loop?
        spearman_correlations = epoch_id_grouped_selected_spikes_df.apply(lambda group: RankOrderAnalyses._subfn_calculate_correlations(group, method='spearman', decoder_names=decoder_names)).reset_index() # Reset index to make 'Probe_Epoch_id' a column
        pearson_correlations = epoch_id_grouped_selected_spikes_df.apply(lambda group: RankOrderAnalyses._subfn_calculate_correlations(group, method='pearson', decoder_names=decoder_names)).reset_index() # Reset index to make 'Probe_Epoch_id' a column

        real_stats_df = pd.concat((spearman_correlations, pearson_correlations), axis='columns')
        real_stats_df = real_stats_df.loc[:, ~real_stats_df.columns.duplicated()] # drop duplicated 'Probe_Epoch_id' column
        # Change column type to uint64 for column: 'Probe_Epoch_id'
        real_stats_df = real_stats_df.astype({'Probe_Epoch_id': 'uint64'})
        # Rename column 'Probe_Epoch_id' to 'label'
        real_stats_df = real_stats_df.rename(columns={'Probe_Epoch_id': 'label'})
        

        ## Compute real values here:
        epoch_id_grouped_selected_spikes_df = selected_spikes_df.groupby('Probe_Epoch_id') # I can even compute this outside the loop?
        
        # Parallelize correlation computations if required
        correlations = []
        for method in ['spearman', 'pearson']:
            correlations.append(
                epoch_id_grouped_selected_spikes_df.apply(
                    lambda group: RankOrderAnalyses._subfn_calculate_correlations(
                        group, method=method, decoder_names=decoder_names)
                )
            )
    
        # Adjust and join all calculated correlations
        real_stats_df = pd.concat(correlations, axis='columns').reset_index()
        real_stats_df = real_stats_df.loc[:, ~real_stats_df.columns.duplicated()] # drop duplicated 'Probe_Epoch_id' column

        real_stats_df.rename(columns={'Probe_Epoch_id': 'label'}, inplace=True)
        real_stats_df['label'] = real_stats_df['label'].astype('uint64')  # in-place type casting
    
        return real_stats_df

    # Determine the number of shuffles you want to do
    @classmethod
    def _new_perform_efficient_shuffle(cls, track_templates, active_selected_spikes_df, decoder_aclu_peak_map_dict, num_shuffles:int=5):
        """ 2024-01-09 - Performs the shuffles in a simple way
        
        """
        unique_Probe_Epoch_IDs = active_selected_spikes_df['Probe_Epoch_id'].unique()

        # Create a list to hold the shuffled dataframes
        shuffled_dfs = []
        shuffled_stats_dfs = []

        for i in range(num_shuffles):
            # Working on a copy of the DataFrame
            shuffled_df = active_selected_spikes_df.copy()

            for a_probe_epoch_ID in unique_Probe_Epoch_IDs:
                mask = (a_probe_epoch_ID == shuffled_df['Probe_Epoch_id'])
                
                # Shuffle 'aclu' values
                shuffled_df.loc[mask, 'aclu'] = shuffled_df.loc[mask, 'aclu'].sample(frac=1).values
                
                # # Apply aclu peak map dictionary to 'aclu' column
                # for a_decoder_name, a_aclu_peak_map in decoder_aclu_peak_map_dict.items():
                #     shuffled_df.loc[mask, f'{a_decoder_name}_pf_peak_x'] = shuffled_df.loc[mask, 'aclu'].map(a_aclu_peak_map)
                

            # end `for a_probe_epoch_ID`
            # Once done, apply the aclu peak maps to shuffled_df's 'aclu' column:
            for a_decoder_name, a_aclu_peak_map in decoder_aclu_peak_map_dict.items():
                shuffled_df[f'{a_decoder_name}_pf_peak_x'] = shuffled_df.aclu.map(a_aclu_peak_map)
                
            a_shuffle_stats_df = cls._compute_single_rank_order_shuffle(track_templates, selected_spikes_df=shuffled_df)
            
            # Adding the shuffled DataFrame to the list
            shuffled_dfs.append(shuffled_df)
            shuffled_stats_dfs.append(a_shuffle_stats_df)
            
        return shuffled_dfs, shuffled_stats_dfs


    @classmethod
    def _subfn_build_pandas_df_based_correlation_computations_column_rename_dict(cls, column_names: List[str], decoder_name_to_column_name_prefix_map:Optional[Dict[str,str]]=None) -> Dict[str,str]:
        """ 2023-12-20 - ensures compatibility with lower-case names to older names

        column_names = ['long_RL_spearman', 'long_LR_pearson', 'short_RL_spearman', 'short_RL_pearson', 'long_LR_spearman', 'short_LR_pearson', 'short_LR_spearman', 'long_RL_pearson', 'long_RL_spearman_Z', 'long_LR_pearson_Z', 'short_RL_spearman_Z', 'short_RL_pearson_Z', 'long_LR_spearman_Z', 'short_LR_pearson_Z', 'short_LR_spearman_Z', 'long_RL_pearson_Z']
        decoder_name_to_column_name_prefix_map = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], ['LR_Long', 'RL_Long', 'LR_Short', 'RL_Short']))

        old_to_new_names = build_column_rename_dict(column_names, decoder_name_to_column_name_prefix_map.copy())
        print(old_to_new_names)

        {'long_RL_spearman': 'RL_Long_spearman', 'long_LR_pearson': 'LR_Long_pearson', 'short_RL_spearman': 'RL_Short_spearman', 'short_RL_pearson': 'RL_Short_pearson', 'long_LR_spearman': 'LR_Long_spearman', 'short_LR_pearson': 'LR_Short_pearson', 'short_LR_spearman': 'LR_Short_spearman', 'long_RL_pearson': 'RL_Long_pearson', 'long_RL_spearman_Z': 'RL_Long_spearman_Z', 'long_LR_pearson_Z': 'LR_Long_pearson_Z', 'short_RL_spearman_Z': 'RL_Short_spearman_Z', 'short_RL_pearson_Z': 'RL_Short_pearson_Z', 'long_LR_spearman_Z': 'LR_Long_spearman_Z', 'short_LR_pearson_Z': 'LR_Short_pearson_Z', 'short_LR_spearman_Z': 'LR_Short_spearman_Z', 'long_RL_pearson_Z': 'RL_Long_pearson_Z'}
        """
        if decoder_name_to_column_name_prefix_map is None:
            decoder_name_to_column_name_prefix_map = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], ['LR_Long', 'RL_Long', 'LR_Short', 'RL_Short'])) # renames lower-case long_, short_ to upper-case variants,  places 'Long' after 'LR'

        old_to_new_names = {}
        for col in column_names:
            for decoder_name, prefix in decoder_name_to_column_name_prefix_map.items():
                if decoder_name in col:
                    new_col = prefix + col.split(decoder_name)[-1]
                    old_to_new_names[col] = new_col
        return old_to_new_names

    @classmethod
    @function_attributes(short_name=None, tags=['active', 'shuffle', 'rank_order', 'main'], input_requires=[], output_provides=[], uses=['_subfn_calculate_correlations', 'build_stacked_arrays', '_subfn_build_pandas_df_based_correlation_computations_column_rename_dict'], used_by=[], creation_date='2023-12-15 14:17', related_items=[])
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
        shuffled_results_output_dict = {}
        
        ## Shuffle each map's aclus, takes `selected_spikes_df`

        # LongShortStatsTuple: Tuple[Zscorer, Zscorer, float, float, bool]

        rng = np.random.default_rng() # seed=13378 #TODO 2023-12-13 05:13: - [ ] DO NOT SET THE SEED! This makes the random permutation/shuffle the same every time!!!

        decoder_aclu_peak_map_dict = track_templates.get_decoder_aclu_peak_map_dict()
        # long_LR_aclu_peak_map, long_RL_aclu_peak_map, short_LR_aclu_peak_map, short_RL_aclu_peak_map = track_templates.get_decoder_aclu_peak_maps()

        ## Restrict to only the relevant columns, and Initialize the dataframe columns to np.nan:
        active_selected_spikes_df: pd.DataFrame = deepcopy(selected_spikes_df[['t_rel_seconds', 'aclu', 'Probe_Epoch_id']]).sort_values(['Probe_Epoch_id', 't_rel_seconds', 'aclu']).astype({'Probe_Epoch_id': RankOrderAnalyses._label_column_type}) # Sort by columns: 'Probe_Epoch_id' (ascending), 't_rel_seconds' (ascending), 'aclu' (ascending)
        # _pf_peak_x_column_names = ['LR_Long_pf_peak_x', 'RL_Long_pf_peak_x', 'LR_Short_pf_peak_x', 'RL_Short_pf_peak_x']
        _pf_peak_x_column_names = [f'{a_decoder_name}_pf_peak_x' for a_decoder_name in track_templates.get_decoder_names()]
        active_selected_spikes_df[_pf_peak_x_column_names] = pd.DataFrame([[RankOrderAnalyses._NaN_Type, RankOrderAnalyses._NaN_Type, RankOrderAnalyses._NaN_Type, RankOrderAnalyses._NaN_Type]], index=active_selected_spikes_df.index)
                
        real_spikes_df = active_selected_spikes_df.copy()
        for a_decoder_name, a_aclu_peak_map in decoder_aclu_peak_map_dict.items():
            real_spikes_df[f'{a_decoder_name}_pf_peak_x'] = real_spikes_df.aclu.map(a_aclu_peak_map)
    
        real_stats_df = cls._compute_single_rank_order_shuffle(track_templates, selected_spikes_df=real_spikes_df) # new `_new_perform_efficient_shuffle`
        # real_stats_df = cls._compute_single_rank_order_shuffle(track_templates, selected_spikes_df=selected_spikes_df) # old
        combined_variable_names = list(set(real_stats_df.columns) - set(['label'])) # ['RL_Short_spearman', 'RL_Long_pearson', 'RL_Short_pearson', 'LR_Long_spearman', 'LR_Short_pearson', 'LR_Long_pearson', 'LR_Short_spearman', 'RL_Long_spearman']
        real_stacked_arrays = real_stats_df[combined_variable_names].to_numpy() # for compatibility

        # ==================================================================================================================== #
        # PERFORM SHUFFLE HERE:                                                                                                #
        # ==================================================================================================================== #


        # _new_perform_efficient_shuffle method: _____________________________________________________________________________ #
        shuffled_dfs, shuffled_stats_dfs = cls._new_perform_efficient_shuffle(track_templates, active_selected_spikes_df, decoder_aclu_peak_map_dict, num_shuffles=num_shuffles) # divide by zeros happening here
        output_active_epoch_computed_values = shuffled_stats_dfs

        # Build the output `stacked_arrays`: _________________________________________________________________________________ #

        stacked_arrays = np.stack([a_shuffle_real_stats_df[combined_variable_names].to_numpy() for a_shuffle_real_stats_df in output_active_epoch_computed_values], axis=0) # for compatibility: .shape (n_shuffles, n_epochs, n_columns)
        # stacked_df = pd.concat(output_active_epoch_computed_values, axis='index')

        ## Drop any shuffle indicies where NaNs are returned for any of the stats values.
        is_valid_row = np.logical_not(np.isnan(stacked_arrays)).all(axis=(1,2)) # row [0, 66, :] is bad, ... so is [1, 66, :], ... [20, 66, :], ... they are repeated!!
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

        # valid_stacked_arrays.shape: (n_shuffles, n_epochs, n_variables)
        assert n_epochs == np.shape(valid_stacked_arrays)[-2]
        assert n_variables == np.shape(valid_stacked_arrays)[-1]

        for variable_IDX, a_column_name in enumerate(combined_variable_z_score_column_names):
            z_scorer_list = [Zscorer.init_from_values(stats_corr_values=np.squeeze(valid_stacked_arrays[:, :, variable_IDX]), real_value=real_stacked_arrays[epoch_IDX, variable_IDX]) for epoch_IDX in np.arange(n_epochs)]
            # shuffled_results_output_dict[a_column_name] = z_scorer_list
            z_score_values = np.array([a_zscorer.z_score_value for a_zscorer in z_scorer_list])
            combined_epoch_stats_df[a_column_name] = z_score_values

        if debug_print:
            print(f'combined_variable_names: {combined_variable_names}')
            print(f'combined_variable_z_score_column_names: {combined_variable_z_score_column_names}')

        # Try to add epoch labels. Only used at the very end: active_epochs_df
        try:
            if (active_epochs_df is not None) and ('label' in active_epochs_df.columns):
                active_epochs_df['label'] = active_epochs_df['label'].astype(cls._label_column_type)
                if (np.shape(active_epochs_df)[0] == np.shape(combined_epoch_stats_df)[0]):
                    combined_epoch_stats_df['label'] = active_epochs_df['label'].astype(cls._label_column_type, copy=True).to_numpy() # .to_numpy() is CRITICAL here, otherwise it tries to apply them using each dataframe's .Index property and they get way off.
                else:
                    print(f'failed to add label column, shapes differ! np.shape(active_epochs_df)[0] : {np.shape(active_epochs_df)[0] }, np.shape(combined_epoch_stats_df)[0]): {np.shape(combined_epoch_stats_df)[0]}')

                # combined_epoch_stats_df = combined_epoch_stats_df.set_index('label')
            else:
                print('invalid active_epochs_df. skipping adding labels')
        except Exception as e:
            print(f'Not giving up: e: {e}')
            pass

        # rename columns for compatibility:
        old_to_new_names = cls._subfn_build_pandas_df_based_correlation_computations_column_rename_dict(column_names=list(combined_epoch_stats_df.columns))
        combined_epoch_stats_df = combined_epoch_stats_df.rename(columns=old_to_new_names)

        return combined_epoch_stats_df, (output_active_epoch_computed_values, combined_variable_names, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles)
    

def validate_has_rank_order_results(curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None, required_included_qclu_values:Optional[List[int]]=None):
    """ Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    # Unpacking:
    rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
    results_minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
    results_included_qclu_values: List[int] = rank_order_results.included_qclu_values
    ripple_result_tuple: Optional[DirectionalRankOrderResult] = rank_order_results.ripple_most_likely_result_tuple
    laps_result_tuple: Optional[DirectionalRankOrderResult] = rank_order_results.laps_most_likely_result_tuple

    
    ## comparing to parameters
    param_typed_parameters = curr_active_pipeline.global_computation_results.computation_config
    if param_typed_parameters is not None:
        rank_order_shuffle_analysis_params = param_typed_parameters.get('rank_order_shuffle_analysis', None)
        if rank_order_shuffle_analysis_params is not None:
            ## has valid rank_order_shuffle_analysis config:
            if required_included_qclu_values is None:
                required_included_qclu_values = rank_order_shuffle_analysis_params.minimum_inclusion_fr_Hz # use the params value
                
            if minimum_inclusion_fr_Hz is None:
                minimum_inclusion_fr_Hz = rank_order_shuffle_analysis_params.minimum_inclusion_fr_Hz
                

            # if (rank_order_shuffle_analysis_params.minimum_inclusion_fr_Hz != results_minimum_inclusion_fr_Hz):
            #     print(f'minimum_inclusion_fr_Hz differs! results_value: {results_minimum_inclusion_fr_Hz}, params_val: {rank_order_shuffle_analysis_params.minimum_inclusion_fr_Hz}')
            #     return False
            
            # # if (rank_order_shuffle_analysis.num_shuffles != rank_order_results.num_shuffles):
            # #     print(f'num_shuffles differs! results_value: {rank_order_results.num_shuffles}, params_val: {rank_order_shuffle_analysis.num_shuffles}')
            # #     return False
            
            # if (set(rank_order_shuffle_analysis_params.included_qclu_values) != set(results_included_qclu_values)):
            #     print(f'included_qclu_values differs! results_value: {results_included_qclu_values}, params_val: {rank_order_shuffle_analysis_params.included_qclu_values}')
            #     return False
            

    ## regardless of whether we have params, we can test the results value against the desired value:
    if minimum_inclusion_fr_Hz is not None:
        if (minimum_inclusion_fr_Hz != results_minimum_inclusion_fr_Hz):
            print(f'minimum_inclusion_fr_Hz differs! results_value: {results_minimum_inclusion_fr_Hz}, specified_val: {minimum_inclusion_fr_Hz}, params_val: {rank_order_shuffle_analysis_params.get("minimum_inclusion_fr_Hz", "err")}')
            return False
    
    
    if required_included_qclu_values is None:
        ## try to get the desired value from the parameters:
        if (set(required_included_qclu_values) != set(results_included_qclu_values)):
            print(f'included_qclu_values differs! results_value: {results_included_qclu_values}, specified_val: {required_included_qclu_values}, params_val: {rank_order_shuffle_analysis_params.get("included_qclu_values", "err")}')
            return False
     

    ## TODO: make sure result is for the current minimimum:

    ## TODO: require same `included_qclu_values` values


    ## Used to be `x[0] for x.long_stats_z_scorer` and `x[1] for x.short_stats_z_scorer`
    # Extract the real spearman-values/p-values:
    # LR_long_relative_real_p_values = np.array([x.long_stats_z_scorer.real_p_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()]) # x is LongShortStatsItem,  # AttributeError: 'dict' object has no attribute 'real_p_value'
    # LR_long_relative_real_values = np.array([x.long_stats_z_scorer.real_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])

    # LR_short_relative_real_p_values = np.array([x.short_stats_z_scorer.real_p_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
    # LR_short_relative_real_values = np.array([x.short_stats_z_scorer.real_value for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])

    LR_template_epoch_actually_included_aclus = [v[1] for v in rank_order_results.LR_ripple.extra_info_dict.values()] # (template_epoch_neuron_IDXs, template_epoch_actually_included_aclus, epoch_neuron_IDX_ranks) ## ERROR: AttributeError: 'NoneType' object has no attribute 'extra_info_dict'
    LR_relative_num_cells = np.array([len(v[1]) for v in rank_order_results.LR_ripple.extra_info_dict.values()])

    # RL_long_relative_real_p_values = np.array([x.long_stats_z_scorer.real_p_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])
    # RL_long_relative_real_values = np.array([x.long_stats_z_scorer.real_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])

    # RL_short_relative_real_p_values = np.array([x.short_stats_z_scorer.real_p_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])
    # RL_short_relative_real_values = np.array([x.short_stats_z_scorer.real_value for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])

    RL_template_epoch_actually_included_aclus = [v[1] for v in rank_order_results.RL_ripple.extra_info_dict.values()] # (template_epoch_neuron_IDXs, template_epoch_actually_included_aclus, epoch_neuron_IDX_ranks)
    RL_relative_num_cells = np.array([len(v[1]) for v in rank_order_results.RL_ripple.extra_info_dict.values()])

    ## z-diffs:
    LR_long_short_z_diff = np.array([x.long_short_z_diff for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
    LR_long_short_naive_z_diff = np.array([x.long_short_naive_z_diff for x in rank_order_results.LR_ripple.ranked_aclus_stats_dict.values()])
    RL_long_short_z_diff = np.array([x.long_short_z_diff for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])
    RL_long_short_naive_z_diff = np.array([x.long_short_naive_z_diff for x in rank_order_results.RL_ripple.ranked_aclus_stats_dict.values()])


    # `LongShortStatsItem` form (2024-01-02):
    # LR_results_real_values = np.array([(a_result_item.long_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.real_value) for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
    # RL_results_real_values = np.array([(a_result_item.long_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.real_value) for epoch_id, a_result_item in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])
    LR_results_long_short_z_diffs = np.array([a_result_item.long_short_z_diff for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
    RL_results_long_short_z_diff = np.array([a_result_item.long_short_z_diff for epoch_id, a_result_item in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])


    laps_merged_complete_epoch_stats_df: pd.DataFrame = rank_order_results.laps_merged_complete_epoch_stats_df ## New method
    ripple_merged_complete_epoch_stats_df: pd.DataFrame = rank_order_results.ripple_merged_complete_epoch_stats_df ## New method

    if ripple_result_tuple is not None:
        rank_order_z_score_df = ripple_result_tuple.rank_order_z_score_df
        # if rank_order_z_score_df is None:
        #     return False # failing here


    # 2023-12-15 - Newest method:
    ripple_combined_epoch_stats_df = rank_order_results.ripple_combined_epoch_stats_df
    if ripple_combined_epoch_stats_df is None:
        return False

    if np.isnan(rank_order_results.ripple_combined_epoch_stats_df.index).any():
        return False # can't have dataframe index that is missing values.

    assert (len(rank_order_results.ripple_new_output_tuple) == 5), f"new_output_tuple must be greater than length 6"
    

    laps_combined_epoch_stats_df = rank_order_results.laps_combined_epoch_stats_df
    if laps_combined_epoch_stats_df is None:
        return False

    assert (len(rank_order_results.laps_new_output_tuple) == 5), f"new_output_tuple must be greater than length 6"
    

    # if 'LongShort_BestDir_quantile_diff' not in ripple_combined_epoch_stats_df:
    #     return False

    if not RankOrderAnalyses.validate_has_rank_order_results_quantiles(curr_active_pipeline, computation_filter_name=computation_filter_name, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz):
        return False
    
    # rank_order_results.included_qclu_values

    if minimum_inclusion_fr_Hz is not None:
        return (minimum_inclusion_fr_Hz == results_minimum_inclusion_fr_Hz) # makes sure same
    else:
        #TODO 2023-11-29 08:42: - [ ] cannot validate minimum because none was passed, eventually reformulate to use parameters
        return True
        




class RankOrderGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'rank_order'
    _computationPrecidence = 1001
    _is_global = True

    @function_attributes(short_name='rank_order_shuffle_analysis', tags=['directional_pf', 'laps', 'rank_order', 'session', 'pf1D', 'pf2D'],
                         input_requires=['DirectionalLaps', 'global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz', 'global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values'], output_provides=['RankOrder'], uses=['RankOrderAnalyses'], used_by=[], creation_date='2023-11-08 17:27', related_items=[],
        requires_global_keys=['DirectionalLaps'], provides_global_keys=['RankOrder'],
        validate_computation_test=validate_has_rank_order_results, is_global=True)
    def perform_rank_order_shuffle_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, num_shuffles:int=500, minimum_inclusion_fr_Hz:float=5.0, included_qclu_values=[1,2,4,6,7,9], skip_laps=False):
        """ Performs the computation of the spearman and pearson correlations for the ripple and lap epochs.

        Does this not depend on the desired_ripple_decoding_time_bin_size?
        
        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['RankOrder']
                ['RankOrder'].odd_ripple
                ['RankOrder'].even_ripple
                ['RankOrder'].odd_laps
                ['RankOrder'].even_laps


        ## Unpack:        
        minimum_inclusion_fr_Hz = global_computation_results.computation_config['rank_order_shuffle_analysis'].minimum_inclusion_fr_Hz
        included_qclu_values = global_computation_results.computation_config['rank_order_shuffle_analysis'].included_qclu_values
        num_shuffles = global_computation_results.computation_config['rank_order_shuffle_analysis'].num_shuffles
        
        
        """
        if include_includelist is not None:
            print(f'WARN: perform_rank_order_shuffle_analysis(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        print(f'####> perform_rank_order_shuffle_analysis(..., num_shuffles={num_shuffles})')

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


        ## Update the `global_computation_results.computation_config`
        global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz = minimum_inclusion_fr_Hz
        global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values = included_qclu_values
        global_computation_results.computation_config.rank_order_shuffle_analysis.num_shuffles = num_shuffles
            

        ## Laps Rank-Order Analysis:
        if not skip_laps:
            print(f'\t##> computing Laps rank-order shuffles:')
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
                track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
                laps_combined_epoch_stats_df, laps_new_output_tuple = RankOrderAnalyses.pandas_df_based_correlation_computations(selected_spikes_df=selected_spikes_df, active_epochs_df=active_epochs, track_templates=track_templates, num_shuffles=num_shuffles)
                # new_output_tuple (output_active_epoch_computed_values, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles) = laps_new_output_tuple
                global_computation_results.computed_data['RankOrder'].laps_combined_epoch_stats_df, global_computation_results.computed_data['RankOrder'].laps_new_output_tuple = laps_combined_epoch_stats_df, laps_new_output_tuple
                print(f'done!')

            except (AssertionError, BaseException) as e:
                print(f'Issue with Laps computation in new method 2023-12-15: e: {e}')
                raise

        ## END `if not skip_laps`


        ## Ripple Rank-Order Analysis:
        print(f'\t##> computing Ripple rank-order shuffles:')
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
            track_templates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
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
            global_computation_results.computed_data['RankOrder'].ripple_most_likely_result_tuple, global_computation_results.computed_data['RankOrder'].laps_most_likely_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(owning_pipeline_reference)
        
        except (AssertionError, BaseException) as e:
            print(f'Issue with `RankOrderAnalyses.most_likely_directional_rank_order_shuffling(...)` e: {e}')
            raise


        print(f'< done with `perform_rank_order_shuffle_analysis(...)`')


        """ Usage:
        
        rank_order_results = curr_active_pipeline.global_computation_results.computed_data['RankOrder']

        odd_laps_epoch_ranked_aclus_stats_dict, odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_laps_long_z_score_values, odd_laps_short_z_score_values, odd_laps_long_short_z_score_diff_values = rank_order_results.odd_laps
        even_laps_epoch_ranked_aclus_stats_dict, even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_laps_long_z_score_values, even_laps_short_z_score_values, even_laps_long_short_z_score_diff_values = rank_order_results.even_laps

        odd_ripple_evts_epoch_ranked_aclus_stats_dict, odd_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, odd_ripple_evts_long_z_score_values, odd_ripple_evts_short_z_score_values, odd_ripple_evts_long_short_z_score_diff_values = rank_order_results.odd_ripple
        even_ripple_evts_epoch_ranked_aclus_stats_dict, even_ripple_evts_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, even_ripple_evts_long_z_score_values, even_ripple_evts_short_z_score_values, even_ripple_evts_long_short_z_score_diff_values = rank_order_results.even_ripple

        """
        return global_computation_results
    

   


# ==================================================================================================================== #
# Display Function Helpers                                                                                             #
# ==================================================================================================================== #


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

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import Zscorer

@function_attributes(short_name=None, tags=['histogram', '1D', 'rank-order'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-12 09:20', related_items=[])
def plot_rank_order_histograms(rank_order_results: RankOrderComputationsContainer, number_of_bins: int = 21, post_title_info: str = '', active_context=None, perform_write_to_file_callback=None) -> Tuple:
    """ plots 1D histograms from the rank-order shuffled data during the ripples.

    https://pandas.pydata.org/pandas-docs/version/0.24.1/user_guide/visualization.html

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import plot_rank_order_histograms

        # Plot histograms:
        post_title_info: str = f'{minimum_inclusion_fr_Hz} Hz\n{curr_active_pipeline.get_session_context().get_description()}'
        _out_z_score, _out_real, _out_most_likely_z = plot_rank_order_histograms(rank_order_results, post_title_info=post_title_info)

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from flexitext import flexitext ## flexitext for formatted matplotlib text

    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText
    # fig = build_or_reuse_figure(fignum=f'1D Histograms')
    # ax1 = fig.add_subplot(3, 1, 1)
    # ax2 = fig.add_subplot(3, 1, 2)
    # ax3 = fig.add_subplot(3, 1, 3)

    try:
        # `LongShortStatsItem` form (2024-01-02):
        LR_results_real_values = np.array([(a_result_item.long_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.real_value) for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
        RL_results_real_values = np.array([(a_result_item.long_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.real_value) for epoch_id, a_result_item in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])
    except AttributeError as e:

        for a_sub_result in (rank_order_results.LR_ripple,  rank_order_results.RL_ripple):
            ## Fix ZScorer types being dicts after loading:
            # this was resulting in `AttributeError: 'dict' object has no attribute 'real_value'`
            for epoch_id, a_result_item in a_sub_result.ranked_aclus_stats_dict.items():
                a_result_item.fixup_types_if_needed()

        print(f'result fixedup.')

    # {epoch_id:a_result_item for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()}

    # `LongShortStatsItem` form (2024-01-02):
    LR_results_real_values = np.array([(a_result_item.long_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.real_value) for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
    RL_results_real_values = np.array([(a_result_item.long_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.real_value) for epoch_id, a_result_item in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])
    # LR_results_long_short_z_diffs = np.array([a_result_item.long_short_z_diff for epoch_id, a_result_item in rank_order_results.LR_ripple.ranked_aclus_stats_dict.items()])
    # RL_results_long_short_z_diff = np.array([a_result_item.long_short_z_diff for epoch_id, a_result_item in rank_order_results.RL_ripple.ranked_aclus_stats_dict.items()])


    if active_context is not None:
            display_context = active_context.adding_context('display_fn', display_fn_name='plot_rank_order_histograms')
            
    with mpl.rc_context({'figure.figsize': (8.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, }):
        # Create a FigureCollector instance
        with FigureCollector(name='plot_rank_order_histograms', base_context=display_context) as collector:

            ## Define common operations to do after making the figure:
            def setup_common_after_creation(a_collector, fig, axes, sub_context, title=f'<size:22> Sig. (>0.95) <weight:bold>Best</> <weight:bold>Quantile Diff</></>'):
                """ Captures:

                t_split
                """
                a_collector.contexts.append(sub_context)                
                for ax in (axes if isinstance(axes, Iterable) else [axes]):
                    # `flexitext` version:
                    text_formatter = FormattedFigureText()
                    ax.set_title('')
                    fig.suptitle('')
                    text_formatter.setup_margins(fig)
                    title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, title, va="bottom", xycoords="figure fraction")
                    footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                                                text_formatter._build_footer_string(active_context=sub_context),
                                                va="top", xycoords="figure fraction")
            
                if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                    perform_write_to_file_callback(sub_context, fig)
                    
            # ax1, ax2, ax3, ax4 = None, None, None, None
            label = ': '.join([f'Ripple Z-scores', post_title_info])
            fig, ax1 = collector.subplots(num=label, clear=True)
            _out_z_score = pd.DataFrame({'LR_long_z_scores': rank_order_results.LR_ripple.long_z_score, 'LR_short_z_scores': rank_order_results.LR_ripple.short_z_score,
                    'RL_long_z_scores': rank_order_results.RL_ripple.long_z_score, 'RL_short_z_scores': rank_order_results.RL_ripple.short_z_score}).hist(bins=number_of_bins, ax=ax1, sharex=True, sharey=True)
            # plt.suptitle(': '.join([f'Ripple Z-scores', post_title_info]))
            setup_common_after_creation(collector, fig=fig, axes=ax1, sub_context=display_context.adding_context('subplot', subplot_name='Ripple Z-scores'), 
                                                    title=f'<size:22> Ripple <weight:bold>Z-scores</> {post_title_info}</>')

            label = ': '.join([f'Ripple real correlations', post_title_info])
            fig, ax2 = collector.subplots(num=label, clear=True)
            _out_real = pd.DataFrame({'LR_long_real_corr': np.squeeze(LR_results_real_values[:,0]), 'LR_short_real_corr': np.squeeze(LR_results_real_values[:,1]),
                    'RL_long_real_corr': np.squeeze(RL_results_real_values[:,0]), 'RL_short_real_corr': np.squeeze(RL_results_real_values[:,1])}).hist(bins=number_of_bins, ax=ax2, sharex=True, sharey=True)
            # plt.suptitle(': '.join([f'Ripple real correlations', post_title_info]))
            setup_common_after_creation(collector, fig=fig, axes=ax2, sub_context=display_context.adding_context('subplot', subplot_name='Ripple real correlations'), 
                                                    title=f'<size:22> Ripple <weight:bold>real correlations</> {post_title_info}</>')
            
            
            label = ': '.join([f'Ripple Most-likely Z-scores', post_title_info])
            fig, ax3 = collector.subplots(num=label, clear=True)
            _out_most_likely_z = pd.DataFrame({'most_likely_long_z_scores': rank_order_results.ripple_most_likely_result_tuple.long_best_dir_z_score_values, 'most_likely_short_z_scores': rank_order_results.ripple_most_likely_result_tuple.short_best_dir_z_score_values}).hist(bins=number_of_bins, ax=ax3, sharex=True, sharey=True)
            # plt.suptitle(': '.join([f'Ripple Most-likely z-scores', post_title_info]))
            setup_common_after_creation(collector, fig=fig, axes=ax3, sub_context=display_context.adding_context('subplot', subplot_name='Ripple Most-likely Z-scores'), 
                                                    title=f'<size:22> Ripple Most-likely <weight:bold>Z-scores</> {post_title_info}</>')

            label = ': '.join([f'Ripple Most-likely Spearman Rho', post_title_info])
            fig, ax4 = collector.subplots(num=label, clear=True)
            _out_most_likely_raw = pd.DataFrame({'most_likely_long_raw_rho': rank_order_results.ripple_combined_epoch_stats_df['Long_BestDir_spearman'].to_numpy(),
                                                'most_likely_short_raw_rho': rank_order_results.ripple_combined_epoch_stats_df['Short_BestDir_spearman'].to_numpy()}).hist(bins=number_of_bins, ax=ax4, sharex=True, sharey=True)
            # plt.suptitle(': '.join([f'Ripple Most-likely Spearman Rho', post_title_info]))
            setup_common_after_creation(collector, fig=fig, axes=ax4, sub_context=display_context.adding_context('subplot', subplot_name='Ripple Most-likely Spearman Rho'), 
                                                    title=f'<size:22> Ripple Most-likely <weight:bold>Spearman Rho</> {post_title_info}</>')
            

    return collector



@function_attributes(short_name=None, tags=['rank-order', 'inst_fr', 'epoch', 'lap', 'replay'], input_requires=[], output_provides=[], uses=['pyqtgraph', 'pyqt'], used_by=[], creation_date='2023-11-16 18:42', related_items=['most_likely_directional_rank_order_shuffling'])
def plot_rank_order_epoch_inst_fr_result_tuples(curr_active_pipeline, result_tuple, analysis_type, included_epoch_idxs=None, active_context=None, perform_write_to_file_callback=None, show=False):
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
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch = curr_active_pipeline.filtered_epochs[long_epoch_name]
    short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]

    active_context = (active_context or curr_active_pipeline.sess.get_context())
    plot_rank_order_epoch_inst_fr_result_tuples_display_context = active_context.adding_context('display_fn', display_fn_name='plot_rank_order_epoch_inst_fr_result_tuples')
    active_display_context = plot_rank_order_epoch_inst_fr_result_tuples_display_context.adding_context('analysis_type', subplot_name=analysis_type)

    if analysis_type == 'Ripple':
        # global_events = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
        global_events = deepcopy(result_tuple.active_epochs)
    elif analysis_type == 'Lap':
        global_events = deepcopy(result_tuple.active_epochs)
    else:
        raise ValueError(f"Invalid analysis type analysis_type: '{analysis_type}'. Choose 'Ripple' or 'Lap'.")

    if isinstance(global_events, pd.DataFrame):
        global_events = Epoch(global_events.epochs.get_valid_df())

    if included_epoch_idxs is not None:
        print(f'filtering global epochs:')
        is_epoch_significant = np.isin(global_events.to_dataframe().index, included_epoch_idxs)
        global_events = deepcopy(global_events).boolean_indicies_slice(is_epoch_significant)

        # significant_ripple_epochs: pd.DataFrame = deepcopy(global_events.epochs.get_valid_df())[is_epoch_significant]
    else:
        # is_epoch_significant = np.arange(global_events.n_epochs)
        is_epoch_significant = np.full_like(result_tuple.long_short_best_dir_z_score_diff_values, fill_value=True, dtype='bool')


    # epoch_identifiers = np.arange(global_events.n_epochs) # these should be labels!
    epoch_identifiers = global_events._df.label.astype({'label': RankOrderAnalyses._label_column_type}).values #.labels
    x_values = global_events.midtimes
    x_axis_name_suffix = 'Mid-time (Sec)'

    _display_z_score_diff_outputs = RankOrderAnalyses._perform_plot_z_score_diff(
        x_values, result_tuple.long_short_best_dir_z_score_diff_values[is_epoch_significant], None,
        variable_name=analysis_type, x_axis_name_suffix=x_axis_name_suffix,
        point_data_values=epoch_identifiers,
        active_display_context=active_display_context, show=show
    )
    _display_z_score_raw_outputs = RankOrderAnalyses._perform_plot_z_score_raw(
        x_values, *[x[is_epoch_significant] for x in result_tuple.masked_z_score_values_list],
        variable_name=analysis_type, x_axis_name_suffix=x_axis_name_suffix,
        point_data_values=epoch_identifiers,
        active_display_context=active_display_context, show=show
    )

    app, diff_win, diff_p1, out_plot_1D, out_hist_stuff, out_label_tuple, diff_sub_context = _display_z_score_diff_outputs
    long_epoch_indicator_region_items, short_epoch_indicator_region_items = PlottingHelpers.helper_pyqtgraph_add_long_short_session_indicator_regions(diff_p1, long_epoch, short_epoch)
    raw_app, raw_win, raw_p1, raw_out_plot_1D, raw_label_tuple, raw_sub_context = _display_z_score_raw_outputs
    long_epoch_indicator_region_items, short_epoch_indicator_region_items = PlottingHelpers.helper_pyqtgraph_add_long_short_session_indicator_regions(raw_p1, long_epoch, short_epoch)

    if (perform_write_to_file_callback is not None):
        if (diff_sub_context is not None):
            perform_write_to_file_callback(diff_sub_context, diff_win)
        if (raw_sub_context is not None):
            perform_write_to_file_callback(raw_sub_context, raw_win)            


    return app, diff_win, diff_p1, out_plot_1D, out_label_tuple, raw_app, raw_win, raw_p1, raw_out_plot_1D, raw_label_tuple




def _plot_significant_event_quantile_fig(curr_active_pipeline, significant_ripple_combined_epoch_stats_df: pd.DataFrame):
    """ 

    Plots a scatterplot that shows the best dir quantile of each replay epoch over time

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import _plot_significant_event_quantile_fig
        # Filter rows based on columns: 'Long_BestDir_quantile', 'Short_BestDir_quantile'
        quantile_significance_threshold: float = 0.95
        significant_ripple_combined_epoch_stats_df = ripple_combined_epoch_stats_df[(ripple_combined_epoch_stats_df['Long_BestDir_quantile'] > quantile_significance_threshold) | (ripple_combined_epoch_stats_df['Short_BestDir_quantile'] > quantile_significance_threshold)]
        _out = _plot_significant_event_quantile_fig(significant_ripple_combined_epoch_stats_df=significant_ripple_combined_epoch_stats_df)
        _out


    """
    # ripple_combined_epoch_stats_df.plot.scatter(x=np.arange(np.shape(ripple_combined_epoch_stats_df)[0]), y='LongShort_BestDir_quantile_diff')
    # ripple_combined_epoch_stats_df['LongShort_BestDir_quantile_diff'].plot.scatter(title='Best Quantile Diff')

    # fig = plt.figure(num='best_quantile')
    marker_style = dict(linestyle='None', color='#ff7f0eff', markersize=6,
                        markerfacecolor='#ff7f0eb4', markeredgecolor='#ff7f0eff')

    # dict(facecolor='#ff7f0eb4', size=8.0)
    # fignum='best_quantiles'
    return significant_ripple_combined_epoch_stats_df[['midtimes', 'LongShort_BestDir_quantile_diff']].plot(x='midtimes', y='LongShort_BestDir_quantile_diff', title='Sig. (>0.95) Best Quantile Diff', **marker_style, marker='o')
    

@function_attributes(short_name=None, tags=['quantile', 'figure', 'seaborn', 'FigureCollector'], input_requires=[], output_provides=[], uses=['FigureCollector'], used_by=[], creation_date='2023-12-22 19:50', related_items=[])
def plot_quantile_diffs(merged_complete_epoch_stats_df, t_start=None, t_split=1000.0, t_end=None, quantile_significance_threshold: float = 0.95, active_context=None, perform_write_to_file_callback=None, include_LR_LR_plot:bool=False, include_RL_RL_plot:bool=False):
    """ Plots three Matplotlib figures displaying the quantile differences
    
    Usage:
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import plot_quantile_diffs

    _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
    global_epoch = curr_active_pipeline.filtered_epochs[global_epoch_name]
    t_start, t_end = global_epoch.start_end_times
    short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
    split_time_t: float = short_epoch.t_start
    active_context = curr_active_pipeline.sess.get_context()

    collector = plot_quantile_diffs(ripple_merged_complete_epoch_stats_df, t_start=t_start, t_split=split_time_t, t_end=t_end, active_context=active_context)

     # sns.relplot(
            #     data=tips, x="total_bill", y="tip",
            #     col="time", hue="day", style="day",
            #     kind="scatter"
            # )
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from flexitext import flexitext ## flexitext for formatted matplotlib text

    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText
    
    ripple_combined_epoch_stats_df = deepcopy(merged_complete_epoch_stats_df)

    # Filter rows based on columns: 'Long_BestDir_quantile', 'Short_BestDir_quantile'
    significant_BestDir_quantile_stats_df = ripple_combined_epoch_stats_df[(ripple_combined_epoch_stats_df['Long_BestDir_quantile'] > quantile_significance_threshold) | (ripple_combined_epoch_stats_df['Short_BestDir_quantile'] > quantile_significance_threshold)]
    LR_likely_active_df = ripple_combined_epoch_stats_df[(ripple_combined_epoch_stats_df['combined_best_direction_indicies']==0) & ((ripple_combined_epoch_stats_df['LR_Long_pearson_percentile'] > quantile_significance_threshold) | (ripple_combined_epoch_stats_df['LR_Short_percentile'] > quantile_significance_threshold))]
    RL_likely_active_df = ripple_combined_epoch_stats_df[(ripple_combined_epoch_stats_df['combined_best_direction_indicies']==1) & ((ripple_combined_epoch_stats_df['RL_Long_percentile'] > quantile_significance_threshold) | (ripple_combined_epoch_stats_df['RL_Short_percentile'] > quantile_significance_threshold))]
   
    if active_context is not None:
        display_context = active_context.adding_context('display_fn', display_fn_name='plot_quantile_diffs')
        
    with mpl.rc_context({'figure.figsize': (12.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, }):
        # Create a FigureCollector instance
        with FigureCollector(name='plot_quantile_diffs', base_context=display_context) as collector:

            ## Define common operations to do after making the figure:
            def setup_common_after_creation(a_collector, fig, axes, sub_context, title=f'<size:22> Sig. (>0.95) <weight:bold>Best</> <weight:bold>Quantile Diff</></>'):
                """ Captures:

                t_split
                """
                a_collector.contexts.append(sub_context)
                
                # Add epoch indicators
                for ax in (axes if isinstance(axes, Iterable) else [axes]):
                    # Update the ylims with the new bounds
                    ax.set_ylim(-1.0, 1.0)
                    # Add epoch indicators
                    PlottingHelpers.helper_matplotlib_add_long_short_epoch_indicator_regions(ax=ax, t_split=t_split, t_start=t_start, t_end=t_end)
                    # Update the xlimits with the new bounds
                    ax.set_xlim(t_start, t_end)
                    # Draw a horizontal line at y=0.5
                    ax.axhline(y=0.0, color=(0,0,0,1))

                    # `flexitext` version:
                    text_formatter = FormattedFigureText()
                    ax.set_title('')
                    fig.suptitle('')
                    text_formatter.setup_margins(fig)
                    title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin,
                                            title,
                                            va="bottom", xycoords="figure fraction")
                    footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                                                text_formatter._build_footer_string(active_context=sub_context),
                                                va="top", xycoords="figure fraction")
            
                if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                    perform_write_to_file_callback(sub_context, fig)
                

            # Plot for BestDir
            fig, ax = collector.subplots(num='LongShort_BestDir_quantile_diff', clear=True)
            _out_BestDir = sns.scatterplot(
                ax=ax,
                data=significant_BestDir_quantile_stats_df,
                x='start',
                y='LongShort_BestDir_quantile_diff',
                # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            )
            setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=display_context.adding_context('subplot', subplot_name='BestDir'), 
                                        title=f'<size:22> Sig. (>0.95) <weight:bold>Best</> Quantile Diff</>')
            

            if include_LR_LR_plot:
                # Create the scatter plot with Seaborn, using 'size' to set marker sizes
                fig, ax = collector.subplots(num='LR-LR_LongShort_LR_quantile_diff', clear=True)
                _out_LR = sns.scatterplot(
                    ax=ax,
                    data=LR_likely_active_df,
                    x='start',
                    y='LongShort_LR_quantile_diff',
                    # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
                )
                setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=display_context.adding_context('subplot', subplot_name='LR-Likely'), 
                                            title=f'<size:22> Sig. (>0.95) <weight:bold>LR-LR (LR-Likely)</> Quantile Diff</>')
            

            if include_RL_RL_plot:
                fig, ax = collector.subplots(num='RL-RL_LongShort_RL_quantile_diff', clear=True)
                _out_RL = sns.scatterplot(
                    ax=ax,
                    data=RL_likely_active_df[RL_likely_active_df['RL_Long_rel_num_cells']>10],
                    x='start',
                    y='LongShort_RL_quantile_diff',
                    # size='RL_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
                )
                setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=display_context.adding_context('subplot', subplot_name='RL-Likely'), 
                                            title=f'<size:22> Sig. (>0.95) <weight:bold>RL-RL (RL-Likely)</> Quantile Diff</>')
            


    # Access the collected figures outside the context manager
    # result = tuple(collector.created_figures)

    return collector
    


def _validate_estimated_lap_dirs(rank_order_results, global_any_laps_epochs_obj):
    """ 2023-12-19 - validstes the estimated lap directions against the ground-truth direction which is known for the laps. 
     """
     
    
    lap_dir_is_LR = deepcopy(rank_order_results.laps_most_likely_result_tuple.directional_likelihoods_tuple.long_best_direction_indices)
    # lap_dir_is_LR = deepcopy(rank_order_results.laps_most_likely_result_tuple.directional_likelihoods_tuple.long_best_direction_indices)
    lap_dir_index = deepcopy(lap_dir_is_LR).astype('int8')
    # lap_dir_index = deepcopy(laps_merged_complete_epoch_stats_df['combined_best_direction_indicies'].to_numpy()).astype('int8')

    # global_laps
    actual_lap_dir = deepcopy(global_any_laps_epochs_obj.to_dataframe()['lap_dir'])
    n_total_laps = np.shape(actual_lap_dir)[0]
    is_correct = (actual_lap_dir.values == lap_dir_index)
    n_correct_direction = is_correct.sum()
    n_correct_direction

    print(f'Lap directions: {n_correct_direction}/{n_total_laps} correct ({100.0*float(n_correct_direction)/float(n_total_laps)}%)') # Lap directions: 76/80 correct (95.0%)

def setup_histogram_common_after_creation(fig, axes, sub_context, title=f'<size:22> Sig. (>0.95) <weight:bold>Best</> <weight:bold>Quantile Diff</></>', perform_write_to_file_callback=None):
    """ Captures:
        perform_write_to_file_callback
    """
    from flexitext import flexitext ## flexitext for formatted matplotlib text

    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText
    
    # `flexitext` version:
    text_formatter = FormattedFigureText()
    fig.suptitle('')
    text_formatter.setup_margins(fig, top_margin=0.740)
    title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, title, va="bottom", xycoords="figure fraction")
    footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                                text_formatter._build_footer_string(active_context=sub_context),
                                va="top", xycoords="figure fraction")

    if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
        perform_write_to_file_callback(sub_context, fig)




from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

# from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper


# ==================================================================================================================== #
# Display Functions                                                                                                    #
# ==================================================================================================================== #

class RankOrderGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """
    @function_attributes(short_name='rank_order_debugger', tags=['rank-order','debugger','shuffle', 'interactive', 'slider'], conforms_to=['output_registering', 'figure_saving'], input_requires=['global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz', 'global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values'], output_provides=[], uses=[], used_by=[], creation_date='2023-11-09 01:12', related_items=[],
                         validate_computation_test=RankOrderAnalyses._validate_can_display_RankOrderRastersDebugger, is_global=True)
    def _display_rank_order_debugger(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """

            """
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger
            
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results: RankOrderComputationsContainer = global_computation_results.computed_data.get('RankOrder', None)
            if rank_order_results is not None:
                minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = rank_order_results.included_qclu_values
            else:        
                ## get from parameters:
                minimum_inclusion_fr_Hz: float = global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
                included_qclu_values: List[int] = global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values
            

            # track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # shared-only
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            global_spikes_df = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].spikes_df) # #TODO 2023-12-08 12:44: - [ ] does ripple_result_tuple contain a spikes_df?

            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
            print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')

            ## RankOrderRastersDebugger:
            _out_rank_order_event_raster_debugger = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, ripple_result_tuple.active_epochs, track_templates, rank_order_results.RL_ripple.selected_spikes_fragile_linear_neuron_IDX_dict, rank_order_results.LR_ripple.selected_spikes_fragile_linear_neuron_IDX_dict)

            return _out_rank_order_event_raster_debugger


    @function_attributes(short_name='rank_order_z_stats', tags=['rank-order','debugger','shuffle'], input_requires=[], output_provides=[], uses=['plot_rank_order_epoch_inst_fr_result_tuples'], used_by=[], creation_date='2023-12-15 21:46', related_items=[],
        validate_computation_test=RankOrderAnalyses._validate_can_display_RankOrderRastersDebugger, is_global=True)
    def _display_rank_order_z_stats_results(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
        """ Plots the z-scores differences and their raw-values

        """
        defer_render: bool = kwargs.pop('defer_render', False)
        should_show: bool = (not defer_render)
        
        # #TODO 2024-01-03 05:24: - [ ] Do something to switch the matplotlib backend to 'AGG' if defer_render == True. Currently only adjusts the pyqtgraph-based figures (`plot_rank_order_epoch_inst_fr_result_tuples`)

        active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        long_epoch_name = include_includelist[0] # 'maze1_PYR'
        short_epoch_name = include_includelist[1] # 'maze2_PYR'
        assert len(include_includelist) > 2
        global_epoch_name = include_includelist[-1] # 'maze_PYR'

        directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
        assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
        rank_order_results: RankOrderComputationsContainer = global_computation_results.computed_data['RankOrder']
        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        ripple_result_tuple, laps_result_tuple = rank_order_results.ripple_most_likely_result_tuple, rank_order_results.laps_most_likely_result_tuple
        laps_merged_complete_epoch_stats_df: pd.DataFrame = rank_order_results.laps_merged_complete_epoch_stats_df ## New method
        ripple_merged_complete_epoch_stats_df: pd.DataFrame = rank_order_results.ripple_merged_complete_epoch_stats_df ## New method


        def _perform_write_to_file_callback(final_context, fig):
            if save_figure:
                return owning_pipeline_reference.output_figure(final_context, fig)
            else:
                pass # do nothing, don't save


        # Quantile Diff Figures: _____________________________________________________________________________________________ #
        global_epoch = owning_pipeline_reference.filtered_epochs[global_epoch_name]
        t_start, t_end = global_epoch.start_end_times
        short_epoch = owning_pipeline_reference.filtered_epochs[short_epoch_name]
        split_time_t: float = short_epoch.t_start
        
        quantile_diffs_collector = plot_quantile_diffs(ripple_merged_complete_epoch_stats_df, t_start=t_start, t_split=split_time_t, t_end=t_end, active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback)


        post_title_info: str = f'{minimum_inclusion_fr_Hz} Hz'
        collector_histograms = plot_rank_order_histograms(rank_order_results, post_title_info=post_title_info, active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback)

        histogram_display_context = active_context.adding_context('display_fn', display_fn_name='plot_histograms')
        _out_ripple_result_tuple_histograms = ripple_result_tuple.plot_histograms() # MatplotlibRenderPlots num='ripple_result_tuple', clear=True
        _out_ripple_result_tuple_histograms.context = histogram_display_context.adding_context('subplot', subplot_name='ripple_result_tuple')
        setup_histogram_common_after_creation(fig=_out_ripple_result_tuple_histograms.figures[0], axes=_out_ripple_result_tuple_histograms.axes, sub_context=_out_ripple_result_tuple_histograms.context,
                                    title=f'<size:22> Histogram: <weight:bold>ripple_result_tuple</></>', perform_write_to_file_callback=_perform_write_to_file_callback)

        _out_laps_result_tuple_histograms = laps_result_tuple.plot_histograms() # num='laps_result_tuple', clear=True
        _out_laps_result_tuple_histograms.context = histogram_display_context.adding_context('subplot', subplot_name='laps_result_tuple')
        setup_histogram_common_after_creation(fig=_out_laps_result_tuple_histograms.figures[0], axes=_out_laps_result_tuple_histograms.axes, sub_context=_out_laps_result_tuple_histograms.context,
                                    title=f'<size:22> Histogram: <weight:bold>laps_result_tuple</></>', perform_write_to_file_callback=_perform_write_to_file_callback)

        ## PyQtGraph Outputs:
        ripple_outputs = plot_rank_order_epoch_inst_fr_result_tuples(owning_pipeline_reference, ripple_result_tuple, 'Ripple', active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback, show=should_show)
        lap_outputs = plot_rank_order_epoch_inst_fr_result_tuples(owning_pipeline_reference, laps_result_tuple, 'Lap', active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback, show=should_show)


        return quantile_diffs_collector, collector_histograms


