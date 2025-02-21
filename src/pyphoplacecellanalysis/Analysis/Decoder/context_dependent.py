# from __future__ import annotations # prevents having to specify types for typehinting as strings
# from typing import TYPE_CHECKING
# import numpy as np
# import pandas as pd
# from attrs import define, field, Factory, asdict, astuple
# from functools import wraps
# from copy import deepcopy
# from collections import namedtuple
# from pathlib import Path
# from datetime import datetime, date, timedelta

# from typing import Dict, List, Tuple, Optional, Callable, Union, Any, Iterable
# from typing_extensions import TypeAlias
# # from nptyping import NDArray
# from numpy.typing import NDArray  # Correct import for NDArray
# # from nptyping import NDArray
# from typing import NewType
# import neuropy.utils.type_aliases as types
# # DecoderName = NewType('DecoderName', str)

# from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
# from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
# from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
# from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
# from pyphocorehelpers.print_helpers import strip_type_str_to_classname
# from pyphocorehelpers.exception_helpers import ExceptionPrintingContext
# from neuropy.utils.indexing_helpers import NumpyHelpers
# from pyphocorehelpers.assertion_helpers import Assert

# from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
# from neuropy.utils.result_context import IdentifyingContext
# from neuropy.utils.dynamic_container import DynamicContainer
# from neuropy.utils.mixins.dict_representable import override_dict # used to build config
# from neuropy.analyses.placefields import PlacefieldComputationParameters
# from neuropy.core.epoch import NamedTimerange, Epoch, ensure_dataframe
# from neuropy.core.epoch import find_data_indicies_from_epoch_times
# from neuropy.utils.indexing_helpers import union_of_arrays # `paired_incremental_sort_neurons`
# from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr
# from neuropy.utils.mixins.HDF5_representable import HDFMixin
# from neuropy.utils.indexing_helpers import PandasHelpers, NumpyHelpers, flatten
 
# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used for `complete_directional_pfs_computations`
# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult # needed in DirectionalPseudo2DDecodersResult
# from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult

# import scipy.stats
# from scipy import ndimage

# from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

# from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability

# decoder_name_str: TypeAlias = str # an string name of a particular decoder, such as 'Long_LR' or 'Short_RL'

# from pyphocorehelpers.programming_helpers import metadata_attributes
# from pyphocorehelpers.function_helpers import function_attributes

# if TYPE_CHECKING:
#     ## typehinting only imports here
#     # from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, SingleEpochDecodedResult #typehinting only
#     # from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
#     from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig


# # DecodedMarginalResultTuple: TypeAlias = Tuple[List[DynamicContainer], NDArray[float], NDArray[int], NDArray[bool]] # used by 
# DecodedMarginalResultTuple: TypeAlias = Tuple[
#     List[DynamicContainer],
#     NDArray[np.float_],
#     NDArray[np.int_],
#     NDArray[np.bool_]
# ]




# # 2025-02-21 - I realized that `TrackTemplates` and `DirectionalLapsResult` are pointlessly hard-coded to require directionality, but a more general solution would be just to allow dicts of decoders.








# # # Define the namedtuple
# LongShortDecodersTuple = namedtuple('LongShortDecodersTuple', ['long', 'short'])

# @metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 13:19', related_items=[])
# class GenericTrackTemplatesMixin:
    
#     def get_decoders(self) -> Tuple[BasePositionDecoder]:
#         """
#         long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
#         """
#         return LongShortDecodersTuple(list(self.decoders.values()))

#     @classmethod
#     def get_decoder_names(cls) -> Tuple[str]:
#         raise NotImplementedError(f'because this is a class method we must override!')
#         # return tuple(list(self.decoders.keys())) # ('long_LR','long_RL','short_LR','short_RL')
    
#     def get_decoders_dict(self) -> Dict[types.DecoderName, BasePositionDecoder]:
#         return self.decoders
    


# @define(slots=False, repr=False, eq=False)
# class BaseTrackTemplates(HDFMixin, AttrsBasedClassHelperMixin, GenericTrackTemplatesMixin):
#     """ Holds the four directional templates for direction placefield analysis.
#     from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

#     History:
#         Based off of `ShuffleHelper` on 2023-10-27
#         TODO: eliminate functional overlap with `ShuffleHelper`
#         TODO: should be moved into `DirectionalPlacefieldGlobalComputation` instead of RankOrder

#     """
#     decoders: Dict[types.DecoderName, BasePositionDecoder] = serialized_field(repr=False)


#     @property
#     def decoder_neuron_IDs_list(self) -> List[NDArray]:
#         """ a list of the neuron_IDs for each decoder (independently) """
#         return [a_decoder.pf.ratemap.neuron_ids for a_decoder in (list(self.decoders.values()))]
    
#     @property
#     def any_decoder_neuron_IDs(self) -> NDArray:
#         """ a list of the neuron_IDs for each decoder (independently) """
#         return np.sort(union_of_arrays(*self.decoder_neuron_IDs_list)) # neuron_IDs as they appear in any list

#     @property
#     def decoder_peak_location_list(self) -> List[NDArray]:
#         """ a list of the peak_tuning_curve_center_of_masses for each decoder (independently) """
#         return [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses for a_decoder in (list(self.decoders.values()))]
    
#     @property
#     def decoder_peak_rank_list_dict(self) -> Dict[str, NDArray]:
#         """ a dict (one for each decoder) of the rank_lists for each decoder (independently) """
#         return {a_decoder_name:scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
#     @property
#     def decoder_aclu_peak_rank_dict_dict(self) -> Dict[str, Dict[types.aclu_index, float]]:
#         """ a Dict (one for each decoder) of aclu-to-rank maps for each decoder (independently) """
#         return {a_decoder_name:dict(zip(a_decoder.pf.ratemap.neuron_ids, scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method))) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
#     @property
#     def decoder_normalized_tuning_curves_dict_dict(self) -> Dict[str, Dict[types.aclu_index, NDArray]]:
#         """ a Dict (one for each decoder) of aclu-to-1D normalized placefields for each decoder (independently) """
#         return {a_name:a_decoder.pf.normalized_tuning_curves_dict for a_name, a_decoder in self.get_decoders_dict().items()}
            

#     @property
#     def decoder_stability_dict_dict(self): # -> Dict[str, Dict[types.aclu_index, NDArray]]:
#         # """ a Dict (one for each decoder) of aclu-to-1D normalized placefields for each decoder (independently) """
#         return {a_name:a_decoder.pf.ratemap.spatial_sparcity for a_name, a_decoder in self.get_decoders_dict().items()}
    

#     def get_decoders_tuning_curve_modes(self, peak_mode='peaks', **find_peaks_kwargs) -> Tuple[Dict[decoder_name_str, Dict[types.aclu_index, NDArray]], Dict[decoder_name_str, Dict[types.aclu_index, int]], Dict[decoder_name_str, pd.DataFrame]]:
#         """ 2023-12-19 - Uses `scipy.signal.find_peaks to find the number of peaks or ("modes") for each of the cells in the ratemap. 
#         Can detect bimodal (or multi-modal) placefields.
        
#         Depends on:
#             self.tuning_curves
        
#         Returns:
#             aclu_n_peaks_dict: Dict[int, int] - A mapping between aclu:n_tuning_curve_modes
#         Usage:    
#             decoder_peaks_dict_dict, decoder_aclu_n_peaks_dict_dict, decoder_peaks_results_df_dict = track_templates.get_decoders_tuning_curve_modes()

#         """
#         decoder_peaks_results_tuples_dict = {a_decoder_name:a_decoder.pf.ratemap.compute_tuning_curve_modes(peak_mode=peak_mode, **find_peaks_kwargs) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
#         # each tuple contains: peaks_dict, aclu_n_peaks_dict, peaks_results_df, so unwrap below
        
#         decoder_peaks_dict_dict = {k:v[0] for k,v in decoder_peaks_results_tuples_dict.items()}
#         decoder_aclu_n_peaks_dict_dict = {k:v[1] for k,v in decoder_peaks_results_tuples_dict.items()}
#         decoder_peaks_results_df_dict = {k:v[2] for k,v in decoder_peaks_results_tuples_dict.items()}

#         # return peaks_dict, aclu_n_peaks_dict, unimodal_peaks_dict, peaks_results_dict
#         return decoder_peaks_dict_dict, decoder_aclu_n_peaks_dict_dict, decoder_peaks_results_df_dict
    

#     @function_attributes(short_name=None, tags=['WORKING', 'peak', 'multi-peak', 'decoder', 'pfs'], input_requires=[], output_provides=[], uses=['get_tuning_curve_peak_positions'], used_by=['add_directional_pf_maximum_peaks'], creation_date='2024-05-21 19:00', related_items=[])
#     def get_directional_pf_maximum_peaks_dfs(self, drop_aclu_if_missing_long_or_short: bool = True) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
#         """ The only version that only gets the maximum peaks appropriate for each decoder.

#         # 2024-05-21 - Replaces `.get_decoders_aclu_peak_location_df(...)` for properly getting peak locations. Is correct (which is why the old result was replaced) but has a potential drawback of not currently accepting `, **find_peaks_kwargs`. I only see `width=None` ever passed in like this though.

#         # 2024-04-09 00:36: - [X] Could be refactored into TrackTemplates

#         #TODO 2024-05-21 22:53: - [ ] Noticed that short track always has all-non-NaN peaks (has a value for each peak) and long track is missing values. This doesn't make sense because many of the peaks indicated for short occur only on the long-track, which makes no sense.

#         Usage:

#             (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df = track_templates.get_directional_pf_maximum_peaks_dfs(drop_aclu_if_missing_long_or_short=False)

#             AnyDir_decoder_aclu_MAX_peak_maps_df
#             LR_only_decoder_aclu_MAX_peak_maps_df
#             RL_only_decoder_aclu_MAX_peak_maps_df


#         """
#         # drop_aclu_if_missing_long_or_short: bool = True ## default=True; Drop entire row if either long/short is missing a value
#         # drop_aclu_if_missing_long_or_short: bool = False
#         from neuropy.utils.indexing_helpers import intersection_of_arrays, union_of_arrays
#         from neuropy.utils.indexing_helpers import unwrap_single_item


#         ## Split into LR/RL groups to get proper peak differences:
#         # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
#         LR_decoder_names = self.get_LR_decoder_names() # ['long_LR', 'short_LR']
#         RL_decoder_names = self.get_RL_decoder_names() # ['long_RL', 'short_RL']

#         ## Only the maximums (height=1 items), guaranteed to be a single (or None) location:
#         decoder_aclu_MAX_peak_maps_dict: Dict[types.DecoderName, Dict[types.aclu_index, Optional[float]]] = {types.DecoderName(a_name):{k:unwrap_single_item(v) for k, v in deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.get_tuning_curve_peak_positions(peak_mode='peaks', height=1)))).items()} for a_name, a_decoder in self.get_decoders_dict().items()}
#         #TODO 2024-05-21 22:59: - [ ] NEed to ensure that `a_decoder.neuron_IDs` and `a_decoder.get_tuning_curve_peak_positions(peak_mode='peaks', height=1)` are returned in the same order for sure
#             # it should because it's dependent only on `pdf_normalized_tuning_curves`, which is in the neuron_IDs order. The only issue could be if the subpeaks sorting issue happens

#         # decoder_aclu_MAX_peak_maps_dict
#         AnyDir_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in (LR_decoder_names + RL_decoder_names)}) # either direction decoder

#         ## Splits by direction:
#         LR_only_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in LR_decoder_names})
#         RL_only_decoder_aclu_MAX_peak_maps_df: pd.DataFrame = pd.DataFrame({k:v for k,v in decoder_aclu_MAX_peak_maps_dict.items() if k in RL_decoder_names})

#         ## Drop entire row if either long/short is missing a value:
#         if drop_aclu_if_missing_long_or_short:
#             LR_only_decoder_aclu_MAX_peak_maps_df = LR_only_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any')
#             RL_only_decoder_aclu_MAX_peak_maps_df = RL_only_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any')

#             AnyDir_decoder_aclu_MAX_peak_maps_df = AnyDir_decoder_aclu_MAX_peak_maps_df.dropna(axis=0, how='any') # might need to think this through a little better. Currently only using the `AnyDir_*` result with `drop_aclu_if_missing_long_or_short == False`

#         ## Compute the difference between the Long/Short peaks: I don't follow this:
#         LR_only_decoder_aclu_MAX_peak_maps_df['peak_diff'] = LR_only_decoder_aclu_MAX_peak_maps_df.diff(axis='columns').to_numpy()[:, -1]
#         RL_only_decoder_aclu_MAX_peak_maps_df['peak_diff'] = RL_only_decoder_aclu_MAX_peak_maps_df.diff(axis='columns').to_numpy()[:, -1]

#         AnyDir_decoder_aclu_MAX_peak_maps_df['peak_diff_LR'] = AnyDir_decoder_aclu_MAX_peak_maps_df[list(LR_decoder_names)].diff(axis='columns').to_numpy()[:, -1]
#         AnyDir_decoder_aclu_MAX_peak_maps_df['peak_diff_RL'] = AnyDir_decoder_aclu_MAX_peak_maps_df[list(RL_decoder_names)].diff(axis='columns').to_numpy()[:, -1]

#         # OUTPUTS: LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df
#         return (LR_only_decoder_aclu_MAX_peak_maps_df, RL_only_decoder_aclu_MAX_peak_maps_df), AnyDir_decoder_aclu_MAX_peak_maps_df


#     @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
#     def get_decoder_aclu_peak_maps(self, peak_mode='CoM') -> DirectionalDecodersTuple:
#         """ returns a tuple of dicts, each containing a mapping between aclu:peak_pf_x for a given decoder. 
         
#         # Naievely:
#         long_LR_aclu_peak_map = deepcopy(dict(zip(self.long_LR_decoder.neuron_IDs, self.long_LR_decoder.peak_locations)))
#         long_RL_aclu_peak_map = deepcopy(dict(zip(self.long_RL_decoder.neuron_IDs, self.long_RL_decoder.peak_locations)))
#         short_LR_aclu_peak_map = deepcopy(dict(zip(self.short_LR_decoder.neuron_IDs, self.short_LR_decoder.peak_locations)))
#         short_RL_aclu_peak_map = deepcopy(dict(zip(self.short_RL_decoder.neuron_IDs, self.short_RL_decoder.peak_locations)))
        
#         """
#         assert peak_mode in ['peaks', 'CoM']
#         if peak_mode == 'peaks':
#             # return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.get_tuning_curve_peak_positions(peak_mode=peak_mode)))) for a_decoder in (list(self.decoders.values()))])
#             return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_locations))) for a_decoder in (list(self.decoders.values()))]) ## #TODO 2024-02-16 04:27: - [ ] This uses .peak_locations which are the positions corresponding to the peak position bin (but not continuously the peak from the curve).
#         elif peak_mode == 'CoM':
#             return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_tuning_curve_center_of_masses))) for a_decoder in (list(self.decoders.values()))])
#         else:
#             raise NotImplementedError(f"peak_mode: '{peak_mode}' is not supported.")
    

#     @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-06 00:00', related_items=[])
#     def get_decoder_aclu_peak_map_dict(self, peak_mode='CoM') -> Dict[decoder_name_str, Dict]:
#         return dict(zip(self.get_decoder_names(), self.get_decoder_aclu_peak_maps(peak_mode=peak_mode)))


#     def __repr__(self):
#         """ 
#         TrackTemplates(long_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             long_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             short_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             short_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
#             shared_LR_aclus_only_neuron_IDs: numpy.ndarray,
#             is_good_LR_aclus: NoneType,
#             shared_RL_aclus_only_neuron_IDs: numpy.ndarray,
#             is_good_RL_aclus: NoneType,
#             decoder_LR_pf_peak_ranks_list: list,
#             decoder_RL_pf_peak_ranks_list: list
#         )
#         """
#         content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
#         return f"{type(self).__name__}({content}\n)"



#     # ==================================================================================================================== #
#     # GenericTrackTemplatesMixin                                                                                           #
#     # ==================================================================================================================== #
#     def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
#         """
#         long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
#         """
#         return DirectionalDecodersTuple(list(self.decoders.values()))

#     @classmethod
#     def get_decoder_names(cls) -> Tuple[str, str, str, str]:
#         return ('long_LR','long_RL','short_LR','short_RL')
    
    
#     def get_decoders_dict(self) -> Dict[types.DecoderName, BasePositionDecoder]:
#         return {'long_LR': self.long_LR_decoder,
#             'long_RL': self.long_RL_decoder,
#             'short_LR': self.short_LR_decoder,
#             'short_RL': self.short_RL_decoder,
#         }
    


#     # # Init/ClassMethods __________________________________________________________________________________________________ #

#     # @classmethod
#     # def init_from_paired_decoders(cls, LR_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], RL_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], rank_method:str='average') -> "TrackTemplates":
#     #     """ 2023-10-31 - Extract from pairs

#     #     """
#     #     long_LR_decoder, short_LR_decoder = LR_decoder_pair
#     #     long_RL_decoder, short_RL_decoder = RL_decoder_pair

#     #     shared_LR_aclus_only_neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs)
#     #     assert np.all(short_LR_decoder.neuron_IDs == shared_LR_aclus_only_neuron_IDs), f"{short_LR_decoder.neuron_IDs} != {shared_LR_aclus_only_neuron_IDs}"

#     #     shared_RL_aclus_only_neuron_IDs = deepcopy(long_RL_decoder.neuron_IDs)
#     #     assert np.all(short_RL_decoder.neuron_IDs == shared_RL_aclus_only_neuron_IDs), f"{short_RL_decoder.neuron_IDs} != {shared_RL_aclus_only_neuron_IDs}"

#     #     # is_good_aclus = np.logical_not(np.isin(shared_aclus_only_neuron_IDs, bimodal_exclude_aclus))
#     #     # shared_aclus_only_neuron_IDs = shared_aclus_only_neuron_IDs[is_good_aclus]

#     #     ## 2023-10-11 - Get the long/short peak locations
#     #     # decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
#     #     ## Compute the ranks:
#     #     # decoder_pf_peak_ranks_list = [scipy.stats.rankdata(a_peaks_com, method='dense') for a_peaks_com in decoder_peak_coms_list]

#     #     #TODO 2023-11-21 13:06: - [ ] Note these are in order of the original entries, and do not reflect any sorts or ordering changes.

#     #     return cls(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, None, shared_RL_aclus_only_neuron_IDs, None,
#     #                 decoder_LR_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_LR_decoder, short_LR_decoder)],
#     #                 decoder_RL_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_RL_decoder, short_RL_decoder)],
#     #                 rank_method=rank_method)

#     # @classmethod
#     # def determine_decoder_aclus_filtered_by_frate_and_qclu(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
#     #     """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
#     #     minimum_inclusion_fr_Hz: float = 5.0
#     #     modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

#     #     individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

#     #     Usage:
#     #         filtered_decoder_list, filtered_direction_shared_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate(decoders_dict=track_templates.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

#     #     """
#     #     decoder_names = cls.get_decoder_names() # ('long_LR', 'long_RL', 'short_LR', 'short_RL')
#     #     modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=decoders_dict, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
#     #     # individual_decoder_filtered_aclus_list = list(modified_neuron_ids_dict.values())
#     #     individual_decoder_filtered_aclus_list = [modified_neuron_ids_dict[a_decoder_name] for a_decoder_name in decoder_names]
#     #     assert len(individual_decoder_filtered_aclus_list) == 4, f"len(individual_decoder_filtered_aclus_list): {len(individual_decoder_filtered_aclus_list)} but expected 4!"
#     #     original_decoder_list = [deepcopy(decoders_dict[a_decoder_name]) for a_decoder_name in decoder_names]
#     #     ## For a given run direction (LR/RL) let's require inclusion in either (OR) long v. short to be included.
#     #     filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])
#     #     filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])
#     #     # build the final shared aclus:
#     #     filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus] # contains the shared aclus for that direction
#     #     filtered_decoder_list = [a_decoder.get_by_id(a_filtered_aclus) for a_decoder, a_filtered_aclus in zip(original_decoder_list, filtered_direction_shared_aclus_list)]
#     #     return filtered_decoder_list, filtered_direction_shared_aclus_list
    
#     # @classmethod
#     # def _perform_determine_decoder_aclus_filtered_by_qclu_and_frate(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
#     #     """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`) and their `qclu` values.

#     #     minimum_inclusion_fr_Hz: float = 5.0
#     #     modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

#     #     individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

#     #     Usage:
#     #         modified_neuron_ids_dict = TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=track_templates.get_decoders_dict())
            
#     #         decoders_dict=self.get_decoders_dict()
            
#     #     """
#     #     # original_neuron_ids_list = [a_decoder.pf.ratemap.neuron_ids for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
#     #     original_neuron_ids_dict = {a_decoder_name:deepcopy(a_decoder.pf.ratemap.neuron_ids) for a_decoder_name, a_decoder in decoders_dict.items()}
#     #     if (minimum_inclusion_fr_Hz is not None) and (minimum_inclusion_fr_Hz > 0.0):
#     #         modified_neuron_ids_dict = {a_decoder_name:np.array(a_decoder.pf.ratemap.neuron_ids)[a_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz] for a_decoder_name, a_decoder in decoders_dict.items()}
#     #     else:            
#     #         modified_neuron_ids_dict = {a_decoder_name:deepcopy(a_decoder_neuron_ids) for a_decoder_name, a_decoder_neuron_ids in original_neuron_ids_dict.items()}
        
#     #     if included_qclu_values is not None:
#     #         # filter by included_qclu_values
#     #         for a_decoder_name, a_decoder in decoders_dict.items():
#     #             # a_decoder.pf.spikes_df
#     #             neuron_identities: pd.DataFrame = deepcopy(a_decoder.pf.filtered_spikes_df).spikes.extract_unique_neuron_identities()
#     #             # filtered_neuron_identities: pd.DataFrame = neuron_identities[neuron_identities.neuron_type == NeuronType.PYRAMIDAL]
#     #             filtered_neuron_identities: pd.DataFrame = deepcopy(neuron_identities)
#     #             filtered_neuron_identities = filtered_neuron_identities[['aclu', 'shank', 'cluster', 'qclu']]
#     #             # filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.aclu, original_neuron_ids_dict[a_decoder_name])]
#     #             filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.aclu, modified_neuron_ids_dict[a_decoder_name])] # require to match to decoders
#     #             filtered_neuron_identities = filtered_neuron_identities[np.isin(filtered_neuron_identities.qclu, included_qclu_values)] # drop [6, 7], which are said to have double fields - 80 remain
#     #             final_included_aclus = filtered_neuron_identities['aclu'].to_numpy()
#     #             modified_neuron_ids_dict[a_decoder_name] = deepcopy(final_included_aclus) #.tolist()
                
#     #     return modified_neuron_ids_dict
    

#     # def determine_decoder_aclus_filtered_by_qclu_and_frate(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None):
#     #     """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`) and their `qclu` values.

#     #     minimum_inclusion_fr_Hz: float = 5.0
#     #     modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

#     #     individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

#     #     Usage:
#     #         modified_neuron_ids_dict = TrackTemplates.determine_decoder_aclus_filtered_by_qclu_and_frate(track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)

#     #     """
#     #     return TrackTemplates._perform_determine_decoder_aclus_filtered_by_qclu_and_frate(decoders_dict=self.get_decoders_dict(), minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)

#     # @classmethod
#     # def determine_active_min_num_unique_aclu_inclusions_requirement(cls, min_num_unique_aclu_inclusions: int, total_num_cells: int, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> int:
#     #     """ 2023-12-21 - Compute the dynamic minimum number of active cells

#     #         active_min_num_unique_aclu_inclusions_requirement: int = cls.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
#     #                                                                                                                                 total_num_cells=len(any_list_neuron_IDs))

#     #     """
#     #     required_min_percentage_of_active_cells = float(required_min_percentage_of_active_cells)
#     #     if debug_print:
#     #         print(f'required_min_percentage_of_active_cells: {required_min_percentage_of_active_cells}') # 20% of active cells
#     #     dynamic_percentage_minimum_num_unique_aclu_inclusions: int = int(round((float(total_num_cells) * required_min_percentage_of_active_cells))) # dynamic_percentage_minimum_num_unique_aclu_inclusions: the percentage-based requirement for the number of active cells
#     #     active_min_num_unique_aclu_inclusions_requirement: int = max(dynamic_percentage_minimum_num_unique_aclu_inclusions, min_num_unique_aclu_inclusions)
#     #     if debug_print:
#     #         print(f'active_min_num_unique_aclu_inclusions_requirement: {active_min_num_unique_aclu_inclusions_requirement}')
#     #     return active_min_num_unique_aclu_inclusions_requirement


#     # def min_num_unique_aclu_inclusions_requirement(self, curr_active_pipeline, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> int:
#     #     """ 2023-12-21 - Compute the dynamic minimum number of active cells

#     #         active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.3333)

#     #     """
#     #     smallest_template_n_neurons: int = np.min([len(v) for v in self.decoder_neuron_IDs_list]) # smallest_template_n_neurons: the fewest number of neurons any template has
#     #     ## Compute the dynamic minimum number of active cells from current num total cells and the `curr_active_pipeline.sess.config.preprocessing_parameters` values:`
#     #     return self.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
#     #                                                                             total_num_cells=smallest_template_n_neurons, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells)


#     # def min_num_unique_aclu_inclusions_requirement_dict(self, curr_active_pipeline, required_min_percentage_of_active_cells: float = 0.3, debug_print=False) -> Dict[str, int]:
#     #     """ 2023-12-21 - Compute the dynamic minimum number of active cells

#     #         active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.3333)

#     #     """
#     #     decoder_neuron_IDs_dict = dict(zip(self.get_decoder_names(), self.decoder_neuron_IDs_list))
#     #     decoder_num_neurons_dict = {k:len(v) for k, v in decoder_neuron_IDs_dict.items()}
#     #     return {k:self.determine_active_min_num_unique_aclu_inclusions_requirement(min_num_unique_aclu_inclusions=curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays.min_num_unique_aclu_inclusions,
#     #                                                                             total_num_cells=a_n_neurons, required_min_percentage_of_active_cells=required_min_percentage_of_active_cells) for k, a_n_neurons in decoder_num_neurons_dict.items()}


#     # @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=['TransitionMatrixComputations'], used_by=[], creation_date='2024-08-02 07:33', related_items=[])
#     # def compute_decoder_transition_matricies(self, n_powers:int=50, use_direct_observations_for_order:bool=True) -> Dict[types.DecoderName, List[NDArray]]:
#     #     """ Computes the position transition matricies for each of the decoders 
#     #     returns a list of length n_powers for each decoder
        
#     #     Usage:
#     #         binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray] = track_templates.compute_decoder_transition_matricies(n_powers=50)
        
#     #     """
#     #     from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
        
#     #     ## INPUTS: track_templates
#     #     decoders_dict: Dict[types.DecoderName, BasePositionDecoder] = self.get_decoders_dict()
#     #     binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray] = {}

#     #     for a_decoder_name, a_decoder in decoders_dict.items():
#     #         a_pf1D = deepcopy(a_decoder.pf)
#     #         binned_x_transition_matrix_higher_order_list_dict[a_decoder_name] = TransitionMatrixComputations._compute_position_transition_matrix(a_pf1D.xbin_labels, binned_x_index_sequence=(a_pf1D.filtered_pos_df['binned_x'].dropna().to_numpy()-1), n_powers=n_powers, use_direct_observations_for_order=use_direct_observations_for_order) # the -1 here is to convert to (binned_x_index_sequence = binned_x - 1)

#     #     # OUTPUTS: binned_x_transition_matrix_higher_order_list_dict
#     #     return binned_x_transition_matrix_higher_order_list_dict






# @define(slots=False, repr=False)
# class DirectionalLapsResult(ComputedResult):
#     """ a container for holding information regarding the computation of directional laps.

#     ## Build a `DirectionalLapsResult` container object to hold the result:
#     directional_laps_result = DirectionalLapsResult()
#     directional_laps_result.directional_lap_specific_configs = directional_lap_specific_configs
#     directional_laps_result.split_directional_laps_dict = split_directional_laps_dict
#     directional_laps_result.split_directional_laps_contexts_dict = split_directional_laps_contexts_dict
#     directional_laps_result.split_directional_laps_config_names = split_directional_laps_config_names
#     directional_laps_result.computed_base_epoch_names = computed_base_epoch_names

#     # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
#     directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
#     directional_laps_result.long_even_shared_aclus_only_one_step_decoder_1D = long_even_shared_aclus_only_one_step_decoder_1D
#     directional_laps_result.short_odd_shared_aclus_only_one_step_decoder_1D = short_odd_shared_aclus_only_one_step_decoder_1D
#     directional_laps_result.short_even_shared_aclus_only_one_step_decoder_1D = short_even_shared_aclus_only_one_step_decoder_1D


#     long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [directional_laps_results.__dict__[k] for k in ['long_LR_shared_aclus_only_one_step_decoder_1D', 'long_RL_shared_aclus_only_one_step_decoder_1D', 'short_LR_shared_aclus_only_one_step_decoder_1D', 'short_RL_shared_aclus_only_one_step_decoder_1D']]

#     """
#     _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
#     directional_lap_specific_configs: Dict = non_serialized_field(default=Factory(dict))
#     split_directional_laps_dict: Dict = non_serialized_field(default=Factory(dict))
#     split_directional_laps_contexts_dict: Dict = non_serialized_field(default=Factory(dict))
#     split_directional_laps_config_names: List[str] = serialized_field(default=Factory(list))
#     computed_base_epoch_names: List[str] = serialized_field(default=Factory(list))

#     long_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
#     long_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
#     short_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
#     short_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)

#     long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_odd_shared_aclus_only_one_step_decoder_1D')
#     long_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_even_shared_aclus_only_one_step_decoder_1D')
#     short_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_odd_shared_aclus_only_one_step_decoder_1D')
#     short_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_even_shared_aclus_only_one_step_decoder_1D')

#     # long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D

#     def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
#         """
#         long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
#         """
#         return DirectionalDecodersTuple(self.long_LR_one_step_decoder_1D, self.long_RL_one_step_decoder_1D, self.short_LR_one_step_decoder_1D, self.short_RL_one_step_decoder_1D)

#     def get_shared_aclus_only_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
#         """
#         long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()
#         """
#         return DirectionalDecodersTuple(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D)


#     def get_templates(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> TrackTemplates:
#         _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_one_step_decoder_1D, self.short_LR_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_one_step_decoder_1D, self.short_RL_one_step_decoder_1D))
#         if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
#             return _obj
#         else:
#             return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
    

#     def get_shared_aclus_only_templates(self, minimum_inclusion_fr_Hz:Optional[float]=None, included_qclu_values:Optional[List]=None) -> TrackTemplates:
#         _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D))
#         if ((minimum_inclusion_fr_Hz is None) and (included_qclu_values is None)):
#             return _obj
#         else:
#             # return _obj.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
#             return _obj.filtered_by_frate_and_qclu(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        

#     def filtered_by_included_aclus(self, included_neuronIDs) -> "DirectionalLapsResult":
#         """ Returns a copy of self with each decoder filtered by the `qclu_included_aclus`
        
#         Usage:
        
#         qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
#         modified_directional_laps_results = directional_laps_results.filtered_by_included_aclus(included_neuronIDs=qclu_included_aclus)
#         modified_directional_laps_results

#         """
#         directional_laps_results = deepcopy(self)
        
#         decoders_list = [directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D,
#                          directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D
#                         ]
#         modified_decoders_list = []
#         for a_decoder in decoders_list:
#             # a_decoder = deepcopy(directional_laps_results.long_LR_one_step_decoder_1D)
#             is_aclu_qclu_included_list = np.isin(a_decoder.pf.ratemap.neuron_ids, included_neuronIDs)
#             included_aclus = np.array(a_decoder.pf.ratemap.neuron_ids)[is_aclu_qclu_included_list]
#             modified_decoder = a_decoder.get_by_id(included_aclus)
#             modified_decoders_list.append(modified_decoder)

#         ## Assign the modified decoders:
#         directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D, directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D = modified_decoders_list

#         return directional_laps_results
    
#     ## For serialization/pickling:
#     def __getstate__(self):
#         # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
#         state = self.__dict__.copy()
#         return state

#     def __setstate__(self, state):
#         # Restore instance attributes (i.e., _mapping and _keys_at_init).
#         self.__dict__.update(state)
#         # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
#         # super(DirectionalLapsResult, self).__init__() # TypeError: super(type, obj): obj must be an instance or subtype of type.



#     @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=[], used_by=['DirectionalPlacefieldGlobalComputationFunctions._split_to_directional_laps'], creation_date='2025-02-13 16:46', related_items=[])
#     @classmethod
#     def init_from_pipeline_natural_epochs(cls, curr_active_pipeline, progress_print=False) -> "DirectionalLapsResult":
#         """ 2023-10-31 - 4pm  - Main computation function, simply extracts the diretional laps from the existing epochs.

#         PURE?: Does not update `curr_active_pipeline` or mess with its filters/configs/etc.

#                 ## {"even": "RL", "odd": "LR"}

#         #TODO 2023-11-10 21:00: - [ ] Convert above "LR/RL" notation to new "LR/RL" versions:

        
#         History 2025-02-13 16:52 used to be called 'DirectionalLapsHelpers.build_global_directional_result_from_natural_epochs'
        
#         Uses:
        
#             curr_active_pipeline.computation_results[an_epoch_name].computed_data.get('pf1D_Decoder', None)
        
        
#         """

#         long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names() # ('maze1_any', 'maze2_any', 'maze_any')
#         # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
#         # long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.

#         # Unwrap the naturally produced directional placefields:
#         long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
#         # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
#         # (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
#         # long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
#         # (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
#         (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
#         # (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
#         # (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
#         # (long_LR_pf2D, long_RL_pf2D, short_LR_pf2D, short_RL_pf2D) = (long_LR_results.pf2D, long_RL_results.pf2D, short_LR_results.pf2D, short_RL_results.pf2D)
#         # (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

#         # Unpack all directional variables:
#         long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name # ('maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any')

#         # Validate:
#         assert not (curr_active_pipeline.computation_results[long_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
#         assert not (curr_active_pipeline.computation_results[short_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
#         # Fix the computation epochs to be constrained to the proper long/short intervals:
#         was_modified = DirectionalLapsHelpers.fix_computation_epochs_if_needed(curr_active_pipeline=curr_active_pipeline) # cls: DirectionalLapsResult
#         was_modified = was_modified or DirectionalLapsHelpers.fixup_directional_pipeline_if_needed(curr_active_pipeline)
#         print(f'DirectionalLapsResult.init_from_pipeline_natural_epochs(...): was_modified: {was_modified}')

#         # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
#         # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

#         ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
#         long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]


#         #TODO 2023-12-07 20:48: - [ ] It looks like I'm still only looking at the intersection here! Do I want this?

#         ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
#         # Kamran says LR and RL sets should be shared
#         ## Odd Laps:
#         LR_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
#         LR_shared_aclus = np.array(list(set.intersection(*map(set,LR_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
#         LR_n_neurons = len(LR_shared_aclus)
#         if progress_print:
#             print(f'LR_n_neurons: {LR_n_neurons}, LR_shared_aclus: {LR_shared_aclus}')

#         ## Even Laps:
#         RL_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
#         RL_shared_aclus = np.array(list(set.intersection(*map(set,RL_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
#         RL_n_neurons = len(RL_shared_aclus)
#         if progress_print:
#             print(f'RL_n_neurons: {RL_n_neurons}, RL_shared_aclus: {RL_shared_aclus}')

#         # Direction Separate shared_aclus decoders: Odd set is limited to LR_shared_aclus and RL set is limited to RL_shared_aclus:
#         long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(LR_shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
#         long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(RL_shared_aclus) for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

#         ## Build a `DirectionalLapsResult` (a `ComputedResult`) container object to hold the result:
#         directional_laps_result = DirectionalLapsResult(is_global=True, result_version=DirectionalLapsResult._VersionedResultMixin_version)
#         directional_laps_result.directional_lap_specific_configs = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # directional_lap_specific_configs
#         directional_laps_result.split_directional_laps_dict = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)}  # split_directional_laps_dict
#         directional_laps_result.split_directional_laps_contexts_dict = {a_name:curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # split_directional_laps_contexts_dict
#         directional_laps_result.split_directional_laps_config_names = [long_LR_name, long_RL_name, short_LR_name, short_RL_name] # split_directional_laps_config_names

#         # use the constrained epochs:
#         directional_laps_result.long_LR_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.long_RL_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_LR_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_RL_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

#         # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
#         directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.long_RL_shared_aclus_only_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_LR_shared_aclus_only_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
#         directional_laps_result.short_RL_shared_aclus_only_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

#         return directional_laps_result



