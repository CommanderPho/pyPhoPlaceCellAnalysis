import numpy as np
import pandas as pd
from attrs import define, field, Factory, asdict, astuple
from functools import wraps
from copy import deepcopy
from collections import namedtuple
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.print_helpers import strip_type_str_to_classname


from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.dynamic_container import DynamicContainer # used to build config
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.core.epoch import NamedTimerange, Epoch
from neuropy.utils.indexing_helpers import union_of_arrays # `paired_incremental_sort_neurons`
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used for `complete_directional_pfs_computations`
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult # needed in DirectionalMergedDecodersResult
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult


import scipy.stats
from scipy import ndimage
from nptyping import NDArray

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

# Define the namedtuple
DirectionalDecodersTuple = namedtuple('DirectionalDecodersTuple', ['long_LR', 'long_RL', 'short_LR', 'short_RL'])

@define(slots=False, repr=False, eq=False)
class TrackTemplates(HDFMixin):
    """ Holds the four directional templates for direction placefield analysis.
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

    History:
        Based off of `ShuffleHelper` on 2023-10-27
        TODO: eliminate functional overlap with `ShuffleHelper`
        TODO: should be moved into `DirectionalPlacefieldGlobalComputation` instead of RankOrder

    """
    long_LR_decoder: BasePositionDecoder = serialized_field(repr=False)
    long_RL_decoder: BasePositionDecoder = serialized_field(repr=False) # keys_only_repr
    short_LR_decoder: BasePositionDecoder = serialized_field(repr=False)
    short_RL_decoder: BasePositionDecoder = serialized_field(repr=False)

    # ## Computed properties
    shared_LR_aclus_only_neuron_IDs: NDArray = serialized_field(repr=True)
    is_good_LR_aclus: NDArray = serialized_field(repr=False)

    shared_RL_aclus_only_neuron_IDs: NDArray = serialized_field(repr=True)
    is_good_RL_aclus: NDArray = serialized_field(repr=False)

    ## Computed properties
    decoder_LR_pf_peak_ranks_list: List = serialized_field(repr=True)
    decoder_RL_pf_peak_ranks_list: List = serialized_field(repr=True)

    rank_method: str = serialized_attribute_field(default="average", is_computable=False, repr=True)


    @property
    def decoder_neuron_IDs_list(self) -> List[NDArray]:
        """ a list of the neuron_IDs for each decoder (independently) """
        return [a_decoder.pf.ratemap.neuron_ids for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)]
    
    @property
    def any_decoder_neuron_IDs(self) -> NDArray:
        """ a list of the neuron_IDs for each decoder (independently) """
        return np.sort(union_of_arrays(*self.decoder_neuron_IDs_list)) # neuron_IDs as they appear in any list

    @property
    def decoder_peak_location_list(self) -> List[NDArray]:
        """ a list of the peak_tuning_curve_center_of_masses for each decoder (independently) """
        return [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)]
    
    @property
    def decoder_peak_rank_list_dict(self) -> Dict[str, NDArray]:
        """ a dict (one for each decoder) of the rank_lists for each decoder (independently) """
        return {a_decoder_name:scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
    @property
    def decoder_aclu_peak_rank_dict_dict(self) -> Dict[str, Dict[int, float]]:
        """ a Dict (one for each decoder) of aclu-to-rank maps for each decoder (independently) """
        return {a_decoder_name:dict(zip(a_decoder.pf.ratemap.neuron_ids, scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=self.rank_method))) for a_decoder_name, a_decoder in self.get_decoders_dict().items()}
    
    
    

    def get_decoder_aclu_peak_maps(self) -> DirectionalDecodersTuple:
        """ returns a tuple of dicts, each containing a mapping between aclu:peak_pf_x for a given decoder. 
         
        # Naievely:
        long_LR_aclu_peak_map = deepcopy(dict(zip(self.long_LR_decoder.neuron_IDs, self.long_LR_decoder.peak_locations)))
        long_RL_aclu_peak_map = deepcopy(dict(zip(self.long_RL_decoder.neuron_IDs, self.long_RL_decoder.peak_locations)))
        short_LR_aclu_peak_map = deepcopy(dict(zip(self.short_LR_decoder.neuron_IDs, self.short_LR_decoder.peak_locations)))
        short_RL_aclu_peak_map = deepcopy(dict(zip(self.short_RL_decoder.neuron_IDs, self.short_RL_decoder.peak_locations)))
        
        """
        # return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_locations))) for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)])
        return DirectionalDecodersTuple(*[deepcopy(dict(zip(a_decoder.neuron_IDs, a_decoder.peak_tuning_curve_center_of_masses))) for a_decoder in (self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)])

    def get_decoder_aclu_peak_map_dict(self) -> Dict[str, Dict]:
        return dict(zip(self.get_decoder_names(), self.get_decoder_aclu_peak_maps()))


    def __repr__(self):
        """ 
        TrackTemplates(long_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            long_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            short_LR_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            short_RL_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder,
            shared_LR_aclus_only_neuron_IDs: numpy.ndarray,
            is_good_LR_aclus: NoneType,
            shared_RL_aclus_only_neuron_IDs: numpy.ndarray,
            is_good_RL_aclus: NoneType,
            decoder_LR_pf_peak_ranks_list: list,
            decoder_RL_pf_peak_ranks_list: list
        )
        """
        # content = ", ".join( [f"{a.name}={v!r}" for a in self.__attrs_attrs__ if (v := getattr(self, a.name)) != a.default] )
        # content = ", ".join([f"{a.name}:{strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        # content = ", ".join([f"{a.name}" for a in self.__attrs_attrs__]) # 'TrackTemplates(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, is_good_LR_aclus, shared_RL_aclus_only_neuron_IDs, is_good_RL_aclus, decoder_LR_pf_peak_ranks_list, decoder_RL_pf_peak_ranks_list)'
        return f"{type(self).__name__}({content}\n)"


    def filtered_by_frate(self, minimum_inclusion_fr_Hz: float = 5.0) -> "TrackTemplates":
        """ Does not modify self! Returns a copy! Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

        Usage:
            minimum_inclusion_fr_Hz: float = 5.0
            filtered_decoder_list = [filtered_by_frate(a_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True) for a_decoder in (track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)]

        """
        filtered_decoder_list, filtered_direction_shared_aclus_list, is_aclu_included_list, individual_decoder_filtered_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate(self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)
        long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = filtered_decoder_list # unpack
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_LR_decoder, short_LR_decoder), RL_decoder_pair=(long_RL_decoder, short_RL_decoder), rank_method=self.rank_method)
        assert np.all(filtered_direction_shared_aclus_list[0] == _obj.shared_LR_aclus_only_neuron_IDs)
        assert np.all(filtered_direction_shared_aclus_list[1] == _obj.shared_RL_aclus_only_neuron_IDs)
        assert len(filtered_direction_shared_aclus_list[0]) == len(_obj.decoder_LR_pf_peak_ranks_list[0])
        assert len(filtered_direction_shared_aclus_list[1]) == len(_obj.decoder_RL_pf_peak_ranks_list[0])
        return _obj

    def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
        """
        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
        """
        return DirectionalDecodersTuple(self.long_LR_decoder, self.long_RL_decoder, self.short_LR_decoder, self.short_RL_decoder)

    def get_decoder_names(self) -> Tuple[str]:
        return ('long_LR','long_RL','short_LR','short_RL')
        

    def get_decoders_dict(self) -> Dict[str, BasePositionDecoder]:
        return {'long_LR': self.long_LR_decoder,
            'long_RL': self.long_RL_decoder,
            'short_LR': self.short_LR_decoder,
            'short_RL': self.short_RL_decoder,
        }

    @classmethod
    def init_from_paired_decoders(cls, LR_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], RL_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], rank_method:str='average') -> "TrackTemplates":
        """ 2023-10-31 - Extract from pairs

        """
        long_LR_decoder, short_LR_decoder = LR_decoder_pair
        long_RL_decoder, short_RL_decoder = RL_decoder_pair

        shared_LR_aclus_only_neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs)
        assert np.all(short_LR_decoder.neuron_IDs == shared_LR_aclus_only_neuron_IDs), f"{short_LR_decoder.neuron_IDs} != {shared_LR_aclus_only_neuron_IDs}"

        shared_RL_aclus_only_neuron_IDs = deepcopy(long_RL_decoder.neuron_IDs)
        assert np.all(short_RL_decoder.neuron_IDs == shared_RL_aclus_only_neuron_IDs), f"{short_RL_decoder.neuron_IDs} != {shared_RL_aclus_only_neuron_IDs}"

        # is_good_aclus = np.logical_not(np.isin(shared_aclus_only_neuron_IDs, bimodal_exclude_aclus))
        # shared_aclus_only_neuron_IDs = shared_aclus_only_neuron_IDs[is_good_aclus]

        ## 2023-10-11 - Get the long/short peak locations
        # decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        ## Compute the ranks:
        # decoder_pf_peak_ranks_list = [scipy.stats.rankdata(a_peaks_com, method='dense') for a_peaks_com in decoder_peak_coms_list]

        #TODO 2023-11-21 13:06: - [ ] Note this are in order of the original entries, and do not reflect any sorts or ordering changes.


        return cls(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, None, shared_RL_aclus_only_neuron_IDs, None,
                    decoder_LR_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_LR_decoder, short_LR_decoder)],
                    decoder_RL_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method=rank_method) for a_decoder in (long_RL_decoder, short_RL_decoder)],
                    rank_method=rank_method)

    @classmethod
    def determine_decoder_aclus_filtered_by_frate(cls, long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, minimum_inclusion_fr_Hz: float = 5.0):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)

        individual_decoder_filtered_aclus_list: list of four lists of aclus, not constrained to have the same aclus as its long/short pair

        Usage:
            filtered_decoder_list, filtered_direction_shared_aclus_list, is_aclu_included_list, individual_decoder_filtered_aclus_list = TrackTemplates.determine_decoder_aclus_filtered_by_frate(track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)

        """
        original_neuron_ids_list = [a_decoder.pf.ratemap.neuron_ids for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        is_aclu_included_list = [a_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]
        individual_decoder_filtered_aclus_list = [np.array(a_decoder.pf.ratemap.neuron_ids)[a_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz] for a_decoder in (long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder)]

        ## For a given run direction (LR/RL) let's require inclusion in either (OR) long v. short to be included.
        filtered_included_LR_aclus = np.union1d(individual_decoder_filtered_aclus_list[0], individual_decoder_filtered_aclus_list[2])
        filtered_included_RL_aclus = np.union1d(individual_decoder_filtered_aclus_list[1], individual_decoder_filtered_aclus_list[3])
        # build the final shared aclus:
        filtered_direction_shared_aclus_list = [filtered_included_LR_aclus, filtered_included_RL_aclus, filtered_included_LR_aclus, filtered_included_RL_aclus] # contains the shared aclus for that direction
        # rebuild the is_aclu_included_list from the shared aclus
        is_aclu_included_list = [np.isin(an_original_neuron_ids, a_filtered_neuron_ids) for an_original_neuron_ids, a_filtered_neuron_ids in zip(original_neuron_ids_list, filtered_direction_shared_aclus_list)]

        filtered_decoder_list = [a_decoder.get_by_id(a_filtered_aclus) for a_decoder, a_filtered_aclus in zip((long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder), filtered_direction_shared_aclus_list)]

        return filtered_decoder_list, filtered_direction_shared_aclus_list, is_aclu_included_list, individual_decoder_filtered_aclus_list


@define(slots=False, repr=False)
class DirectionalLapsResult(ComputedResult):
    """ a container for holding information regarding the computation of directional laps.

    ## Build a `DirectionalLapsResult` container object to hold the result:
    directional_laps_result = DirectionalLapsResult()
    directional_laps_result.directional_lap_specific_configs = directional_lap_specific_configs
    directional_laps_result.split_directional_laps_dict = split_directional_laps_dict
    directional_laps_result.split_directional_laps_contexts_dict = split_directional_laps_contexts_dict
    directional_laps_result.split_directional_laps_config_names = split_directional_laps_config_names
    directional_laps_result.computed_base_epoch_names = computed_base_epoch_names

    # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
    directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.long_even_shared_aclus_only_one_step_decoder_1D = long_even_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.short_odd_shared_aclus_only_one_step_decoder_1D = short_odd_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.short_even_shared_aclus_only_one_step_decoder_1D = short_even_shared_aclus_only_one_step_decoder_1D


    long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [directional_laps_results.__dict__[k] for k in ['long_LR_shared_aclus_only_one_step_decoder_1D', 'long_RL_shared_aclus_only_one_step_decoder_1D', 'short_LR_shared_aclus_only_one_step_decoder_1D', 'short_RL_shared_aclus_only_one_step_decoder_1D']]

    """
    directional_lap_specific_configs: Dict = non_serialized_field(default=Factory(dict))
    split_directional_laps_dict: Dict = non_serialized_field(default=Factory(dict))
    split_directional_laps_contexts_dict: Dict = non_serialized_field(default=Factory(dict))
    split_directional_laps_config_names: List[str] = serialized_field(default=Factory(list))
    computed_base_epoch_names: List[str] = serialized_field(default=Factory(list))

    long_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
    long_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
    short_LR_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)
    short_RL_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None)

    long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_odd_shared_aclus_only_one_step_decoder_1D')
    long_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_even_shared_aclus_only_one_step_decoder_1D')
    short_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_odd_shared_aclus_only_one_step_decoder_1D')
    short_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_even_shared_aclus_only_one_step_decoder_1D')

    # long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D

    def get_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
        """
        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
        """
        return DirectionalDecodersTuple(self.long_LR_one_step_decoder_1D, self.long_RL_one_step_decoder_1D, self.short_LR_one_step_decoder_1D, self.short_RL_one_step_decoder_1D)

    def get_shared_aclus_only_decoders(self) -> Tuple[BasePositionDecoder, BasePositionDecoder, BasePositionDecoder, BasePositionDecoder]:
        """
        long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()
        """
        return DirectionalDecodersTuple(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D)


    def get_templates(self, minimum_inclusion_fr_Hz: Optional[float] = None) -> TrackTemplates:
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_one_step_decoder_1D, self.short_LR_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_one_step_decoder_1D, self.short_RL_one_step_decoder_1D))
        if minimum_inclusion_fr_Hz is None:
            return _obj
        else:
            return _obj.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)

    def get_shared_aclus_only_templates(self, minimum_inclusion_fr_Hz: Optional[float] = None) -> TrackTemplates:
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(self.long_LR_shared_aclus_only_one_step_decoder_1D, self.short_LR_shared_aclus_only_one_step_decoder_1D), RL_decoder_pair=(self.long_RL_shared_aclus_only_one_step_decoder_1D, self.short_RL_shared_aclus_only_one_step_decoder_1D))
        if minimum_inclusion_fr_Hz is None:
            return _obj
        else:
            return _obj.filtered_by_frate(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz)


    def filtered_by_included_aclus(self, qclu_included_aclus) -> "DirectionalLapsResult":
        """ Returns a copy of self with each decoder filtered by the `qclu_included_aclus`
        
        Usage:
        
        qclu_included_aclus = curr_active_pipeline.determine_good_aclus_by_qclu(included_qclu_values=[1,2,4,9])
        modified_directional_laps_results = directional_laps_results.filtered_by_included_aclus(qclu_included_aclus)
        modified_directional_laps_results

        """
        directional_laps_results = deepcopy(self)
        
        decoders_list = [directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D,
                         directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D
                        ]
        modified_decoders_list = []
        for a_decoder in decoders_list:
            # a_decoder = deepcopy(directional_laps_results.long_LR_one_step_decoder_1D)
            is_aclu_qclu_included_list = np.isin(a_decoder.pf.ratemap.neuron_ids, qclu_included_aclus)
            included_aclus = np.array(a_decoder.pf.ratemap.neuron_ids)[is_aclu_qclu_included_list]
            modified_decoder = a_decoder.get_by_id(included_aclus)
            modified_decoders_list.append(modified_decoder)

        ## Assign the modified decoders:
        directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D, directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D = modified_decoders_list

        return directional_laps_results
    
    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        super(DirectionalLapsResult, self).__init__() # from




class DirectionalLapsHelpers:
    """ 2023-10-24 - Directional Placefields Computations

    use_direction_dependent_laps

    from neuropy.core.laps import Laps
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

    curr_active_pipeline, directional_lap_specific_configs = DirectionalLapsHelpers.split_to_directional_laps(curr_active_pipeline=curr_active_pipeline, add_created_configs_to_pipeline=True)




    Computing:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

        # Run directional laps and set the global result:
        curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] = DirectionalLapsHelpers.complete_directional_pfs_computations(curr_active_pipeline)


    """


    lap_direction_suffix_list = ['_odd', '_even', '_any'] # ['maze1_odd', 'maze1_even', 'maze1_any', 'maze2_odd', 'maze2_even', 'maze2_any', 'maze_odd', 'maze_even', 'maze_any']
    # lap_direction_suffix_list = ['_odd', '_even', ''] # no '_any' prefix, instead reuses the existing names
    split_directional_laps_name_parts = ['odd_laps', 'even_laps'] # , 'any_laps'

    split_all_laps_name_parts = ['odd_laps', 'even_laps', 'any']
    # ['maze_even_laps', 'maze_odd_laps']

    @classmethod
    def validate_has_directional_laps(cls, curr_active_pipeline, computation_filter_name='maze'):
        # Unpacking:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = directional_laps_results.directional_lap_specific_configs, directional_laps_results.split_directional_laps_dict, directional_laps_results.split_directional_laps_config_names, directional_laps_results.computed_base_epoch_names

        long_LR_one_step_decoder_1D, long_RL_one_step_decoder_1D, short_LR_one_step_decoder_1D, short_RL_one_step_decoder_1D = directional_laps_results.get_decoders()
        long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = directional_laps_results.get_shared_aclus_only_decoders()

        # assert (computation_filter_name in computed_base_epoch_names), f'computation_filter_name: {computation_filter_name} is missing from computed_base_epoch_names: {computed_base_epoch_names} '
        return (computation_filter_name in split_directional_laps_config_names)
        # return (computation_filter_name in computed_base_epoch_names)

    @classmethod
    def has_duplicated_memory_references(cls, *args) -> bool:
        # Check for duplicated memory references in the configs first:
        memory_ids = [id(a_config) for a_config in args] # YUP, they're different for odd/even but duplicated for long/short
        has_duplicated_reference: bool = len(np.unique(memory_ids)) < len(memory_ids)
        return has_duplicated_reference

    @classmethod
    def deduplicate_memory_references(cls, *args) -> list:
        """ Ensures that all entries in the args list point to unique memory addresses, deduplicating them with `deepcopy` if needed.

        Usage:

            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

            args = DirectionalLapsHelpers.deduplicate_memory_references(args)

        """
        has_duplicated_reference: bool = cls.has_duplicated_memory_references(*args)
        if has_duplicated_reference:
            de_deuped_args = [deepcopy(v) for v in args]
            assert not cls.has_duplicated_memory_references(*de_deuped_args), f"duplicate memory references still exist even after de-duplicating with deepcopy!!!"
            return de_deuped_args
        else:
            return args

    @classmethod
    def post_fixup_filtered_contexts(cls, curr_active_pipeline, debug_print=False) -> bool:
        """ 2023-10-24 - tries to update misnamed `curr_active_pipeline.filtered_contexts`

            curr_active_pipeline.filtered_contexts with correct filter_names

            Uses: `curr_active_pipeline.filtered_epoch`
            Updates: `curr_active_pipeline.filtered_contexts`

        Still needed for 2023-11-29 to add back in the 'lap_dir' key

        History: factored out of BatchCompletionHandler

        NOTE: works for non-directional contexts as well, fixing `filter_name` as needed.


        """
        was_updated = False
        for a_name, a_named_timerange in curr_active_pipeline.filtered_epochs.items():
            # `desired_filter_name`: the correct name to be set as the .filter_name in the context
            # 2023-11-29 - as of right now, I think the full name including the lap_dir 'maze1_any' (mode 2) should be used as this is literally what the name of the corresponding filtering function is.
            # desired_filter_name:str = a_named_timerange.name # mode 1: uses the period name 'maze1' without the lap_dir part, probably best for compatibility in most places
            desired_filter_name:str = a_name  # mode 2: uses the config_name: 'maze1_any', includes the lap_dir part

            if debug_print:
                print(f'"{a_name}" - desired_filter_name: "{desired_filter_name}"')
            a_filtered_ctxt = curr_active_pipeline.filtered_contexts[a_name]
            ## Parse the name into the parts:
            _split_parts = a_name.split('_')
            if (len(_split_parts) >= 2):
                # also have lap_dir:
                a_split_name, lap_dir, *remainder_list = a_name.split('_') # successfully splits 'maze_odd_laps' into good
                if (a_filtered_ctxt.filter_name != desired_filter_name):
                    was_updated = True
                    print(f"WARNING: filtered_contexts['{a_name}']'s actual context name is incorrect. \n\ta_filtered_ctxt.filter_name: '{a_filtered_ctxt.filter_name}' != desired_filter_name: '{desired_filter_name}'\n\tUpdating it. (THIS IS A HACK)")
                    a_filtered_ctxt = a_filtered_ctxt.overwriting_context(filter_name=desired_filter_name, lap_dir=lap_dir)

                if not a_filtered_ctxt.has_keys('lap_dir'):
                    print(f'WARNING: context {a_name} is missing the "lap_dir" key despite directional laps being detected from the name! Adding missing context key! lap_dir="{lap_dir}"')
                    a_filtered_ctxt = a_filtered_ctxt.adding_context_if_missing(lap_dir=lap_dir) # Add the lap_dir context if it was missing
                    was_updated = True

            else:
                if a_filtered_ctxt.filter_name != desired_filter_name:
                    was_updated = True
                    print(f"WARNING: filtered_contexts['{a_name}']'s actual context name is incorrect. \n\ta_filtered_ctxt.filter_name: '{a_filtered_ctxt.filter_name}' != desired_filter_name: '{desired_filter_name}'\n\tUpdating it. (THIS IS A HACK)")
                    a_filtered_ctxt = a_filtered_ctxt.overwriting_context(filter_name=desired_filter_name)

            if debug_print:
                print(f'\t{a_filtered_ctxt.to_dict()}')
            curr_active_pipeline.filtered_contexts[a_name] = a_filtered_ctxt # correct the context

        # end for
        return was_updated


    @classmethod
    def fix_computation_epochs_if_needed(cls, curr_active_pipeline, debug_print=False):
        """2023-11-10 - WORKING NOW - decouples the configs and constrains the computation_epochs to the relevant long/short periods. Will need recomputations if was_modified """
        #TODO 2023-11-10 23:32: - [ ] WORKING NOW!
        # 2023-11-10 21:15: - [X] Not yet finished! Does not work due to shared memory issue. Changes to the first two affect the next two

        was_modified: bool = False
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.
        if debug_print:
            print(f'long_epoch_obj: {long_epoch_obj}, short_epoch_obj: {short_epoch_obj}')
        assert short_epoch_obj.n_epochs > 0
        assert long_epoch_obj.n_epochs > 0

        ## {"even": "RL", "odd": "LR"}
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']

        (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]

        # Check for duplicated memory references in the configs first:
        has_duplicated_reference: bool = cls.has_duplicated_memory_references(long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config)
        if has_duplicated_reference:
            long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config = [deepcopy(a_config) for a_config in (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config)]
            assert not cls.has_duplicated_memory_references(long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config), f"duplicate memory references still exist even after de-duplicating with deepcopy!!!"
            was_modified = was_modified or True # duplicated references fixed!
            # re-assign:
            for an_epoch_name, a_deduplicated_config in zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config)):
                curr_active_pipeline.computation_results[an_epoch_name].computation_config = a_deduplicated_config
            print(f'deduplicated references!')

        original_num_epochs = np.array([curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.n_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)])
        if debug_print:
            print(f'original_num_epochs: {original_num_epochs}')
        assert np.all(original_num_epochs > 0)
        # Fix the computation epochs to be constrained to the proper long/short intervals:
        # relys on: long_epoch_obj, short_epoch_obj
        for an_epoch_name in (long_LR_name, long_RL_name):
            curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs = deepcopy(curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.time_slice(long_epoch_obj.t_start, long_epoch_obj.t_stop))

        for an_epoch_name in (short_LR_name, short_RL_name):
            curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs = deepcopy(curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.time_slice(short_epoch_obj.t_start, short_epoch_obj.t_stop))

        modified_num_epochs = np.array([curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs.n_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)])
        if debug_print:
            print(f'modified_num_epochs: {modified_num_epochs}')
        was_modified = was_modified or np.any(original_num_epochs != modified_num_epochs)
        assert np.all(modified_num_epochs > 0)

        return was_modified


    @classmethod
    def fixup_directional_pipeline_if_needed(cls, curr_active_pipeline, debug_print=False):
        """2023-11-29 - Updates the filtered context and decouples the configs and constrains the computation_epochs to the relevant long/short periods as needed. Will need recomputations if was_modified """
        #TODO 2023-11-10 23:32: - [ ] WORKING NOW!
        # 2023-11-10 21:15: - [X] Not yet finished! Does not work due to shared memory issue. Changes to the first two affect the next two

        was_modified: bool = False
        was_modified = was_modified or DirectionalLapsHelpers.post_fixup_filtered_contexts(curr_active_pipeline)
        was_modified = was_modified or DirectionalLapsHelpers.fix_computation_epochs_if_needed(curr_active_pipeline)
        return was_modified


    @classmethod
    def build_global_directional_result_from_natural_epochs(cls, curr_active_pipeline, progress_print=False) -> "DirectionalLapsResult":
        """ 2023-10-31 - 4pm  - Main computation function, simply extracts the diretional laps from the existing epochs.

        Does not update `curr_active_pipeline` or mess with its filters/configs/etc.

                ## {"even": "RL", "odd": "LR"}

        #TODO 2023-11-10 21:00: - [ ] Convert above "LR/RL" notation to new "LR/RL" versions:

        """

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.

        # Unwrap the naturally produced directional placefields:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
        (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
        (long_LR_pf2D, long_RL_pf2D, short_LR_pf2D, short_RL_pf2D) = (long_LR_results.pf2D, long_RL_results.pf2D, short_LR_results.pf2D, short_RL_results.pf2D)
        (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

        # Unpack all directional variables:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name

        # Validate:
        assert not (curr_active_pipeline.computation_results[long_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
        assert not (curr_active_pipeline.computation_results[short_LR_name].computation_config['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_RL_name].computation_config['pf_params'].computation_epochs)
        # Fix the computation epochs to be constrained to the proper long/short intervals:
        was_modified = cls.fix_computation_epochs_if_needed(curr_active_pipeline=curr_active_pipeline)
        was_modified = was_modified or DirectionalLapsHelpers.post_fixup_filtered_contexts(curr_active_pipeline)
        print(f'build_global_directional_result_from_natural_epochs(...): was_modified: {was_modified}')

        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
        long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]


        #TODO 2023-12-07 20:48: - [ ] It looks like I'm still only looking at the intersection here! Do I want this?

        # # ## Version 2023-10-30 - All four templates with same shared_aclus version:
        # # # Prune to the shared aclus in both epochs (short/long):
        # active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        # # Find only the common aclus amongst all four templates:
        # shared_aclus = np.array(list(set.intersection(*map(set,active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        # n_neurons = len(shared_aclus)
        # print(f'n_neurons: {n_neurons}, shared_aclus: {shared_aclus}')
        # # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

        ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
        # Kamran says LR and RL sets should be shared
        ## Odd Laps:
        LR_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        LR_shared_aclus = np.array(list(set.intersection(*map(set,LR_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        LR_n_neurons = len(LR_shared_aclus)
        if progress_print:
            print(f'LR_n_neurons: {LR_n_neurons}, LR_shared_aclus: {LR_shared_aclus}')

        ## Even Laps:
        RL_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        RL_shared_aclus = np.array(list(set.intersection(*map(set,RL_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        RL_n_neurons = len(RL_shared_aclus)
        if progress_print:
            print(f'RL_n_neurons: {RL_n_neurons}, RL_shared_aclus: {RL_shared_aclus}')

        # Direction Separate shared_aclus decoders: Odd set is limited to LR_shared_aclus and RL set is limited to RL_shared_aclus:
        long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(LR_shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(RL_shared_aclus) for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]


        # ## Encode/Decode from global result:
        # # Unpacking:
        # directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        # # split_directional_laps_config_names

        ## Build a `ComputedResult` container object to hold the result:
        directional_laps_result = DirectionalLapsResult()
        directional_laps_result.directional_lap_specific_configs = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # directional_lap_specific_configs
        directional_laps_result.split_directional_laps_dict = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)}  # split_directional_laps_dict
        directional_laps_result.split_directional_laps_contexts_dict = {a_name:curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)} # split_directional_laps_contexts_dict
        directional_laps_result.split_directional_laps_config_names = [long_LR_name, long_RL_name, short_LR_name, short_RL_name] # split_directional_laps_config_names

        # # use the non-constrained epochs:
        # directional_laps_result.long_LR_one_step_decoder_1D = long_LR_laps_one_step_decoder_1D
        # directional_laps_result.long_RL_one_step_decoder_1D = long_RL_laps_one_step_decoder_1D
        # directional_laps_result.short_LR_one_step_decoder_1D = short_LR_laps_one_step_decoder_1D
        # directional_laps_result.short_RL_one_step_decoder_1D = short_RL_laps_one_step_decoder_1D

        # use the constrained epochs:
        directional_laps_result.long_LR_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_RL_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_LR_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_RL_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
        directional_laps_result.long_LR_shared_aclus_only_one_step_decoder_1D = long_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_RL_shared_aclus_only_one_step_decoder_1D = long_RL_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_LR_shared_aclus_only_one_step_decoder_1D = short_LR_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_RL_shared_aclus_only_one_step_decoder_1D = short_RL_shared_aclus_only_one_step_decoder_1D

        return directional_laps_result




@define(slots=False, repr=False)
class DirectionalMergedDecodersResult(ComputedResult):
    """ a container for holding information regarding the computation of merged directional placefields.

    result_instance = DirectionalMergedDecodersResult()

    all_directional_decoder_dict_value = result_instance.all_directional_decoder_dict
    all_directional_pf1D_Decoder_value = result_instance.all_directional_pf1D_Decoder
    long_directional_pf1D_Decoder_value = result_instance.long_directional_pf1D_Decoder
    long_directional_decoder_dict_value = result_instance.long_directional_decoder_dict
    short_directional_pf1D_Decoder_value = result_instance.short_directional_pf1D_Decoder
    short_directional_decoder_dict_value = result_instance.short_directional_decoder_dict

    all_directional_laps_filter_epochs_decoder_result_value = result_instance.all_directional_laps_filter_epochs_decoder_result
    all_directional_ripple_filter_epochs_decoder_result_value = result_instance.all_directional_ripple_filter_epochs_decoder_result


    """
    all_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
    all_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    long_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    long_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
    short_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    short_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)

    all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)
    all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)


    # long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_odd_shared_aclus_only_one_step_decoder_1D')
    # long_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='long_even_shared_aclus_only_one_step_decoder_1D')
    # short_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_odd_shared_aclus_only_one_step_decoder_1D')
    # short_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = serialized_field(default=None, alias='short_even_shared_aclus_only_one_step_decoder_1D')

    @classmethod
    def validate_has_directional_merged_placefields(cls, curr_active_pipeline, computation_filter_name='maze'):
        """ 
            DirectionalMergedDecodersResult.validate_has_directional_merged_placefields
        """
        # Unpacking:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
        
        # extract properties:
        all_directional_decoder_dict_value = directional_merged_decoders_result.all_directional_decoder_dict
        all_directional_pf1D_Decoder_value = directional_merged_decoders_result.all_directional_pf1D_Decoder
        long_directional_pf1D_Decoder_value = directional_merged_decoders_result.long_directional_pf1D_Decoder
        long_directional_decoder_dict_value = directional_merged_decoders_result.long_directional_decoder_dict
        short_directional_pf1D_Decoder_value = directional_merged_decoders_result.short_directional_pf1D_Decoder
        short_directional_decoder_dict_value = directional_merged_decoders_result.short_directional_decoder_dict

        all_directional_laps_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result
        all_directional_ripple_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result

        return True


    @classmethod
    def build_custom_marginal_over_direction(cls, filter_epochs_decoder_result, debug_print=False):
        """ only works for the all-directional coder with the four items
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            active_decoder = all_directional_pf1D_Decoder
            laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_LAPS',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: build_custom_marginal_over_direction(filter_epochs_decoder_result),
                                                        )
                                    
                                                        
        0: LR
        1: RL
        
        """
        custom_curr_unit_marginal_list = []
        
        for a_p_x_given_n in filter_epochs_decoder_result.p_x_given_n_list:
            # an_array = all_directional_laps_filter_epochs_decoder_result.p_x_given_n_list[0] # .shape # (62, 4, 236)
            curr_array_shape = np.shape(a_p_x_given_n)
            if debug_print:
                print(f'a_p_x_given_n.shape: {curr_array_shape}')
            # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
            # (['long', 'long', 'short', 'short'])
            # (n_neurons, is_long, is_LR, pos_bins)
            assert curr_array_shape[1] == 4, f"only works with the all-directional decoder with ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "

            out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 2, :]) # LR_marginal = long_LR + short_LR
            out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 1, :] + a_p_x_given_n[:, 3, :]) # RL_marginal = long_RL + short_RL

            # normalized_out_p_x_given_n = out_p_x_given_n / np.sum(out_p_x_given_n, axis=1) # , keepdims=True

            normalized_out_p_x_given_n = out_p_x_given_n
            # reshaped_p_x_given_n = np.reshape(a_p_x_given_n, (curr_array_shape[0], 2, 2, curr_array_shape[-1]))
            # assert np.array_equiv(reshaped_p_x_given_n[:,0,0,:], a_p_x_given_n[:, 0, :]) # long_LR
            # assert np.array_equiv(reshaped_p_x_given_n[:,1,0,:], a_p_x_given_n[:, 2, :]) # short_LR

            # print(f'np.shape(reshaped_p_x_given_n): {np.shape(reshaped_p_x_given_n)}')

            # normalized_reshaped_p_x_given_n = np.squeeze(np.sum(reshaped_p_x_given_n, axis=(1), keepdims=False)) / np.sum(reshaped_p_x_given_n, axis=(0,1), keepdims=False)
            # print(f'np.shape(normalized_reshaped_p_x_given_n): {np.shape(normalized_reshaped_p_x_given_n)}')

            # restored_shape_p_x_given_n = np.reshape(normalized_reshaped_p_x_given_n, curr_array_shape)
            # print(f'np.shape(restored_shape_p_x_given_n): {np.shape(restored_shape_p_x_given_n)}')

            # np.sum(reshaped_array, axis=2) # axis=2 means sum over both long and short for LR/RL

            # to sum over both long/short for LR
            # np.sum(reshaped_p_x_given_n, axis=1).shape # axis=2 means sum over both long and short for LR/RL
            

            # input_array = a_p_x_given_n
            # input_array = normalized_reshaped_p_x_given_n
            input_array = normalized_out_p_x_given_n

            if debug_print:
                print(f'np.shape(input_array): {np.shape(input_array)}')
            # custom marginal over long/short, leaving only LR/RL:
            curr_unit_marginal_y = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_y.p_x_given_n = input_array
            
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            # curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, 1)) # sum over all y. Result should be [x_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
        
            # y-axis marginal:
            curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, axis=0)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=1, keepdims=True) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

            curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)

            # curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, axis=1)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            if debug_print:
                print(f'np.shape(curr_unit_marginal_y.p_x_given_n): {np.shape(curr_unit_marginal_y.p_x_given_n)}')
            
            ## Ensures that the marginal posterior is at least 2D:
            # print(f"curr_unit_marginal_y.p_x_given_n.ndim: {curr_unit_marginal_y.p_x_given_n.ndim}")
            # assert curr_unit_marginal_y.p_x_given_n.ndim >= 2
            if curr_unit_marginal_y.p_x_given_n.ndim == 0:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_y.p_x_given_n.ndim == 1:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_y.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_y)
        return custom_curr_unit_marginal_list

    @classmethod
    def determine_directional_likelihoods(cls, all_directional_laps_filter_epochs_decoder_result):
        """ 

        determine_directional_likelihoods

        directional_marginals, directional_all_epoch_bins_marginal, most_likely_direction_from_decode, is_most_likely_direction_LR_dirr = DirectionalMergedDecodersResult.determine_directional_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)

        0: LR
        1: RL
        
        """
        directional_marginals = cls.build_custom_marginal_over_direction(all_directional_laps_filter_epochs_decoder_result)

        # gives the likelihood of [LR, RL] for each epoch using information from both Long/Short:
        directional_all_epoch_bins_marginal = np.stack([np.sum(v.p_x_given_n, axis=-1)/np.sum(v.p_x_given_n, axis=(-2, -1)) for v in directional_marginals], axis=0) # sum over all time-bins within the epoch to reach a consensus
        # directional_all_epoch_bins_marginal

        # Find the indicies via this method:
        most_likely_direction_from_decoder = np.argmax(directional_all_epoch_bins_marginal, axis=1) # consistent with 'lap_dir' columns. for LR_dir, values become more positive with time
        is_most_likely_direction_LR_dir = np.logical_not(most_likely_direction_from_decoder) # consistent with 'is_LR_dir' column. for LR_dir, values become more positive with time

        # most_likely_direction_from_decoder
        return directional_marginals, directional_all_epoch_bins_marginal, most_likely_direction_from_decoder, is_most_likely_direction_LR_dir

    @classmethod
    def validate_lap_dir_estimations(cls, global_session, active_global_laps_df, laps_is_most_likely_direction_LR_dir):
        def _subfn_compute_lap_dir_from_smoothed_velocity(global_session, active_global_laps_df):
            """ uses the smoothed velocity to determine the proper lap direction

            for LR_dir, values become more positive with time

            global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
            global_laps = compute_lap_dir_from_smoothed_velocity(global_session)
            global_laps

            """
            # global_session.laps.to_dataframe()
            # if active_global_laps_df is None:
            #     active_global_laps = deepcopy(global_session.laps)
            #     active_global_laps_df = global_laps._df

            n_laps = np.shape(active_global_laps_df)[0]

            global_pos = global_session.position
            global_pos.compute_higher_order_derivatives()
            global_pos.compute_smoothed_position_info()
            pos_df: pd.DataFrame = global_pos.to_dataframe()

            # Filter rows based on column: 'lap'
            pos_df = pos_df[pos_df['lap'].notna()]
            # Performed 1 aggregation grouped on column: 'lap'
            is_LR_dir = ((pos_df.groupby(['lap']).agg(speed_mean=('velocity_x_smooth', 'mean'))).reset_index()['speed_mean'] > 0.0).to_numpy() # increasing values => LR_dir
            active_global_laps_df['is_LR_dir'] = is_LR_dir
            # global_laps._df['direction_consistency'] = 0.0
            assert np.all(active_global_laps_df[(active_global_laps_df['is_LR_dir'].astype(int) == np.logical_not(active_global_laps_df['lap_dir'].astype(int)))])
            return active_global_laps_df

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
        active_global_laps_df = _subfn_compute_lap_dir_from_smoothed_velocity(global_session, active_global_laps_df=active_global_laps_df)
        # Validate Laps:
        # ground_truth_lap_dirs = active_global_laps_df['lap_dir'].to_numpy()
        ground_truth_lap_is_LR_dir = active_global_laps_df['is_LR_dir'].to_numpy()
        n_laps = np.shape(active_global_laps_df)[0]
        assert len(laps_is_most_likely_direction_LR_dir) == n_laps
        percent_laps_estimated_correctly = (np.sum(ground_truth_lap_is_LR_dir == laps_is_most_likely_direction_LR_dir) / n_laps)
        print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
        return percent_laps_estimated_correctly


class DirectionalPlacefieldGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'directional_pfs'
    _computationPrecidence = 1000
    _is_global = True

    @function_attributes(short_name='split_to_directional_laps', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'], input_requires=[], output_provides=[], uses=['_perform_PBE_stats'], used_by=[], creation_date='2023-10-25 09:33', related_items=[],
        validate_computation_test=DirectionalLapsHelpers.validate_has_directional_laps, is_global=True)
    def _split_to_directional_laps(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['DirectionalLaps']
                ['DirectionalLaps']['directional_lap_specific_configs']
                ['DirectionalLaps']['split_directional_laps_dict']
                ['DirectionalLaps']['split_directional_laps_contexts_dict']
                ['DirectionalLaps']['split_directional_laps_names']
                ['DirectionalLaps']['computed_base_epoch_names']


        """
        if include_includelist is not None:
            print(f'WARN: _split_to_directional_laps(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

        # if include_includelist is None:
            # long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            # include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
            # include_includelist = [global_epoch_name] # ['maze'] # only for maze
            # include_includelist = [long_epoch_name, short_epoch_name] # ['maze1', 'maze2'] # only for maze

        ## Adds ['*_even_laps', '*_odd_laps'] pseduofilters

        # owning_pipeline_reference, directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = DirectionalLapsHelpers.split_to_directional_laps(owning_pipeline_reference, include_includelist=include_includelist, add_created_configs_to_pipeline=True)
        # curr_active_pipeline, directional_lap_specific_configs = constrain_to_laps(curr_active_pipeline)
        # list(directional_lap_specific_configs.keys())

        # Set the global result:
        global_computation_results.computed_data['DirectionalLaps'] = DirectionalLapsHelpers.build_global_directional_result_from_natural_epochs(owning_pipeline_reference)


        # global_computation_results.computed_data['DirectionalLaps'] = DynamicParameters.init_from_dict({
        #     'directional_lap_specific_configs': directional_lap_specific_configs,
        #     'split_directional_laps_dict': split_directional_laps_dict,
        #     'split_directional_laps_contexts_dict': split_directional_laps_contexts_dict,
        #     'split_directional_laps_names': split_directional_laps_config_names,
        #     'computed_base_epoch_names': computed_base_epoch_names,
        # })

        ## Needs to call `owning_pipeline_reference.prepare_for_display()` before display functions can be used with new directional results

        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_contexts_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        """
        return global_computation_results



    @function_attributes(short_name='merged_directional_placefields', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'], input_requires=[], output_provides=[], uses=['PfND.build_merged_directional_placefields('], used_by=[], creation_date='2023-10-25 09:33', related_items=[],
        validate_computation_test=DirectionalMergedDecodersResult.validate_has_directional_merged_placefields, is_global=True)
    def _build_merged_directional_placefields(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
        """

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['DirectionalMergedDecoders']
                ['DirectionalMergedDecoders']['directional_lap_specific_configs']
                ['DirectionalMergedDecoders']['split_directional_laps_dict']
                ['DirectionalMergedDecoders']['split_directional_laps_contexts_dict']
                ['DirectionalMergedDecoders']['split_directional_laps_names']
                ['DirectionalMergedDecoders']['computed_base_epoch_names']


                directional_merged_decoders_result: "DirectionalMergedDecodersResult" = global_computation_results.computed_data['DirectionalMergedDecoders']

        """
        from neuropy.analyses.placefields import PfND
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(owning_pipeline_reference.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in owning_pipeline_reference.sess.epochs is 'maze1' without the '_any' part.
        global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used in 

        # Unwrap the naturally produced directional placefields:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj, global_any_laps_epochs_obj = [owning_pipeline_reference.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name, global_any_name)] # note has global also
        (long_LR_session, long_RL_session, short_LR_session, short_RL_session) = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # sessions are correct at least, seems like just the computation parameters are messed up
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [owning_pipeline_reference.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_computation_config, long_RL_computation_config, short_LR_computation_config, short_RL_computation_config) = [owning_pipeline_reference.computation_results[an_epoch_name].computation_config for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_pf1D, long_RL_pf1D, short_LR_pf1D, short_RL_pf1D) = (long_LR_results.pf1D, long_RL_results.pf1D, short_LR_results.pf1D, short_RL_results.pf1D)
       
        # Unpack all directional variables:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name

        # Use the four epochs to make to a pseudo-y:
        all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        all_directional_decoder_dict = dict(zip(all_directional_decoder_names, [deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D), deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D)]))
        all_directional_pf1D = PfND.build_merged_directional_placefields(all_directional_decoder_dict, debug_print=False)
        all_directional_pf1D_Decoder = BasePositionDecoder(all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        ## Combine the non-directional PDFs and renormalize to get the directional PDF:
        # Inputs: long_LR_pf1D, long_RL_pf1D
        long_directional_decoder_names = ['long_LR', 'long_RL']
        long_directional_decoder_dict = dict(zip(long_directional_decoder_names, [deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D)]))
        long_directional_pf1D = PfND.build_merged_directional_placefields(long_directional_decoder_dict, debug_print=False)
        long_directional_pf1D_Decoder = BasePositionDecoder(long_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        # Inputs: short_LR_pf1D, short_RL_pf1D
        short_directional_decoder_names = ['short_LR', 'short_RL']
        short_directional_decoder_dict = dict(zip(short_directional_decoder_names, [deepcopy(short_LR_pf1D), deepcopy(short_RL_pf1D)]))
        short_directional_pf1D = PfND.build_merged_directional_placefields(short_directional_decoder_dict, debug_print=False)
        short_directional_pf1D_Decoder = BasePositionDecoder(short_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        # takes 6.3 seconds


        _out_result = global_computation_results.computed_data.get('DirectionalMergedDecoders', DirectionalMergedDecodersResult(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
                                                      long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
                                                      short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder))


        _out_result.__dict__.update(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
                                                      long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
                                                      short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder)
        
        

        # Do decodings:
        ## Decode Laps:
        laps_decoding_time_bin_size: float = 0.05

        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_any_name].computation_config.pf_params.computation_epochs) # global_any_name='maze_any' (? same as global_epoch_name?)

        all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, debug_print=False)
        _out_result.all_directional_laps_filter_epochs_decoder_result = all_directional_laps_filter_epochs_decoder_result
        # directional_marginals, directional_all_epoch_bins_marginal, most_likely_direction_from_decoder, is_most_likely_direction_LR_dir = determine_directional_likelihoods(all_directional_laps_filter_epochs_decoder_result)
        laps_marginals = DirectionalMergedDecodersResult.determine_directional_likelihoods(_out_result.all_directional_laps_filter_epochs_decoder_result)
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = laps_marginals

        # ground_truth_lap_dirs = global_any_laps_epochs_obj.to_dataframe()['lap_dir'].to_numpy()
        # n_laps = global_any_laps_epochs_obj.n_epochs
        # assert len(laps_most_likely_direction_from_decoder) == n_laps
        # percent_laps_estimated_correctly = (np.sum(ground_truth_lap_dirs == laps_most_likely_direction_from_decoder) / n_laps)
        # print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
        
        # Validate Laps:
        percent_laps_estimated_correctly = DirectionalMergedDecodersResult.validate_lap_dir_estimations(global_session, active_global_laps_df=global_any_laps_epochs_obj.to_dataframe(), laps_is_most_likely_direction_LR_dir=laps_is_most_likely_direction_LR_dir)
        print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')

        ## Decode Ripples:
        # Decode using long_directional_decoder
        ripple_decoding_time_bin_size: float = 0.002        
        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
        all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = all_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(owning_pipeline_reference.sess.spikes_df), global_replays, decoding_time_bin_size=ripple_decoding_time_bin_size)
        _out_result.all_directional_ripple_filter_epochs_decoder_result = all_directional_ripple_filter_epochs_decoder_result
        
        ripple_marginals = DirectionalMergedDecodersResult.determine_directional_likelihoods(_out_result.all_directional_ripple_filter_epochs_decoder_result)
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = ripple_marginals


        # Set the global result:
        # global_computation_results.computed_data['DirectionalMergedDecoders']
        
        
        # Only update what has changed:
        global_computation_results.computed_data['DirectionalMergedDecoders'] = _out_result
        

        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_contexts_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        """
        return global_computation_results


# ==================================================================================================================== #
# Display Functions/Plotting                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyqtgraph as pg
import pyqtgraph.exporters
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons, paired_incremental_sort_neurons # _display_directional_template_debugger
from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays


class DirectionalPlacefieldGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='directional_laps_overview', tags=['directional','laps','overview'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-09 12:03', related_items=[], is_global=True)
    def _display_directional_laps_overview(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
            """ Renders a window with the position/laps displayed in the middle and the four templates displayed to the left and right of them.

            #TODO 2023-12-07 09:29: - [ ] This function's rasters have not been updated (as `_display_directional_template_debugger` on 2023-12-07) and when filtering the unit sort order and their labels will probably become incorrect.

            """

            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import EpochsEditor # perform_plot_laps_diagnoser
            from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
            from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
            from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
            from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
            from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color

            # raise NotImplementedError
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)

            figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={})


            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

            assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results = global_computation_results.computed_data['RankOrder'] # RankOrderComputationsContainer
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz

            # track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # shared-only
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

            # uses `global_session`
            epochs_editor = EpochsEditor.init_from_session(global_session, include_velocity=False, include_accel=False)
            root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(epochs_editor.plots.win, None, title='Pho Directional Laps Templates')

            decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

            # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`

            # INCRIMENTAL SORTING:
            if use_incremental_sorting:
                ref_decoder_name: str = list(decoders_dict.keys())[0] # name of the reference coder. Should be 'long_LR'
                sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_incremental_sort_neurons(decoders_dict, included_any_context_neuron_ids)
            else:
                # INDIVIDUAL SORTING:
                # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
                sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in decoders_dict.items()} # tuning_curve CoM location
                sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids, sortable_values_list_dict=sortable_values_list_dict)

            sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]

            ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
            _out_pf1D_heatmaps = {}
            for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
                if use_incremental_sorting:
                    title_str = f'{a_decoder_name}_pf1Ds [sort: {ref_decoder_name}]'
                else:
                    title_str = f'{a_decoder_name}_pf1Ds'

                _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=title_str, show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
                # _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(_get_decoder_sorted_pfs(a_decoder), title=f'{a_decoder_name}_pf1Ds', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Individual Sort

                # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
                curr_win, curr_img = _out_pf1D_heatmaps[a_decoder_name] # win, img
                a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)

                # Coloring the heatmap data for each row of the 1D heatmap:
                curr_data = deepcopy(sorted_pf_tuning_curves[i])
                if debug_print:
                    print(f'np.shape(curr_data): {np.shape(curr_data)}, np.nanmax(curr_data): {np.nanmax(curr_data)}, np.nanmin(curr_data): {np.nanmin(curr_data)}') # np.shape(curr_data): (34, 62), np.nanmax(curr_data): 0.15320444716258447, np.nanmin(curr_data): 0.0

                _temp_curr_out_colors_heatmap_image = [] # used to accumulate the rows so they can be built into a color image in `out_colors_heatmap_image_matrix`

                for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
                    # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
                    text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
                    text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
                    curr_win.addItem(text)

                    # modulate heatmap color for this row (`curr_data[i, :]`):
                    heatmap_base_color = pg.mkColor(a_color_vector)
                    out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
                    _temp_curr_out_colors_heatmap_image.append(out_colors_row)

                ## Build the colored heatmap:
                out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
                if debug_print:
                    print(f"np.shape(out_colors_heatmap_image_matrix): {np.shape(out_colors_heatmap_image_matrix)}") # (34, 62, 4) - (n_cells, n_pos_bins, n_channels_RGBA)

                # Ensure the data is in the correct range [0, 1]
                out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
                curr_img.updateImage(out_colors_heatmap_image_matrix)
                _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix


            ## Build Dock Widgets:
            # decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
            _out_dock_widgets = {}
            dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                            CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
            # dock_add_locations = (['left'], ['left'], ['right'], ['right'])
            dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))

            for i, (a_decoder_name, a_heatmap) in enumerate(_out_pf1D_heatmaps.items()):
                _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_heatmap[0], dockSize=(300,200), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name])


            # Outputs: root_dockAreaWindow, app, epochs_editor, _out_pf1D_heatmaps, _out_dock_widgets
            graphics_output_dict = {'win': root_dockAreaWindow, 'app': app,  'ui': (epochs_editor, _out_dock_widgets), 'plots': _out_pf1D_heatmaps, 'data': _out_data}

            # Saving/Exporting to file ___________________________________________________________________________________________ #
            #TODO 2023-11-16 22:16: - [ ] Figure out how to save

            def save_figure(): # export_file_base_path: Path = Path(f'output').resolve()
                """ captures: epochs_editor, _out_pf1D_heatmaps

                TODO: note output paths are currently hardcoded. Needs to add the animal's context at least. Probably needs to be integrated into pipeline.
                import pyqtgraph as pg
                import pyqtgraph.exporters
                from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
                """
                ## Get main laps plotter:
                # print_keys_if_possible('_out', _out, max_depth=4)
                # plots = _out['plots']

                ## Already have: epochs_editor, _out_pf1D_heatmaps
                epochs_editor = graphics_output_dict['ui'][0]

                shared_output_file_prefix = f'output/2023-11-20'
                # print(list(plots.keys()))
                # pg.GraphicsLayoutWidget
                main_graphics_layout_widget = epochs_editor.plots.win
                export_file_path = Path(f'{shared_output_file_prefix}_test_main_position_laps_line_plot').with_suffix('.svg').resolve()
                export_pyqtgraph_plot(main_graphics_layout_widget, savepath=export_file_path) # works

                _out_pf1D_heatmaps = graphics_output_dict['plots']
                for a_decoder_name, a_decoder_heatmap_tuple in _out_pf1D_heatmaps.items():
                    a_win, a_img = a_decoder_heatmap_tuple
                    # a_win.export_image(f'{a_decoder_name}_heatmap.png')
                    print(f'a_win: {type(a_win)}')

                    # create an exporter instance, as an argument give it the item you wish to export
                    exporter = pg.exporters.ImageExporter(a_win.plotItem)
                    # exporter = pg.exporters.SVGExporter(a_win.plotItem)
                    # set export parameters if needed
                    # exporter.parameters()['width'] = 300   # (note this also affects height parameter)

                    # save to file
                    export_file_path = Path(f'{shared_output_file_prefix}_test_{a_decoder_name}_heatmap').with_suffix('.png').resolve() # '.svg' # .resolve()

                    exporter.export(str(export_file_path)) # '.png'
                    print(f'exporting to {export_file_path}')
                    # .scene()


            #TODO 2023-11-16 22:23: - [ ] The other display functions using matplotlib do things like this:
            # final_context = active_context
            # graphics_output_dict['context'] = final_context
            # graphics_output_dict['plot_data'] |= {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate,
            #     'time_variable_name':time_variable_name, 'fignum':curr_fig_num}

            # def _perform_write_to_file_callback():
            #     ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            #     return owning_pipeline_reference.output_figure(final_context, graphics_output_dict.figures[0])

            # if save_figure:
            #     active_out_figure_paths = _perform_write_to_file_callback()
            # else:
            #     active_out_figure_paths = []

            # graphics_output_dict['saved_figures'] = active_out_figure_paths


            return graphics_output_dict


    @function_attributes(short_name='directional_template_debugger', tags=['directional','template','debug', 'overview'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-28 10:13', related_items=[], is_global=True)
    def _display_directional_template_debugger(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
            """ Renders a window with the four template heatmaps displayed to the left and right of center, and the ability to filter the actively included aclus via `included_any_context_neuron_ids`

            enable_cell_colored_heatmap_rows: bool - uses the cell's characteristic assigned color to shade the 1D heatmap row value for that cell. NOTE: there are some perceptual non-uniformities with luminance how it is being applied now.

            use_incremental_sorting: bool = False - incremental sorting refers to the method of sorting where plot A is sorted first, all of those cells retain their position for all subsequent plots, but the B-unique cells are sorted for B, ... and so on.
                The alternative (use_incremental_sorting = False) is *individual* sorting, where each is sorted independently.

            """

            # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger
            
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)

            enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
            use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)

            figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
            # _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)

            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

            assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results = global_computation_results.computed_data['RankOrder'] # RankOrderComputationsContainer
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            assert minimum_inclusion_fr_Hz is not None
            if (use_shared_aclus_only_templates):
                track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # shared-only
            else:
                track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only

            template_debugger: TemplateDebugger = TemplateDebugger.init_templates_debugger(track_templates=track_templates, included_any_context_neuron_ids=included_any_context_neuron_ids,
                                                      use_incremental_sorting=use_incremental_sorting, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, use_shared_aclus_only_templates=use_shared_aclus_only_templates,
                                                      figure_name=figure_name, debug_print=debug_print, defer_render=defer_render, **kwargs)

            # decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }


            # # build the window with the dock widget in it:
            # root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho Directional Template Debugger: {figure_name}', defer_show=False)
            # _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, text_items_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)
            # root_dockAreaWindow.resize(900, 700)

            # _out_data, _out_plots, _out_ui = TemplateDebugger._subfn_buildUI_directional_template_debugger_data(included_any_context_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict)
            # update_callback_fn = (lambda included_neuron_ids: TemplateDebugger._subfn_update_directional_template_debugger_data(included_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict))
            # _out_ui.on_update_callback = update_callback_fn

            # Outputs: root_dockAreaWindow, app, epochs_editor, _out_pf1D_heatmaps, _out_dock_widgets
            # graphics_output_dict = {'win': root_dockAreaWindow, 'app': app,  'ui': _out_ui, 'plots': _out_plots, 'data': _out_data}

            graphics_output_dict = {'win': template_debugger.ui.root_dockAreaWindow, 'app': template_debugger.ui.app,  'ui': template_debugger.ui, 'plots': template_debugger.plots, 'data': template_debugger.plots_data, 'obj': template_debugger}


            

            # def on_update(included_any_context_neuron_ids):
            #     """ call to update when `included_any_context_neuron_ids` changes.

            #      captures: `decoders_dict`, `_out_plots`, 'enable_cell_colored_heatmap_rows'

            #     """
            #     decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
            #     sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=included_any_context_neuron_ids)
            #     # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`

            #     ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
            #     _out_plots.pf1D_heatmaps = {}
            #     for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            #         _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=f'{a_decoder_name}_pf1Ds [sort: long_RL]', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
            #         # _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(_get_decoder_sorted_pfs(a_decoder), title=f'{a_decoder_name}_pf1Ds', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Individual Sort

            #         # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
            #         curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img

            #         a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)

            #         # Coloring the heatmap data for each row of the 1D heatmap:
            #         curr_data = deepcopy(sorted_pf_tuning_curves[i])
            #         if debug_print:
            #             print(f'np.shape(curr_data): {np.shape(curr_data)}, np.nanmax(curr_data): {np.nanmax(curr_data)}, np.nanmin(curr_data): {np.nanmin(curr_data)}') # np.shape(curr_data): (34, 62), np.nanmax(curr_data): 0.15320444716258447, np.nanmin(curr_data): 0.0

            #         _temp_curr_out_colors_heatmap_image = [] # used to accumulate the rows so they can be built into a color image in `out_colors_heatmap_image_matrix`

            #         for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
            #             # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
            #             text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
            #             text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
            #             curr_win.addItem(text)

            #             # modulate heatmap color for this row (`curr_data[i, :]`):
            #             heatmap_base_color = pg.mkColor(a_color_vector)
            #             out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
            #             _temp_curr_out_colors_heatmap_image.append(out_colors_row)

            #         ## Build the colored heatmap:
            #         out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            #         if debug_print:
            #             print(f"np.shape(out_colors_heatmap_image_matrix): {np.shape(out_colors_heatmap_image_matrix)}") # (34, 62, 4) - (n_cells, n_pos_bins, n_channels_RGBA)

            #         # Ensure the data is in the correct range [0, 1]
            #         out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            #         if enable_cell_colored_heatmap_rows:
            #             curr_img.updateImage(out_colors_heatmap_image_matrix) # use the color image only if `enable_cell_colored_heatmap_rows==True`
            #         _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix

            # graphics_output_dict['ui'].on_update_callback = on_update

            #TODO 2023-11-16 22:23: - [ ] The other display functions using matplotlib do things like this:
            # final_context = active_context
            # graphics_output_dict['context'] = final_context
            # graphics_output_dict['plot_data'] |= {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate,
            #     'time_variable_name':time_variable_name, 'fignum':curr_fig_num}

            # def _perform_write_to_file_callback():
            #     ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
            #     return owning_pipeline_reference.output_figure(final_context, graphics_output_dict.figures[0])

            # if save_figure:
            #     active_out_figure_paths = _perform_write_to_file_callback()
            # else:
            #     active_out_figure_paths = []

            # graphics_output_dict['saved_figures'] = active_out_figure_paths

            return graphics_output_dict


    @function_attributes(short_name='directional_track_template_pf1Ds', tags=['directional','template','debug', 'overview'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-22 10:41', related_items=[], is_global=True)
    def _display_directional_track_template_pf1Ds(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
            """ Plots each template's pf1Ds side-by-side in subplots. 
            """

            # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger
            from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables
            import matplotlib.pyplot as plt

            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)

            enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
            use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)

            figure_name: str = kwargs.pop('figure_name', 'directional_track_template_pf1Ds')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
            # _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)

            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

            assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
            rank_order_results = global_computation_results.computed_data['RankOrder'] # RankOrderComputationsContainer
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            assert minimum_inclusion_fr_Hz is not None
            if (use_shared_aclus_only_templates):
                track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # shared-only
            else:
                track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only

            ## {"even": "RL", "odd": "LR"}
            long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
            (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]

            decoders_dict = track_templates.get_decoders_dict()
            decoders_dict_keys = list(decoders_dict.keys())
            decoder_context_dict = dict(zip(decoders_dict_keys, (long_LR_context, long_RL_context, short_LR_context, short_RL_context)))

            # print(f'decoders_dict_keys: {decoders_dict_keys}')
            plot_kwargs = {}
            mosaic = [
                    ["ax_pf_tuning_curve"],
                    ["ax_pf_occupancy"],
                ]
            fig = plt.figure(layout="constrained")
            subfigures_dict = dict(zip(list(decoders_dict.keys()), fig.subfigures(nrows=1, ncols=4)))
            display_outputs = {}
            
            for a_name, a_subfigure in subfigures_dict.items():
                axd = a_subfigure.subplot_mosaic(mosaic, sharex=True, height_ratios=[8, 1], gridspec_kw=dict(wspace=0, hspace=0.15))
                a_decoder = decoders_dict[a_name]
                active_context = decoder_context_dict[a_name]
                active_display_ctx = active_context.adding_context('display_fn', display_fn_name='plot_ratemaps_1D')
                # active_display_fn_identifying_ctx = curr_active_pipeline.build_display_context_for_filtered_session(filtered_session_name=a_name, display_fn_name='plot_directional_pf1Ds')
                # active_display_fn_identifying_ctx
                ax_pf_1D = a_decoder.pf.plot_ratemaps_1D(ax=axd["ax_pf_tuning_curve"], active_context=active_display_ctx)
                active_display_ctx = active_context.adding_context('display_fn', display_fn_name='plot_occupancy_1D')
                # active_display_ctx_string = active_display_ctx.get_description(separator='|')
                
                display_outputs[a_name] = a_decoder.pf.plot_occupancy(fig=a_subfigure, ax=axd["ax_pf_occupancy"], active_context=active_display_ctx, **({} | plot_kwargs))
                
                # plot_variable_name = ({'plot_variable': None} | kwargs)
                plot_variable_name = plot_kwargs.get('plot_variable', enumTuningMap2DPlotVariables.OCCUPANCY).name
                active_display_ctx = active_display_ctx.adding_context(None, plot_variable=plot_variable_name)

            return fig, subfigures_dict, display_outputs


            # decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

            # # build the window with the dock widget in it:
            # root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho Directional Template Debugger: {figure_name}', defer_show=False)
            # _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, text_items_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)
            # root_dockAreaWindow.resize(900, 700)

            # _out_data, _out_plots, _out_ui = TemplateDebugger._subfn_buildUI_directional_template_debugger_data(included_any_context_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict)
            # update_callback_fn = (lambda included_neuron_ids: TemplateDebugger._subfn_update_directional_template_debugger_data(included_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict))
            # _out_ui.on_update_callback = update_callback_fn

            # Outputs: root_dockAreaWindow, app, epochs_editor, _out_pf1D_heatmaps, _out_dock_widgets
            # graphics_output_dict = {'win': root_dockAreaWindow, 'app': app,  'ui': _out_ui, 'plots': _out_plots, 'data': _out_data}

            graphics_output_dict = {'win': template_debugger.ui.root_dockAreaWindow, 'app': template_debugger.ui.app,  'ui': template_debugger.ui, 'plots': template_debugger.plots, 'data': template_debugger.plots_data, 'obj': template_debugger}


    @function_attributes(short_name='directional_merged_pfs', tags=['display'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-04 03:27', related_items=[], is_global=True)
    def _display_directional_merged_pfs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
        """ Plots the merged pseduo-2D pfs/ratemaps. Plots: All-Directions, Long-Directional, Short-Directional in seperate windows. 
        """
        # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow

        ## Post 2022-10-22 display_all_pf_2D_pyqtgraph_binned_image_rendering-based method:

        # Visualization:
        # from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap, visualize_heatmap_pyqtgraph
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, display_all_pf_2D_pyqtgraph_binned_image_rendering
        # from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability

        defer_render = kwargs.pop('defer_render', False)

        # active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())
        
        directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

        active_merged_pf_plots_data_dict = {
                                       owning_pipeline_reference.build_display_context_for_session(track_config='All-Directions', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering'):directional_merged_decoders_result.all_directional_pf1D_Decoder.pf, # all-directions
                                       owning_pipeline_reference.build_display_context_for_session(track_config='Long-Directional', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering'):directional_merged_decoders_result.long_directional_pf1D_Decoder.pf, # Long-only:
                                       owning_pipeline_reference.build_display_context_for_session(track_config='Short-Directional', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering'):directional_merged_decoders_result.short_directional_pf1D_Decoder.pf, # Short-only:
                                    }

        out_plots_dict = {}
        
        for active_context, active_pf_2D in active_merged_pf_plots_data_dict.items():
            # figure_format_config = {} # empty dict for config
            figure_format_config = {} # kwargs # kwargs as default figure_format_config
            out_all_pf_2D_pyqtgraph_binned_image_fig = display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config) # output is BasicBinnedImageRenderingWindow
            # Set the window title from the context
            out_all_pf_2D_pyqtgraph_binned_image_fig.setWindowTitle(f'{active_context.get_description()}')
            out_plots_dict[active_context] = out_all_pf_2D_pyqtgraph_binned_image_fig

            if not defer_render:
                out_all_pf_2D_pyqtgraph_binned_image_fig.show()

        return out_plots_dict


    @function_attributes(short_name='directional_merged_decoder_decoded_epochs', tags=['directional_merged_decoder_decoded_epochs', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_decoded_epoch_slices'], used_by=[], creation_date='2024-01-04 02:59', related_items=[], is_global=True)
    def _display_directional_merged_pf_decoded_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ Renders to windows, one with the decoded laps and another with the decoded ripple posteriors, computed using the merged pseudo-2D decoder.

            """
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            # raise NotImplementedError
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)

            # figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={})

            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

            # requires `laps_is_most_likely_direction_LR_dir` from `laps_marginals`
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
            
            # Marginal
            global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
            active_decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
            laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_LAPS',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
                                                        )


            global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
            active_decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
            ripples_plot_tuple = plot_decoded_epoch_slices(global_replays,  directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_Ripples',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
                                                        )



            # active_decoder = long_directional_pf1D_Decoder
            # laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, directional_merged_decoders_result.long_only_laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
            #                                             name='long_only_lstacked_epoch_slices_matplotlib_subplots_LAPS',
            #                                             # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
            #                                             active_marginal_fn = lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
            #                                             )



            graphics_output_dict = {'laps_plot_tuple': laps_plot_tuple, 'ripples_plot_tuple': ripples_plot_tuple}


            return graphics_output_dict
