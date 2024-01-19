import numpy as np
import pandas as pd
from attrs import define, field, Factory, asdict, astuple
from functools import wraps
from copy import deepcopy
from collections import namedtuple
from typing import Iterable, List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, date, timedelta

from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.print_helpers import strip_type_str_to_classname

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.dynamic_container import DynamicContainer, override_dict # used to build config
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

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability

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
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
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

    ## Get the result after computation:
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    
    all_directional_decoder_dict_value = directional_merged_decoders_result.all_directional_decoder_dict
    all_directional_pf1D_Decoder_value = directional_merged_decoders_result.all_directional_pf1D_Decoder
    # long_directional_pf1D_Decoder_value = directional_merged_decoders_result.long_directional_pf1D_Decoder
    # long_directional_decoder_dict_value = directional_merged_decoders_result.long_directional_decoder_dict
    # short_directional_pf1D_Decoder_value = directional_merged_decoders_result.short_directional_pf1D_Decoder
    # short_directional_decoder_dict_value = directional_merged_decoders_result.short_directional_decoder_dict

    all_directional_laps_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result
    all_directional_ripple_filter_epochs_decoder_result_value = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result

    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = directional_merged_decoders_result.laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = directional_merged_decoders_result.laps_track_identity_marginals_tuple
    ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = directional_merged_decoders_result.ripple_directional_marginals_tuple
    ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = directional_merged_decoders_result.ripple_track_identity_marginals_tuple
    
    ripple_decoding_time_bin_size: float = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result.decoding_time_bin_size
    ripple_decoding_time_bin_size
    laps_decoding_time_bin_size: float = directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result.decoding_time_bin_size
    laps_decoding_time_bin_size

    laps_all_epoch_bins_marginals_df = directional_merged_decoders_result.laps_all_epoch_bins_marginals_df
    ripple_all_epoch_bins_marginals_df = directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df

    """
    _VersionedResultMixin_version: str = "2024.01.10_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    all_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
    all_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    long_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    long_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
    short_directional_pf1D_Decoder: BasePositionDecoder = serialized_field(default=None)
    short_directional_decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)

    # Posteriors computed via the all_directional decoder:
    all_directional_laps_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)
    all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = serialized_field(default=None)

    # Marginalized posteriors computed from above posteriors:
    laps_directional_marginals_tuple: Tuple = serialized_field(default=None) # laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
    laps_track_identity_marginals_tuple: Tuple = serialized_field(default=None)
    
    ripple_directional_marginals_tuple: Tuple = serialized_field(default=None)
    ripple_track_identity_marginals_tuple: Tuple = serialized_field(default=None) 
    



    # Computed Properties ________________________________________________________________________________________________ #
    @property
    def laps_epochs_df(self) -> pd.DataFrame:
        return deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs).to_dataframe()

    @property
    def ripple_epochs_df(self) -> pd.DataFrame:
        return deepcopy(self.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)

    @property
    def laps_decoding_time_bin_size(self) -> float:
        return self.all_directional_laps_filter_epochs_decoder_result.decoding_time_bin_size

    @property
    def ripple_decoding_time_bin_size(self) -> float:
        return self.all_directional_ripple_filter_epochs_decoder_result.decoding_time_bin_size

    @property
    def laps_all_epoch_bins_marginals_df(self) -> pd.DataFrame:
        """ same quantities computed by `compute_and_export_marginals_df_csvs(...)` 

        all_epoch_bins_marginal

        """
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
        laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = self.laps_track_identity_marginals_tuple

        laps_marginals_df = pd.DataFrame(np.hstack((laps_directional_all_epoch_bins_marginal, laps_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        laps_marginals_df['lap_idx'] = laps_marginals_df.index.to_numpy()
        laps_marginals_df['lap_start_t'] = self.laps_epochs_df['start'].to_numpy()
        return laps_marginals_df

    @property
    def ripple_all_epoch_bins_marginals_df(self) -> pd.DataFrame:
        """ same quantities computed by `compute_and_export_marginals_df_csvs(...)` 
        """
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = self.ripple_directional_marginals_tuple
        ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = self.ripple_track_identity_marginals_tuple

        ## Ripple marginals_df:
        ripple_marginals_df = pd.DataFrame(np.hstack((ripple_directional_all_epoch_bins_marginal, ripple_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
        ripple_marginals_df['ripple_start_t'] = self.ripple_epochs_df['start'].to_numpy()
        return ripple_marginals_df


    def __attrs_post_init__(self):
        # Computes and initializes the marginal properties:
        # laps_epochs_df = deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs).to_dataframe()
        marginals_needs_purge = True
        if all([v is not None for v in [self.all_directional_laps_filter_epochs_decoder_result, self.all_directional_ripple_filter_epochs_decoder_result]]):
            try:
                self.perform_compute_marginals()
                marginals_needs_purge = False
            except AttributeError as e:				
                raise e

        if marginals_needs_purge:
            self.laps_directional_marginals_tuple = None
            self.laps_track_identity_marginals_tuple = None
            self.ripple_directional_marginals_tuple = None
            self.ripple_track_identity_marginals_tuple = None


    def perform_compute_marginals(self):
        """ 
        Called after [self.all_directional_laps_filter_epochs_decoder_result, self.all_directional_ripple_filter_epochs_decoder_result] are updated to compute the four marginal tuples

        Requires: self.all_directional_laps_filter_epochs_decoder_result, self.all_directional_ripple_filter_epochs_decoder_result

        """
        # Computes and initializes the marginal properties:
        laps_epochs_df = deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs).to_dataframe()
        self.laps_directional_marginals_tuple = DirectionalMergedDecodersResult.determine_directional_likelihoods(self.all_directional_laps_filter_epochs_decoder_result)
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = self.laps_directional_marginals_tuple
        self.laps_track_identity_marginals_tuple = DirectionalMergedDecodersResult.determine_long_short_likelihoods(self.all_directional_laps_filter_epochs_decoder_result)
        laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = self.laps_track_identity_marginals_tuple

        laps_marginals_df = pd.DataFrame(np.hstack((laps_directional_all_epoch_bins_marginal, laps_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        laps_marginals_df['lap_idx'] = laps_marginals_df.index.to_numpy()
        laps_marginals_df['lap_start_t'] = laps_epochs_df['start'].to_numpy()
        laps_marginals_df

        ## Decode Ripples:
        ripple_epochs_df = deepcopy(self.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
        self.ripple_directional_marginals_tuple = DirectionalMergedDecodersResult.determine_directional_likelihoods(self.all_directional_ripple_filter_epochs_decoder_result)
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = self.ripple_directional_marginals_tuple
        self.ripple_track_identity_marginals_tuple = DirectionalMergedDecodersResult.determine_long_short_likelihoods(self.all_directional_ripple_filter_epochs_decoder_result)
        ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = self.ripple_track_identity_marginals_tuple

        ## Ripple marginals_df:
        ripple_marginals_df = pd.DataFrame(np.hstack((ripple_directional_all_epoch_bins_marginal, ripple_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
        ripple_marginals_df['ripple_start_t'] = ripple_epochs_df['start'].to_numpy()
        ripple_marginals_df




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

        laps_epochs_df = directional_merged_decoders_result.laps_epochs_df
        ripple_epochs_df = directional_merged_decoders_result.ripple_epochs_df

        return True


    @classmethod
    def build_non_marginalized_raw_posteriors(cls, filter_epochs_decoder_result, debug_print=False):
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

            assert curr_array_shape[1] == 4, f"only works with the all-directional decoder with ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "

            if debug_print:
                print(f'np.shape(a_p_x_given_n): {np.shape(curr_array_shape)}')
                
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=a_p_x_given_n, most_likely_positions_1D=None)
            
            if debug_print:
                print(f'np.shape(curr_unit_posterior_list.p_x_given_n): {np.shape(curr_unit_marginal_x.p_x_given_n)}')
            
            # y-axis marginal:
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(a_p_x_given_n, axis=0)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=1, keepdims=True) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)

            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_x.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_x)
        return custom_curr_unit_marginal_list


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

            assert curr_array_shape[1] == 4, f"only works with the all-directional decoder with ['long_LR', 'long_RL', 'short_LR', 'short_RL'] "

            out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 2, :]) # LR_marginal = long_LR + short_LR
            out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 1, :] + a_p_x_given_n[:, 3, :]) # RL_marginal = long_RL + short_RL

            normalized_out_p_x_given_n = out_p_x_given_n

            input_array = normalized_out_p_x_given_n

            if debug_print:
                print(f'np.shape(input_array): {np.shape(input_array)}')
            # custom marginal over long/short, leaving only LR/RL:
            curr_unit_marginal_y = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_y.p_x_given_n = input_array
            
            # y-axis marginal:
            curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, axis=0)) # sum over all x. Result should be [y_bins x time_bins]

            curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)

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
    def build_custom_marginal_over_long_short(cls, filter_epochs_decoder_result, debug_print=False):
        """ only works for the all-directional coder with the four items
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            active_decoder = all_directional_pf1D_Decoder
            laps_plot_tuple = plot_decoded_epoch_slices(global_any_laps_epochs_obj, laps_filter_epochs_decoder_result, global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                                                        name='stacked_epoch_slices_matplotlib_subplots_LAPS',
                                                        # active_marginal_fn = lambda filter_epochs_decoder_result: filter_epochs_decoder_result.marginal_y_list,
                                                        active_marginal_fn = lambda filter_epochs_decoder_result: build_custom_marginal_over_long_short(filter_epochs_decoder_result),
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

            # out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            # out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 2, :]) # LR_marginal = long_LR + short_LR
            # out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 1, :] + a_p_x_given_n[:, 3, :]) # RL_marginal = long_RL + short_RL

            # Extract the Long/Short items
            out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
            out_p_x_given_n[:, 0, :] = (a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 1, :]) # Long_marginal = long_LR + long_RL 
            out_p_x_given_n[:, 1, :] = (a_p_x_given_n[:, 2, :] + a_p_x_given_n[:, 3, :]) # Short_marginal = short_LR + short_RL
            

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
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_x.p_x_given_n = input_array
            
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            # curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, 1)) # sum over all y. Result should be [x_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
        
            # y-axis marginal:
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(input_array, axis=0)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=1, keepdims=True) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)

            # curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(input_array, axis=1)) # sum over all x. Result should be [y_bins x time_bins]
            # curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            if debug_print:
                print(f'np.shape(curr_unit_marginal_y.p_x_given_n): {np.shape(curr_unit_marginal_x.p_x_given_n)}')
            
            ## Ensures that the marginal posterior is at least 2D:
            # print(f"curr_unit_marginal_y.p_x_given_n.ndim: {curr_unit_marginal_y.p_x_given_n.ndim}")
            # assert curr_unit_marginal_y.p_x_given_n.ndim >= 2
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_x.p_x_given_n.shape}')
            custom_curr_unit_marginal_list.append(curr_unit_marginal_x)
        return custom_curr_unit_marginal_list
    

    
    @classmethod
    def determine_directional_likelihoods(cls, all_directional_laps_filter_epochs_decoder_result):
        """ 

        determine_directional_likelihoods

        directional_marginals, directional_all_epoch_bins_marginal, most_likely_direction_from_decode, is_most_likely_direction_LR_dir = DirectionalMergedDecodersResult.determine_directional_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)

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
    def determine_long_short_likelihoods(cls, all_directional_laps_filter_epochs_decoder_result):
        """ 
        
        laps_track_identity_marginals = DirectionalMergedDecodersResult.determine_long_short_likelihoods(directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result)
        track_identity_marginals, track_identity_all_epoch_bins_marginal, most_likely_track_identity_from_decoder, is_most_likely_track_identity_Long = laps_track_identity_marginals
        
        0: Long
        1: Short
        
        """
        track_identity_marginals = cls.build_custom_marginal_over_long_short(all_directional_laps_filter_epochs_decoder_result)
        
        # gives the likelihood of [LR, RL] for each epoch using information from both Long/Short:
        track_identity_all_epoch_bins_marginal = np.stack([np.sum(v.p_x_given_n, axis=-1)/np.sum(v.p_x_given_n, axis=(-2, -1)) for v in track_identity_marginals], axis=0) # sum over all time-bins within the epoch to reach a consensus
        # directional_all_epoch_bins_marginal

        # Find the indicies via this method:
        most_likely_track_identity_from_decoder = np.argmax(track_identity_all_epoch_bins_marginal, axis=1) # consistent with 'lap_dir' columns. for LR_dir, values become more positive with time
        is_most_likely_track_identity_Long = np.logical_not(most_likely_track_identity_from_decoder) # consistent with 'is_LR_dir' column. for LR_dir, values become more positive with time

        # most_likely_direction_from_decoder
        return track_identity_marginals, track_identity_all_epoch_bins_marginal, most_likely_track_identity_from_decoder, is_most_likely_track_identity_Long

    @classmethod
    def validate_lap_dir_estimations(cls, global_session, active_global_laps_df, laps_is_most_likely_direction_LR_dir):
        """ validates the lap direction and track estimations. 
        """
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name])
        # active_global_laps_df = _subfn_compute_lap_dir_from_smoothed_velocity(global_session, active_global_laps_df=active_global_laps_df)
        active_global_laps_df = Laps._compute_lap_dir_from_smoothed_velocity(active_global_laps_df, global_session=global_session) ## NOTE: global_session does not work, use curr_active_pipeline.sess instead (unfiltered session) otherwise it clips the last two laps

        # Validate Laps:
        # ground_truth_lap_dirs = active_global_laps_df['lap_dir'].to_numpy()
        ground_truth_lap_is_LR_dir = active_global_laps_df['is_LR_dir'].to_numpy()
        n_laps = np.shape(active_global_laps_df)[0]
        assert len(laps_is_most_likely_direction_LR_dir) == n_laps
        percent_laps_estimated_correctly = (np.sum(ground_truth_lap_is_LR_dir == laps_is_most_likely_direction_LR_dir) / n_laps)
        print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
        return percent_laps_estimated_correctly


    def compute_and_export_marginals_df_csvs(self, parent_output_path: Path, active_context):
        """ 
        
        active_context = curr_active_pipeline.get_session_context()
        """
        output_date_str: str = get_now_rounded_time_str()

        # Export CSVs:
        def export_marginals_df_csv(marginals_df: pd.DataFrame, data_identifier_str: str = f'(laps_marginals_df)'):
            """ captures `active_context`, `parent_output_path`. 'output_date_str'
            """
            # parent_output_path: Path = Path('output').resolve()
            # active_context = curr_active_pipeline.get_session_context()
            session_identifier_str: str = active_context.get_description()
            assert output_date_str is not None
            out_basename = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-01-04|kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
            out_filename = f"{out_basename}.csv"
            out_path = parent_output_path.joinpath(out_filename).resolve()
            marginals_df.to_csv(out_path)
            return out_path 


        ## Laps:
        laps_epochs_df = deepcopy(self.all_directional_laps_filter_epochs_decoder_result.filter_epochs).to_dataframe()
        laps_directional_marginals_tuple = DirectionalMergedDecodersResult.determine_directional_likelihoods(self.all_directional_laps_filter_epochs_decoder_result)
        laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = laps_directional_marginals_tuple
        laps_track_identity_marginals = DirectionalMergedDecodersResult.determine_long_short_likelihoods(self.all_directional_laps_filter_epochs_decoder_result)
        track_identity_marginals, track_identity_all_epoch_bins_marginal, most_likely_track_identity_from_decoder, is_most_likely_track_identity_Long = laps_track_identity_marginals
        
        laps_marginals_df = pd.DataFrame(np.hstack((laps_directional_all_epoch_bins_marginal, track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        laps_marginals_df['lap_idx'] = laps_marginals_df.index.to_numpy()
        laps_marginals_df['lap_start_t'] = laps_epochs_df['start'].to_numpy()
        laps_marginals_df

        laps_out_path = export_marginals_df_csv(laps_marginals_df, data_identifier_str=f'(laps_marginals_df)')


        ## Ripples:
        ripple_epochs_df = deepcopy(self.all_directional_ripple_filter_epochs_decoder_result.filter_epochs)
        all_directional_ripple_filter_epochs_decoder_result: DecodedFilterEpochsResult = self.all_directional_ripple_filter_epochs_decoder_result
        ripple_marginals = DirectionalMergedDecodersResult.determine_directional_likelihoods(all_directional_ripple_filter_epochs_decoder_result)
        ripple_directional_marginals, ripple_directional_all_epoch_bins_marginal, ripple_most_likely_direction_from_decoder, ripple_is_most_likely_direction_LR_dir  = ripple_marginals
        ripple_track_identity_marginals = DirectionalMergedDecodersResult.determine_long_short_likelihoods(all_directional_ripple_filter_epochs_decoder_result)
        ripple_track_identity_marginals, ripple_track_identity_all_epoch_bins_marginal, ripple_most_likely_track_identity_from_decoder, ripple_is_most_likely_track_identity_Long = ripple_track_identity_marginals

        ## Ripple marginals_df:
        ripple_marginals_df = pd.DataFrame(np.hstack((ripple_directional_all_epoch_bins_marginal, ripple_track_identity_all_epoch_bins_marginal)), columns=['P_LR', 'P_RL', 'P_Long', 'P_Short'])
        ripple_marginals_df['ripple_idx'] = ripple_marginals_df.index.to_numpy()
        ripple_marginals_df['ripple_start_t'] = ripple_epochs_df['start'].to_numpy()
        ripple_marginals_df

        ripple_out_path = export_marginals_df_csv(ripple_marginals_df, data_identifier_str=f'(ripple_marginals_df)')

        return (laps_marginals_df, laps_out_path), (ripple_marginals_df, ripple_out_path)


@define(slots=False, repr=False)
class DirectionalDecodersDecodedResult(ComputedResult):
    """ a container containing a dict containing the four pf1D_Decoders and a Dict that can be used to cache the results of decoding across all time-bins (continuously) for each decoder in the dict.
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersDecodedResult

    ## Get the result after computation:
    directional_decoders_decode_result: DirectionalDecodersDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
    all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
    # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
    time_bin_size: float = directional_decoders_decode_result.last_decoding_time_bin_size
    print(f'time_bin_size: {time_bin_size}')
    continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_dict
    
        
    """
    _VersionedResultMixin_version: str = "2024.01.16_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = serialized_field(default=None)
    
    # Posteriors computed via the all_directional decoder:
    continuously_decoded_result_cache_dict: Dict[float, Dict[str, DecodedFilterEpochsResult]] = serialized_field(default=None) # key is the t_bin_size in seconds
    
    @property
    def last_decoding_time_bin_size(self) -> Optional[float]:
        """Gets the last cached continuously_decoded_dict property."""
        if ((self.continuously_decoded_result_cache_dict is None) or (len(self.continuously_decoded_result_cache_dict or {}) < 1)):
            return None
        else:
            last_time_bin_size: float = list(self.continuously_decoded_result_cache_dict.keys())[-1]
            return last_time_bin_size   
        

    @property
    def continuously_decoded_dict(self) -> Optional[Dict[str, DecodedFilterEpochsResult]]:
        """Gets the last cached continuously_decoded_dict property."""
        last_time_bin_size = self.last_decoding_time_bin_size
        if (last_time_bin_size is None):
            return None
        else:
            # otherwise return the result            
            return self.continuously_decoded_result_cache_dict[last_time_bin_size]         


    @classmethod
    def validate_has_directional_decoded_continuous_epochs(cls, curr_active_pipeline, computation_filter_name='maze') -> bool:
        """ Validates that the decoding is complete
        """
        directional_decoders_decode_result: DirectionalDecodersDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        if len(continuously_decoded_result_cache_dict) < 1:
            return False
    
        time_bin_size: float = directional_decoders_decode_result.last_decoding_time_bin_size
        if time_bin_size is None:
            return False

        continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_dict
        if continuously_decoded_dict is None:
            return False

        return True


import pandas as pd
import plotly.express as px
from pathlib import Path

def plot_all_sessions(directory, save_figures=False, figure_save_extension='.png'):
    """ takes the directory containing the .csv pairs that were exported by `export_marginals_df_csv`
    Produces and then saves figures out the the f'{directory}/figures/' subfolder

    """
    if not isinstance(directory, Path):
        directory = Path(directory).resolve()
    assert directory.exists()
    print(f'plot_all_sessions(directory: {directory})')
    if save_figures:
        # Create a 'figures' subfolder if it doesn't exist
        figures_folder = Path(directory, 'figures')
        figures_folder.mkdir(parents=False, exist_ok=True)
        assert figures_folder.exists()
        print(f'\tfigures_folder: {figures_folder}')
    
    # Get all CSV files in the specified directory
    # all_csv_files = Path(directory).glob('*-(laps|ripple)_marginals_df).csv')
    all_csv_files = sorted(Path(directory).glob('*_marginals_df).csv'))

    # Separate the CSV files into laps and ripple lists
    laps_files = [file for file in all_csv_files if 'laps' in file.stem]
    ripple_files = [file for file in all_csv_files if 'ripple' in file.stem]

    # Create an empty list to store the figures
    all_figures = []

    # Iterate through the pairs and create figures
    for laps_file, ripple_file in zip(laps_files, ripple_files):
        session_name = laps_file.stem.split('-')[3]  # Extract session name from the filename
        print(f'processing session_name: {session_name}')
        
        laps_df = pd.read_csv(laps_file)
        ripple_df = pd.read_csv(ripple_file)

        # SEPERATELY _________________________________________________________________________________________________________ #
        # Create a bubble chart for laps
        fig_laps = px.scatter(laps_df, x='lap_start_t', y='P_Long', title=f"Laps - Session: {session_name}")

        # Create a bubble chart for ripples
        fig_ripples = px.scatter(ripple_df, x='ripple_start_t', y='P_Long', title=f"Ripples - Session: {session_name}")

        if save_figures:
            # Save the figures to the 'figures' subfolder
            print(f'\tsaving figures...')
            fig_laps_name = Path(figures_folder, f"{session_name}_laps_marginal{figure_save_extension}").resolve()
            print(f'\tsaving "{fig_laps_name}"...')
            fig_laps.write_image(fig_laps_name)
            fig_ripple_name = Path(figures_folder, f"{session_name}_ripples_marginal{figure_save_extension}").resolve()
            print(f'\tsaving "{fig_ripple_name}"...')
            fig_ripples.write_image(fig_ripple_name)
        
        # Append both figures to the list
        all_figures.append((fig_laps, fig_ripples))
        
        # # COMBINED ___________________________________________________________________________________________________________ #
        # # Create a subplot with laps and ripples stacked vertically
        # fig_combined = px.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        #                                         subplot_titles=[f"Laps - Session: {session_name}", f"Ripples - Session: {session_name}"])

        # # Add scatter traces to the subplots
        # fig_combined.add_trace(px.scatter(laps_df, x='lap_start_t', y='P_Long').data[0], row=1, col=1)
        # fig_combined.add_trace(px.scatter(ripple_df, x='ripple_start_t', y='P_Long').data[0], row=2, col=1)

        # # Update layout for better visualization
        # fig_combined.update_layout(height=600, width=800, title_text=f"Combined Plot - Session: {session_name}")

        # # Save the figure to the 'figures' subfolder
        # figure_filename = Path(figures_folder, f"{session_name}_marginal.png")
        # fig_combined.write_image(figure_filename)
        
        # all_figures.append(fig_combined)
        
    return all_figures

# # Example usage:
# # directory = '/home/halechr/FastData/collected_outputs/'
# # directory = r'C:\Users\pho\Desktop\collected_outputs'
# directory = r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs'

# all_session_figures = plot_all_sessions(directory, save_figures=True)

# # Show figures for all sessions
# for fig_laps, fig_ripples in all_session_figures:
#     fig_laps.show()
#     fig_ripples.show()



def plot_all_epoch_bins_marginal_predictions(directional_merged_decoders_result, t_start=None, t_split=1000.0, t_end=None, active_context=None, perform_write_to_file_callback=None):
    """ Plots three Matplotlib figures displaying the quantile differences
    
    
    Usage:
    
    ripple_all_epoch_bins_marginals_df
    laps_all_epoch_bins_marginals_df
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import plot_all_epoch_bins_marginal_predictions

    _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
    global_epoch = curr_active_pipeline.filtered_epochs[global_epoch_name]
    t_start, t_end = global_epoch.start_end_times
    short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
    split_time_t: float = short_epoch.t_start
    active_context = curr_active_pipeline.sess.get_context()

    ## Get the result after computation:
    directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
    
    collector = plot_all_epoch_bins_marginal_predictions(directional_merged_decoders_result, t_start=t_start, t_split=split_time_t, t_end=t_end, active_context=active_context, perform_write_to_file_callback=None)

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from flexitext import flexitext ## flexitext for formatted matplotlib text

    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText
    
    laps_all_epoch_bins_marginals_df = deepcopy(directional_merged_decoders_result.laps_all_epoch_bins_marginals_df)
    ripple_all_epoch_bins_marginals_df = deepcopy(directional_merged_decoders_result.ripple_all_epoch_bins_marginals_df)

    if active_context is not None:
        display_context = active_context.adding_context('display_fn', display_fn_name='plot_all_epoch_bins_marginal_predictions')
        
    # These subset contexts are used to filter out lap/ripple only keys.
    # e.g. active_context=curr_active_pipeline.build_display_context_for_session('directional_merged_pf_decoded_epochs', laps_t_bin=laps_decoding_time_bin_size, ripple_t_bin=ripple_decoding_time_bin_size)
        # only want laps_t_bin on the laps plot and ripple_t_bin on the ripples plot
    laps_only_keys = [item for item in display_context.keys() if 'lap' in item] # items exclusive to laps: ['laps_t_bin']
    ripple_only_keys = [item for item in display_context.keys() if 'ripple' in item]
    laps_display_context = display_context.get_subset(subset_excludelist=ripple_only_keys) # laps specific context filtering out the ripple keys
    ripple_display_context = display_context.get_subset(subset_excludelist=laps_only_keys) # ripple specific context filtering out the laps keys


    with mpl.rc_context({'figure.figsize': (12.4, 4.8), 'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42,
                          "axes.spines.left": False, "axes.spines.right": False, "axes.spines.bottom": False, "axes.spines.top": False,
                          "axes.edgecolor": "none", "xtick.bottom": False, "xtick.top": False, "ytick.left": False, "ytick.right": False}):
        # Create a FigureCollector instance
        with FigureCollector(name='plot_all_epoch_bins_marginal_predictions', base_context=display_context) as collector:

            ## Define common operations to do after making the figure:
            def setup_common_after_creation(a_collector, fig, axes, sub_context, title=f'<size:22> Sig. (>0.95) <weight:bold>Best</> <weight:bold>Quantile Diff</></>'):
                """ Captures:

                t_split, t_start, t_end)
                """
                a_collector.contexts.append(sub_context)
                
                for ax in (axes if isinstance(axes, Iterable) else [axes]):
                    # Update the xlimits with the new bounds
                    ax.set_ylim(0.0, 1.0)
                    # Add epoch indicators
                    _tmp_output_dict = PlottingHelpers.helper_matplotlib_add_long_short_epoch_indicator_regions(ax=ax, t_split=t_split, t_start=t_start, t_end=t_end)
                    # Update the xlimits with the new bounds
                    ax.set_xlim(t_start, t_end)
                    # Draw a horizontal line at y=0.5
                    ax.axhline(y=0.5, color=(0,0,0,1)) # , linestyle='--'
                    ## This is figure level stuff and only needs to be done once:
                    # `flexitext` version:
                    text_formatter = FormattedFigureText()
                    ax.set_title('')
                    fig.suptitle('')
                    # top=0.84, bottom=0.125, left=0.07, right=0.97,
                    # text_formatter.setup_margins(fig, top_margin=1.0, left_margin=0.0, right_margin=1.0, bottom_margin=0.05)
                    text_formatter.setup_margins(fig, top_margin=0.84, left_margin=0.07, right_margin=0.97, bottom_margin=0.125)
                    # fig.subplots_adjust(top=top_margin, left=left_margin, right=right_margin, bottom=bottom_margin)
                    # title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, title, va="bottom", xycoords="figure fraction")
                    title_text_obj = flexitext(text_formatter.left_margin, 0.98, title, va="top", xycoords="figure fraction") # 0.98, va="top" means the top edge of the title will be aligned to the fig_y=0.98 mark of the figure.
                    # footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                    #                             text_formatter._build_footer_string(active_context=sub_context),
                    #                             va="top", xycoords="figure fraction")

                    footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (0.0025), ## (va="bottom", (0.0025)) - this means that the bottom edge of the footer text is aligned with the fig_y=0.0025 in figure space
                                                text_formatter._build_footer_string(active_context=sub_context),
                                                va="bottom", xycoords="figure fraction")
            
                if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                    perform_write_to_file_callback(sub_context, fig)
                
            # Plot for BestDir
            fig, ax = collector.subplots(num='Laps_Marginal', clear=True)
            _out_Laps = sns.scatterplot(
                ax=ax,
                data=laps_all_epoch_bins_marginals_df,
                x='lap_start_t',
                y='P_Long',
                # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            )
            setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=laps_display_context.adding_context('subplot', subplot_name='Laps all_epoch_binned Marginals'), 
                                        title=f'<size:22> Laps <weight:bold>all_epoch_binned</> Marginals</>')
            
            fig, ax = collector.subplots(num='Ripple_Marginal', clear=True)
            _out_Ripple = sns.scatterplot(
                ax=ax,
                data=ripple_all_epoch_bins_marginals_df,
                x='ripple_start_t',
                y='P_Long',
                # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            )
            setup_common_after_creation(collector, fig=fig, axes=ax, sub_context=ripple_display_context.adding_context('subplot', subplot_name='Ripple all_epoch_binned Marginals'), 
                            title=f'<size:22> Ripple <weight:bold>all_epoch_binned</> Marginals</>')


            # # Plot for Both Laps/Ripple on the same figure using subplots:
            # fig, axs = collector.subplots(num='all_epoch_binned_Marginals', nrows = 2, ncols = 1, sharex=True, sharey=True, clear=True)
            # _out_Laps = sns.scatterplot(
            #     ax=axs[0],
            #     data=laps_all_epoch_bins_marginals_df,
            #     x='lap_start_t',
            #     y='P_Long',
            #     # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            # )
            
            # # Ripple_Marginal
            # _out_Ripple = sns.scatterplot(
            #     ax=axs[1],
            #     data=ripple_all_epoch_bins_marginals_df,
            #     x='ripple_start_t',
            #     y='P_Long',
            #     # size='LR_Long_rel_num_cells',  # Use the 'size' parameter for variable marker sizes
            # )
            # setup_common_after_creation(collector, fig=fig, axes=axs, sub_context=display_context.adding_context('subplot', subplot_name='Laps and Ripple all_epoch_binned Marginals'), 
            #                             title=f'<size:22> Laps+Ripple <weight:bold>all_epoch_binned</> Marginals</>')




    
    # Access the collected figures outside the context manager
    # result = tuple(collector.created_figures)

    return collector




class DirectionalPlacefieldGlobalComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    """ functions related to directional placefield computations. """
    _computationGroupName = 'directional_pfs'
    _computationPrecidence = 1000
    _is_global = True

    @function_attributes(short_name='split_to_directional_laps', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'], input_requires=[], output_provides=[], uses=['_perform_PBE_stats'], used_by=[], creation_date='2023-10-25 09:33', related_items=[],
        provides_global_keys=['DirectionalLaps'],
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
        requires_global_keys=['DirectionalLaps'], provides_global_keys=['DirectionalMergedDecoders'],
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
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        # long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(owning_pipeline_reference.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in owning_pipeline_reference.sess.epochs is 'maze1' without the '_any' part.
        
        unfiltered_session = deepcopy(owning_pipeline_reference.sess)
        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used in 

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

        ## Great or update the global directional_merged_decoders_result:
        directional_merged_decoders_result = global_computation_results.computed_data.get('DirectionalMergedDecoders', DirectionalMergedDecodersResult(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
                                                      long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
                                                      short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder))


        directional_merged_decoders_result.__dict__.update(all_directional_decoder_dict=all_directional_decoder_dict, all_directional_pf1D_Decoder=all_directional_pf1D_Decoder, 
                                                      long_directional_decoder_dict=long_directional_decoder_dict, long_directional_pf1D_Decoder=long_directional_pf1D_Decoder, 
                                                      short_directional_decoder_dict=short_directional_decoder_dict, short_directional_pf1D_Decoder=short_directional_pf1D_Decoder)
        
        

        # Decode Epochs (Laps/Ripples) Using the merged all-directional decoder): ____________________________________________ #

        ## Decode Laps:
        laps_decoding_time_bin_size: float = 0.075
        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_any_name].computation_config.pf_params.computation_epochs) # global_any_name='maze_any' (? same as global_epoch_name?)
        

        directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, debug_print=False)
        
        ## Decode Ripples:        
        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
        min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
        ripple_decoding_time_bin_size: float = min(0.010, min_possible_time_bin_size) # 10ms # 0.002
        directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(owning_pipeline_reference.sess.spikes_df), global_replays, decoding_time_bin_size=ripple_decoding_time_bin_size)

        ## Post Compute Validations:
        directional_merged_decoders_result.perform_compute_marginals()
        
         # Validate Laps:
        try:
            laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = directional_merged_decoders_result.laps_directional_marginals_tuple
            percent_laps_estimated_correctly = DirectionalMergedDecodersResult.validate_lap_dir_estimations(unfiltered_session, active_global_laps_df=global_any_laps_epochs_obj.to_dataframe(), laps_is_most_likely_direction_LR_dir=laps_is_most_likely_direction_LR_dir)
            print(f'percent_laps_estimated_correctly: {percent_laps_estimated_correctly}')
        except (AssertionError, ValueError) as err:
            print(F'fails due to some types thing?')
            print(f'\terr: {err}')
            pass
        
        # Set the global result:
        global_computation_results.computed_data['DirectionalMergedDecoders'] = directional_merged_decoders_result        
        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_contexts_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        ripple_filter_epochs_decoder_result = directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result
        
        """
        return global_computation_results
    

    @function_attributes(short_name='directional_decoders_decode_continuous', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-01-17 09:05', related_items=[],
        requires_global_keys=['DirectionalLaps', 'RankOrder'], provides_global_keys=['DirectionalDecodersDecoded'],
        validate_computation_test=DirectionalDecodersDecodedResult.validate_has_directional_decoded_continuous_epochs, is_global=True)
    def _decode_continuous_using_directional_decoders(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, time_bin_size: Optional[float]=None):
        """

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['pf1D_Decoder_dict']
                ['DirectionalDecodersDecoded']['directional_lap_specific_configs']
                ['DirectionalDecodersDecoded']['continuously_decoded_result_cache_dict']


                from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersDecodedResult

                directional_decoders_decode_result: DirectionalDecodersDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
                all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
                # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
                time_bin_size: float = directional_decoders_decode_result.last_decoding_time_bin_size
                print(f'time_bin_size: {time_bin_size}')
                continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_dict


        """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
        from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
        from neuropy.core.epoch import Epoch
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder
                
        # directional_decoders_decode_result = global_computation_results.computed_data.get('DirectionalDecodersDecoded', DirectionalDecodersDecodedResult(pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict, continuously_decoded_result_cache_dict=continuously_decoded_result_cache_dict))

        # Unpack all directional variables:
        long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [owning_pipeline_reference.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

        all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = dict(zip(all_directional_decoder_names, [deepcopy(long_LR_pf1D_Decoder), deepcopy(long_RL_pf1D_Decoder), deepcopy(short_LR_pf1D_Decoder), deepcopy(short_RL_pf1D_Decoder)]))


        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        single_global_epoch: Epoch = Epoch(single_global_epoch_df)
        ## Build Epoch object across whole sessions:
        if time_bin_size is None:
            # use default time_bin_size from the previous decoder
            time_bin_size = long_LR_pf1D_Decoder.time_bin_size
            
        # time_binning_container: BinningContainer = deepcopy(long_LR_pf1D_Decoder.time_binning_container)
        # time_binning_container
        # time_binning_container.edges # array([31.8648, 31.8978, 31.9308, ..., 1203.56, 1203.6, 1203.63])
        # time_binning_container.centers # array([31.8813, 31.9143, 31.9473, ..., 1203.55, 1203.58, 1203.61])
        print(f'time_bin_size: {time_bin_size}')

        # Get proper global_spikes_df:
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        rank_order_results = owning_pipeline_reference.global_computation_results.computed_data['RankOrder'] # "RankOrderComputationsContainer"
        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        included_qclu_values: List[int] = rank_order_results.included_qclu_values
        directional_laps_results: DirectionalLapsResult = owning_pipeline_reference.global_computation_results.computed_data['DirectionalLaps']
        track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
        any_list_neuron_IDs = track_templates.any_decoder_neuron_IDs # neuron_IDs as they appear in any list
        global_spikes_df = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].spikes_df).spikes.sliced_by_neuron_id(any_list_neuron_IDs) # Cut spikes_df down to only the neuron_IDs that appear at least in one decoder:        
        # alternatively if importing `RankOrderAnalyses` is okay, we can do:
        # global_spikes_df, _, _ = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=owning_pipeline_reference, num_shuffles=0) # does not do shuffling
        
        spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)

        # print(f'add_directional_decoder_decoded_epochs(...): decoding continuous epochs for each directional decoder.')
        # t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        # single_global_epoch: Epoch = Epoch(pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})) # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session
        # global_spikes_df, _, _ = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=owning_pipeline_reference, num_shuffles=0) # does not do shuffling
        # spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)
        all_directional_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = {k:v.decode_specific_epochs(spikes_df=spikes_df, filter_epochs=single_global_epoch, decoding_time_bin_size=time_bin_size, debug_print=False) for k,v in all_directional_pf1D_Decoder_dict.items()}
        # print(f'\t computation done.')
        
        continuously_decoded_result_cache_dict = {time_bin_size:all_directional_continuously_decoded_dict}

        # Store all_directional_pf1D_Decoder_dict, all_directional_continuously_decoded_dict
        
        ## Create or update the global directional_decoders_decode_result:
        directional_decoders_decode_result = global_computation_results.computed_data.get('DirectionalDecodersDecoded', DirectionalDecodersDecodedResult(pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict, continuously_decoded_result_cache_dict=continuously_decoded_result_cache_dict))
        directional_decoders_decode_result.__dict__.update(pf1D_Decoder_dict=all_directional_pf1D_Decoder_dict)
        directional_decoders_decode_result.continuously_decoded_result_cache_dict[time_bin_size] = all_directional_continuously_decoded_dict # update the entry for this time_bin_size

        # Set the global result:
        global_computation_results.computed_data['DirectionalDecodersDecoded'] = directional_decoders_decode_result
        

        """ Usage:
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersDecodedResult

        directional_decoders_decode_result: DirectionalDecodersDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.last_decoding_time_bin_size
        print(f'time_bin_size: {time_bin_size}')
        continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_dict
        
        """
        return global_computation_results






    


        # # Unpack all directional variables:
        # ## {"even": "RL", "odd": "LR"}
        # long_LR_name, short_LR_name, global_LR_name, long_RL_name, short_RL_name, global_RL_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']
        # # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        # (long_LR_context, long_RL_context, short_LR_context, short_RL_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]
        # (long_LR_pf1D_Decoder, long_RL_pf1D_Decoder, short_LR_pf1D_Decoder, short_RL_pf1D_Decoder) = (long_LR_results.pf1D_Decoder, long_RL_results.pf1D_Decoder, short_LR_results.pf1D_Decoder, short_RL_results.pf1D_Decoder)

        # all_directional_decoder_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = dict(zip(all_directional_decoder_names, [deepcopy(long_LR_pf1D_Decoder), deepcopy(long_RL_pf1D_Decoder), deepcopy(short_LR_pf1D_Decoder), deepcopy(short_RL_pf1D_Decoder)]))


        # print(f'add_directional_decoder_decoded_epochs(...): decoding continuous epochs for each directional decoder.')
        # t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # single_global_epoch: Epoch = Epoch(pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})) # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session
        # global_spikes_df, _, _ = RankOrderAnalyses.common_analysis_helper(curr_active_pipeline=curr_active_pipeline, num_shuffles=0) # does not do shuffling
        # spikes_df = deepcopy(global_spikes_df) #.spikes.sliced_by_neuron_id(track_templates.shared_aclus_only_neuron_IDs)
        # all_directional_continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = {k:v.decode_specific_epochs(spikes_df=spikes_df, filter_epochs=single_global_epoch, decoding_time_bin_size=time_bin_size, debug_print=False) for k,v in all_directional_pf1D_Decoder_dict.items()}
        # print(f'\t computation done.')
        


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
            fig = plt.figure() # layout="constrained"
            subfigures_dict = dict(zip(list(decoders_dict.keys()), fig.subfigures(nrows=1, ncols=4)))
            # plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
            # plt.tight_layout()
            
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
    def _display_directional_merged_pfs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None,
                                        plot_all_directions=True, plot_long_directional=False, plot_short_directional=False, **kwargs):
        """ Plots the merged pseduo-2D pfs/ratemaps. Plots: All-Directions, Long-Directional, Short-Directional in seperate windows. 
        
        History: this is the Post 2022-10-22 display_all_pf_2D_pyqtgraph_binned_image_rendering-based method:
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, display_all_pf_2D_pyqtgraph_binned_image_rendering

        defer_render = kwargs.pop('defer_render', False)
        directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

        if plot_all_directions:
            active_merged_pf_plots_data_dict[owning_pipeline_reference.build_display_context_for_session(track_config='All-Directions', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering')] = directional_merged_decoders_result.all_directional_pf1D_Decoder.pf # all-directions
        if plot_long_directional:
            active_merged_pf_plots_data_dict[owning_pipeline_reference.build_display_context_for_session(track_config='Long-Directional', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering')] = directional_merged_decoders_result.long_directional_pf1D_Decoder.pf # Long-only
        if plot_short_directional:
            active_merged_pf_plots_data_dict[owning_pipeline_reference.build_display_context_for_session(track_config='Short-Directional', display_fn_name='display_all_pf_2D_pyqtgraph_binned_image_rendering')] = directional_merged_decoders_result.short_directional_pf1D_Decoder.pf # Short-only

        out_plots_dict = {}
        
        for active_context, active_pf_2D in active_merged_pf_plots_data_dict.items():
            # figure_format_config = {} # empty dict for config
            figure_format_config = {'scrollability_mode': LayoutScrollability.NON_SCROLLABLE} # kwargs # kwargs as default figure_format_config
            out_all_pf_2D_pyqtgraph_binned_image_fig = display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config) # output is BasicBinnedImageRenderingWindow
            # Set the window title from the context
            out_all_pf_2D_pyqtgraph_binned_image_fig.setWindowTitle(f'{active_context.get_description()}')
            out_plots_dict[active_context] = out_all_pf_2D_pyqtgraph_binned_image_fig

            if not defer_render:
                out_all_pf_2D_pyqtgraph_binned_image_fig.show()

        return out_plots_dict


    @function_attributes(short_name='directional_merged_decoder_decoded_epochs', tags=['yellow-blue-plots', 'directional_merged_decoder_decoded_epochs', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_decoded_epoch_slices'], used_by=[], creation_date='2024-01-04 02:59', related_items=[], is_global=True)
    def _display_directional_merged_pf_decoded_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None,
                                                    single_plot_fixed_height=50.0, max_num_lap_epochs: int = 25, max_num_ripple_epochs: int = 45, size=(15,7), dpi=72, constrained_layout=True, scrollable_figure=True,
                                                    skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, **kwargs):
            """ Renders to windows, one with the decoded laps and another with the decoded ripple posteriors, computed using the merged pseudo-2D decoder.

            """
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
            from neuropy.utils.matplotlib_helpers import FormattedFigureText
        

            active_context = kwargs.pop('active_context', None)
            if active_context is not None:
                # Update the existing context:
                display_context = active_context.adding_context('display_fn', display_fn_name='directional_merged_pf_decoded_epochs')
            else:
                active_context = owning_pipeline_reference.sess.get_context()
                # Build the active context directly:
                display_context = owning_pipeline_reference.build_display_context_for_session('directional_merged_pf_decoded_epochs')

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)
            active_config_name: bool = kwargs.pop('active_config_name', None)

            perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: owning_pipeline_reference.output_figure(final_context, fig)))
            # Extract kwargs for figure rendering
            render_merged_pseudo2D_decoder_laps = kwargs.pop('render_merged_pseudo2D_decoder_laps', False)
            
            render_directional_marginal_laps = kwargs.pop('render_directional_marginal_laps', True)
            render_directional_marginal_ripples = kwargs.pop('render_directional_marginal_ripples', False)
            render_track_identity_marginal_laps = kwargs.pop('render_track_identity_marginal_laps', False)
            render_track_identity_marginal_ripples = kwargs.pop('render_track_identity_marginal_ripples', False)

            directional_merged_decoders_result = kwargs.pop('directional_merged_decoders_result', None)
            if directional_merged_decoders_result is not None:
                print("WARN: User provided a custom directional_merged_decoders_result as a kwarg. This will be used instead of the computed result global_computation_results.computed_data['DirectionalMergedDecoders'].")
                
            else:
                directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']


            # get the time bin size from the decoder:
            laps_decoding_time_bin_size: float = directional_merged_decoders_result.laps_decoding_time_bin_size
            ripple_decoding_time_bin_size: float = directional_merged_decoders_result.ripple_decoding_time_bin_size



            # figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={})

            # Recover from the saved global result:
            # directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            # directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

            # requires `laps_is_most_likely_direction_LR_dir` from `laps_marginals`
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()

            graphics_output_dict = {}

            # Shared active_decoder, global_session:
            active_decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
            global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 

            # 'figure.constrained_layout.use': False, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
            # 'figure.constrained_layout.use': constrained_layout, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
            with mpl.rc_context({'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, }): # 'figure.figsize': (12.4, 4.8), 
                # Create a FigureCollector instance
                with FigureCollector(name='plot_directional_merged_pf_decoded_epochs', base_context=display_context) as collector:

                    ## Define the overriden plot function that internally calls the normal plot function but also permits doing operations before and after, such as building titles or extracting figures to save them:
                    def _mod_plot_decoded_epoch_slices(*args, **subfn_kwargs):
                        """ implicitly captures: owning_pipeline_reference, collector, perform_write_to_file_callback, save_figure, skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True

                        NOTE: each call requires adding the additional kwarg: `_main_context=_main_context`
                        """
                        assert '_mod_plot_kwargs' in subfn_kwargs
                        _mod_plot_kwargs = subfn_kwargs.pop('_mod_plot_kwargs')
                        assert 'final_context' in _mod_plot_kwargs
                        _main_context = _mod_plot_kwargs['final_context']
                        assert _main_context is not None
                        # Build the rest of the properties:
                        sub_context: IdentifyingContext = owning_pipeline_reference.build_display_context_for_session('directional_merged_pf_decoded_epochs', **_main_context)
                        sub_context_str: str = sub_context.get_description(subset_includelist=['t_bin'], include_property_names=True) # 't-bin_0.5' # str(sub_context.get_description())
                        modified_name: str = subfn_kwargs.pop('name', '')
                        if len(sub_context_str) > 0:
                            modified_name = f"{modified_name}_{sub_context_str}"
                        subfn_kwargs['name'] = modified_name # update the name by appending 't-bin_0.5'
                        
                        # Call the main plot function:
                        out_plot_tuple = plot_decoded_epoch_slices(*args, skip_plotting_measured_positions=skip_plotting_measured_positions, skip_plotting_most_likely_positions=skip_plotting_most_likely_positions, **subfn_kwargs)
                        # Post-plot call:
                        assert len(out_plot_tuple) == 4
                        params, plots_data, plots, ui = out_plot_tuple # [2] corresponds to 'plots' in params, plots_data, plots, ui = laps_plots_tuple
                        # post_hoc_append to collector
                        mw = ui.mw # MatplotlibTimeSynchronizedWidget
                        if mw is not None:
                            fig = mw.getFigure()
                            collector.post_hoc_append(figs=mw.fig, axes=mw.axes, contexts=sub_context)
                            title = mw.params.name
                        else:
                            fig = plots.fig
                            collector.post_hoc_append(figs=fig, axes=plots.axs, contexts=sub_context)
                            title = params.name

                        # Recover the proper title:
                        assert title is not None, f"title: {title}"
                        print(f'title: {title}')

                        # # `flexitext` version:
                        # text_formatter = FormattedFigureText()
                        # fig.suptitle('')
                        # text_formatter.setup_margins(fig) # , top_margin=0.740
                        # title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, title, va="bottom", xycoords="figure fraction")
                        # footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                        # 							text_formatter._build_footer_string(active_context=sub_context),
                        # 							va="top", xycoords="figure fraction")

                        # # Add epoch indicators
                        # for ax in (axes if isinstance(axes, Iterable) else [axes]):
                        # 	PlottingHelpers.helper_matplotlib_add_long_short_epoch_indicator_regions(ax=ax, t_split=t_split)
                            
                            # # `flexitext` version:
                            # text_formatter = FormattedFigureText()
                            # ax.set_title('')
                            # fig.suptitle('')
                            # text_formatter.setup_margins(fig)
                            # title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin,
                            # 						title,
                            # 						va="bottom", xycoords="figure fraction")
                            # footer_text_obj = flexitext((text_formatter.left_margin * 0.1), (text_formatter.bottom_margin * 0.25),
                            # 							text_formatter._build_footer_string(active_context=sub_context),
                            # 							va="top", xycoords="figure fraction")
                    
                        if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                            if save_figure:
                                perform_write_to_file_callback(sub_context, fig)
                            
                        # Close if defer_render
                        if defer_render:
                            if mw is not None:
                                mw.close()

                        return out_plot_tuple
                    
                    if render_merged_pseudo2D_decoder_laps:
                        # Merged Pseduo2D Decoder Posteriors:
                        _main_context = {'decoded_epochs': 'Laps', 'Pseudo2D': 'Posterior', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        graphics_output_dict['raw_posterior_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Posterior',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_non_marginalized_raw_posteriors(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context),
                            **deepcopy(kwargs)
                        )
                        

                    if render_directional_marginal_laps:
                        # Laps Direction (LR/RL) Marginal:
                        _main_context = {'decoded_epochs': 'Laps', 'Marginal': 'Direction', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        graphics_output_dict['directional_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Marginal_LAPS',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context),
                            **deepcopy(kwargs)
                        )

                    if render_directional_marginal_ripples:
                        # Ripple Direction (LR/RL) Marginal:
                        _main_context = {'decoded_epochs': 'Ripple', 'Marginal': 'Direction', 't_bin': ripple_decoding_time_bin_size}
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
                        graphics_output_dict['directional_ripples_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_replays, directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Marginal_Ripples',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_direction(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_ripple_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context),
                            **deepcopy(kwargs)
                        )

                    if render_track_identity_marginal_laps:
                        # Laps Track-identity (Long/Short) Marginal:
                        _main_context = {'decoded_epochs': 'Laps', 'Marginal': 'TrackID', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        graphics_output_dict['track_identity_marginal_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='TrackIdentity_Marginal_LAPS',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context),
                            **deepcopy(kwargs)
                        )


                    if render_track_identity_marginal_ripples:
                        # Ripple Track-identity (Long/Short) Marginal:
                        _main_context = {'decoded_epochs': 'Ripple', 'Marginal': 'TrackID', 't_bin': ripple_decoding_time_bin_size}
                        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        graphics_output_dict['track_identity_marginal_ripples_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_replays, directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='TrackIdentity_Marginal_Ripples',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalMergedDecodersResult.build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_ripple_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context),
                            **deepcopy(kwargs)
                        )

            graphics_output_dict['collector'] = collector

            return graphics_output_dict



    @function_attributes(short_name='directional_decoded_epochs_marginals', tags=['directional_merged_decoder_decoded_epochs','directional','marginal'], input_requires=[], output_provides=[], uses=['plot_rank_order_epoch_inst_fr_result_tuples'], used_by=[], creation_date='2023-12-15 21:46', related_items=[], is_global=True)
    def _display_directional_merged_pf_decoded_epochs_marginals(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
        """ Plots two scatter plot figures of the decoded marginals from the merged decoder

        """
        defer_render: bool = kwargs.pop('defer_render', False)
        should_show: bool = (not defer_render)
        
        # #TODO 2024-01-03 05:24: - [ ] Do something to switch the matplotlib backend to 'AGG' if defer_render == True. Currently only adjusts the pyqtgraph-based figures (`plot_rank_order_epoch_inst_fr_result_tuples`)

        active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        directional_merged_decoders_result = kwargs.pop('directional_merged_decoders_result', None)
        if directional_merged_decoders_result is not None:
            print("WARN: User provided a custom `directional_merged_decoders_result` as a kwarg. This will be used instead of the computed result global_computation_results.computed_data['DirectionalMergedDecoders'].")
            
        else:
            directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

        def _perform_write_to_file_callback(final_context, fig):
            if save_figure:
                return owning_pipeline_reference.output_figure(final_context, fig)
            else:
                pass # do nothing, don't save


        # Quantile Diff Figures: _____________________________________________________________________________________________ #
        t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
        all_epoch_bins_marginals_collector = plot_all_epoch_bins_marginal_predictions(directional_merged_decoders_result, t_start=t_start, t_split=t_delta, t_end=t_end, active_context=active_context, perform_write_to_file_callback=_perform_write_to_file_callback)

        return all_epoch_bins_marginals_collector



## Build Dock Widgets:
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuCommand


@define(slots=False)
class AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand(BaseMenuCommand):
    """ 2024-01-17 
    Adds four rows to the SpikeRaster2D showing the continuously decoded posterior for each of the four 1D decoders


    Usage:
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand

    """
    _spike_raster_window = field()
    _active_pipeline = field(alias='curr_active_pipeline')
    _active_config_name = field(default=None)
    _context = field(default=None, alias="active_context")
    _display_output = field(default=Factory(dict))

    @classmethod
    def _perform_add_new_decoded_row(cls, curr_active_pipeline, active_2d_plot, a_dock_config, a_decoder_name: str, a_decoder, a_decoded_result=None):
        """ adds a single decoded row to the matplotlib dynamic output
        
        # a_decoder_name: str = "long_LR"

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        
        ## ✅ Add a new row for each of the four 1D directional decoders:
        identifier_name: str = f'{a_decoder_name}_ContinuousDecode'
        print(f'identifier_name: {identifier_name}')
        widget, matplotlib_fig, matplotlib_fig_axes = active_2d_plot.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(65, 200), display_config=a_dock_config)
        an_ax = matplotlib_fig_axes[0]

        # _active_config_name = None
        variable_name: str = a_decoder_name
        active_decoder = deepcopy(a_decoder)
        
        if a_decoded_result is not None:
            active_result = deepcopy(a_decoded_result) # already decoded
            assert (active_result.num_filter_epochs == 1), f"currently only supports decoded results (DecodedFilterEpochsResult) computed with a single epoch for all time bins, but active_result.num_filter_epochs: {active_result.num_filter_epochs}"
            active_marginals = active_result.marginal_x_list[0]
        else:
            # no previously decoded result, fallback to the decoder's internal properties        
            active_marginals = active_decoder.marginal.x
            

        active_bins = active_decoder.xbin

        # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        active_most_likely_positions = None
        active_posterior = active_marginals.p_x_given_n

        # most_likely_positions_mode: 'standard'|'corrected'
        # fig, curr_ax = curr_active_pipeline.display('_display_plot_marginal_1D_most_likely_position_comparisons', _active_config_name, variable_name='x', most_likely_positions_mode='corrected', ax=an_ax) # ax=active_2d_plot.ui.matplotlib_view_widget.ax
        ## Actual plotting portion:
        fig, curr_ax = plot_1D_most_likely_position_comparsions(None, time_window_centers=active_decoder.time_window_centers, xbin=active_bins,
                                                                posterior=active_posterior,
                                                                active_most_likely_positions_1D=active_most_likely_positions,
                                                                ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False)

        widget.draw() # alternative to accessing through full path?
        active_2d_plot.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
        return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes

    @classmethod
    def add_directional_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, debug_print=False):
        """ adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses
        
        showCloseButton = True
        dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=showCloseButton), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=showCloseButton),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=showCloseButton), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=showCloseButton))))


        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        directional_decoders_decode_result: DirectionalDecodersDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        # continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        time_bin_size: float = directional_decoders_decode_result.last_decoding_time_bin_size
        print(f'time_bin_size: {time_bin_size}')
        continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.continuously_decoded_dict
        all_directional_continuously_decoded_dict = continuously_decoded_dict or {}

        # Need all_directional_pf1D_Decoder_dict
        output_dict = {}

        for a_decoder_name, a_decoder in all_directional_pf1D_Decoder_dict.items():
            a_dock_config = dock_configs[a_decoder_name]
            a_decoded_result = all_directional_continuously_decoded_dict.get(a_decoder_name, None) # already decoded
            _out_tuple = cls._perform_add_new_decoded_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=a_dock_config, a_decoder_name=a_decoder_name, a_decoder=a_decoder, a_decoded_result=a_decoded_result)
            # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = _out_tuple
            output_dict[a_decoder_name] = _out_tuple

        return output_dict


    def validate_can_display(self) -> bool:
        """ returns True if the item is enabled, otherwise returns false """
        try:
            curr_active_pipeline = self._active_pipeline
            # assert curr_active_pipeline is not None
            if curr_active_pipeline is None:
                raise ValueError("Current active pipeline is None!")
            active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
            # assert active_2d_plot is not None
            if active_2d_plot is None:
                raise ValueError("active_2d_plot is None!")

            return DirectionalDecodersDecodedResult.validate_has_directional_decoded_continuous_epochs(curr_active_pipeline=curr_active_pipeline)
            
        except Exception as e:
            print(f'Exception {e} occured in validate_can_display(), returning False')
            return False

    def execute(self, *args, **kwargs) -> None:
        ## To begin, the destination plot must have a matplotlib widget plot to render to:
        # print(f'AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand.execute(...)')
        active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
        # If no plot to render on, do this:
        output_dict = self.add_directional_decoder_decoded_epochs(self._active_pipeline, active_2d_plot) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        # Update display output dict:
        for a_decoder_name, an_output_tuple in output_dict.items():
            identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = an_output_tuple
            self._display_output[identifier_name] = an_output_tuple

        print(f'\t AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand.execute() is done.')
        
