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

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.dynamic_container import DynamicContainer # used to build config
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.core.epoch import NamedTimerange, Epoch

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used for `complete_directional_pfs_computations`
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult

import scipy.stats
from scipy import ndimage
from nptyping import NDArray

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

# Define the namedtuple
DirectionalDecodersTuple = namedtuple('DirectionalDecodersTuple', ['long_LR', 'long_RL', 'short_LR', 'short_RL'])

@define(slots=False, repr=False, eq=False)
class TrackTemplates:
    """ Holds the four directional templates for direction placefield analysis.
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
    
    History:
        Based off of `ShuffleHelper` on 2023-10-27
        TODO: eliminate functional overlap with `ShuffleHelper`
        TODO: should be moved into `DirectionalPlacefieldGlobalComputation` instead of RankOrder
        
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
        _obj = TrackTemplates.init_from_paired_decoders(LR_decoder_pair=(long_LR_decoder, short_LR_decoder), RL_decoder_pair=(long_RL_decoder, short_RL_decoder))
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
    

    def get_decoders_dict(self) -> Dict[str, BasePositionDecoder]:
        return {'long_LR': self.long_LR_decoder,
            'long_RL': self.long_RL_decoder,
            'short_LR': self.short_LR_decoder,
            'short_RL': self.short_RL_decoder,
        }

    @classmethod
    def init_from_paired_decoders(cls, LR_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder], RL_decoder_pair: Tuple[BasePositionDecoder, BasePositionDecoder]) -> "TrackTemplates":
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
                    decoder_LR_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense') for a_decoder in (long_LR_decoder, short_LR_decoder)],
                    decoder_RL_pf_peak_ranks_list=[scipy.stats.rankdata(a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, method='dense') for a_decoder in (long_RL_decoder, short_RL_decoder)] )

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
    directional_lap_specific_configs: Dict = field(default=Factory(dict))
    split_directional_laps_dict: Dict = field(default=Factory(dict))
    split_directional_laps_contexts_dict: Dict = field(default=Factory(dict))
    split_directional_laps_config_names: List[str] = field(default=Factory(list))
    computed_base_epoch_names: List[str] = field(default=Factory(list))

    long_LR_one_step_decoder_1D: BasePositionDecoder = field(default=None)
    long_RL_one_step_decoder_1D: BasePositionDecoder = field(default=None)
    short_LR_one_step_decoder_1D: BasePositionDecoder = field(default=None)
    short_RL_one_step_decoder_1D: BasePositionDecoder = field(default=None)

    long_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None, alias='long_odd_shared_aclus_only_one_step_decoder_1D')
    long_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None, alias='long_even_shared_aclus_only_one_step_decoder_1D')
    short_LR_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None, alias='short_odd_shared_aclus_only_one_step_decoder_1D')
    short_RL_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None, alias='short_even_shared_aclus_only_one_step_decoder_1D')

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



            
    # 

    # shared_aclus: np.ndarray
    # long_short_pf_neurons_diff: SetPartition
    # n_neurons: int

    

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
        print(f'build_global_directional_result_from_natural_epochs(...): was_modified: {was_modified}')
        
        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
        long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]

        # ## Version 2023-10-30 - All four templates with same shared_aclus version:
        # # Prune to the shared aclus in both epochs (short/long):
        active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        # Find only the common aclus amongst all four templates:
        shared_aclus = np.array(list(set.intersection(*map(set,active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        n_neurons = len(shared_aclus)
        print(f'n_neurons: {n_neurons}, shared_aclus: {shared_aclus}')
        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        long_LR_shared_aclus_only_one_step_decoder_1D, long_RL_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

        # ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
        # # Kamran says LR and RL sets should be shared
        # ## Odd Laps:
        # LR_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        # LR_shared_aclus = np.array(list(set.intersection(*map(set,LR_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        # LR_n_neurons = len(LR_shared_aclus)
        # if progress_print:
        #     print(f'LR_n_neurons: {LR_n_neurons}, LR_shared_aclus: {LR_shared_aclus}')

        # ## Even Laps:
        # RL_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]
        # RL_shared_aclus = np.array(list(set.intersection(*map(set,RL_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        # RL_n_neurons = len(RL_shared_aclus)
        # if progress_print:
        #     print(f'RL_n_neurons: {RL_n_neurons}, RL_shared_aclus: {RL_shared_aclus}')

        # # Direction Separate shared_aclus decoders: Odd set is limited to LR_shared_aclus and RL set is limited to RL_shared_aclus:
        # long_LR_shared_aclus_only_one_step_decoder_1D, short_LR_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(LR_shared_aclus) for a_decoder in (long_LR_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D)]
        # long_RL_shared_aclus_only_one_step_decoder_1D, short_RL_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(RL_shared_aclus) for a_decoder in (long_RL_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D)]

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



# ==================================================================================================================== #
# Display Functions/Plotting                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyqtgraph as pg
import pyqtgraph.exporters
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_incremental_sort_neurons # _display_directional_template_debugger
from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays


class DirectionalPlacefieldGlobalDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='directional_laps_overview', tags=['directional','laps','overview'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-09 12:03', related_items=[], is_global=True)
    def _display_directional_laps_overview(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ Renders a window with the position/laps displayed in the middle and the four templates displayed to the left and right of them.

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
            # track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=None) # shared-only
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=None) # non-shared-only
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

            # uses `global_session`
            epochs_editor = EpochsEditor.init_from_session(global_session, include_velocity=False, include_accel=False)
            root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(epochs_editor.plots.win, None, title='Pho Directional Laps Templates')
            
            def _get_decoder_sorted_pfs(a_decoder):
                """ used only when viewing with individual sorts (instead of all four decoder's pfs aligned to the first decoder's sort) """
                ratemap = a_decoder.pf.ratemap
                CoM_sort_indicies = np.argsort(ratemap.peak_tuning_curve_center_of_masses) # get the indicies to sort the placefields by their center-of-mass (CoM) location # CoM_sort_indicies.shape # (n_neurons,)
                return ratemap.pdf_normalized_tuning_curves[CoM_sort_indicies, :]


            decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

            # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`               
            sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=included_any_context_neuron_ids)

            ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
            _out_pf1D_heatmaps = {}
            for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
                _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=f'{a_decoder_name}_pf1Ds [sort: long_RL]', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
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


            even_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_even_dock_colors)
            odd_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_odd_dock_colors)

            _out_dock_widgets = {}
            dock_configs = (even_dock_config, odd_dock_config, even_dock_config, odd_dock_config)
            dock_add_locations = (['left'], ['left'], ['right'], ['right'])

            for i, (a_decoder_name, a_heatmap) in enumerate(_out_pf1D_heatmaps.items()):
                _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_heatmap[0], dockSize=(300,200), dockAddLocationOpts=dock_add_locations[i], display_config=dock_configs[i])

            
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
    def _display_directional_template_debugger(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ Renders a window with the four templates displayed to the left and right of center, and the ability to filter the actively included aclus via `included_any_context_neuron_ids`

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


            enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)

            figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')  
            _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sorted_pf_tuning_curves=None)
            _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)
            
            
            # Recover from the saved global result:
            directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            # track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=None) # shared-only
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=None) # non-shared-only
            
            # build the window with the dock widget in it:
            root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho Directional Template Debugger: {figure_name}', defer_show=False)
            _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, dock_widgets=None, on_update_callback=None)
            root_dockAreaWindow.resize(800, 400)

            def _get_decoder_sorted_pfs(a_decoder):
                """ used only when viewing with individual sorts (instead of all four decoder's pfs aligned to the first decoder's sort) """
                ratemap = a_decoder.pf.ratemap
                CoM_sort_indicies = np.argsort(ratemap.peak_tuning_curve_center_of_masses) # get the indicies to sort the placefields by their center-of-mass (CoM) location # CoM_sort_indicies.shape # (n_neurons,)
                return ratemap.pdf_normalized_tuning_curves[CoM_sort_indicies, :]
            
            # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting` 
            decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
            sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=included_any_context_neuron_ids)
            # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
            _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
            _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
            _out_data.sorted_pf_tuning_curves = sorted_pf_tuning_curves

            # saveData('output/2023-11-28_debug_paired_incremental_sort_neurons_data.pkl', (decoders_dict, included_any_context_neuron_ids, sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves))

            # testNone_included_any_context_neuron_ids = None
            # testNone_sorted_neuron_IDs_lists, testNone_sort_helper_neuron_id_to_neuron_colors_dicts, testNone_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=testNone_included_any_context_neuron_ids)
            # all_neuron_ids = np.sort(union_of_arrays(*testNone_sorted_neuron_IDs_lists))

            # test0_included_any_context_neuron_ids = np.array([25])
            # test_sorted_neuron_IDs_lists, test_sort_helper_neuron_id_to_neuron_colors_dicts, test_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=test0_included_any_context_neuron_ids)
            
            # test1_included_any_context_neuron_ids = np.array([9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
            # test1_sorted_neuron_IDs_lists, test1_sort_helper_neuron_id_to_neuron_colors_dicts, test1_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=test1_included_any_context_neuron_ids)
            # assert np.all([testNone_sorted_pf_tuning_curves[i] == test1_sorted_pf_tuning_curves[i] for i in np.arange(len(test1_sorted_pf_tuning_curves))])
            # assert np.all([testNone_sorted_neuron_IDs_lists[i] == test1_sorted_neuron_IDs_lists[i] for i in np.arange(len(test1_sorted_pf_tuning_curves))])

            # # test2 omits only one element: aclu == 25
            # test2_included_any_context_neuron_ids = np.array([9, 10, 11,  15,  16,  18,  24,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
            # test2_sorted_neuron_IDs_lists, test2_sort_helper_neuron_id_to_neuron_colors_dicts, test2_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=test2_included_any_context_neuron_ids)


            ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
            _out_plots.pf1D_heatmaps = {}
            for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
                _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=f'{a_decoder_name}_pf1Ds [sort: long_RL]', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
                # _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(_get_decoder_sorted_pfs(a_decoder), title=f'{a_decoder_name}_pf1Ds', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Individual Sort

                # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
                curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img
                
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
                if enable_cell_colored_heatmap_rows:
                    curr_img.updateImage(out_colors_heatmap_image_matrix) # use the color image only if `enable_cell_colored_heatmap_rows==True`
                _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix


            ## These are one-time configs:
            even_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_even_dock_colors)
            odd_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_odd_dock_colors)

            _out_ui.dock_widgets = {}
            dock_configs = (even_dock_config, odd_dock_config, even_dock_config, odd_dock_config)
            dock_add_locations = (['left'], ['left'], ['right'], ['right'])

            for i, (a_decoder_name, a_heatmap) in enumerate(_out_plots.pf1D_heatmaps.items()):
                _out_ui.dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_heatmap[0], dockSize=(300,200), dockAddLocationOpts=dock_add_locations[i], display_config=dock_configs[i])

            
            # Outputs: root_dockAreaWindow, app, epochs_editor, _out_pf1D_heatmaps, _out_dock_widgets
            graphics_output_dict = {'win': root_dockAreaWindow, 'app': app,  'ui': _out_ui, 'plots': _out_plots, 'data': _out_data}

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

            def on_update(included_any_context_neuron_ids):
                """ call to update when `included_any_context_neuron_ids` changes.

                 captures: `decoders_dict`, `_out_plots`, 'enable_cell_colored_heatmap_rows'

                """
                decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
                sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=included_any_context_neuron_ids)
                # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`

                ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
                _out_plots.pf1D_heatmaps = {}
                for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
                    _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(sorted_pf_tuning_curves[i], title=f'{a_decoder_name}_pf1Ds [sort: long_RL]', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
                    # _out_pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(_get_decoder_sorted_pfs(a_decoder), title=f'{a_decoder_name}_pf1Ds', show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Individual Sort

                    # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
                    curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img
                    
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
                    if enable_cell_colored_heatmap_rows:
                        curr_img.updateImage(out_colors_heatmap_image_matrix) # use the color image only if `enable_cell_colored_heatmap_rows==True`
                    _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix

            graphics_output_dict['ui'].on_update_callback = on_update

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
