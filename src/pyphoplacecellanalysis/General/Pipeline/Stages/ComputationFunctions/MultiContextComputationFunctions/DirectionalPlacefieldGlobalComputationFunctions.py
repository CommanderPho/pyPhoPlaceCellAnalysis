import numpy as np
import pandas as pd
from attrs import define, field, Factory, asdict, astuple
from functools import wraps
from copy import deepcopy
from typing import List, Dict, Optional

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.analyses.laps import build_lap_computation_epochs # used in `DirectionalLapsHelpers.split_to_directional_laps`
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.dynamic_container import DynamicContainer # used to build config
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.core.epoch import NamedTimerange, Epoch

from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import ComputedPipelineStage # for building computation result
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import compute_long_short_constrained_decoders


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder # used for `complete_directional_pfs_computations`
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult



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
    directional_laps_result.long_odd_shared_aclus_only_one_step_decoder_1D = long_odd_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.long_even_shared_aclus_only_one_step_decoder_1D = long_even_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.short_odd_shared_aclus_only_one_step_decoder_1D = short_odd_shared_aclus_only_one_step_decoder_1D
    directional_laps_result.short_even_shared_aclus_only_one_step_decoder_1D = short_even_shared_aclus_only_one_step_decoder_1D

        
    """
    directional_lap_specific_configs: Dict = field(default=Factory(dict))
    split_directional_laps_dict: Dict = field(default=Factory(dict))
    split_directional_laps_contexts_dict: Dict = field(default=Factory(dict))
    split_directional_laps_config_names: List[str] = field(default=Factory(list))
    computed_base_epoch_names: List[str] = field(default=Factory(list))
    long_odd_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None)
    long_even_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None)
    short_odd_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None)
    short_even_shared_aclus_only_one_step_decoder_1D: BasePositionDecoder = field(default=None)

    # shared_aclus: np.ndarray
    # long_short_pf_neurons_diff: SetPartition
    # n_neurons: int

    


class DirectionalLapsHelpers:
    """ 2023-10-24 - Directional Placefields Computations

    use_direction_dependent_laps

    from neuropy.core.laps import Laps
    from neuropy.analyses.laps import build_lap_computation_epochs
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers


    curr_active_pipeline, directional_lap_specific_configs = DirectionalLapsHelpers.split_to_directional_laps(curr_active_pipeline=curr_active_pipeline, add_created_configs_to_pipeline=True)

    
    
    
    Computing:
    
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers
        
        # Run directional laps and set the global result:
        curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] = DirectionalLapsHelpers.complete_directional_pfs_computations(curr_active_pipeline)

    
    """


    # lap_direction_suffix_list = ['_odd', '_even', '_any'] # ['maze1_odd', 'maze1_even', 'maze1_any', 'maze2_odd', 'maze2_even', 'maze2_any', 'maze_odd', 'maze_even', 'maze_any']
    # lap_direction_suffix_list = ['_odd', '_even', ''] # no '_any' prefix, instead reuses the existing names
    split_directional_laps_name_parts = ['odd_laps', 'even_laps'] # , 'any_laps'

    split_all_laps_name_parts = ['odd_laps', 'even_laps', 'any']
    # ['maze_even_laps', 'maze_odd_laps']

    @classmethod
    def validate_has_directional_laps(cls, curr_active_pipeline, computation_filter_name='maze'):
        # Unpacking:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        # assert (computation_filter_name in computed_base_epoch_names), f'computation_filter_name: {computation_filter_name} is missing from computed_base_epoch_names: {computed_base_epoch_names} '
        return (computation_filter_name in split_directional_laps_config_names)
        # return (computation_filter_name in computed_base_epoch_names)


    @classmethod
    def build_global_directional_result_from_natural_epochs(cls, curr_active_pipeline):
        """ 2023-10-31 - 4pm 

        Does not update `curr_active_pipeline` or mess with its filters/configs/etc.

        """
        # Unwrap the naturally produced directional placefields:
        long_odd_name, short_odd_name, global_odd_name, long_even_name, short_even_name, global_even_name, long_any_name, short_any_name, global_any_name = ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any']

        # Most popular
        # long_odd_name, long_even_name, short_odd_name, short_even_name, global_any_name
        long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name = long_odd_name, long_even_name, short_odd_name, short_even_name

        # Unpacking for `(long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)`
        (long_odd_laps_context, long_even_laps_context, short_odd_laps_context, short_even_laps_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        long_odd_laps_obj, long_even_laps_obj, short_odd_laps_obj, short_even_laps_obj, global_any_laps_obj = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'].pf_params.computation_epochs for an_epoch_name in (long_odd_name, long_even_name, short_odd_name, short_even_name, global_any_name)] # note has global also
        (long_odd_laps_session, long_even_laps_session, short_odd_laps_session, short_even_laps_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_results, long_even_laps_results, short_odd_laps_results, short_even_laps_results) = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_computation_config, long_even_laps_computation_config, short_odd_laps_computation_config, short_even_laps_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_pf1D, long_even_laps_pf1D, short_odd_laps_pf1D, short_even_laps_pf1D) = (long_odd_laps_results.pf1D, long_even_laps_results.pf1D, short_odd_laps_results.pf1D, short_even_laps_results.pf1D)
        (long_odd_laps_pf2D, long_even_laps_pf2D, short_odd_laps_pf2D, short_even_laps_pf2D) = (long_odd_laps_results.pf2D, long_even_laps_results.pf2D, short_odd_laps_results.pf2D, short_even_laps_results.pf2D)

        # Validate:
        assert not (curr_active_pipeline.computation_results[long_odd_laps_name]['computation_config']['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_even_laps_name]['computation_config']['pf_params'].computation_epochs)
        assert not (curr_active_pipeline.computation_results[short_odd_laps_name]['computation_config']['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_even_laps_name]['computation_config']['pf_params'].computation_epochs)


        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D  = DirectionalLapsHelpers.build_directional_constrained_decoders(curr_active_pipeline)

        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
        long_odd_laps_one_step_decoder_1D, long_even_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_odd_laps_results, long_even_laps_results, short_odd_laps_results, short_even_laps_results)]

        # ## Version 2023-10-30 - All four templates with same shared_aclus version:
        # # Prune to the shared aclus in both epochs (short/long):
        # active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_odd_laps_one_step_decoder_1D, long_even_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D)]
        # # Find only the common aclus amongst all four templates:
        # shared_aclus = np.array(list(set.intersection(*map(set,active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        # n_neurons = len(shared_aclus)
        # print(f'n_neurons: {n_neurons}, shared_aclus: {shared_aclus}')
        # # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        # long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(shared_aclus) for a_decoder in (long_odd_laps_one_step_decoder_1D, long_even_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D)]


        ## Version 2023-10-31 - 4pm - Two sets of templates for (Odd/Even) shared aclus:
        # Kamran says odd and even sets should be shared
        ## Odd Laps:
        odd_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_odd_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D)]
        odd_shared_aclus = np.array(list(set.intersection(*map(set,odd_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        odd_n_neurons = len(odd_shared_aclus)
        print(f'odd_n_neurons: {odd_n_neurons}, odd_shared_aclus: {odd_shared_aclus}')

        ## Even Laps:
        even_active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_even_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D)]
        even_shared_aclus = np.array(list(set.intersection(*map(set,even_active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        even_n_neurons = len(even_shared_aclus)
        print(f'even_n_neurons: {even_n_neurons}, even_shared_aclus: {even_shared_aclus}')

        # Direction Separate shared_aclus decoders: Odd set is limited to odd_shared_aclus and even set is limited to even_shared_aclus:
        long_odd_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(odd_shared_aclus) for a_decoder in (long_odd_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D)]
        long_even_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(even_shared_aclus) for a_decoder in (long_even_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D)]

        # ## Encode/Decode from global result:
        # # Unpacking:
        # directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        # # split_directional_laps_config_names

        ## Build a `ComputedResult` container object to hold the result:
        directional_laps_result = DirectionalLapsResult()
        directional_laps_result.directional_lap_specific_configs = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)} # directional_lap_specific_configs
        directional_laps_result.split_directional_laps_dict = {an_epoch_name:curr_active_pipeline.computation_results[an_epoch_name]['computation_config'].pf_params.computation_epochs for an_epoch_name in (long_odd_name, long_even_name, short_odd_name, short_even_name)}  # split_directional_laps_dict
        directional_laps_result.split_directional_laps_contexts_dict = {a_name:curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)} # split_directional_laps_contexts_dict
        directional_laps_result.split_directional_laps_config_names = (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name) # split_directional_laps_config_names
        # directional_laps_result.computed_base_epoch_names = computed_base_epoch_names

        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names
        directional_laps_result.long_odd_shared_aclus_only_one_step_decoder_1D = long_odd_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.long_even_shared_aclus_only_one_step_decoder_1D = long_even_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_odd_shared_aclus_only_one_step_decoder_1D = short_odd_shared_aclus_only_one_step_decoder_1D
        directional_laps_result.short_even_shared_aclus_only_one_step_decoder_1D = short_even_shared_aclus_only_one_step_decoder_1D
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

