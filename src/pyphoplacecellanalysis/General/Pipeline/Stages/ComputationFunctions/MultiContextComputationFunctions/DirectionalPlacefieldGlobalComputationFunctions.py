import numpy as np
import pandas as pd
from functools import wraps
from copy import deepcopy

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from neuropy.core.laps import Laps # used in `DirectionalLapsHelpers`
from neuropy.analyses.laps import build_lap_computation_epochs # used in `DirectionalLapsHelpers.split_to_directional_laps`
from neuropy.utils.result_context import IdentifyingContext

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import compute_long_short_constrained_decoders


class DirectionalLapsHelpers:
    """ 2023-10-24 - Directional Placefields Computations

    use_direction_dependent_laps

    from neuropy.core.laps import Laps
    from neuropy.analyses.laps import build_lap_computation_epochs
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers


    curr_active_pipeline, directional_lap_specific_configs = DirectionalLapsHelpers.split_to_directional_laps(curr_active_pipeline=curr_active_pipeline, add_created_configs_to_pipeline=True)

    """


    # lap_direction_suffix_list = ['_odd', '_even', '_any'] # ['maze1_odd', 'maze1_even', 'maze1_any', 'maze2_odd', 'maze2_even', 'maze2_any', 'maze_odd', 'maze_even', 'maze_any']
    # lap_direction_suffix_list = ['_odd', '_even', ''] # no '_any' prefix, instead reuses the existing names
    split_directional_laps_name_parts = ['odd_laps', 'even_laps'] # , 'any_laps'

    # ['maze_even_laps', 'maze_odd_laps']

    @classmethod
    def format_directional_laps_context(cls, a_context: IdentifyingContext, a_maze_name: str = 'maze1', a_directional_epoch_name: str, a_lap_dir_name: str) -> IdentifyingContext:
        """ Builds the correct context for a lap-direction-specific epoch from the base epoch

        originally:
            'maze2_even_laps': IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'maze2')>


        a_lap_dir_name: str - like "odd_laps" or "even_laps"


        Usage:
            a_context = deepcopy(curr_active_pipeline.filtered_contexts[a_name])
            a_context = DirectionalLapsHelpers.format_directional_laps_context(a_context, a_split_directional_laps_config_name, a_lap_dir)
            curr_active_pipeline.filtered_contexts[a_split_directional_laps_config_name] = a_context


        """
        a_context = a_context.overwriting_context(filter_name=a_directional_epoch_name, maze_name=a_maze_name, lap_dir=a_lap_dir_name)
        return a_context

    @classmethod
    def split_specific_epoch_to_directional_laps(cls, a_name: str, a_sess, a_result, curr_active_pipeline, add_created_configs_to_pipeline:bool=True, use_direction_dependent_laps=True, debug_print=False):
        """ 

            a_name, a_sess, a_result = global_epoch_name, global_session, global_results
            curr_epoch_directional_lap_specific_configs, curr_epoch_split_directional_laps_dict, curr_epoch_split_directional_laps_config_names = cls.split_specific_epoch_to_directional_laps(a_name, a_sess, a_result, curr_active_pipeline, add_created_configs_to_pipeline=add_created_configs_to_pipeline)

        """
        directional_lap_specific_configs = {}

        split_directional_laps_config_names = [f'{a_name}_{a_lap_dir_description}' for a_lap_dir_description in cls.split_directional_laps_name_parts] # ['maze_odd_laps', 'maze_even_laps']
        if debug_print:
            print(f'\tsplit_directional_laps_config_names: {split_directional_laps_config_names}')

        # 'build_lap_computation_epochs(...)' based mode:
        desired_computation_epochs = build_lap_computation_epochs(a_sess, use_direction_dependent_laps=use_direction_dependent_laps)
        even_lap_specific_epochs, odd_lap_specific_epochs, any_lap_specific_epochs = desired_computation_epochs

        split_directional_laps_dict = dict(zip(split_directional_laps_config_names, (even_lap_specific_epochs, odd_lap_specific_epochs)))

        # # manual mode:
        # lap_specific_epochs = a_sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0) # set this to the laps object
        # any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(a_sess.laps.lap_id))])
        # even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(a_sess.laps.lap_id), 2)])
        # odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(a_sess.laps.lap_id), 2)])

        # split_directional_laps_dict = {'even_laps': even_lap_specific_epochs, 'odd_laps': odd_lap_specific_epochs, 'any_laps': any_lap_specific_epochs}

        if debug_print:
            print(f'any_lap_specific_epochs: {any_lap_specific_epochs}\n\teven_lap_specific_epochs: {even_lap_specific_epochs}\n\todd_lap_specific_epochs: {odd_lap_specific_epochs}\n') # lap_specific_epochs: {lap_specific_epochs}\n\t

        for i, (a_split_directional_laps_config_name, lap_dir_epochs) in enumerate(split_directional_laps_dict.items()):
            a_lap_dir: str = cls.split_directional_laps_name_parts[i]
            if debug_print:
                print(f'\ta_split_directional_laps_config_name: {a_split_directional_laps_config_name}, a_lap_dir: {a_lap_dir}')
            active_config_copy = deepcopy(curr_active_pipeline.active_configs[a_name])
            # active_config_copy.computation_config.pf_params.computation_epochs = active_config_copy.computation_config.pf_params.computation_epochs.label_slice(odd_lap_specific_epochs.labels)
            ## Just overwrite directly:
            active_config_copy.computation_config.pf_params.computation_epochs = lap_dir_epochs
            directional_lap_specific_configs[a_split_directional_laps_config_name] = active_config_copy
            if add_created_configs_to_pipeline:
                curr_active_pipeline.active_configs[a_split_directional_laps_config_name] = active_config_copy
                # When a new config is added, new results and stuff should be added too.
                curr_active_pipeline.filtered_sessions[a_split_directional_laps_config_name] = curr_active_pipeline.filtered_sessions[a_name]
                curr_active_pipeline.filtered_epochs[a_split_directional_laps_config_name] = curr_active_pipeline.filtered_epochs[a_name]

                a_context = deepcopy(curr_active_pipeline.filtered_contexts[a_name])
                a_context = cls.format_directional_laps_context(a_context, a_maze_name=a_name, a_directional_epoch_name=a_split_directional_laps_config_name, a_lap_dir_name=a_lap_dir)
                curr_active_pipeline.filtered_contexts[a_split_directional_laps_config_name] = a_context

                curr_active_pipeline.computation_results[a_split_directional_laps_config_name] = None # empty

            # end loop over split_directional_lap types:
        return directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names

    @classmethod
    def split_to_directional_laps(cls, curr_active_pipeline, include_includelist=None, add_created_configs_to_pipeline:bool=True, debug_print=False):
        """ 2023-10-23 - Duplicates the global_epoch and all of its derived properties (filtered*, computed*, etc) but restricts its computation_config.computation_epochs to be either the odd or even laps
                (restricting the motion to one of the two directions) allowing us to observe the directional placefields 

        if add_created_configs_to_pipeline is False, just returns the built configs and doesn't add them to the pipeline.

        """
        if include_includelist is None:
            use_global_epoch_only_mode: bool = True # 2023-10-24 - 4:19pm - Duplicates only the `global_epoch_name` results for the directional laps and then filters from there
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            # include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
            include_includelist = [global_epoch_name] # ['maze'] # only for maze
        
        computed_base_epoch_names = []
        lap_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
        assert lap_estimation_parameters is not None
        use_direction_dependent_laps: bool = lap_estimation_parameters.get('use_direction_dependent_laps', True)
        if debug_print:
            print(f'split_to_directional_laps(...): use_direction_dependent_laps: {use_direction_dependent_laps}')
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        ## After all top-level computations are done, compute the subsets for direction laps
        directional_lap_specific_configs = {}
        split_directional_laps_dict = {}
        split_directional_laps_config_names = []

        if use_direction_dependent_laps:
            if debug_print:
                print(f'split_to_directional_laps(...) processing for directional laps...')
            # for a_name, a_sess, a_result in zip((long_epoch_name, short_epoch_name, global_epoch_name), (long_session, short_session, global_session), (long_results, short_results, global_results)):

            for an_epoch_name in include_includelist:
                # for `use_global_epoch_only_mode == True` mode:
                # an_epoch_name, a_sess, a_result = global_epoch_name, global_session, global_results
                a_sess, a_result = curr_active_pipeline.filtered_sessions[an_epoch_name], curr_active_pipeline.computation_results[an_epoch_name]['computed_data']

                curr_epoch_directional_lap_specific_configs, curr_epoch_split_directional_laps_dict, curr_epoch_split_directional_laps_config_names = cls.split_specific_epoch_to_directional_laps(an_epoch_name, a_sess, a_result, curr_active_pipeline, add_created_configs_to_pipeline=add_created_configs_to_pipeline)

                directional_lap_specific_configs |= curr_epoch_directional_lap_specific_configs
                split_directional_laps_dict |= curr_epoch_split_directional_laps_dict
                split_directional_laps_config_names.extend(curr_epoch_split_directional_laps_config_names)

                # end loop over filter epochs:
                if debug_print:
                    print(f'split_directional_laps_config_names: {split_directional_laps_config_names}')

                ## Actually do the filtering now. We have 
                updated_active_session_pseudo_filter_configs = {k:curr_active_pipeline.active_configs[k].filter_config['filter_function'] for k in split_directional_laps_config_names} # split_directional_laps_config_names: ['maze_odd_laps', 'maze_even_laps']
                curr_active_pipeline.filter_sessions(updated_active_session_pseudo_filter_configs, changed_filters_ignore_list=['maze1','maze2','maze'], debug_print=False)

                ## Perform the computations which builds the computation results:
                _out = curr_active_pipeline.perform_computations(computation_functions_name_includelist=['pf_computation', 'firing_rate_trends', 'position_decoding'], # , 'pfdt_computation'
                    enabled_filter_names=split_directional_laps_config_names,
                    # computation_kwargs_list=[dict(ndim=1)],
                    fail_on_exception=True, debug_print=True)

                computed_base_epoch_names.append(an_epoch_name)
                
            # end for an_epoch_name
        # end if use_direction_dependent_laps
        return curr_active_pipeline, directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names

    @classmethod
    def validate_has_directional_laps(cls, curr_active_pipeline, computation_filter_name='maze'):
        return (
                    curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'],
                    curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']['computed_base_epoch_names'],
                    (computation_filter_name in curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']['computed_base_epoch_names']) 
                )



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
                ['DirectionalLaps']['split_directional_laps_names']
                ['DirectionalLaps']['computed_base_epoch_names']

        
        """
        if include_includelist is None:
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            # include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
            include_includelist = [global_epoch_name] # ['maze'] # only for maze
        
        ## Adds ['*_even_laps', '*_odd_laps'] pseduofilters
        owning_pipeline_reference, directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = DirectionalLapsHelpers.split_to_directional_laps(owning_pipeline_reference, include_includelist=include_includelist, add_created_configs_to_pipeline=True)
        # curr_active_pipeline, directional_lap_specific_configs = constrain_to_laps(curr_active_pipeline)
        # list(directional_lap_specific_configs.keys())

        global_computation_results.computed_data['DirectionalLaps'] = DynamicParameters.init_from_dict({
            'directional_lap_specific_configs': directional_lap_specific_configs,
            'split_directional_laps_dict': split_directional_laps_dict,
            'split_directional_laps_names': split_directional_laps_config_names,
            'computed_base_epoch_names': computed_base_epoch_names,
        })

        owning_pipeline_reference.prepare_for_display() # TODO: pass a display config

        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        """
        return global_computation_results

