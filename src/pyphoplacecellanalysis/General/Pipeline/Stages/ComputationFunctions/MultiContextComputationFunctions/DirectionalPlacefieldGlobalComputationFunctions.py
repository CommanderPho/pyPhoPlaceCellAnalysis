import numpy as np
import pandas as pd
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
    def format_directional_laps_context(cls, a_context: IdentifyingContext, a_maze_name: str, a_directional_epoch_name: str, a_lap_dir_name: str) -> IdentifyingContext:
        """ Builds the correct context for a lap-direction-specific epoch from the base epoch

        originally:
            'maze2_even_laps': IdentifyingContext<('kdiba', 'gor01', 'two', '2006-6-07_16-40-19', 'maze2')>

         a_maze_name: str = 'maze1'
        a_lap_dir_name: str - like "odd_laps" or "even_laps"


        Usage:
            a_context = deepcopy(curr_active_pipeline.filtered_contexts[a_name])
            a_context = DirectionalLapsHelpers.format_directional_laps_context(a_context, a_split_directional_laps_config_name, a_lap_dir)
            curr_active_pipeline.filtered_contexts[a_split_directional_laps_config_name] = a_context


        """
        a_context = a_context.overwriting_context(filter_name=a_directional_epoch_name, maze_name=a_maze_name, lap_dir=a_lap_dir_name)
        return a_context


    @classmethod
    def split_to_directional_laps(cls, curr_active_pipeline, include_includelist=None, add_created_configs_to_pipeline:bool=True, debug_print=False):
        """ 2023-10-23 - Duplicates the global_epoch and all of its derived properties (filtered*, computed*, etc) but restricts its computation_config.computation_epochs to be either the odd or even laps
                (restricting the motion to one of the two directions) allowing us to observe the directional placefields 

        if add_created_configs_to_pipeline is False, just returns the built configs and doesn't add them to the pipeline.

        """
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations # required for re-computation

        if include_includelist is None:
            use_global_epoch_only_mode: bool = True # 2023-10-24 - 4:19pm - Duplicates only the `global_epoch_name` results for the directional laps and then filters from there
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            # include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
            include_includelist = [global_epoch_name] # ['maze'] # only for maze
        

        ## Ensure no doublely-directional
        include_includelist = [an_epoch_name for an_epoch_name in include_includelist if np.all([not an_epoch_name.endswith(a_lap_dir_description) for a_lap_dir_description in cls.split_directional_laps_name_parts])]

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
        split_directional_laps_contexts_dict = {} # new
        split_directional_laps_dict = {}
        split_directional_laps_config_names = []

        directional_lap_specific_configs = {} # old

        if use_direction_dependent_laps:
            if debug_print:
                print(f'split_to_directional_laps(...) processing for directional laps...')
            # for a_name, a_sess, a_result in zip((long_epoch_name, short_epoch_name, global_epoch_name), (long_session, short_session, global_session), (long_results, short_results, global_results)):

            for an_epoch_name in include_includelist:
                curr_epoch_split_directional_laps_config_names: List[str] = [f'{an_epoch_name}_{a_lap_dir_description}' for a_lap_dir_description in cls.split_directional_laps_name_parts] # ['maze_odd_laps', 'maze_even_laps']
                if debug_print:
                    print(f'\tcurr_epoch_split_directional_laps_config_names: {curr_epoch_split_directional_laps_config_names}')


                # for `use_global_epoch_only_mode == True` mode:
                # an_epoch_name, a_sess, a_result = global_epoch_name, global_session, global_results
                # a_sess, a_result = curr_active_pipeline.filtered_sessions[an_epoch_name], curr_active_pipeline.computation_results[an_epoch_name]['computed_data']

                a_sess = curr_active_pipeline.filtered_sessions[an_epoch_name]

                # curr_epoch_directional_lap_specific_configs, curr_epoch_split_directional_laps_dict, curr_epoch_split_directional_laps_config_names = cls.split_specific_epoch_to_directional_laps(an_epoch_name, a_sess, a_result, curr_active_pipeline, add_created_configs_to_pipeline=add_created_configs_to_pipeline)
                # split_directional_laps_contexts_dict, split_directional_laps_dict = cls.split_specific_epoch_to_directional_laps(an_epoch_name, a_sess, a_result, curr_active_pipeline, add_created_configs_to_pipeline=add_created_configs_to_pipeline)

                

                # 'build_lap_computation_epochs(...)' based mode:
                desired_computation_epochs = build_lap_computation_epochs(a_sess, use_direction_dependent_laps=use_direction_dependent_laps)
                even_lap_specific_epochs, odd_lap_specific_epochs, any_lap_specific_epochs = desired_computation_epochs

                curr_epoch_split_directional_laps_contexts_dict: Dict[IdentifyingContext,str] = {cls.format_directional_laps_context(deepcopy(curr_active_pipeline.filtered_contexts[an_epoch_name]), a_maze_name=an_epoch_name, a_directional_epoch_name=a_split_directional_laps_config_name, a_lap_dir_name=a_lap_dir_description):a_split_directional_laps_config_name for a_lap_dir_description, a_split_directional_laps_config_name in zip(cls.split_directional_laps_name_parts, curr_epoch_split_directional_laps_config_names)}
                curr_epoch_split_directional_laps_dict: Dict[IdentifyingContext, Epoch] = dict(zip(list(curr_epoch_split_directional_laps_contexts_dict.keys()), (even_lap_specific_epochs, odd_lap_specific_epochs)))

                # Update the accumulators:
                split_directional_laps_contexts_dict |= curr_epoch_split_directional_laps_contexts_dict
                split_directional_laps_dict |= curr_epoch_split_directional_laps_dict
                split_directional_laps_config_names.extend(curr_epoch_split_directional_laps_config_names)

                # directional_lap_specific_configs |= curr_epoch_directional_lap_specific_configs # old

                # end loop over filter epochs:


                ## Actually do the filtering now. We have 
                updated_active_session_pseudo_filter_configs = {k:deepcopy(curr_active_pipeline.active_configs[an_epoch_name].filter_config['filter_function']) for k in split_directional_laps_config_names}
                # updated_active_session_pseudo_filter_configs = {k:curr_active_pipeline.active_configs[k].filter_config['filter_function'] for k in split_directional_laps_config_names} # split_directional_laps_config_names: ['maze_odd_laps', 'maze_even_laps']
                curr_active_pipeline.filter_sessions(updated_active_session_pseudo_filter_configs, changed_filters_ignore_list=['maze1','maze2','maze'], debug_print=True) # filter_sessions resets `self.filtered_sessions, self.filtered_epochs, self.filtered_contexts, self.active_configs self.computation_results`

                # UPDATING PIPELINE CONFIGS MUST BE DONE HERE ________________________________________________________________________ #
                for i, a_split_directional_laps_config_name in enumerate(curr_epoch_split_directional_laps_config_names):
                    a_lap_dir: str = cls.split_directional_laps_name_parts[i]

                    if debug_print:
                        print(f'\ta_split_directional_laps_config_name: {a_split_directional_laps_config_name}, a_lap_dir: {a_lap_dir}')

                    ## Build the IdentifyingContext for this epoch. Could also get it from 
                    a_context = deepcopy(curr_active_pipeline.filtered_contexts[an_epoch_name])
                    a_context = cls.format_directional_laps_context(a_context, a_maze_name=an_epoch_name, a_directional_epoch_name=a_split_directional_laps_config_name, a_lap_dir_name=a_lap_dir)
                    # split_directional_laps_contexts.append(a_context)

                    # Indexed by context now:
                    directional_lap_specific_configs[a_context] = deepcopy(curr_active_pipeline.active_configs[an_epoch_name])
                    ## Just overwrite directly:
                    directional_lap_specific_configs[a_context].computation_config = DynamicContainer.init_from_dict(deepcopy(curr_active_pipeline.active_configs[an_epoch_name].computation_config.to_dict()))
                    # directional_lap_specific_configs[a_split_directional_laps_config_name].computation_config.pf_params.computation_epochs = deepcopy(directional_lap_specific_configs[a_split_directional_laps_config_name].computation_config.pf_params.computation_epochs.label_slice(lap_dir_epochs.labels)) # does this work?
                    directional_lap_specific_configs[a_context].computation_config.pf_params.computation_epochs = deepcopy(split_directional_laps_dict[a_context])
                    # directional_lap_specific_configs[a_split_directional_laps_config_name] = active_config_copy

                    if add_created_configs_to_pipeline:
                        curr_active_pipeline.active_configs[a_split_directional_laps_config_name] = deepcopy(directional_lap_specific_configs[a_context])
                        # When a new config is added, new results and stuff should be added too.
                        # Set the laps... AGAIN
                        curr_active_pipeline.active_configs[a_split_directional_laps_config_name].computation_config.pf_params.computation_epochs = deepcopy(split_directional_laps_dict[a_context])

                        curr_active_pipeline.filtered_sessions[a_split_directional_laps_config_name] = curr_active_pipeline.filtered_sessions[an_epoch_name]
                        curr_active_pipeline.filtered_epochs[a_split_directional_laps_config_name] = curr_active_pipeline.filtered_epochs[an_epoch_name]


                        curr_active_pipeline.filtered_contexts[a_split_directional_laps_config_name] = a_context

                        curr_active_pipeline.computation_results[a_split_directional_laps_config_name] = None # empty

                        active_computation_params = deepcopy(directional_lap_specific_configs[a_context].computation_config)
                        curr_active_pipeline.computation_results[a_split_directional_laps_config_name] = ComputedPipelineStage._build_initial_computationResult(curr_active_pipeline.filtered_sessions[a_split_directional_laps_config_name], active_computation_params)
                        print(f'\t\n\tcomputation_epochs: {curr_active_pipeline.active_configs[a_split_directional_laps_config_name].computation_config.pf_params.computation_epochs}\n\n')


                    # end loop over split_directional_lap types:


                ## Perform the computations which builds the computation results:
                # _out = curr_active_pipeline.perform_computations(computation_functions_name_includelist=['pf_computation'], # , 'pfdt_computation', 'firing_rate_trends', 'position_decoding'
                #     enabled_filter_names=split_directional_laps_config_names,
                #     # computation_kwargs_list=[dict(ndim=1)],
                #     fail_on_exception=False, debug_print=True) # does not work to recompute anything because it detects that nothing has changed.


                computed_base_epoch_names.append(an_epoch_name)
                
            # end for an_epoch_name


            # COMPUTATION DONE HERE ______________________________________________________________________________________________ #
            #TODO 2023-10-26 05:56: - [ ] Can't work because `AttributeError: 'DisplayPipelineStage' object has no attribute 'get_merged_computation_function_validators'`, the stage is never going to work.
            # newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=['pf_computation'],
            #         included_computation_filter_names=split_directional_laps_config_names,
            #         include_global_functions=True, fail_on_exception=True, progress_print=True, force_recompute=True, debug_print=True)
            # print(f'newly_computed_values: {newly_computed_values}')



        # end if use_direction_dependent_laps
        return curr_active_pipeline, directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names

    @classmethod
    def validate_has_directional_laps(cls, curr_active_pipeline, computation_filter_name='maze'):
        # Unpacking:
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        # assert (computation_filter_name in computed_base_epoch_names), f'computation_filter_name: {computation_filter_name} is missing from computed_base_epoch_names: {computed_base_epoch_names} '
        return (computation_filter_name in computed_base_epoch_names)


    @classmethod
    def build_directional_constrained_decoders(cls, curr_active_pipeline):
        """ 2023-10-26 - Builds the four templates for the directioanl placefields 
        Uses: cls.split_directional_laps_name_parts, cls.split_to_directional_laps
        
        """
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        odd_laps_suffix, even_laps_suffix = cls.split_directional_laps_name_parts

        long_odd_laps_name, long_even_laps_name = [f'{long_epoch_name}_{a_suffix}' for a_suffix in (odd_laps_suffix, even_laps_suffix)]
        short_odd_laps_name, short_even_laps_name = [f'{short_epoch_name}_{a_suffix}' for a_suffix in (odd_laps_suffix, even_laps_suffix)]

        print(long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name) # ('maze1_odd_laps', 'maze1_even_laps', 'maze2_odd_laps', 'maze2_even_laps')

        # Unpacking for non-direction specific epochs (to get the replays, etc):
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [a_session.replay for a_session in [long_session, short_session, global_session]]

        # Unpacking for `(long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)`
        (long_odd_laps_context, long_even_laps_context, short_odd_laps_context, short_even_laps_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_obj, long_even_laps_obj, short_odd_laps_obj, short_even_laps_obj) = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name)) for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_session, long_even_laps_session, short_odd_laps_session, short_even_laps_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_results, long_even_laps_results, short_odd_laps_results, short_even_laps_results) = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_computation_config, long_even_laps_computation_config, short_odd_laps_computation_config, short_even_laps_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_pf1D, long_even_laps_pf1D, short_odd_laps_pf1D, short_even_laps_pf1D) = (long_odd_laps_results.pf1D, long_even_laps_results.pf1D, short_odd_laps_results.pf1D, short_even_laps_results.pf1D)
        (long_odd_laps_pf2D, long_even_laps_pf2D, short_odd_laps_pf2D, short_even_laps_pf2D) = (long_odd_laps_results.pf2D, long_even_laps_results.pf2D, short_odd_laps_results.pf2D, short_even_laps_results.pf2D)

        # Validate:
        assert not (curr_active_pipeline.computation_results[long_odd_laps_name]['computation_config']['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_even_laps_name]['computation_config']['pf_params'].computation_epochs)
        assert not (curr_active_pipeline.computation_results[short_odd_laps_name]['computation_config']['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_even_laps_name]['computation_config']['pf_params'].computation_epochs)

        ## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
        long_odd_laps_one_step_decoder_1D, long_even_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_odd_laps_results, long_even_laps_results, short_odd_laps_results, short_even_laps_results)]

        # Prune to the shared aclus in both epochs (short/long):
        active_neuron_IDs_list = [a_decoder.neuron_IDs for a_decoder in (long_odd_laps_one_step_decoder_1D, long_even_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D)]

        # Find only the common aclus amongst all four templates:
        shared_aclus = np.array(list(set.intersection(*map(set,active_neuron_IDs_list)))) # array([ 6,  7,  8, 11, 15, 16, 20, 24, 25, 26, 31, 33, 34, 35, 39, 40, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64])
        n_neurons = len(shared_aclus)
        print(f'n_neurons: {n_neurons}, shared_aclus: {shared_aclus}')

        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D = [a_decoder.get_by_id(shared_aclus) for a_decoder in (long_odd_laps_one_step_decoder_1D, long_even_laps_one_step_decoder_1D, short_odd_laps_one_step_decoder_1D, short_even_laps_one_step_decoder_1D)]

        return long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D 



    @classmethod
    def complete_directional_pfs_computations(cls, curr_active_pipeline):
        """ Explicit recomputation of laps: ^^top-level computation^^
            Calls `cls.build_directional_constrained_decoders(...)` 
        
        """
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations
        # Explicit recomputation of laps
        curr_active_pipeline, directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = cls.split_to_directional_laps(curr_active_pipeline, include_includelist=['maze1', 'maze2'], add_created_configs_to_pipeline=True)
        ## Explicit recomputation for the placefields of the given epochs:
        newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=['pf_computation', 'position_decoding'], included_computation_filter_names=split_directional_laps_config_names, # , 'position_decoding', 'firing_rate_trends'
                                                            include_global_functions=True, fail_on_exception=False, progress_print=True, force_recompute=False, debug_print=True)
        # prepare for display:
        # curr_active_pipeline.prepare_for_display(root_output_dir=global_data_root_parent_path.joinpath('Output'), should_smooth_maze=True) # TODO: pass a display config
        print(f'newly_computed_values: {newly_computed_values}')
        curr_active_pipeline.prepare_for_display()


        ### 2023-10-26 - Extract Directional Laps Outputs and computed items:
        # (long_one_step_decoder_1D, short_one_step_decoder_1D), (long_one_step_decoder_2D, short_one_step_decoder_2D) = compute_short_long_constrained_decoders(curr_active_pipeline, recalculate_anyway=True)
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_epoch_obj, short_epoch_obj = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name)) for an_epoch_name in [long_epoch_name, short_epoch_name]]
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
        long_pf2D, short_pf2D, global_pf2D = long_results.pf2D, short_results.pf2D, global_results.pf2D


        # long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        odd_laps_suffix, even_laps_suffix = cls.split_directional_laps_name_parts

        long_odd_laps_name, long_even_laps_name = [f'{long_epoch_name}_{a_suffix}' for a_suffix in (odd_laps_suffix, even_laps_suffix)]
        short_odd_laps_name, short_even_laps_name = [f'{short_epoch_name}_{a_suffix}' for a_suffix in (odd_laps_suffix, even_laps_suffix)]

        print(long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name) # ('maze1_odd_laps', 'maze1_even_laps', 'maze2_odd_laps', 'maze2_even_laps')

        # Unpacking for non-direction specific epochs (to get the replays, etc):
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [a_session.replay for a_session in [long_session, short_session, global_session]]

        # Unpacking for `(long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)`
        (long_odd_laps_context, long_even_laps_context, short_odd_laps_context, short_even_laps_context) = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_obj, long_even_laps_obj, short_odd_laps_obj, short_even_laps_obj) = [Epoch(curr_active_pipeline.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name)) for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_session, long_even_laps_session, short_odd_laps_session, short_even_laps_session) = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_results, long_even_laps_results, short_odd_laps_results, short_even_laps_results) = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_computation_config, long_even_laps_computation_config, short_odd_laps_computation_config, short_even_laps_computation_config) = [curr_active_pipeline.computation_results[an_epoch_name]['computation_config'] for an_epoch_name in (long_odd_laps_name, long_even_laps_name, short_odd_laps_name, short_even_laps_name)]
        (long_odd_laps_pf1D, long_even_laps_pf1D, short_odd_laps_pf1D, short_even_laps_pf1D) = (long_odd_laps_results.pf1D, long_even_laps_results.pf1D, short_odd_laps_results.pf1D, short_even_laps_results.pf1D)
        (long_odd_laps_pf2D, long_even_laps_pf2D, short_odd_laps_pf2D, short_even_laps_pf2D) = (long_odd_laps_results.pf2D, long_even_laps_results.pf2D, short_odd_laps_results.pf2D, short_even_laps_results.pf2D)

        # Validate:
        assert not (curr_active_pipeline.computation_results[long_odd_laps_name]['computation_config']['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_even_laps_name]['computation_config']['pf_params'].computation_epochs)
        assert not (curr_active_pipeline.computation_results[short_odd_laps_name]['computation_config']['pf_params'].computation_epochs is curr_active_pipeline.computation_results[long_even_laps_name]['computation_config']['pf_params'].computation_epochs)

        # build the four `*_shared_aclus_only_one_step_decoder_1D` versions of the decoders constrained only to common aclus:
        long_odd_shared_aclus_only_one_step_decoder_1D, long_even_shared_aclus_only_one_step_decoder_1D, short_odd_shared_aclus_only_one_step_decoder_1D, short_even_shared_aclus_only_one_step_decoder_1D  = cls.build_directional_constrained_decoders(curr_active_pipeline)

        # ## Encode/Decode from global result:
        # # Unpacking:
        # directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        # directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]
        # # split_directional_laps_config_names
        
        ## Build a `ComputedResult` container object to hold the result:
        directional_laps_result = ComputedResult()
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

        ## Set the global result:
        # curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps'] = directional_laps_result

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
        if include_includelist is None:
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            # include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']
            # include_includelist = [global_epoch_name] # ['maze'] # only for maze
            include_includelist = [long_epoch_name, short_epoch_name] # ['maze1', 'maze2'] # only for maze
        
        ## Adds ['*_even_laps', '*_odd_laps'] pseduofilters

        owning_pipeline_reference, directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = DirectionalLapsHelpers.split_to_directional_laps(owning_pipeline_reference, include_includelist=include_includelist, add_created_configs_to_pipeline=True)
        # curr_active_pipeline, directional_lap_specific_configs = constrain_to_laps(curr_active_pipeline)
        # list(directional_lap_specific_configs.keys())

        global_computation_results.computed_data['DirectionalLaps'] = DynamicParameters.init_from_dict({
            'directional_lap_specific_configs': directional_lap_specific_configs,
            'split_directional_laps_dict': split_directional_laps_dict,
            'split_directional_laps_contexts_dict': split_directional_laps_contexts_dict,
            'split_directional_laps_names': split_directional_laps_config_names,
            'computed_base_epoch_names': computed_base_epoch_names,
        })

        ## Needs to call `owning_pipeline_reference.prepare_for_display()` before display functions can be used with new directional results

        """ Usage:
        
        directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
        directional_lap_specific_configs, split_directional_laps_dict, split_directional_laps_contexts_dict, split_directional_laps_config_names, computed_base_epoch_names = [directional_laps_results[k] for k in ['directional_lap_specific_configs', 'split_directional_laps_dict', 'split_directional_laps_contexts_dict', 'split_directional_laps_names', 'computed_base_epoch_names']]

        """
        return global_computation_results

