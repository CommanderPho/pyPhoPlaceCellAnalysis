import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np

## MATPLOTLIB Imports:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends import backend_pdf


from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session
from neuropy.utils.misc import compute_paginated_grid_config # for paginating shared aclus
from neuropy.utils.result_context import IdentifyingContext

# pyphocorehelpers
from pyphocorehelpers.indexing_helpers import reorder_keys_relative
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.exception_helpers import CapturedException, ExceptionPrintingContext

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import SpecificComputationValidator
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import SplitPartitionMembership # needed for batch_extended_computations, batch_programmatic_figures
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import programmatic_display_to_PDF, programmatic_render_to_file
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_metadata_from_display_context # newer version of build_pdf_export_metadata
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline, PipelineSavingScheme # for batch_load_session

from pyphoplacecellanalysis.SpecificResults.fourthYearPresentation import export_active_relative_entropy_results_videos
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPlacefieldGlobalComputationFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import SequenceBasedComputationsGlobalComputationFunctions


""" 

filters should be checkable to express whether we want to build that one or not


"""
def get_all_batch_computation_names():
    """ Gets the hardcoded or dynamically loaded computation names
    
    
    non_global_comp_names, global_comp_names = get_all_batch_computation_names()
    
    
    """
    ## functions to exclude from the outputs:
    non_global_all_exclude_list = ['EloyAnalysis', '_DEP_ratemap_peaks', '_perform_specific_epochs_decoding', 'extended_stats', 'placefield_overlap', 'recursive_latent_pf_decoding', 'velocity_vs_pf_simplified_count_density']
    global_all_exclude_list = ['PBE_stats']
    return ComputationFunctionRegistryHolder.get_all_computation_fn_names(non_global_all_exclude_list=non_global_all_exclude_list,
                                                                           global_all_exclude_list=global_all_exclude_list)





# ==================================================================================================================== #
# 2022-12-07 - batch_load_session - Computes Entire Pipeline                                                           #
# ==================================================================================================================== #


@function_attributes(short_name='batch_load_session', tags=['main', 'batch', 'automated', 'session', 'load'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2022-12-07 00:00')
def batch_load_session(global_data_root_parent_path, active_data_mode_name, basedir, force_reload=False, saving_mode=PipelineSavingScheme.SKIP_SAVING, fail_on_exception=True, skip_extended_batch_computations=False, **kwargs):
    """Loads and runs the entire pipeline for a session folder located at the path 'basedir'.

    Args:
        global_data_root_parent_path (_type_): _description_
        active_data_mode_name (_type_): _description_
        basedir (_type_): _description_

    Returns:
        _type_: _description_
    """
    saving_mode = PipelineSavingScheme.init(saving_mode)
    epoch_name_includelist = kwargs.get('epoch_name_includelist', ['maze1','maze2','maze'])
    debug_print = kwargs.get('debug_print', False)
    assert 'skip_save' not in kwargs, f"use saving_mode=PipelineSavingScheme.SKIP_SAVING instead"
    # skip_save = kwargs.get('skip_save', False)
    active_pickle_filename = kwargs.get('active_pickle_filename', 'loadedSessPickle.pkl')

    active_session_computation_configs = kwargs.get('active_session_computation_configs', None)

    computation_functions_name_includelist = kwargs.get('computation_functions_name_includelist', None)


    known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    ## Begin main run of the pipeline (load or execute):
    curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties,
        override_basepath=Path(basedir), force_reload=force_reload, active_pickle_filename=active_pickle_filename, skip_save_on_initial_load=True)
    
    was_loaded_from_file: bool =  curr_active_pipeline.has_associated_pickle # True if pipeline was loaded from an existing file, False if it was created fresh
    
    # Get the previous configs:
    # curr_active_pipeline.filtered_sessions
    # ['filtered_session_names', 'filtered_contexts', 'filtered_epochs', 'filtered_sessions']
    # loaded_session_filter_configurations = {k:v.filter_config['filter_function'] for k,v in curr_active_pipeline.active_configs.items()}
    # loaded_pipeline_computation_configs = {k:v.computation_config for k,v in curr_active_pipeline.active_configs.items()}


    ## Build updated ones from the current configs:
    active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_includelist=epoch_name_includelist) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
    if debug_print:
        print(f'active_session_filter_configurations: {active_session_filter_configurations}')
    
    ## Skip the filtering, it used to be performed bere but NOT NOW

    ## TODO 2023-05-16 - set `curr_active_pipeline.active_configs[a_name].computation_config.pf_params.computation_epochs = curr_laps_obj` equivalent
    ## TODO 2023-05-16 - determine appropriate binning from `compute_short_long_constrained_decoders` so it's automatically from the long

    if active_session_computation_configs is None:
        """
        If there are is provided computation config, get the default:
        """
        # ## Compute shared grid_bin_bounds for all epochs from the global positions:
        # global_unfiltered_session = curr_active_pipeline.sess
        # # ((22.736279243974774, 261.696733348342), (49.989466271998936, 151.2870218547401))
        # first_filtered_session = curr_active_pipeline.filtered_sessions[curr_active_pipeline.filtered_session_names[0]]
        # # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))
        # second_filtered_session = curr_active_pipeline.filtered_sessions[curr_active_pipeline.filtered_session_names[1]]
        # # ((71.67666779621361, 224.37820920766043), (110.51617463644946, 151.2870218547401))

        # grid_bin_bounding_session = first_filtered_session
        # grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(grid_bin_bounding_session.position.x, grid_bin_bounding_session.position.y)

        ## OR use no grid_bin_bounds meaning they will be determined dynamically for each epoch:
        # grid_bin_bounds = None
        # time_bin_size = 0.03333 #1.0/30.0 # decode at 30fps to match the position sampling frequency
        # time_bin_size = 0.1 # 10 fps
        time_bin_size = kwargs.get('time_bin_size', 0.03333) # 0.03333 = 1.0/30.0 # decode at 30fps to match the position sampling frequency
        # time_bin_size = kwargs.get('time_bin_size', 0.1) # 10 fps
    
        # lap_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
        # assert lap_estimation_parameters is not None
        active_session_computation_configs = active_data_mode_registered_class.build_active_computation_configs(sess=curr_active_pipeline.sess, time_bin_size=time_bin_size) # , grid_bin_bounds=grid_bin_bounds
    
    else:
        # Use the provided `active_session_computation_configs`:
        assert 'time_bin_size' not in kwargs, f"time_bin_size kwarg provided but will not be used because a custom active_session_computation_configs was provided as well."


    ## Setup Computation Functions to be executed:
    if computation_functions_name_includelist is None:
        # includelist Mode:
        computation_functions_name_includelist=['_perform_baseline_placefield_computation', '_perform_time_dependent_placefield_computation', '_perform_extended_statistics_computation',
                                            '_perform_position_decoding_computation', 
                                            '_perform_firing_rate_trends_computation',
                                            '_perform_pf_find_ratemap_peaks_computation',
                                            # '_perform_time_dependent_pf_sequential_surprise_computation'
                                            # '_perform_two_step_position_decoding_computation',
                                            # '_perform_recursive_latent_placefield_decoding'
                                        ]  # '_perform_pf_find_ratemap_peaks_peak_prominence2d_computation'
        computation_functions_name_excludelist=None
    else:
        print(f'using provided computation_functions_name_includelist: {computation_functions_name_includelist}')
        computation_functions_name_excludelist=None

    ## For every computation config we build a fake (duplicate) filter config).
    # OVERRIDE WITH TRUE:
    # curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.laps['use_direction_dependent_laps'] = True # override with True
    lap_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
    assert lap_estimation_parameters is not None
    use_direction_dependent_laps: bool = lap_estimation_parameters.get('use_direction_dependent_laps', False) # whether to split the laps into left and right directions
    # use_direction_dependent_laps: bool = lap_estimation_parameters.get('use_direction_dependent_laps', True) # whether to split the laps into left and right directions
    
    if (use_direction_dependent_laps or (len(active_session_computation_configs) > 3)):
        lap_direction_suffix_list = ['_odd', '_even', '_any'] # ['maze1_odd', 'maze1_even', 'maze1_any', 'maze2_odd', 'maze2_even', 'maze2_any', 'maze_odd', 'maze_even', 'maze_any']
        # lap_direction_suffix_list = ['_odd', '_even', ''] # no '_any' prefix, instead reuses the existing names
        # assert len(lap_direction_suffix_list) == len(active_session_computation_configs), f"len(lap_direction_suffix_list): {len(lap_direction_suffix_list)}, len(active_session_computation_configs): {len(active_session_computation_configs)}, "
    else:
        print(f'not using direction-dependent laps.')
        lap_direction_suffix_list = ['']


    # active_session_computation_configs: this should contain three configs, one for each Epoch

    active_session_computation_configs = [deepcopy(a_config) for a_config in active_session_computation_configs]
    assert len(lap_direction_suffix_list) == len(active_session_computation_configs)
    updated_active_session_pseudo_filter_configs = {} # empty list, woot!


    for a_computation_suffix_name, a_computation_config in zip(lap_direction_suffix_list, active_session_computation_configs): # these should NOT be the same length: lap_direction_suffix_list: ['_odd', '_even', '_any']
        # We need to filter and then compute with the appropriate config iteratively.
        for a_filter_config_name, a_filter_config_fn in active_session_filter_configurations.items():
            # TODO: Build a context:
            a_combined_name: str = f'{a_filter_config_name}{a_computation_suffix_name}'
            # if a_computation_suffix_name != '':
            updated_active_session_pseudo_filter_configs[a_combined_name] = deepcopy(a_filter_config_fn) # this copy is just so that the values are recomputed with the appropriate config. This is a HACK
        # end for filter_configs

        ## Actually do the filtering now. We have 
        curr_active_pipeline.filter_sessions(updated_active_session_pseudo_filter_configs, changed_filters_ignore_list=['maze1','maze2','maze'], debug_print=False)

        ## TODO 2023-01-15 - perform_computations for all configs!!
        #TODO 2023-10-31 14:58: - [ ] This is where the computations are being done multiple times!
        #TODO 2023-11-13 14:23: - [ ] With this approach, we can't actually properly filter the computation_configs for the relevant sessions ahead of time because they are calculated for a single computation config but across all sessions at once.
        curr_active_pipeline.perform_computations(a_computation_config, computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, debug_print=debug_print) #, overwrite_extant_results=False  ], fail_on_exception=True, debug_print=False)


    if not skip_extended_batch_computations:
        batch_extended_computations(curr_active_pipeline, include_global_functions=False, fail_on_exception=fail_on_exception, progress_print=True, debug_print=False)
    # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_excludelist=['_perform_spike_burst_detection_computation'], debug_print=False, fail_on_exception=False) # includelist: ['_perform_baseline_placefield_computation']


    try:
        curr_active_pipeline.prepare_for_display(root_output_dir=global_data_root_parent_path.joinpath('Output'), should_smooth_maze=True) # TODO: pass a display config
    except Exception as e:
        exception_info = sys.exc_info()
        an_error = CapturedException(e, exception_info, curr_active_pipeline)
        print(f'WARNING: Failed to do `curr_active_pipeline.prepare_for_display(...)` with error: {an_error}')
        if fail_on_exception:
            raise

    try:
        curr_active_pipeline.save_pipeline(saving_mode=saving_mode)
    except Exception as e:
        exception_info = sys.exc_info()
        an_error = CapturedException(e, exception_info, curr_active_pipeline)
        print(f'WARNING: Failed to save pipeline via `curr_active_pipeline.save_pipeline(...)` with error: {an_error}')
        if fail_on_exception:
            raise

    if not saving_mode.shouldSave:
        print(f'saving_mode.shouldSave == False, so not saving at the end of batch_load_session')

    ## Load pickled global computations:
    # If previously pickled global results were saved, they will typically no longer be relevent if the pipeline was recomputed. We need a system of invalidating/versioning the global results when the other computations they depend on change.
    # Maybe move into `batch_extended_computations(...)` or integrate with that somehow
    # curr_active_pipeline.load_pickled_global_computation_results()

    return curr_active_pipeline


def batch_evaluate_required_computations(curr_active_pipeline, include_includelist=None, included_computation_filter_names=None, include_global_functions=False, fail_on_exception=False, progress_print=True, debug_print=False,
                                force_recompute:bool=False, force_recompute_override_computations_includelist=None):
    """ determines how many of the specified computations are already valid, vs. how many will need to be recomputed. 

    force_recompute:bool=False
    force_recompute_override_computations_includelist: Optional[List[str]] a list of computation function names to recompute regardless of their validator's status.
    
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_evaluate_required_computations

    needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=['lap_direction_determination'], include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                        force_recompute=False, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
    needs_computation_output_dict
    

    WARNING: `force_recompute_override_computations_includelist` must be a subset of the items in `include_includelist`
    include_includelist = ['pf_computation', 'split_to_directional_laps', 'merged_directional_placefields', 'rank_order_shuffle_analysis']
    force_recompute_override_computations_includelist = ['pf_computation', 'split_to_directional_laps']
    
    The whole function is only called if  ((_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist)) is true.
    
    """
    #TODO 2023-09-08 07:48: - [ ] Currently only executes functions with a valid `validate_computation_test` set and silently skips functions that don't exist or are missing a validator.
    #TODO 2023-08-31 11:05: - [X] Do local computations first for all valid filter_epochs, then do global

    force_recompute_override_computations_includelist = force_recompute_override_computations_includelist or []

    ## Get the names of the global and non-global computations:
    non_global_comp_names, global_comp_names = get_all_batch_computation_names()
    
    if include_includelist is None:
        # include all:
        include_includelist = non_global_comp_names + global_comp_names
    else:
        print(f'included includelist is specified: {include_includelist}, so only performing these extended computations.')

    #TODO 2024-01-09 21:41: - [ ] I don't enforce that every member of  `force_recompute_override_computations_includelist` is in `include_includelist` because one might be specified in a "_comp_specifier.short_name" format while the other a "_comp_specifier.computation_fn_name", so they wouldn't literally match.
    # If the function is specified in `force_recompute_override_computations_includelist` it should NOT be omitted from `include_includelist`.
    # #  The whole function is only called if  ((_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist)) is true, so check.
    # for a_name in force_recompute_override_computations_includelist:
    #     assert (not (a_name not in ((_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist)))), f'WARNING: function "{a_name}" specified in `force_recompute_override_computations_includelist` is missing from `include_includelist`: ({include_includelist})! If this is the case, the function will never be computed (forced or otherwise).\nIf the function is specified in `force_recompute_override_computations_includelist` it should not be omitted from `included_computation_filter_names`.'            
    #     # print(f'WARNING: function "{a_name}" specified in `force_recompute_override_computations_includelist` is missing from `include_includelist`: ({include_includelist})! If this is the case, the function will never be computed (forced or otherwise).')
    assert len(force_recompute_override_computations_includelist) <= len(include_includelist), f"READ THE NOTE ABOUT force_recompute_override_computations_includelist being a subset of include_includelist in the code above!! include_includelist: {include_includelist}\nforce_recompute_override_computations_includelist: {force_recompute_override_computations_includelist}"

    ## Get computed relative entropy measures:
    _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # global_epoch_name = curr_active_pipeline.active_completed_computation_result_names[-1] # 'maze'

    if included_computation_filter_names is None:
        included_computation_filter_names = [global_epoch_name] # use only the global epoch: e.g. ['maze']
        if progress_print:
            print(f'Running batch_evaluate_required_computations(...) with global_epoch_name: "{global_epoch_name}"')
    else:
        if progress_print:
            print(f'Running batch_evaluate_required_computations(...) with included_computation_filter_names: "{included_computation_filter_names}"')


    ## Specify the computations and the requirements to validate them.

    ## Hardcoded comp_specifiers
    _comp_specifiers = list(curr_active_pipeline.get_merged_computation_function_validators().values())
    # ## Execution order is currently determined by `_comp_specifiers` order and not the order the `include_includelist` lists them (which is good) but the `curr_active_pipeline.registered_merged_computation_function_dict` has them registered in *REVERSE* order for the specific computation function called, so we need to reverse these

    remaining_include_function_names = {k:False for k in include_includelist.copy()}

    valid_computed_results_output_list = [] # items that have valid results (that pass the validator)
    needs_computation_output_dict = {} # items that need to be computed

    for _comp_specifier in _comp_specifiers:
        if (not _comp_specifier.is_global) or include_global_functions:
            # if (_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist):
            if _comp_specifier.is_name_in(include_includelist):
                if (not _comp_specifier.is_global):
                    # Not Global-only, need to compute for all `included_computation_filter_names`:
                    for a_computation_filter_name in included_computation_filter_names:
                        has_valid_computation: bool = _comp_specifier.try_validate_is_computation_valid(curr_active_pipeline, computation_filter_name=a_computation_filter_name, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
                        if has_valid_computation:
                            valid_computed_results_output_list.append((a_computation_filter_name, _comp_specifier.short_name))

                        needs_compute: bool = (not has_valid_computation)
                        if needs_compute:
                            if _comp_specifier.short_name not in needs_computation_output_dict:
                                needs_computation_output_dict[_comp_specifier.short_name] = {}

                            needs_computation_output_dict[_comp_specifier.short_name][a_computation_filter_name] = _comp_specifier.computation_precidence

                else:
                    # Global-Only:
                    _curr_force_recompute = force_recompute or _comp_specifier.is_name_in(force_recompute_override_computations_includelist) # force_recompute for this specific result if either of its name is included in `force_recompute_override_computations_includelist`
                    has_valid_computation: bool = _comp_specifier.try_validate_is_computation_valid(curr_active_pipeline, computation_filter_name=global_epoch_name, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=_curr_force_recompute)
                    
                    if has_valid_computation:
                        valid_computed_results_output_list.append(_comp_specifier.short_name)

                    needs_compute: bool = (not has_valid_computation)
                    if needs_compute:
                        needs_computation_output_dict[_comp_specifier.short_name] = _comp_specifier.computation_precidence

                        

                ## ENDIF is_global
                        
                ## remove from the found name from the list of remaining function names
                if (_comp_specifier.short_name in include_includelist):
                    del remaining_include_function_names[_comp_specifier.short_name]
                elif (_comp_specifier.computation_fn_name in include_includelist):
                    del remaining_include_function_names[_comp_specifier.computation_fn_name]
                else:
                    raise NotImplementedError

    if len(remaining_include_function_names) > 0:
        print(f'WARNING: after checking all_comp_specifiers found the functions: {remaining_include_function_names} still remain! Are they correct and do they have proper validator decorators?')
    if progress_print:
        print('done with all batch_evaluate_required_computations(...).')

    return needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names




@function_attributes(short_name='batch_extended_computations', tags=['batch', 'automated', 'session', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 04:46')
def batch_extended_computations(curr_active_pipeline, include_includelist=None, included_computation_filter_names=None, include_global_functions=False, fail_on_exception=False, progress_print=True, debug_print=False,
                                force_recompute:bool=False, force_recompute_override_computations_includelist=None, dry_run:bool=False):
    """ performs the remaining required global computations

    force_recompute:bool=False
    force_recompute_override_computations_includelist: Optional[List[str]] a list of computation function names to recompute regardless of their validator's status.
    
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations
    batch_extended_computations(include_includelist=['merged_directional_placefields'], include_global_functions=True)
    
    WARNING: `force_recompute_override_computations_includelist` must be a subset of the items in `include_includelist`
    include_includelist = ['pf_computation', 'split_to_directional_laps', 'merged_directional_placefields', 'rank_order_shuffle_analysis']
    force_recompute_override_computations_includelist = ['pf_computation', 'split_to_directional_laps']
    
    The whole function is only called if  ((_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist)) is true.
    
    """
    #TODO 2023-09-08 07:48: - [ ] Currently only executes functions with a valid `validate_computation_test` set and silently skips functions that don't exist or are missing a validator.
    #TODO 2023-08-31 11:05: - [X] Do local computations first for all valid filter_epochs, then do global

    def _subfn_on_already_computed(_comp_name, computation_filter_name):
        """ captures: `progress_print`, `force_recompute`
        raises AttributeError if force_recompute is true to trigger recomputation """
        if progress_print:
            print(f'{_comp_name}, {computation_filter_name} already computed.')
        if force_recompute:
            if progress_print:
                print(f'\tforce_recompute is true so recomputing anyway')
            raise AttributeError # just raise an AttributeError to trigger recomputation    

    newly_computed_values = []
    force_recompute_override_computations_includelist = force_recompute_override_computations_includelist or []

    ## Get the names of the global and non-global computations:
    non_global_comp_names, global_comp_names = get_all_batch_computation_names()
    
    if include_includelist is None:
        # include all:
        include_includelist = non_global_comp_names + global_comp_names
    else:
        print(f'included includelist is specified: {include_includelist}, so only performing these extended computations.')

    #TODO 2024-01-09 21:41: - [ ] I don't enforce that every member of  `force_recompute_override_computations_includelist` is in `include_includelist` because one might be specified in a "_comp_specifier.short_name" format while the other a "_comp_specifier.computation_fn_name", so they wouldn't literally match.
    # If the function is specified in `force_recompute_override_computations_includelist` it should NOT be omitted from `include_includelist`.
    # #  The whole function is only called if  ((_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist)) is true, so check.
    # for a_name in force_recompute_override_computations_includelist:
    #     assert (not (a_name not in ((_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist)))), f'WARNING: function "{a_name}" specified in `force_recompute_override_computations_includelist` is missing from `include_includelist`: ({include_includelist})! If this is the case, the function will never be computed (forced or otherwise).\nIf the function is specified in `force_recompute_override_computations_includelist` it should not be omitted from `included_computation_filter_names`.'            
    #     # print(f'WARNING: function "{a_name}" specified in `force_recompute_override_computations_includelist` is missing from `include_includelist`: ({include_includelist})! If this is the case, the function will never be computed (forced or otherwise).')
    assert len(force_recompute_override_computations_includelist) <= len(include_includelist), f"READ THE NOTE ABOUT force_recompute_override_computations_includelist being a subset of include_includelist in the code above!! include_includelist: {include_includelist}\nforce_recompute_override_computations_includelist: {force_recompute_override_computations_includelist}"

    ## Get computed relative entropy measures:
    _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # global_epoch_name = curr_active_pipeline.active_completed_computation_result_names[-1] # 'maze'

    if included_computation_filter_names is None:
        included_computation_filter_names = [global_epoch_name] # use only the global epoch: e.g. ['maze']
        if progress_print:
            print(f'Running batch_extended_computations(...) with global_epoch_name: "{global_epoch_name}"')
    else:
        if progress_print:
            print(f'Running batch_extended_computations(...) with included_computation_filter_names: "{included_computation_filter_names}"')



    ## Specify the computations and the requirements to validate them.

    ## Hardcoded comp_specifiers
    _comp_specifiers = list(curr_active_pipeline.get_merged_computation_function_validators().values())
    ## Execution order is currently determined by `_comp_specifiers` order and not the order the `include_includelist` lists them (which is good) but the `curr_active_pipeline.registered_merged_computation_function_dict` has them registered in *REVERSE* order for the specific computation function called, so we need to reverse these


    ## apply arbitrary sort:
    _comp_specifier_dict = {_comp_specifier.short_name:_comp_specifier for _comp_specifier in _comp_specifiers}
    # move_to_end_names = ['directional_train_test_split', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring']
    # _comp_specifier_dict = reorder_keys_relative(_comp_specifier_dict, key_names=move_to_end_names, relative_mode='end') # list(filter(lambda column: column.endswith('_peak_heights'), existing_columns))
    _comp_specifier_dict = dict(sorted(_comp_specifier_dict.items(), key=lambda item: item[1].computation_precidence))
    # back to list
    _comp_specifiers = list(_comp_specifier_dict.values())

    remaining_include_function_names = {k:False for k in include_includelist.copy()}

    for _comp_specifier in _comp_specifiers:
        if (not _comp_specifier.is_global) or include_global_functions:
            if (_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist):
                if (not _comp_specifier.is_global):
                    # Not Global-only, need to compute for all `included_computation_filter_names`:
                    for a_computation_filter_name in included_computation_filter_names:
                        if not dry_run:
                            newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, computation_filter_name=a_computation_filter_name, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
                        else:
                            print(f'dry-run: {_comp_specifier.short_name}, computation_filter_name={a_computation_filter_name}, force_recompute={force_recompute}')

                else:
                    # Global-Only:
                    _curr_force_recompute = force_recompute or _comp_specifier.is_name_in(force_recompute_override_computations_includelist) # force_recompute for this specific result if either of its name is included in `force_recompute_override_computations_includelist`
                    if not dry_run:
                        newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, computation_filter_name=global_epoch_name, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=_curr_force_recompute)
                    else:
                        print(f'dry-run: {_comp_specifier.short_name}, force_recompute={force_recompute}, curr_force_recompute={_curr_force_recompute}')
                        # Check for existing result:
                        is_known_missing_provided_keys: bool = _comp_specifier.try_check_missing_provided_keys(curr_active_pipeline)
                        if is_known_missing_provided_keys:
                            print(f'{_comp_specifier.short_name} -- is_known_missing_provided_keys = True!')

                if (_comp_specifier.short_name in include_includelist):
                    del remaining_include_function_names[_comp_specifier.short_name]
                elif (_comp_specifier.computation_fn_name in include_includelist):
                    del remaining_include_function_names[_comp_specifier.computation_fn_name]
                else:
                    raise NotImplementedError

    if len(remaining_include_function_names) > 0:
        print(f'WARNING: after execution of all _comp_specifiers found the functions: {remaining_include_function_names} still remain! Are they correct and do they have proper validator decorators?')
    if progress_print:
        print('done with all batch_extended_computations(...).')

    return newly_computed_values



            
# ==================================================================================================================== #
# Batch Programmatic Figures - 2022-12-08 Batch Programmatic Figures (Currently only Jonathan-style)                                                                                           #
# ==================================================================================================================== #
@function_attributes(short_name='batch_programmatic_figures', tags=['batch', 'automated', 'session', 'display', 'figures'], input_requires=[], output_provides=[], uses=[], used_by=['batch_perform_all_plots'], creation_date='2023-03-28 04:46')
def batch_programmatic_figures(curr_active_pipeline, debug_print=False):
    """ programmatically generates and saves the batch figures 2022-12-07 
        curr_active_pipeline is the pipeline for a given session with all computations done.

    ## TODO: curr_session_parent_out_path

    active_identifying_session_ctx, active_out_figures_dict = batch_programmatic_figures(curr_active_pipeline)
    
    """
    ## 🗨️🟢 2022-10-26 - Jonathan Firing Rate Analyses
    # Perform missing global computations                                                                                  #
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_jonathan_replay_firing_rate_analyses', '_perform_long_short_pf_overlap_analyses'], fail_on_exception=True, debug_print=True)

    ## Get global 'jonathan_firing_rate_analysis' results:
    curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
    neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis.neuron_replay_stats_df, curr_jonathan_firing_rate_analysis.rdf.rdf, curr_jonathan_firing_rate_analysis.rdf.aclu_to_idx, curr_jonathan_firing_rate_analysis.irdf.irdf

    # ==================================================================================================================== #
    # Batch Output of Figures                                                                                              #
    # ==================================================================================================================== #
    ## 🗨️🟢 2022-11-05 - Pho-Jonathan Batch Outputs of Firing Rate Figures
    # %matplotlib qt
    short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
    short_only_aclus = short_only_df.index.values.tolist()
    long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
    long_only_aclus = long_only_df.index.values.tolist()
    shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
    shared_aclus = shared_df.index.values.tolist()
    if debug_print:
        print(f'shared_aclus: {shared_aclus}')
        print(f'long_only_aclus: {long_only_aclus}')
        print(f'short_only_aclus: {short_only_aclus}')

    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'    
    ## MODE: this mode creates a special folder to contain the outputs for this session.

    # ==================================================================================================================== #
    # Output Figures to File                                                                                               #
    # ==================================================================================================================== #
    active_out_figures_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, n_max_page_rows=10)
    return active_identifying_session_ctx, active_out_figures_dict


# import matplotlib as mpl
# import matplotlib.pyplot as plt
@function_attributes(short_name='batch_extended_programmatic_figures', tags=['batch', 'automated', 'session', 'display', 'figures', 'extended', 'matplotlib', 'main'], input_requires=[], output_provides=[], uses=['_display_batch_pho_jonathan_replay_firing_rate_comparison'], used_by=[], creation_date='2023-03-28 04:46')
def batch_extended_programmatic_figures(curr_active_pipeline, write_vector_format=False, write_png=True, debug_print=False):
    """ 
    
    Generation and display of figures should produce as many as possible, not stopping after failing on one.
    
    """
    _bak_rcParams = mpl.rcParams.copy()
    mpl.rcParams['toolbar'] = 'None' # disable toolbars
    matplotlib.use('AGG') # non-interactive backend ## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!
    from neuropy.plotting.ratemaps import BackgroundRenderingOptions
    
    # active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
    programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_1d_placefields', write_vector_format=write_vector_format, write_png=write_png, debug_print=debug_print) # 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    
    # '_display_1d_placefield_validations' can't use `programmatic_render_to_file` yet and must rely on `programmatic_display_to_PDF` because it doesn't get a new session for each figure and it overwrites itself a bunch
    programmatic_display_to_PDF(curr_active_pipeline, curr_display_function_name='_display_1d_placefield_validations', debug_print=debug_print) # , filter_name=active_config_name 🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear. Moderate visual improvements can still be made (titles overlap and stuff). Works with %%capture
    # programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_1d_placefield_validations', write_vector_format=write_vector_format, write_png=write_png, debug_print=debug_print) #  UNTESTED 2023-05-29 
    
    programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_result_plot_ratemaps_2D', write_vector_format=write_vector_format, write_png=write_png, debug_print=debug_print, bg_rendering_mode=BackgroundRenderingOptions.EMPTY) #  🟢✅ Now seems to be working and saving to PDF!! Still using matplotlib.use('Qt5Agg') mode and plots still appear.
    programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_2d_placefield_occupancy', write_vector_format=write_vector_format, write_png=write_png, debug_print=debug_print) #  🟢✅ 2023-05-25
   
    #TODO 2023-06-14 05:30: - [ ] Refactor these (the global placefields) into a form compatible with the local ones using some sort of shortcut method like `programmatic_render_to_file`
    save_figure = (write_vector_format or write_png)

    # Plot long|short firing rate index:
    try:
        _out = curr_active_pipeline.display('_display_short_long_firing_rate_index_comparison', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _display_short_long_firing_rate_index_comparison failed with error: {e}\n skipping.')
    
    try:
        _out = curr_active_pipeline.display('_display_long_short_expected_v_observed_firing_rate', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _prepare_plot_expected_vs_observed failed with error: {e}\n skipping.')
    
    try:
        _out = curr_active_pipeline.display('_display_grid_bin_bounds_validation', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _display_grid_bin_bounds_validation failed with error: {e}\n skipping.')
        
    try:
        _out = curr_active_pipeline.display('_display_running_and_replay_speeds_over_time', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _display_running_and_replay_speeds_over_time failed with error: {e}\n skipping.')
        
    try:
        _out = curr_active_pipeline.display('_display_long_short_laps', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure) # batch_extended_programmatic_figures(...): _display_long_short_laps failed with error: plot_position_curves_figure() got an unexpected keyword argument 'active_config_name'
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _display_long_short_laps failed with error: {e}\n skipping.')
        
    try:
        _out = curr_active_pipeline.display('_display_long_and_short_firing_rate_replays_v_laps', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _display_long_and_short_firing_rate_replays_v_laps failed with error: {e}\n skipping.')
                
    try:
        long_single_cell_pfmap_processing_fn = None
        short_single_cell_pfmap_processing_fn = None
        sort_idx = 'peak_long'
        _out = curr_active_pipeline.display('_display_long_short_pf1D_comparison', curr_active_pipeline.get_session_context(), single_figure=False, debug_print=False, fignum='Short v Long pf1D Comparison',
                                   long_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': long_single_cell_pfmap_processing_fn},
                                   short_kwargs={'sortby': sort_idx, 'single_cell_pfmap_processing_fn': short_single_cell_pfmap_processing_fn, 'curve_hatch_style': {'hatch':'///', 'edgecolor':'k'}},
                                   save_figure=save_figure) # defer_render=True, 
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _display_long_short_pf1D_comparison failed with error: {e}\n skipping.')

    ## TODO 2023-06-02 NOW, NEXT: this might not work in 'AGG' mode because it tries to render it with QT, but we can see.
    try:
        #TODO 2023-07-06 14:46: - [ ] This is quite slow - can I do defer_render=True?
        _out = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', curr_active_pipeline.get_session_context(), defer_render=False, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): _prepare_plot_long_and_short_epochs failed with error: {e}\n skipping.')
        
    ## Exports the video of the surprise:
    try:
        # Relative Entropy/Surprise Results:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        active_relative_entropy_results = global_results['extended_stats']['pf_dt_sequential_surprise'] # DynamicParameters
        historical_snapshots = active_relative_entropy_results['historical_snapshots']
        ## Get the placefield dt matrix:
        if 'snapshot_occupancy_weighted_tuning_maps' not in active_relative_entropy_results:
            ## Compute it if missing:
            occupancy_weighted_tuning_maps_over_time = np.stack([placefield_snapshot.occupancy_weighted_tuning_maps_matrix for placefield_snapshot in historical_snapshots.values()])
            active_relative_entropy_results['snapshot_occupancy_weighted_tuning_maps'] = occupancy_weighted_tuning_maps_over_time
        else:
            occupancy_weighted_tuning_maps_over_time = active_relative_entropy_results['snapshot_occupancy_weighted_tuning_maps'] # (n_post_update_times, n_neurons, n_xbins)

        video_output_parent_path = export_active_relative_entropy_results_videos(active_relative_entropy_results, active_context=curr_active_pipeline.get_session_context())

    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): export_active_relative_entropy_results_videos failed with error: {e}\n skipping.')


    ## Exports all cells to the `BatchPhoJonathanFiguresHelper` but without the top part:
    try:
        # PhoJonathan Results:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        jonathan_firing_rate_analysis_result = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
        neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions(frs_index_inclusion_magnitude=0.2)
        ## all cells:
        fig_1c_figures_all_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=None, n_max_page_rows=20, write_vector_format=False, write_png=True, show_only_refined_cells=False, disable_top_row=True)
    
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): BatchPhoJonathanFiguresHelper.run(...) failed for all cells. failed with error: {e}\n skipping.')
        

    # _display_directional_merged_pf_decoded_epochs ______________________________________________________________________ #
    # this open the Yellow-Blue plots and various marginals
    try:
        # # Interactive-mode parameters:
        # _interactive_mode_kwargs = dict(should_use_MatplotlibTimeSynchronizedWidget=True, scrollable_figure=True, defer_render=False)
        # _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
        # _curr_interaction_mode_kwargs = _interactive_mode_kwargs # interactive mode

        # Non-interactive:
        _non_interactive_mode_kwargs = dict(should_use_MatplotlibTimeSynchronizedWidget=False, scrollable_figure=False, defer_render=True)
        # _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=False, backend='AGG')
        _curr_interaction_mode_kwargs = _non_interactive_mode_kwargs # non-interactive mode

        _out = curr_active_pipeline.display('_display_directional_merged_pf_decoded_epochs', curr_active_pipeline.get_session_context(),
                    max_num_lap_epochs = 240, max_num_ripple_epochs = 500,
                    render_directional_marginal_laps=True, render_directional_marginal_ripples=True, render_track_identity_marginal_laps=True, render_track_identity_marginal_ripples=True,
                    # render_directional_marginal_laps=True, render_directional_marginal_ripples=False, render_track_identity_marginal_laps=False, render_track_identity_marginal_ripples=False,
                    # constrained_layout=True, # layout='none',
                    # build_fn='basic_view', constrained_layout=True, 
                    build_fn='insets_view', constrained_layout=None, layout='none', # , constrained_layout=False constrained_layout=None, layout='none', # , constrained_layout=None, layout='none' extrodinarily fast
                    **_curr_interaction_mode_kwargs, # interactive mode
                    skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, save_figure=save_figure)
        
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): "_display_directional_merged_pf_decoded_epochs" failed with error: {e}\n skipping.')

    # _display_directional_merged_pf_decoded_epochs_marginals ________________________________________________________________________________ #
    try:
        _out = curr_active_pipeline.display('_display_directional_merged_pf_decoded_epochs_marginals', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): "_display_directional_merged_pf_decoded_epochs_marginals" failed with error: {e}\n skipping.')



    # _display_rank_order_z_stats_results ________________________________________________________________________________ #
    try:
        _out = curr_active_pipeline.display('_display_rank_order_z_stats_results', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): "_display_rank_order_z_stats_results" failed with error: {e}\n skipping.')


    try:
        _out = curr_active_pipeline.display('_display_directional_track_remapping_diagram', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure, is_dark_mode=False)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): "_display_directional_track_remapping_diagram" failed with error: {e}\n skipping.')


    try:
        _out = curr_active_pipeline.display('_display_trial_to_trial_reliability', curr_active_pipeline.get_session_context(), defer_render=True, save_figure=save_figure, is_dark_mode=False)
    except BaseException as e:
        print(f'batch_extended_programmatic_figures(...): "_display_trial_to_trial_reliability" failed with error: {e}\n skipping.')





@metadata_attributes(short_name=None, tags=['PhoJonathan', 'figure', 'batch', 'LxC', 'SxC', 'LongShort'], input_requires=[], output_provides=[], uses=[], used_by=['batch_programmatic_figures', 'batch_extended_programmatic_figures'],
                     creation_date='2024-09-16 17:34', related_items=['PhoJonathanPlotHelpers'])
class BatchPhoJonathanFiguresHelper:
    """Private methods that help with batch figure generator for PhoJonathanPlotHelpers.

    In .run(...) it builds the plot_kwargs (`active_kwarg_list`) ahead of time that will be passed to the specific plot function using `cls._build_batch_plot_kwargs(...)`
        It then calls `active_out_figures_list = cls._perform_batch_plot(...)` to do the plotting, getting the list of figures and output paths
            -> `cls._subfn_batch_plot_automated()`

            Ultimately, it calls the display function '_display_batch_pho_jonathan_replay_firing_rate_comparison'


    2022-12-08 - Batch Programmatic Figures (Currently only Jonathan-style) 
    2022-12-01 - Automated programmatic output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`

    Currently meant to generate and save figures in the background.
    
    
    """
    _display_fn_name = '_display_batch_pho_jonathan_replay_firing_rate_comparison' # used as the display function called in `_subfn_batch_plot_automated(...)`
    _display_fn_context_display_name = 'BatchPhoJonathanReplayFRC' # used in `_build_batch_plot_kwargs` as the display_fn_name for the generated context. Affects the output names of the figures like f'kdiba_gor01_one_2006-6-09_1-22-43_{cls._display_fn_context_display_name}_long_only_[5, 23, 29, 38, 70, 85, 97, 103].pdf'. 


    @classmethod
    def run(cls, curr_active_pipeline, neuron_replay_stats_df, included_unit_neuron_IDs=None, n_max_page_rows=10, write_vector_format=False, write_png=True, progress_print=True, debug_print=False,
             show_only_refined_cells:bool=False, disable_top_row=False, split_by_short_long_shared: bool = True):
        """ The only public function. Performs the batch plotting.
        
        # split_by_short_long_shared, bool: whether to create separate figures for the short/long exclusive cells and the shared. If False all will be treated as "shared"
        
        """
        if included_unit_neuron_IDs is not None:
            ## pre-filter the `neuron_replay_stats_df` by the included_unit_neuron_IDs only:
            neuron_replay_stats_df = neuron_replay_stats_df[np.isin(neuron_replay_stats_df.index, included_unit_neuron_IDs)]
            
        ## 🗨️🟢 2022-11-05 - Pho-Jonathan Batch Outputs of Firing Rate Figures
        # %matplotlib qt

         
        if split_by_short_long_shared:
            ## 2023-09-28 - Refined Outputs
            if show_only_refined_cells:
                short_only_df = neuron_replay_stats_df[neuron_replay_stats_df['is_refined_SxC']]
                short_only_aclus = short_only_df.index.values.tolist()
                long_only_df = neuron_replay_stats_df[neuron_replay_stats_df['is_refined_LxC']]
                long_only_aclus = long_only_df.index.values.tolist()

                ## Note that the above ("refined" LxC and SxC) omit long_only and short_only place cells that don't meant the "exclusive" criteria. That means these cells will be omitted entirely (and not even included in the shared)
                # shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
                # shared_aclus = shared_df.index.values.tolist()
                shared_aclus = [] # skip shared aclus for this mode
                print(f"WARNING: 2023-09-28 - Note that the above ('refined' LxC and SxC) omit long_only and short_only place cells that don't meant the 'exclusive' criteria. That means these cells will be omitted entirely (and not even included in the shared)")

            else:
                # original (non-refined) mode
                short_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.RIGHT_ONLY]
                short_only_aclus = short_only_df.index.values.tolist()
                long_only_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.LEFT_ONLY]
                long_only_aclus = long_only_df.index.values.tolist()
                shared_df = neuron_replay_stats_df[neuron_replay_stats_df.track_membership == SplitPartitionMembership.SHARED]
                shared_aclus = shared_df.index.values.tolist()
        else:
            # don't split the short/long/shared, treat all as shared.
            short_only_aclus = []
            long_only_aclus = []
            shared_df = neuron_replay_stats_df.copy()
            shared_aclus = shared_df.index.values.tolist()


        if debug_print:
            print(f'shared_aclus: {shared_aclus}')
            print(f'long_only_aclus: {long_only_aclus}')
            print(f'short_only_aclus: {short_only_aclus}')

        active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        
        _batch_plot_kwargs_list = cls._build_batch_plot_kwargs(long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=n_max_page_rows, _extra_kwargs=dict(disable_top_row=disable_top_row))
        active_out_figures_dict = cls._perform_batch_plot(curr_active_pipeline, _batch_plot_kwargs_list, write_vector_format=write_vector_format, write_png=write_png, progress_print=progress_print, debug_print=debug_print)
        
        # ## find the output figures from the `curr_active_pipeline.registered_output_files`
        # exports_any_figures_to_disk: bool = (write_png or write_vector_format and (len(active_out_figures_dict)>0))
        # if exports_any_figures_to_disk:
        #     _found_contexts_dict: Dict[IdentifyingContext, Path] = {}
        #     for a_figure_path, an_output_dict in curr_active_pipeline.registered_output_files.items():
        #         a_ctxt = an_output_dict['context']
        #         _found_contexts_dict[a_ctxt] = a_figure_path

        #     relevant_figures_dict: Dict[IdentifyingContext, Path] = IdentifyingContext.matching(_found_contexts_dict, criteria={'display_fn_name': 'BatchPhoJonathanReplayFRC'})
        #     relevant_figures_dict

        return active_out_figures_dict

    @classmethod
    def _subfn_batch_plot_automated(cls, curr_active_pipeline, included_unit_neuron_IDs=None, active_identifying_ctx=None, fignum=None, fig_idx=0, n_max_page_rows=10, disable_top_row=False):
        """ the a programmatic wrapper for automated output using `_display_batch_pho_jonathan_replay_firing_rate_comparison`. The specific plot function called. 
        Called ONLY by `_perform_batch_plot(...)`

        Calls `curr_active_pipeline.display(cls._display_fn_name, ...)
        """
        # size_dpi = 100.0,
        # single_subfigure_size_px = np.array([1920.0, 220.0])
        single_subfigure_size_inches = np.array([19.2,  2.2])

        num_cells = len(included_unit_neuron_IDs or [])
        desired_figure_size_inches = single_subfigure_size_inches.copy()
        desired_figure_size_inches[1] = desired_figure_size_inches[1] * num_cells
        graphics_output_dict = curr_active_pipeline.display(cls._display_fn_name, active_identifying_ctx,
                                                            n_max_plot_rows=n_max_page_rows, included_unit_neuron_IDs=included_unit_neuron_IDs,
                                                            show_inter_replay_frs=True, spikes_color=(0.1, 0.0, 0.1), spikes_alpha=0.5, fignum=fignum, fig_idx=fig_idx, figsize=desired_figure_size_inches, save_figure=False, defer_render=True, disable_top_row=disable_top_row) # save_figure will be false because we're saving afterwards
        # fig, subfigs, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['subfigs'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
        fig, subfigs, axs, plot_data = graphics_output_dict.figures[0], graphics_output_dict.subfigs, graphics_output_dict.axes, graphics_output_dict.plot_data
        fig.suptitle(active_identifying_ctx.get_description()) # 'kdiba_2006-6-08_14-26-15_[4, 13, 36, 58, 60]'
        return fig

    @classmethod
    def _build_batch_plot_kwargs(cls, long_only_aclus, short_only_aclus, shared_aclus, active_identifying_session_ctx, n_max_page_rows=10, _extra_kwargs=None):
        """ builds the list of kwargs for all figures (and their aclus). """
        _batch_plot_kwargs_list = [] # empty list to start

        _extra_kwargs = _extra_kwargs or {}
        
        # _aclu_indicies_list_str_formatter = lambda a_list: ('[' + ','.join(map(str, a_list)) + ']') # Prints an array of integer aclu indicies without spaces and comma separated, surrounded by hard brackets. e.g. '[5,6,7,8,11,12,15,17,18,19,20,21,24,25,26,27,28,31,34,35,39,40,41,43,44,45,48,49,50,51,52,53,55,56,60,62,63,64,65]'
        _aclu_indicies_list_str_formatter = lambda a_list: ('(' + ','.join(map(str, a_list)) + ')')

        ## {long_only, short_only} plot configs (doesn't include the shared_aclus)
        if len(long_only_aclus) > 0:        
            _batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=long_only_aclus,
            active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
                display_fn_name=cls._display_fn_context_display_name, plot_result_set='long_only', aclus=f"{_aclu_indicies_list_str_formatter(long_only_aclus)}"
            ),
            fignum='long_only', n_max_page_rows=len(long_only_aclus), **_extra_kwargs))
        else:
            print(f'WARNING: long_only_aclus is empty, so not adding kwargs for these.')
        
        if len(short_only_aclus) > 0:
            _batch_plot_kwargs_list.append(dict(included_unit_neuron_IDs=short_only_aclus,
            active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test',
                display_fn_name=cls._display_fn_context_display_name, plot_result_set='short_only', aclus=f"{_aclu_indicies_list_str_formatter(short_only_aclus)}"
            ),
            fignum='short_only', n_max_page_rows=len(short_only_aclus), **_extra_kwargs))
        else:
            print(f'WARNING: short_only_aclus is empty, so not adding kwargs for these.')

        ## Build Pages for Shared ACLUS:    
        nAclusToShow = len(shared_aclus)
        if nAclusToShow > 0:        
            # Paging Management: Constrain the subplots values to just those that you need
            subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nAclusToShow, max_num_columns=1, max_subplots_per_page=n_max_page_rows, data_indicies=shared_aclus, last_figure_subplots_same_layout=True)
            num_pages = len(included_combined_indicies_pages)
            ## paginated outputs for shared cells
            included_unit_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items
            paginated_shared_cells_kwarg_list = [dict(included_unit_neuron_IDs=curr_included_unit_indicies,
                active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test', display_fn_name=cls._display_fn_context_display_name, plot_result_set='shared', page=f'{page_idx+1}of{num_pages}', aclus=f"{_aclu_indicies_list_str_formatter(curr_included_unit_indicies)}"),
                fignum=f'shared_{page_idx}', fig_idx=page_idx, n_max_page_rows=n_max_page_rows, **_extra_kwargs) for page_idx, curr_included_unit_indicies in enumerate(included_unit_indicies_pages)]
            _batch_plot_kwargs_list.extend(paginated_shared_cells_kwarg_list) # add paginated_shared_cells_kwarg_list to the list
        else:
            print(f'WARNING: shared_aclus is empty, so not adding kwargs for these.')
        return _batch_plot_kwargs_list

    @classmethod
    def _perform_batch_plot(cls, curr_active_pipeline, active_kwarg_list, subset_includelist=None, subset_excludelist=None, write_vector_format=False, write_png=True, progress_print=True, debug_print=False):
        """ Plots everything by calling `cls._subfn_batch_plot_automated` using the kwargs provided in `active_kwarg_list`

        Args:
            active_kwarg_list (_type_): generated by `_build_batch_plot_kwargs(...)`
            figures_parent_out_path (_type_, optional): _description_. Defaults to None.
            write_vector_format (bool, optional): _description_. Defaults to False.
            write_png (bool, optional): _description_. Defaults to True.
            progress_print (bool, optional): _description_. Defaults to True.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # if figures_parent_out_path is None:
        #     figures_parent_out_path = create_daily_programmatic_display_function_testing_folder_if_needed()

        # 2023-06-14 21:53: - [X] REPLACE `create_daily_programmatic_display_function_testing_folder_if_needed` with `curr_active_pipeline.get_figure_manager()` approach
        fig_man = curr_active_pipeline.get_output_manager()

        active_out_figures_dict = {} # empty dict to hold figures
        num_pages = len(active_kwarg_list)

        # Each figure:
        for i, curr_batch_plot_kwargs in enumerate(active_kwarg_list):
            curr_active_identifying_ctx = curr_batch_plot_kwargs['active_identifying_ctx'] # this context is good
            
            ## 2023-06-14 - New way using `fig_man.get_figure_save_file_path(...)` - this is correct
            final_figure_file_path = fig_man.get_figure_save_file_path(curr_active_identifying_ctx, make_folder_if_needed=True) # complete file path without extension: e.g. '/ProgrammaticDisplayFunctionTesting/2023-06-15/kdiba_pin01_one_11-03_12-3-25_BatchPhoJonathanReplayFRC_short_only_[13, 22, 28]'

            # Perform the plotting:
            a_fig = cls._subfn_batch_plot_automated(curr_active_pipeline, **curr_batch_plot_kwargs)
            active_out_figures_dict[curr_active_identifying_ctx] = a_fig

            # One plot at a time to PDF:
            if write_vector_format:
                active_pdf_metadata, _UNUSED_pdf_save_filename = build_pdf_metadata_from_display_context(curr_active_identifying_ctx, subset_includelist=subset_includelist, subset_excludelist=subset_excludelist)
                curr_pdf_save_path = final_figure_file_path.with_suffix('.pdf')
                with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=active_pdf_metadata) as pdf:
                    # Save out PDF page:
                    pdf.savefig(a_fig)
                    curr_active_pipeline.register_output_file(output_path=curr_pdf_save_path, output_metadata={'context': curr_active_identifying_ctx, 'fig': (a_fig), 'pdf_metadata': active_pdf_metadata})
                    if progress_print:
                        print(f'\t saved {curr_pdf_save_path}')


            # Also save .png versions:
            if write_png:
                # curr_page_str = f'pg{i+1}of{num_pages}'
                fig_png_out_path = final_figure_file_path.with_suffix('.png')
                # fig_png_out_path = fig_png_out_path.with_stem(f'{curr_pdf_save_path.stem}_{curr_page_str}') # note this replaces the current .pdf extension with .png, resulting in a good filename for a .png
                a_fig.savefig(fig_png_out_path)
                curr_active_pipeline.register_output_file(output_path=fig_png_out_path, output_metadata={'context': curr_active_identifying_ctx, 'fig': (a_fig)})
                if progress_print:
                    print(f'\t saved {fig_png_out_path}')


        return active_out_figures_dict


# ==================================================================================================================== #
# Main Public Plot Function                                                                                            #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['active', 'batch', 'public', 'neptune', 'figures', 'output'], input_requires=[], output_provides=[],
                      uses=['batch_programmatic_figures', 'batch_extended_programmatic_figures'], used_by=[], creation_date='2023-06-12 14:23', related_items=[])
def batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None):
    """ 2023-05-25 - Performs all the batch plotting commands. 
    
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots
    
    """
    try:
        from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, Neptuner
        if neptuner is None:
            neptuner = Neptuner.init_with_pipeline(curr_active_pipeline) 
    except BaseException as e:
        print(f'in `batch_perform_all_plots(...)`: Neptuner.init_with_pipeline(curr_active_pipeline)(...) failed with exception: {e}. Continuing but disabling neptune.')
        enable_neptune = False
        print(f'\t since neptune setup failed, neptune will be disabled going forward.')
        neptuner = None
         
    curr_active_pipeline.reload_default_display_functions()
    try:
        active_identifying_session_ctx, active_out_figures_dict = batch_programmatic_figures(curr_active_pipeline)
    except BaseException as e:
        print(f'in `batch_perform_all_plots(...)`: batch_programmatic_figures(...) failed with exception: {e}. Continuing.')
    
    try:
        batch_extended_programmatic_figures(curr_active_pipeline=curr_active_pipeline)
    except BaseException as e:
        print(f'in `batch_perform_all_plots(...)`: batch_extended_programmatic_figures(...) failed with exception: {e}. Continuing.')
    
    if enable_neptune:
        try:
            succeeded_fig_paths, failed_fig_paths = neptuner.upload_figures(curr_active_pipeline)
            # neptune_output_figures(curr_active_pipeline)
        except BaseException as e:
            print(f'in `batch_perform_all_plots(...)`: neptune_output_figures(...) failed with exception: {e}. Continuing.')
        finally:
            neptuner.stop()
        neptuner.stop()
        
    return neptuner
    





# ==================================================================================================================== #
# 2023-05-25 - Pipeline Preprocessing Parameter Saving                                                                 #
# ==================================================================================================================== #

# from neuropy.utils.dynamic_container import DynamicContainer
# from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

# ==================================================================================================================== #
# 2023-05-25 - Separate Generalized Plot Saving/Registering Function                                                   #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context

                        

#TODO 2023-07-11 01:52: - [ ] Batch Load (No compute allowed)

