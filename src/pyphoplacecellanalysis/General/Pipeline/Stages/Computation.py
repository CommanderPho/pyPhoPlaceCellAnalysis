from collections import OrderedDict
import sys
import typing
from typing import Optional
from warnings import warn
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum # for EvaluationActions


# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters # to replace simple PlacefieldComputationParameters, `load_pickled_global_computation_results`
from pyphocorehelpers.function_helpers import compose_functions, compose_functions_with_error_handling

from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import LoadableInput, LoadableSessionInput, LoadedPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData # used for `load_pickled_global_computation_results`
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData # used for `save_global_computation_results`
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

import pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions
# from General.Pipeline.Stages.ComputationFunctions import ComputationFunctionRegistryHolder # should include ComputationFunctionRegistryHolder and all specifics
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _wrap_multi_context_computation_function

from pyphocorehelpers.print_helpers import CapturedException # used in _execute_computation_functions for error handling



class EvaluationActions(Enum):
    """An enum specifying the available commands that can be performed in the ComputedPipelineStage in regards to computations. Allows generalizing a previously confusing set of functions."""
    EVALUATE_COMPUTATIONS = "evaluate_computations" # replaces .evaluate_computations_for_single_params(...)
    RUN_SPECIFIC = "run_specific" # replaces .run_specific_computations(...)
    RERUN_FAILED = "rerun_failed" # replaces .rerun_failed_computations(...)


class FunctionsSearchMode(Enum):
    """An enum specifying which of the registered functions should be returned."""
    GLOBAL_ONLY = "global_functions_only"
    NON_GLOBAL_ONLY = "non_global_functions_only" 
    ANY = "any_functions" 

    @classmethod
    def initFromIsGlobal(cls, is_global):
        if is_global is None:
            return cls.ANY
        else:
            assert isinstance(is_global, bool)
            if is_global:
                return cls.GLOBAL_ONLY
            else:
                return cls.NON_GLOBAL_ONLY

# ==================================================================================================================== #
# PIPELINE STAGE                                                                                                       #
# ==================================================================================================================== #
class ComputedPipelineStage(LoadableInput, LoadableSessionInput, FilterablePipelineStage, BaseNeuropyPipelineStage):
    """Docstring for ComputedPipelineStage.

    global_comparison_results has keys of type IdentifyingContext
    """
    identity: PipelineStage = PipelineStage.Computed
    filtered_sessions: Optional[DynamicParameters] = None
    filtered_epochs: Optional[DynamicParameters] = None
    filtered_contexts: Optional[DynamicParameters] = None
    active_configs: Optional[DynamicParameters] = None
    computation_results: Optional[DynamicParameters] = None
    global_computation_results: Optional[ComputationResult] = None

    def __init__(self, loaded_stage: LoadedPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = loaded_stage.stage_name
        self.basedir = loaded_stage.basedir
        self.loaded_data = loaded_stage.loaded_data

        # Initialize custom fields:
        self.filtered_sessions = DynamicParameters()
        self.filtered_epochs = DynamicParameters()
        self.filtered_contexts = DynamicParameters()
        self.active_configs = DynamicParameters() # active_config corresponding to each filtered session/epoch
        self.computation_results = DynamicParameters() # computation_results is a DynamicParameters with keys of type IdentifyingContext and values of type ComputationResult
        self.global_computation_results = ComputedPipelineStage._build_initial_computationResult(self.sess, None) # proper type setup

        self.registered_computation_function_dict = OrderedDict()
        self.registered_global_computation_function_dict = OrderedDict()
        self.reload_default_computation_functions() # registers the default

    @property
    def active_completed_computation_result_names(self):
        """The this list of all computed configs."""
        return self._get_computation_results_progress()[0] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict

    @property
    def active_incomplete_computation_result_status_dicts(self):
        """The this dict containing all the incompletely computed configs and their reason for being incomplete."""
        return self._get_computation_results_progress()[1] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict

    @property
    def registered_computation_functions(self):
        return list(self.registered_computation_function_dict.values())
    @property
    def registered_computation_function_names(self):
        return list(self.registered_computation_function_dict.keys()) 


    @property
    def registered_global_computation_functions(self):
        return list(self.registered_global_computation_function_dict.values())
    @property
    def registered_global_computation_function_names(self):
        return list(self.registered_global_computation_function_dict.keys()) 


    # 'merged' refers to the fact that both global and non-global computation functions are included _____________________ #
    @property
    def registered_merged_computation_function_dict(self):
        """build a merged function dictionary containing both global and non-global functions:"""
        return (self.registered_global_computation_function_dict | self.registered_computation_function_dict)
    @property
    def registered_merged_computation_functions(self):
        return list(self.registered_merged_computation_function_dict.values())
    @property
    def registered_merged_computation_function_names(self):
        return list(self.registered_merged_computation_function_dict.keys()) 


    def reload_default_computation_functions(self):
        """ reloads/re-registers the default display functions after adding a new one
         Note: execution ORDER MATTERS for the computation functions, unlike the display functions, so they need to be enumerated in the correct order and not sorted alphabetically        
        # Sort by precidence:
            _computationPrecidence
        """
        # Non-Global Items:
        for (a_computation_class_name, a_computation_class) in reversed(ComputationFunctionRegistryHolder.get_non_global_registry_items().items()):
            for (a_computation_fn_name, a_computation_fn) in reversed(a_computation_class.get_all_functions(use_definition_order=True)):
                self.register_computation(a_computation_fn_name, a_computation_fn, is_global=False)
        # Global Items:
        for (a_computation_class_name, a_computation_class) in reversed(ComputationFunctionRegistryHolder.get_global_registry_items().items()):
            for (a_computation_fn_name, a_computation_fn) in reversed(a_computation_class.get_all_functions(use_definition_order=True)):
                self.register_computation(a_computation_fn_name, a_computation_fn, is_global=True)

        
    def register_computation(self, registered_name, computation_function, is_global:bool):
        # Set the .is_global attribute on the function object itself, since functions are 1st-class objects in Python:
        computation_function.is_global = is_global

        if is_global:
            try:
                self.registered_global_computation_function_dict[registered_name] = computation_function
            except AttributeError as e:
                # Create a new global dictionary if needed and then try re-register:
                self.registered_global_computation_function_dict = OrderedDict()
                self.registered_global_computation_function_dict[registered_name] = computation_function            
        else:
            # non-global:
            try:
                self.registered_computation_function_dict[registered_name] = computation_function
            except AttributeError as e:
                # Create a new non-global dictionary if needed and then try re-register:
                self.registered_computation_function_dict = OrderedDict()
                self.registered_computation_function_dict[registered_name] = computation_function
        

    def unregister_all_computation_functions(self):
        ## Drops all registered computationf functions (global and non-global) so they can be reloaded fresh:
        self.registered_global_computation_function_dict = OrderedDict()
        self.registered_computation_function_dict = OrderedDict()


    def find_registered_computation_functions(self, registered_names_list, search_mode:FunctionsSearchMode=FunctionsSearchMode.ANY, names_list_is_blacklist:bool=False):
        ''' Finds the list of actual function objects associated with the registered_names_list by using the appropriate dictionary of registered functions depending on whether are_global is True or not.

        registered_names_list: list<str> - a list of function names to be used to fetch the appropriate functions
        are_global: bool - If True, the registered_global_computation_function_dict is used instead of the registered_computation_function_dict
        names_list_is_blacklist: bool - if True, registered_names_list is treated as a blacklist, and all functions are returned EXCEPT those that are in registered_names_list

        Usage:
            active_computation_functions = self.find_registered_computation_functions(computation_functions_name_whitelist, are_global=are_global)
        '''
        # We want to reload the new/modified versions of the functions:
        self.reload_default_computation_functions()

        if search_mode.name == FunctionsSearchMode.GLOBAL_ONLY.name:
            active_registered_computation_function_dict = self.registered_global_computation_function_dict
        elif search_mode.name == FunctionsSearchMode.NON_GLOBAL_ONLY.name:
            active_registered_computation_function_dict = self.registered_computation_function_dict
        elif search_mode.name == FunctionsSearchMode.ANY.name:
            # build a merged function dictionary containing both global and non-global functions:
            merged_computation_function_dict = (self.registered_global_computation_function_dict | self.registered_computation_function_dict)
            active_registered_computation_function_dict = merged_computation_function_dict
        else:
            raise NotImplementedError

        if names_list_is_blacklist:
            # blacklist-style operation: treat the registered_names_list as a blacklist and return all registered functions EXCEPT those that are in registered_names_list
            active_computation_function_dict = {a_computation_fn_name:a_computation_fn for (a_computation_fn_name, a_computation_fn) in active_registered_computation_function_dict.items() if a_computation_fn_name not in registered_names_list}
        else:
            # default whitelist-style operation:
            active_computation_function_dict = {a_computation_fn_name:a_computation_fn for (a_computation_fn_name, a_computation_fn) in active_registered_computation_function_dict.items() if a_computation_fn_name in registered_names_list}
        return list(active_computation_function_dict.values())
        


    # ==================================================================================================================== #
    # Specific Context Computation Helpers                                                                                 #
    # ==================================================================================================================== #
    def perform_registered_computations_single_context(self, previous_computation_result=None, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ Executes all registered computations for a single filter
        
        The return value should be set to the self.computation_results[a_select_config_name]
        """
        search_mode=FunctionsSearchMode.ANY
        # can also do:
        # search_mode=FunctionsSearchMode.initFromIsGlobal(are_global)

        # Need to exclude any computation functions specified in omitted_computation_functions_dict
        if computation_functions_name_whitelist is not None:
            active_computation_functions = self.find_registered_computation_functions(computation_functions_name_whitelist, search_mode=search_mode, names_list_is_blacklist=False)
            print(f'due to whitelist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')

        elif computation_functions_name_blacklist is not None:
            active_computation_functions = self.find_registered_computation_functions(computation_functions_name_whitelist, search_mode=search_mode, names_list_is_blacklist=True)
            print(f'due to blacklist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')
            # TODO: do something about the previous_computation_result?
        else:
            # Both are None:            
            if are_global:
                active_computation_functions = self.registered_global_computation_functions
            else:
                active_computation_functions = self.registered_computation_functions

        # Perform the computations:
        return ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result=previous_computation_result, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)
    
    def rerun_failed_computations_single_context(self, previous_computation_result, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result
        TODO: doesn't yet work with global functions due to relying on self.registered_computation_function_dict
         """
        active_computation_errors = previous_computation_result.accumulated_errors
        # Get potentially updated references to all computation functions that had failed in the previous run of the pipeline:
        potentially_updated_failed_functions = [self.registered_computation_function_dict[failed_computation_fn.__name__] for failed_computation_fn, error in active_computation_errors.items()]
        # Perform the computations:
        return ComputedPipelineStage._execute_computation_functions(potentially_updated_failed_functions, previous_computation_result=previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)

    def run_specific_computations_single_context(self, previous_computation_result, computation_functions_name_whitelist, computation_kwargs_list=None, fail_on_exception:bool=False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ re-runs just a specific computation provided by computation_functions_name_whitelist """
        active_computation_functions = self.find_registered_computation_functions(computation_functions_name_whitelist, search_mode=FunctionsSearchMode.initFromIsGlobal(are_global))
        if progress_logger_callback is not None:
            progress_logger_callback(f'run_specific_computations_single_context(including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions): active_computation_functions: {active_computation_functions}...')
        # Perform the computations:
        return ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

    # ==================================================================================================================== #
    # Other                                                                                                                #
    # ==================================================================================================================== #


    def _get_computation_results_progress(self, debug_print=False):
        """ returns the names of all the configs (usually epochs, like 'maze1' or 'maze2') that have been completely computed
        Returns:
            computed_epochs_list: the names of all the configs (usually epochs, like 'maze1' or 'maze2') that have been computed
        """
        computed_config_names_list = list(self.computation_results.keys()) # ['maze1', 'maze2']
        if debug_print:
            print(f'computed_config_names_list: {computed_config_names_list}') 

        complete_computed_config_names_list = []
        incomplete_computed_config_dict = {}
        for curr_config_name in computed_config_names_list:
            # Try to see if the current config is valid or incomplete
            curr_config_incomplete_reason = None
            active_computation_results = self.computation_results.get(curr_config_name, None)
            if active_computation_results is None:
                curr_config_incomplete_reason = 'MISSING_computation_results'
            else:
                # Check the members:
                active_computed_data = self.computation_results[curr_config_name].computed_data
                if active_computed_data is None:
                    curr_config_incomplete_reason = 'INVALID_computation_results_computed_data'
                active_computation_config = self.computation_results[curr_config_name].computation_config
                if active_computation_config is None:
                    curr_config_incomplete_reason = 'INVALID_computation_results_computation_config'

            if curr_config_incomplete_reason is not None:
                if debug_print:
                    print(f'curr_config_incomplete_reason: {curr_config_incomplete_reason}')
                ## Add the incomplete config to the incomplete dict with its reason for being incomplete:
                incomplete_computed_config_dict[curr_config_name] = curr_config_incomplete_reason
            else:
                complete_computed_config_names_list.append(curr_config_name)
                
        return complete_computed_config_names_list, incomplete_computed_config_dict

    def find_LongShortGlobal_epoch_names(self):
        """ Helper function to returns the [long, short, global] epoch names. They must exist.
        Usage:
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            long_results = curr_active_pipeline.computation_results[long_epoch_name]['computed_data']
            short_results = curr_active_pipeline.computation_results[short_epoch_name]['computed_data']
            global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

        """
        include_whitelist = self.active_completed_computation_result_names # ['maze', 'sprinkle']
        assert (len(include_whitelist) >= 3), "Must have at least 3 completed computation results to find the long, short, and global epoch names."
        long_epoch_name = include_whitelist[0] # 'maze1_PYR'
        short_epoch_name = include_whitelist[1] # 'maze2_PYR'
        global_epoch_name = include_whitelist[-1] # 'maze_PYR'
        return long_epoch_name, short_epoch_name, global_epoch_name


    def rerun_failed_computations(self, enabled_filter_names=None, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result """
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                print(f'Performing rerun_failed_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                previous_computation_result = self.computation_results[a_select_config_name]
                self.computation_results[a_select_config_name] = self.rerun_failed_computations_single_context(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)    


    def perform_action_for_all_contexts(self, action: EvaluationActions, enabled_filter_names=None, active_computation_params: Optional[DynamicParameters]=None, overwrite_extant_results=False, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None,
                                                 fail_on_exception:bool=False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ Aims to generalize the `evaluate_computations_for_single_params(...)` function's functionality (such as looping over each context and passing/updating appropriate results, to all three of the computation functions:


        'single' here refers to the fact that it evaluates only one of the active_computation_params
        Takes its filtered_session and applies the provided active_computation_params to it. The results are stored in self.computation_results under the same key as the filtered session. 
        Called only by the pipeline's .perform_computations(...) function
        
        History:
            created to generalize the `stage.evaluate_computations_for_single_params(...)` function
        """
        assert (len(self.filtered_sessions.keys()) > 0), "Must have at least one filtered session before calling evaluate_computations_for_single_params(...). Call self.select_filters(...) first."
        # self.active_computation_results = dict()
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified

        if are_global:
            active_computation_results = self.global_computation_results
        else:
            active_computation_results = self.computation_results

        ## Here's where we loop through all possible configs:
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                if debug_print:
                    print(f'Performing perform_action_for_all_contexts with action {action} on filtered_session with filter named "{a_select_config_name}"...')
                if progress_logger_callback is not None:
                    progress_logger_callback(f'Performing perform_action_for_all_contexts with action {action} on filtered_session with filter named "{a_select_config_name}"...')
                
                # TODO 2023-01-15 - ASSUMES 1:1 correspondence between self.filtered_sessions's config names and computation_configs:
                if active_computation_params is None:
                    active_computation_params = self.active_configs[a_select_config_name].computation_config # get the previously set computation configs
                else:
                    # set/update the computation configs:
                    self.active_configs[a_select_config_name].computation_config = active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.
                
                if action.name == EvaluationActions.EVALUATE_COMPUTATIONS.name:
                    # active_function = self.perform_registered_computations_single_context
                    skip_computations_for_this_result = False
                    if overwrite_extant_results or (active_computation_results.get(a_select_config_name, None) is None):
                        # If we're supposed to overwrite the previous result OR the previous result is already empty/not yet calculated, initialize a new one:
                        active_computation_results[a_select_config_name] = ComputedPipelineStage._build_initial_computationResult(a_filtered_session, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
                        skip_computations_for_this_result = False # need to compute the result
                    else:
                        # Otherwise it already exists and is not None, so don't overwrite it:
                        if progress_logger_callback is not None:
                            progress_logger_callback(f'WARNING: skipping computation because overwrite_extant_results={overwrite_extant_results} and active_computation_results[{a_select_config_name}] already exists and is non-None')
                            progress_logger_callback('\t TODO: this will prevent recomputation even when the blacklist/whitelist or computation function definitions change. Rework so that this is smarter.')
                        
                        print(f'WARNING: skipping computation because overwrite_extant_results={overwrite_extant_results} and active_computation_results[{a_select_config_name}] already exists and is non-None')
                        print('\t TODO: this will prevent recomputation even when the blacklist/whitelist or computation function definitions change. Rework so that this is smarter.')
                        # active_computation_results.setdefault(a_select_config_name, ComputedPipelineStage._build_initial_computationResult(a_filtered_session, active_computation_params)) # returns a computation result. This stores the computation config used to compute it.
                        skip_computations_for_this_result = True

                    if not skip_computations_for_this_result:
                        # call to perform any registered computations:
                        active_computation_results[a_select_config_name] = self.perform_registered_computations_single_context(active_computation_results[a_select_config_name],
                            computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

                elif action.name == EvaluationActions.RUN_SPECIFIC.name:
                    print(f'Performing run_specific_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                    # active_function = self.run_specific_computations_single_context
                    previous_computation_result = active_computation_results[a_select_config_name]
                    active_computation_results[a_select_config_name] = self.run_specific_computations_single_context(previous_computation_result, computation_functions_name_whitelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

                elif action.name == EvaluationActions.RERUN_FAILED.name:
                    # active_function = self.rerun_failed_computations_single_context
                    print(f'Performing rerun_failed_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                    previous_computation_result = active_computation_results[a_select_config_name]
                    active_computation_results[a_select_config_name] = self.rerun_failed_computations_single_context(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)    


                else:
                    active_function = None
                    print(f'ERROR: {action}, {action.name}')
                    raise NotImplementedError

                    # # call to perform any registered computations:
                    # active_computation_results[a_select_config_name] = self.perform_registered_computations_single_context(active_computation_results[a_select_config_name],
                    #     computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)
            else:
                # this filter is excluded from the enabled list, no computations will we performed on it
                if overwrite_extant_results:
                    active_computation_results.pop(a_select_config_name, None) # remove the computation results from previous runs from the dictionary to indicate that it hasn't been computed
                else:
                    # no *additional* computations will be performed on it, but it will be pass through and not removed form the active_computation_results
                    pass

            # Re-apply changes when done:
            print(f'updating computation_results...')
            if are_global:
                self.global_computation_results = active_computation_results
            else:
                self.computation_results = active_computation_results
            print(f'done.')


    def perform_specific_computation(self, active_computation_params=None, enabled_filter_names=None, computation_functions_name_whitelist=None, computation_kwargs_list=None, fail_on_exception:bool=False, debug_print=False):
        """ perform a specific computation (specified in computation_functions_name_whitelist) in a minimally destructive manner using the previously recomputed results:
        Ideally would already have access to the:
        - Previous computation result
        - Previous computation config (the input parameters)


        computation_kwargs_list: Optional<list>: a list of kwargs corresponding to each function name in computation_functions_name_whitelist

        Internally calls: `run_specific_computations_single_context`.

        Updates:
            curr_active_pipeline.computation_results
        """
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified

        if computation_kwargs_list is None:
            computation_kwargs_list = [{} for _ in computation_functions_name_whitelist]
            assert len(computation_kwargs_list) == len(computation_functions_name_whitelist)


        active_computation_functions = self.find_registered_computation_functions(computation_functions_name_whitelist, search_mode=FunctionsSearchMode.ANY) # find_registered_computation_functions is a pipeline.stage property
        contains_any_global_functions = np.any([v.is_global for v in active_computation_functions])
        if contains_any_global_functions:
            assert np.all([v.is_global for v in active_computation_functions]), 'ERROR: cannot mix global and non-global functions in a single call to perform_specific_computation'

            if self.global_computation_results is None or not isinstance(self.global_computation_results, ComputationResult):
                print(f'global_computation_results is None. Building initial global_computation_results...')
                self.global_computation_results = None # clear existing results
                self.global_computation_results = ComputedPipelineStage._build_initial_computationResult(self.sess, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
                

        if contains_any_global_functions:
            # global computation functions:
            if self.global_computation_results is None or not isinstance(self.global_computation_results, ComputationResult):
                print(f'global_computation_results is None. Building initial global_computation_results...')
                self.global_computation_results = None # clear existing results
                self.global_computation_results = ComputedPipelineStage._build_initial_computationResult(self.sess, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
            ## TODO: what is this about?
            previous_computation_result = self.global_computation_results

            ## TODO: ERROR: `owning_pipeline_reference=self` is not CORRECT as self is of type `ComputedPipelineStage` (or `DisplayPipelineStage`) and not `NeuropyPipeline`
                # this has been fine for all the global functions so far because the majority of the properties are defined on the stage anyway, but any pipeline properties will be missing! 
            global_kwargs = dict(owning_pipeline_reference=self, global_computation_results=previous_computation_result, computation_results=self.computation_results, active_configs=self.active_configs, include_whitelist=enabled_filter_names, debug_print=debug_print)

            self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_whitelist=computation_functions_name_whitelist, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print)
        else:
            # Non-global functions:
            for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
                if a_select_config_name in enabled_filter_names:
                    print(f'Performing run_specific_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                    if active_computation_params is None:
                        curr_active_computation_params = self.active_configs[a_select_config_name].computation_config # get the previously set computation configs
                    else:
                        # set/update the computation configs:
                        curr_active_computation_params = active_computation_params 
                        self.active_configs[a_select_config_name].computation_config = curr_active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.

                    ## Here is an issue, we need to get the appropriate computation result depending on whether it's global or not 
                    previous_computation_result = self.computation_results[a_select_config_name]
                    self.computation_results[a_select_config_name] = self.run_specific_computations_single_context(previous_computation_result, computation_functions_name_whitelist=computation_functions_name_whitelist, computation_kwargs_list=computation_kwargs_list, are_global=False, fail_on_exception=fail_on_exception, debug_print=debug_print)
        
        ## IMPLEMENTATION FAULT: the global computations/results should not be ran within the filter/config loop. It applies to all config names and should be ran last. Also don't allow mixing local/global functions.


    # ==================================================================================================================== #
    # CLASS/STATIC METHODS                                                                                                 #
    # ==================================================================================================================== #


    @classmethod
    def _build_initial_computationResult(cls, active_session, computation_config) -> ComputationResult:
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): this is the filtered data session
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        # only requires that active_session has the .spikes_df and .position  properties
        output_result = ComputationResult(active_session, computation_config, computed_data=DynamicParameters(), accumulated_errors=DynamicParameters()) # Note that this active_session should be correctly filtered
        
        return output_result

    @staticmethod
    def _execute_computation_functions(active_computation_functions, previous_computation_result=None, computation_kwargs_list=None, fail_on_exception:bool = False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ actually performs the provided computations in active_computation_functions """
        if computation_kwargs_list is None:
            computation_kwargs_list = [{} for _ in active_computation_functions]
        assert len(computation_kwargs_list) == len(active_computation_functions)

        if (len(active_computation_functions) > 0):
            if debug_print:
                print(f'Performing _execute_computation_functions(...) with {len(active_computation_functions)} registered_computation_functions...')
            if progress_logger_callback is not None:
                progress_logger_callback(f'Performing _execute_computation_functions(...) with {len(active_computation_functions)} registered_computation_functions...')


            if are_global:
                assert isinstance(previous_computation_result, (dict, DynamicParameters)), 'ERROR: previous_computation_result must be a dict or DynamicParameters object when are_global=True'
                # global_kwargs = dict(owning_pipeline_reference=self, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False)
                previous_computation_result = list(previous_computation_result.values()) # get the list of values since the global computation functions expects positional arguments
                # Wrap the active functions in the wrapper that extracts their arguments:
                active_computation_functions = [_wrap_multi_context_computation_function(a_global_fcn) for a_global_fcn in active_computation_functions]
            
            if fail_on_exception:
                ## normal version that fails on any exception:
                total_num_funcs = len(active_computation_functions)
                for i, f in enumerate(reversed(active_computation_functions)):
                    if progress_logger_callback is not None:
                        progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
                    previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i])

                # Since there's no error handling, gettin ghere means that there were no accumulated errors
                accumulated_errors = None
            else:
                ## Use exception-tolerant version of function composition (functions are composed left-to-right):
                if progress_logger_callback is not None:
                    error_logger = (lambda x: progress_logger_callback(f'ERROR: {x}'))
                else:
                    error_logger = (lambda x: print(f'ERROR: {x}'))
                accumulated_errors = dict() # empty list for keeping track of exceptions
                total_num_funcs = len(active_computation_functions)
                for i, f in enumerate(reversed(active_computation_functions)):
                    if progress_logger_callback is not None:
                        progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
                    try:
                        temp_result = f(previous_computation_result, **computation_kwargs_list[i]) # evaluate the function 'f' using the result provided from the previous output or the initial input
                    except (TypeError, ValueError, NameError, AttributeError, KeyError, NotImplementedError) as e:
                        exception_info = sys.exc_info()
                        accumulated_errors[f] = CapturedException(e, exception_info, previous_computation_result)
                        # accumulated_errors.append(e) # add the error to the accumulated error array
                        temp_result = previous_computation_result # restore the result from prior to the calculations?
                        # result shouldn't be updated unless there wasn't an error, so it should be fine to move on to the next function
                        if error_logger is not None:
                            error_logger(f'\t Encountered error: {accumulated_errors[f]} continuing.')
                    else:
                        # only if no error occured do we commit the temp_result to result
                        previous_computation_result = temp_result
                        if progress_logger_callback is not None:
                            progress_logger_callback('\t done.')

            
            if debug_print:
                print(f'_execute_computation_functions(...): \n\taccumulated_errors: {accumulated_errors}')
            

            if are_global:
                # Extract the global_computation_results from the returned list for global computations:
                # owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, debug_print=False
                assert isinstance(previous_computation_result, list)
                previous_computation_result = previous_computation_result[1] # get the global_computation_results object
                assert isinstance(previous_computation_result, ComputationResult)

            # Add the function to the computation result:
            previous_computation_result.accumulated_errors = accumulated_errors
            if len(accumulated_errors or {}) > 0:
                if progress_logger_callback is not None:
                    progress_logger_callback(f'WARNING: there were {len(accumulated_errors)} that occurred during computation. Check these out by looking at computation_result.accumulated_errors.')
                    
                warn(f'WARNING: there were {len(accumulated_errors)} that occurred during computation. Check these out by looking at computation_result.accumulated_errors.')
                error_desc_str = f'{len(accumulated_errors or {})} errors.'
            else:
                error_desc_str = f'no errors!'
            
            if progress_logger_callback is not None:
                progress_logger_callback(f'\t all computations complete! (Computed {len(active_computation_functions)} with {error_desc_str}.')
                                
            return previous_computation_result
            
        else:
            if progress_logger_callback is not None:
                progress_logger_callback(f'No registered_computation_functions, skipping extended computations.')
                
            if debug_print:
                print(f'No registered_computation_functions, skipping extended computations.')
            return previous_computation_result # just return the unaltered result
                
    @classmethod    
    def continue_computations_if_needed(cls, curr_active_pipeline, active_computation_params=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, debug_print=False):
        """ continues computations for a pipeline 

            NOTE: TODO: this is not yet implemented.
            Calls perform_specific_context_registered_computations(...) to do the actual comptuations

        
            TODO: the rest of the system can't work until we have a way of associating the previously computed results with the functions that compute them. As it stands we don't know anything about whether a new function was registered after the computations were complete, etc.
                DESIGN GOAL: don't make this too complicated.
        
        Usage:
            continue_computations_if_needed(curr_active_pipeline, active_session_computation_configs[0], overwrite_extant_results=False, computation_functions_name_blacklist=['_perform_spike_burst_detection_computation'], debug_print=True)

        """
        ## First look for incomplete computation results (that have never been computed):
        # curr_incomplete_status_dicts = curr_active_pipeline.active_incomplete_computation_result_status_dicts
        complete_computed_config_names_list, incomplete_computed_config_dict = curr_active_pipeline.stage._get_computation_results_progress()

        for an_incomplete_config_name, a_reason in incomplete_computed_config_dict.items():
            a_filtered_session = curr_active_pipeline.filtered_sessions[an_incomplete_config_name] # get the filtered session
            if active_computation_params is None:
                active_computation_params = curr_active_pipeline.active_configs[an_incomplete_config_name].computation_config # get the previously set computation configs
            else:
                # set/update the computation configs:
                curr_active_pipeline.active_configs[an_incomplete_config_name].computation_config = active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.
            
            if overwrite_extant_results or (curr_active_pipeline.computation_results.get(an_incomplete_config_name, None) is None):
                curr_active_pipeline.computation_results[an_incomplete_config_name] = cls._build_initial_computationResult(a_filtered_session, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
            else:
                # Otherwise it already exists and is not None, so don't overwrite it:
                curr_active_pipeline.computation_results.setdefault(an_incomplete_config_name, cls._build_initial_computationResult(a_filtered_session, active_computation_params)) # returns a computation result. This stores the computation config used to compute it.

            # call to perform any registered computations:
            curr_active_pipeline.computation_results[an_incomplete_config_name] = curr_active_pipeline.perform_specific_context_registered_computations(curr_active_pipeline.computation_results[an_incomplete_config_name], computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, debug_print=debug_print)

            ## TODO: initially compute incomplete_computed_config_dict items...

            ## Next look for previously failed computation results:

            ## Next look for previously complete computation results that lack computations for functions explicitly specified in the whitelist (if provided):

            ## Then look for previously complete computation results that are missing computations that have been registered after they were computed, or that were previously part of the blacklist but now are not:

        
# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
class PipelineWithComputedPipelineStageMixin:
    """ To be added to the pipeline to enable conveninece access ot its pipeline stage post Computed stage. """
    ## Computed Properties:
    @property
    def is_computed(self):
        """The is_computed property. TODO: Needs validation/Testing """
        return (self.can_compute and (self.computation_results is not None) and (len(self.computation_results) > 0))
        # return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage) and (self.computation_results is not None) and (len(self.computation_results) > 0))

    @property
    def can_compute(self):
        """The can_compute property."""
        return (self.last_completed_stage >= PipelineStage.Filtered)

    @property
    def computation_results(self):
        """The computation_results property, accessed through the stage."""
        return self.stage.computation_results

    @property
    def active_completed_computation_result_names(self):
        """The this list of all computed configs."""
        return self.stage.active_completed_computation_result_names
    @property
    def active_incomplete_computation_result_status_dicts(self):
        """The this dict containing all the incompletely computed configs and their reason for being incomplete."""
        return self.stage.active_incomplete_computation_result_status_dicts
    
    @property
    def registered_computation_functions(self):
        """The registered_computation_functions property."""
        return self.stage.registered_computation_functions
        
    @property
    def registered_computation_function_names(self):
        """The registered_computation_function_names property."""
        return self.stage.registered_computation_function_names
    
    @property
    def registered_computation_function_dict(self):
        """The registered_computation_function_dict property can be used to get the corresponding function from the string name."""
        return self.stage.registered_computation_function_dict
    
    @property
    def registered_computation_function_docs_dict(self):
        """Returns the doc strings for each registered computation function. This is taken from their docstring at the start of the function defn, and provides an overview into what the function will do."""
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_computation_function_dict.items()}

    # Global Computation Properties ______________________________________________________________________________________ #
    @property
    def global_computation_results(self):
        """The global_computation_results property, accessed through the stage."""
        return self.stage.global_computation_results
    

    # @property
    # def active_completed_global_computation_result_names(self):
    #     """The this list of all computed configs."""        
    #     # return self.stage._get_valid_global_computation_results_config_names()
    #     return self.stage._get_global_computation_results_progress()[0] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict
    
    # @property
    # def active_incomplete_global_computation_result_status_dicts(self):
    #     """The this dict containing all the incompletely computed configs and their reason for being incomplete."""
    #     return self.stage._get_global_computation_results_progress()[1] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict
    
    @property
    def registered_global_computation_functions(self):
        """The registered_global_computation_functions property."""
        return self.stage.registered_global_computation_functions
        
    @property
    def registered_global_computation_function_names(self):
        """The registered_global_computation_function_names property."""
        return self.stage.registered_global_computation_function_names
    
    @property
    def registered_global_computation_function_dict(self):
        """The registered_global_computation_function_dict property can be used to get the corresponding function from the string name."""
        return self.stage.registered_global_computation_function_dict
    
    @property
    def registered_global_computation_function_docs_dict(self):
        """Returns the doc strings for each registered computation function. This is taken from their docstring at the start of the function defn, and provides an overview into what the function will do."""
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_global_computation_function_dict.items()}
    
    def reload_default_computation_functions(self):
        """ reloads/re-registers the default display functions after adding a new one """
        self.stage.reload_default_computation_functions()
        
    def register_computation(self, registered_name, computation_function, is_global:bool):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_computation(registered_name, computation_function, is_global)

        
    ## Computation Helpers: 
    # perform_computations: The main computation function for the pipeline
    def perform_computations(self, active_computation_params: Optional[DynamicParameters]=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, debug_print=False):
        """The main computation function for the pipeline.

        Internally updates the
            .computation_results


        Args:
            active_computation_params (Optional[DynamicParameters], optional): _description_. Defaults to None.
            enabled_filter_names (_type_, optional): _description_. Defaults to None.
            overwrite_extant_results (bool, optional): _description_. Defaults to False.
            computation_functions_name_whitelist (_type_, optional): _description_. Defaults to None.
            computation_functions_name_blacklist (_type_, optional): _description_. Defaults to None.
            fail_on_exception (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        History:
            perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS
        """
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        progress_logger_callback=(lambda x: self.logger.info(x))

        self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
            computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)
        # self.stage.evaluate_computations_for_single_params(active_computation_params, enabled_filter_names=enabled_filter_names, overwrite_extant_results=overwrite_extant_results,
        #     computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)

        # Global MultiContext computations will be done here:
        if progress_logger_callback is not None:
            progress_logger_callback(f'Performing global computations...')

        ## TODO: BUG: WHY IS THIS CALLED TWICE? Was this supposed to be the global implementation or something?
        # self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
        #     computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=(lambda x: self.logger.info(x)), debug_print=debug_print)
        # self.stage.evaluate_computations_for_single_params(active_computation_params, enabled_filter_names=enabled_filter_names, overwrite_extant_results=overwrite_extant_results,
            # computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=(lambda x: self.logger.info(x)), debug_print=debug_print)


    def rerun_failed_computations(self, previous_computation_result, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result """
        # return self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, ... # TODO: refactor to use new layout
        return self.stage.rerun_failed_computations(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)
    
    def perform_specific_computation(self, active_computation_params=None, enabled_filter_names=None, computation_functions_name_whitelist=None, computation_kwargs_list=None, fail_on_exception:bool=False, debug_print=False):
        """ perform a specific computation (specified in computation_functions_name_whitelist) in a minimally destructive manner using the previously recomputed results:
        Passthrough wrapper to self.stage.perform_specific_computation(...) with the same arguments.

        Updates:
            curr_active_pipeline.computation_results
        """
        # self.stage is of type ComputedPipelineStage
        return self.stage.perform_specific_computation(active_computation_params=active_computation_params, enabled_filter_names=enabled_filter_names, computation_functions_name_whitelist=computation_functions_name_whitelist, computation_kwargs_list=computation_kwargs_list, fail_on_exception=fail_on_exception, debug_print=debug_print)
    
    # Utility/Debugging Functions:
    def perform_drop_entire_computed_config(self, config_names_to_drop = ['maze1_rippleOnly', 'maze2_rippleOnly']):
        """ Loops through all the configs and drops all results of the specified configs
        2022-09-13 - Unfinished 
        2022-10-23 - This seems to drop ALL the computed items for a specified set of configs/contexts, not a specific computed item across configs/contexts        
        """
        # config_names_to_drop
        print(f'_drop_computed_items(config_names_to_drop: {config_names_to_drop}):\n\tpre keys: {list(self.active_configs.keys())}')
        
        for a_config_name in config_names_to_drop:
            a_config_to_drop = self.active_configs.pop(a_config_name, None)
            if a_config_to_drop is not None:
                print(f'\tpreparing to drop: {a_config_name}')
                ## TODO: filtered_sessions, filtered_epochs
                # curr_active_pipeline.active_configs
                # curr_active_pipeline.filtered_contexts[a_config_name]
                _dropped_computation_results = self.computation_results.pop(a_config_name, None)
                a_filter_context_to_drop = self.filtered_contexts.pop(a_config_name, None)
                if a_filter_context_to_drop is not None:
                    _dropped_display_items = self.display_output.pop(a_filter_context_to_drop, None)

            print(f'\t dropped.')
            
        print(f'\tpost keys: {list(self.active_configs.keys())}')

    def perform_drop_computed_result(self, computed_data_keys_to_drop, config_names_whitelist=None, debug_print=False):
        """ Loops through all computed items and drops a specific result across all configs/contexts  
        Inputs:
            computed_data_keys_to_drop: list of specific results to drop for each context
            config_names_whitelist: optional list of names to operate on. No changes will be made to results for configs not in the whitelist
        """
        # config_names_to_drop
        if debug_print:
            print(f'perform_drop_computed_result(computed_data_keys_to_drop: {computed_data_keys_to_drop}, config_names_whitelist: {config_names_whitelist})')

        if config_names_whitelist is None:
            # if no whitelist specified, get all computed keys:
            config_names_whitelist = self.active_completed_computation_result_names # ['maze1_PYR', 'maze2_PYR', 'maze_PYR']
        
        ## Loop across all computed contexts
        for a_config_name, curr_computed_results in self.computation_results.items():
            if a_config_name in config_names_whitelist:            
                # remove the results from this config
                for a_key_to_drop in computed_data_keys_to_drop:
                    a_result_to_drop = curr_computed_results.pop(a_key_to_drop, None)
                    ## TODO: Should we drop from curr_computed_results.accumulated_errors in addition to curr_computed_results.computed_data? Probably fine not to.
                    if a_result_to_drop is not None:
                        # Successfully dropped
                        print(f"\t Dropped computation_results['{a_config_name}'].computed_data['{a_key_to_drop}'].")
                    else:
                        print(f"\t computation_results['{a_config_name}'].computed_data['{a_key_to_drop}'] did not exist.")
                        pass
            else:
                # Otherwise skip it if it isn't in the whitelist
                if debug_print:
                    print(f'skipping {a_config_name} because it is not in the context whitelist.')

    def find_LongShortGlobal_epoch_names(self):
        """ Returns the [long, short, global] epoch names. They must exist.
        Usage:
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            long_results = curr_active_pipeline.computation_results[long_epoch_name]['computed_data']
            short_results = curr_active_pipeline.computation_results[short_epoch_name]['computed_data']
            global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

        """
        return self.stage.find_LongShortGlobal_epoch_names()

        # print(f'\tpost keys: {list(self.active_configs.keys())}')

    def get_output_path(self) -> Path:
        """ returns the appropriate output path to store the outputs for this session. Usually '$session_folder/outputs/' """
        return self.sess.get_output_path()

    @property
    def global_computation_results_pickle_path(self) -> Path:
        """ The path to pickle the global_computation_results """
        return self.get_output_path().joinpath(f'global_computation_results.pkl').resolve()

    ## Global Computation Result Persistance Hacks:
    def save_global_computation_results(self):
        """Save out the `global_computation_results` which are not currently saved with the pipeline
        Usage:
            curr_active_pipeline.save_global_computation_results()
        """
        print(f'global_computation_results_pickle_path: {self.global_computation_results_pickle_path}')
        saveData(self.global_computation_results_pickle_path, (self.global_computation_results.to_dict()))

    def load_pickled_global_computation_results(self, override_global_computation_results_pickle_path=None):
        """ loads the previously pickled `global_computation_results` into `self.global_computation_results`, replacing the current values.
        Usage:
            curr_active_pipeline.load_pickled_global_computation_results()
        """
        if override_global_computation_results_pickle_path is None:
            # Use the default if no override is provided.
            global_computation_results_pickle_path = self.global_computation_results_pickle_path
        else:
            global_computation_results_pickle_path = override_global_computation_results_pickle_path
        loaded_global_computation_results = DynamicParameters(**loadData(global_computation_results_pickle_path))
        self.stage.global_computation_results = loaded_global_computation_results # TODO 2023-05-19 - Merge results instead of replacing. Requires checking parameters.
