from collections import OrderedDict
import sys
import typing
from typing import Optional
from warnings import warn
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters # to replace simple PlacefieldComputationParameters
from pyphocorehelpers.function_helpers import compose_functions, compose_functions_with_error_handling

from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import LoadableInput, LoadableSessionInput, LoadedPipelineStage    
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

import pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions
# from General.Pipeline.Stages.ComputationFunctions import ComputationFunctionRegistryHolder # should include ComputationFunctionRegistryHolder and all specifics
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


# ==================================================================================================================== #
# PIPELINE STAGE                                                                                                       #
# ==================================================================================================================== #
class ComputedPipelineStage(LoadableInput, LoadableSessionInput, FilterablePipelineStage, BaseNeuropyPipelineStage):
    """Docstring for ComputedPipelineStage."""
    identity: PipelineStage = PipelineStage.Computed
    filtered_sessions: Optional[DynamicParameters] = None
    filtered_epochs: Optional[DynamicParameters] = None
    active_configs: Optional[DynamicParameters] = None
    computation_results: Optional[DynamicParameters] = None
    
    def __init__(self, loaded_stage: LoadedPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = loaded_stage.stage_name
        self.basedir = loaded_stage.basedir
        self.loaded_data = loaded_stage.loaded_data

        # Initialize custom fields:
        self.filtered_sessions = DynamicParameters()
        self.filtered_epochs = DynamicParameters()
        self.active_configs = DynamicParameters() # active_config corresponding to each filtered session/epoch
        self.computation_results = DynamicParameters()
        self.global_computation_results = DynamicParameters()

        
        self.registered_computation_function_dict = OrderedDict()
        self.reload_default_computation_functions() # registers the default
        
    @property
    def registered_computation_functions(self):
        """The registered_computation_functions property."""
        return list(self.registered_computation_function_dict.values())

    @property
    def registered_computation_function_names(self):
        """The registered_computation_function_names property."""
        return list(self.registered_computation_function_dict.keys()) 
    
    def reload_default_computation_functions(self):
        """ reloads/re-registers the default display functions after adding a new one
         Note: execution ORDER MATTERS for the computation functions, unlike the display functions, so they need to be enumerated in the correct order and not sorted alphabetically        
        # Sort by precidence:
            _computationPrecidence
        """    
        for (a_computation_class_name, a_computation_class) in reversed(ComputationFunctionRegistryHolder.get_registry().items()):
            for (a_computation_fn_name, a_computation_fn) in reversed(a_computation_class.get_all_functions(use_definition_order=True)):
                self.register_computation(a_computation_fn_name, a_computation_fn)
        
    def register_computation(self, registered_name, computation_function):
        self.registered_computation_function_dict[registered_name] = computation_function
        
    def perform_specific_context_registered_computations(self, previous_computation_result=None, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, progress_logger_callback=None, debug_print=False):
        """ Executes all registered computations for a single filter
        
        The return value should be set to the self.computation_results[a_select_config_name]
        """
        # Need to exclude any computation functions specified in omitted_computation_functions_dict
        if computation_functions_name_whitelist is not None:
            active_computation_function_dict = {a_computation_fn_name:a_computation_fn for (a_computation_fn_name, a_computation_fn) in self.registered_computation_function_dict.items() if a_computation_fn_name in computation_functions_name_whitelist}
            active_computation_functions = list(active_computation_function_dict.values())
            print(f'due to whitelist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')

        elif computation_functions_name_blacklist is not None:
            active_computation_function_dict = {a_computation_fn_name:a_computation_fn for (a_computation_fn_name, a_computation_fn) in self.registered_computation_function_dict.items() if a_computation_fn_name not in computation_functions_name_blacklist}
            active_computation_functions = list(active_computation_function_dict.values())
            print(f'due to blacklist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')
            # TODO: do something about the previous_computation_result?
            
        else:
            active_computation_functions = self.registered_computation_functions
        
        # Perform the computations:
        return ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result=previous_computation_result, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)
    
    def rerun_failed_computations(self, previous_computation_result, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result """
        active_computation_errors = previous_computation_result.accumulated_errors
        # Get potentially updated references to all computation functions that had failed in the previous run of the pipeline:
        potentially_updated_failed_functions = [self.registered_computation_function_dict[failed_computation_fn.__name__] for failed_computation_fn, error in active_computation_errors.items()]
        # Perform the computations:
        return ComputedPipelineStage._execute_computation_functions(potentially_updated_failed_functions, previous_computation_result=previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)
        
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
    
    def evaluate_computations_for_single_params(self, active_computation_params: Optional[DynamicParameters]=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None,
                                                 fail_on_exception:bool=False, progress_logger_callback=None, debug_print=False):
        """ 'single' here refers to the fact that it evaluates only one of the active_computation_params
        
        Takes its filtered_session and applies the provided active_computation_params to it. The results are stored in self.computation_results under the same key as the filtered session. 
        
        Called only by the pipeline's .perform_computations(...) function
        
        """
        assert (len(self.filtered_sessions.keys()) > 0), "Must have at least one filtered session before calling evaluate_computations_for_single_params(...). Call self.select_filters(...) first."
        # self.active_computation_results = dict()
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified

        ## Here's where we loop through all possible configs:
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                print(f'Performing evaluate_computations_for_single_params on filtered_session with filter named "{a_select_config_name}"...')
                if progress_logger_callback is not None:
                    progress_logger_callback(f'Performing evaluate_computations_for_single_params on filtered_session with filter named "{a_select_config_name}"...')
                
                if active_computation_params is None:
                    active_computation_params = self.active_configs[a_select_config_name].computation_config # get the previously set computation configs
                else:
                    # set/update the computation configs:
                    self.active_configs[a_select_config_name].computation_config = active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.
                
                
                skip_computations_for_this_result = False
                if overwrite_extant_results or (self.computation_results.get(a_select_config_name, None) is None):
                    # If we're supposed to overwrite the previous result OR the previous result is already empty/not yet calculated, initialize a new one:
                    self.computation_results[a_select_config_name] = ComputedPipelineStage._build_initial_computationResult(a_filtered_session, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
                    skip_computations_for_this_result = False # need to compute the result
                else:
                    # Otherwise it already exists and is not None, so don't overwrite it:
                    if progress_logger_callback is not None:
                        progress_logger_callback(f'WARNING: skipping computation because overwrite_extant_results={overwrite_extant_results} and self.computation_results[{a_select_config_name}] already exists and is non-None')
                        progress_logger_callback('\t TODO: this will prevent recomputation even when the blacklist/whitelist or computation function definitions change. Rework so that this is smarter.')
                    
                    print(f'WARNING: skipping computation because overwrite_extant_results={overwrite_extant_results} and self.computation_results[{a_select_config_name}] already exists and is non-None')
                    print('\t TODO: this will prevent recomputation even when the blacklist/whitelist or computation function definitions change. Rework so that this is smarter.')                    
                    # self.computation_results.setdefault(a_select_config_name, ComputedPipelineStage._build_initial_computationResult(a_filtered_session, active_computation_params)) # returns a computation result. This stores the computation config used to compute it.
                    skip_computations_for_this_result = True

                if not skip_computations_for_this_result:
                    # call to perform any registered computations:
                    self.computation_results[a_select_config_name] = self.perform_specific_context_registered_computations(self.computation_results[a_select_config_name], computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)
            else:
                # this filter is excluded from the enabled list, no computations will we performed on it
                if overwrite_extant_results:
                    self.computation_results.pop(a_select_config_name, None) # remove the computation results from previous runs from the dictionary to indicate that it hasn't been computed
                else:
                    # no *additional* computations will be performed on it, but it will be pass through and not removed form the self.computation_results
                    pass

    def rerun_failed_computations(self, enabled_filter_names=None, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result """
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                print(f'Performing rerun_failed_computations on filtered_session with filter named "{a_select_config_name}"...')
                previous_computation_result = self.computation_results[a_select_config_name]
                self.computation_results[a_select_config_name] = self.rerun_failed_computations(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)    
                
    @classmethod    
    def continue_computations_if_needed(cls, curr_active_pipeline, active_computation_params=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, debug_print=False):
        """ continues computations for a pipeline 

            NOTE: TODO: this is not yet implemented.
            Calls perform_registered_computations(...) to do the actual comptuations

        
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
            curr_active_pipeline.computation_results[an_incomplete_config_name] = curr_active_pipeline.perform_registered_computations(curr_active_pipeline.computation_results[an_incomplete_config_name], computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, debug_print=debug_print)

        ## TODO: initially compute incomplete_computed_config_dict items...

        ## Next look for previously failed computation results:

        ## Next look for previously complete computation results that lack computations for functions explicitly specified in the whitelist (if provided):

        ## Then look for previously complete computation results that are missing computations that have been registered after they were computed, or that were previously part of the blacklist but now are not:

    
    # ==================================================================================================================== #
    # CLASS/STATIC METHODS                                                                                                 #
    # ==================================================================================================================== #
    
    @classmethod
    def _build_initial_computationResult(cls, active_session, computation_config):
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
    def _execute_computation_functions(active_computation_functions, previous_computation_result=None, fail_on_exception:bool = False, progress_logger_callback=None, debug_print=False):
        """ actually performs the provided computations in active_computation_functions """
        if (len(active_computation_functions) > 0):
            if debug_print:
                print(f'Performing _execute_computation_functions(...) with {len(active_computation_functions)} registered_computation_functions...')
            if progress_logger_callback is not None:
                progress_logger_callback(f'Performing _execute_computation_functions(...) with {len(active_computation_functions)} registered_computation_functions...')
            
            if fail_on_exception:
                ## normal version that fails on any exception:
                composed_registered_computations_function = compose_functions(*active_computation_functions, progress_logger=progress_logger_callback, error_logger=None) # functions are composed left-to-right
                previous_computation_result = composed_registered_computations_function(previous_computation_result)
                accumulated_errors = None
            else:
                ## Use exception-tolerant version of function composition (functions are composed left-to-right):
                composed_registered_computations_function = compose_functions_with_error_handling(*active_computation_functions, progress_logger=progress_logger_callback, error_logger=(lambda x: progress_logger_callback(f'ERROR: {x}'))) # functions are composed left-to-right, exception-tolerant version
                previous_computation_result, accumulated_errors = composed_registered_computations_function(previous_computation_result)
            
            if debug_print:
                print(f'_execute_computation_functions(...): \n\taccumulated_errors: {accumulated_errors}')
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
    def global_computation_results(self):
        """The global_computation_results property, accessed through the stage."""
        return self.stage.global_computation_results
    


    @property
    def active_completed_computation_result_names(self):
        """The this list of all computed configs."""        
        # return self.stage._get_valid_computation_results_config_names()
        return self.stage._get_computation_results_progress()[0] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict
    
    @property
    def active_incomplete_computation_result_status_dicts(self):
        """The this dict containing all the incompletely computed configs and their reason for being incomplete."""
        return self.stage._get_computation_results_progress()[1] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict
    
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
    
    def reload_default_computation_functions(self):
        """ reloads/re-registers the default display functions after adding a new one """
        self.stage.reload_default_computation_functions()
        
    def register_computation(self, registered_name, computation_function):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_computation(registered_name, computation_function)

        
    ## Computation Helpers: 
    # perform_computations: The main computation function for the pipeline
    def perform_computations(self, active_computation_params: Optional[DynamicParameters]=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, debug_print=False):
        """The main computation function for the pipeline.

        Args:
            active_computation_params (Optional[DynamicParameters], optional): _description_. Defaults to None.
            enabled_filter_names (_type_, optional): _description_. Defaults to None.
            overwrite_extant_results (bool, optional): _description_. Defaults to False.
            computation_functions_name_whitelist (_type_, optional): _description_. Defaults to None.
            computation_functions_name_blacklist (_type_, optional): _description_. Defaults to None.
            fail_on_exception (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.
        """
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.evaluate_computations_for_single_params(active_computation_params, enabled_filter_names=enabled_filter_names, overwrite_extant_results=overwrite_extant_results, computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, progress_logger_callback=(lambda x: self.logger.info(x)), debug_print=debug_print)
        
    def _perform_registered_computations(self, previous_computation_result=None, computation_functions_name_whitelist=None, computation_functions_name_blacklist=None, fail_on_exception:bool=False, debug_print=False):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.perform_computations to reach this step."
        self.stage.perform_registered_computations(previous_computation_result, computation_functions_name_whitelist=computation_functions_name_whitelist, computation_functions_name_blacklist=computation_functions_name_blacklist, fail_on_exception=fail_on_exception, debug_print=debug_print)
    
    def rerun_failed_computations(self, previous_computation_result, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result """
        return self.stage.rerun_failed_computations(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)
    
    
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

            
        # print(f'\tpost keys: {list(self.active_configs.keys())}')

