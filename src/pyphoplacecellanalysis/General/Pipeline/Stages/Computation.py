from collections import OrderedDict
from pickle import PicklingError
import sys
from copy import deepcopy
from datetime import datetime, timedelta
import typing

from typing import Any, Callable, Optional, Dict, List, Tuple, Union
from warnings import warn
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum # for EvaluationActions
from datetime import datetime
from attrs import define, field, Factory
from functools import partial

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
from neuropy.core.epoch import Epoch
from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from neuropy.utils.result_context import IdentifyingContext, DisplaySpecifyingIdentifyingContext

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters # to replace simple PlacefieldComputationParameters, `load_pickled_global_computation_results`
from pyphocorehelpers.function_helpers import compose_functions, compose_functions_with_error_handling
from pyphocorehelpers.print_helpers import format_seconds_human_readable
from pyphocorehelpers.programming_helpers import MemoryManagement

from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import LoadableInput, LoadableSessionInput, LoadedPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData # used for `load_pickled_global_computation_results`
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData # used for `save_global_computation_results`
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode
from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import SpecificComputationValidator


import pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions import ComputationFunctionRegistryHolder # should include ComputationFunctionRegistryHolder and all specifics
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _wrap_multi_context_computation_function

from pyphocorehelpers.exception_helpers import CapturedException, ExceptionPrintingContext # used in _execute_computation_functions for error handling
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.assertion_helpers import Assert


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



def session_context_filename_formatting_fn(ctxt: DisplaySpecifyingIdentifyingContext, subset_includelist=None, subset_excludelist=None, parts_separator:str='-') -> str:
    """ `neuropy.utils.result_context.ContextFormatRenderingFn` protocol format callable 
    specific_purpose='filename_prefix'
    renders a custom_prefix from the context
    
        
        final_filename like "2024-10-31_1020PM-kdiba_pin01_one_11-03_12-3-25__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]-(ripple_simple_pf_pearson_merged_df)_tbin-0.025.csv"
    only handles the "_withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]" part
    
    History: duplicated from `_get_custom_suffix_for_filename_from_computation_metadata` 2024-11-01 08:16 
    """
    to_filename_conversion_dict = {'compute_diba_quiescent_style_replay_events':'_withNewComputedReplays', 'diba_evt_file':'_withNewKamranExportedReplays', 'initial_loaded': '_withOldestImportedReplays', 'normal_computed': '_withNormalComputedReplays'}

    if subset_includelist is None:
        assert subset_includelist is None, f"subset_includelist is not supported for this formatting function but was provided as subset_includelist: {subset_includelist}"
        # subset_includelist = []
    if subset_excludelist is None:
        subset_excludelist = []
    
    custom_suffix_string_parts = []
    custom_suffix: str = ''
    basic_session_property_names = ['format_name', 'animal', 'exper_name', 'session_name']
    for a_key in basic_session_property_names:
        if (ctxt.get(a_key, None) is not None) and (len(str(ctxt.get(a_key, None))) > 0) and (a_key not in subset_excludelist):
            # custom_suffix_string_parts.append(f"{a_key}_{ctxt.get(a_key, None)}")
            custom_suffix_string_parts.append(f"{ctxt.get(a_key, None)}") ## no key names

    if (ctxt.get('epochs_source', None) is not None) and (len(str(ctxt.get('epochs_source', None))) > 0) and ('epochs_source' not in subset_excludelist):
        custom_suffix_string_parts.append(to_filename_conversion_dict[ctxt.get('epochs_source', None)])
    if (ctxt.get('included_qclu_values', None) is not None) and (len(str(ctxt.get('included_qclu_values', None))) > 0) and ('included_qclu_values' not in subset_excludelist):
        custom_suffix_string_parts.append(f"qclu_{ctxt.get('included_qclu_values', None)}")
    if (ctxt.get('minimum_inclusion_fr_Hz', None) is not None) and (len(str(ctxt.get('minimum_inclusion_fr_Hz', None))) > 0) and ('minimum_inclusion_fr_Hz' not in subset_excludelist):
        custom_suffix_string_parts.append(f"frateThresh_{ctxt.get('minimum_inclusion_fr_Hz', None):.1f}")
    # custom_suffix = parts_separator.join([custom_suffix, *custom_suffix_string_parts])
    custom_suffix = parts_separator.join(custom_suffix_string_parts)
    return custom_suffix



# ==================================================================================================================== #
# PIPELINE STAGE                                                                                                       #
# ==================================================================================================================== #
@define(slots=False, repr=False)
class ComputedPipelineStage(FilterablePipelineStage, LoadedPipelineStage):
    """Docstring for ComputedPipelineStage.

    global_comparison_results has keys of type IdentifyingContext
    """
    @classmethod
    def get_stage_identity(cls) -> PipelineStage:
        return PipelineStage.Computed

    identity: PipelineStage = field(default=PipelineStage.Computed)
    # identity: PipelineStage = PipelineStage.Computed

    filtered_sessions: Optional[DynamicParameters] = field(default=None)
    filtered_epochs: Optional[DynamicParameters] = field(default=None)
    filtered_contexts: Optional[DynamicParameters] = field(default=None)
    active_configs: Optional[DynamicParameters] = field(default=None)
    computation_results: Optional[DynamicParameters] = field(default=None)
    global_computation_results: Optional[ComputationResult] = field(default=None)

    registered_computation_function_dict: OrderedDict = field(default=Factory(OrderedDict))
    registered_global_computation_function_dict: OrderedDict = field(default=Factory(OrderedDict))


    @classmethod
    def init_from_previous_stage(cls, loaded_stage: LoadedPipelineStage):
        _obj = cls()
        _obj.stage_name = loaded_stage.stage_name
        _obj.basedir = loaded_stage.basedir
        _obj.loaded_data = loaded_stage.loaded_data

        # Initialize custom fields:
        _obj.filtered_sessions = DynamicParameters()
        _obj.filtered_epochs = DynamicParameters()
        _obj.filtered_contexts = DynamicParameters()

        _obj.active_configs = DynamicParameters() # active_config corresponding to each filtered session/epoch
        _obj.computation_results = DynamicParameters() # computation_results is a DynamicParameters with keys of type IdentifyingContext and values of type ComputationResult

        # _obj.global_computation_results = ComputedPipelineStage._build_initial_computationResult(_obj.sess, None) # proper type setup
        # _obj.global_computation_results = ComputedPipelineStage._build_initial_global_computationResult(curr_active_pipeline=_obj, active_session=_obj.sess, computation_config=None) # proper type setup
        _obj.global_computation_results = _obj.build_initial_global_computationResult(computation_config=None)
        

        _obj.registered_computation_function_dict = OrderedDict()
        _obj.registered_global_computation_function_dict = OrderedDict()
        _obj.reload_default_computation_functions() # registers the default
        return _obj

    # Filtered Properties: _______________________________________________________________________________________________ #
    @property
    def is_filtered(self):
        """The is_filtered property."""
        # return isinstance(self, ComputedPipelineStage) # this is redundant
        return True

    # Computation Properties: _______________________________________________________________________________________________ #
    
    @property
    def can_compute(self):
        """The can_compute property."""
        return True # (self.last_completed_stage >= PipelineStage.Filtered)
    
    @property
    def active_completed_computation_result_names(self):
        """The this list of all computed configs."""
        return self._get_computation_results_progress()[0] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict

    @property
    def active_incomplete_computation_result_status_dicts(self):
        """The this dict containing all the incompletely computed configs and their reason for being incomplete."""
        return self._get_computation_results_progress()[1] # get [0] because it returns complete_computed_config_names_list, incomplete_computed_config_dict

    @property
    def registered_computation_functions(self) -> List[Callable]:
        return list(self.registered_computation_function_dict.values())
    @property
    def registered_computation_function_names(self) -> List[str]:
        return list(self.registered_computation_function_dict.keys()) 


    @property
    def registered_global_computation_functions(self) -> List[Callable]:
        return list(self.registered_global_computation_function_dict.values())
    @property
    def registered_global_computation_function_names(self) -> List[str]:
        return list(self.registered_global_computation_function_dict.keys()) 


    # 'merged' refers to the fact that both global and non-global computation functions are included _____________________ #
    @property
    def registered_merged_computation_function_dict(self) -> Dict[str, Callable]:
        """build a merged function dictionary containing both global and non-global functions:"""
        return (self.registered_global_computation_function_dict | self.registered_computation_function_dict)
    @property
    def registered_merged_computation_functions(self) -> List[Callable]:
        return list(self.registered_merged_computation_function_dict.values())
    @property
    def registered_merged_computation_function_names(self) -> List[str]:
        return list(self.registered_merged_computation_function_dict.keys()) 


    def reload_default_computation_functions(self):
        """ reloads/re-registers the default display functions after adding a new one
         Note: execution ORDER MATTERS for the computation functions, unlike the display functions, so they need to be enumerated in the correct order and not sorted alphabetically        
        # Sort by precidence:
            _computationPrecidence
        """
        for (a_computation_fn_key, a_computation_fn) in ComputationFunctionRegistryHolder.get_ordered_registry_items_functions(include_local=True, include_global=True).items():
                self.register_computation(a_computation_fn.__name__, a_computation_fn, is_global=a_computation_fn.is_global)


        
    def register_computation(self, registered_name: str, computation_function, is_global:bool):
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


    def find_registered_computation_functions(self, registered_names_list, search_mode:FunctionsSearchMode=FunctionsSearchMode.ANY, names_list_is_excludelist:bool=False):
        ''' Finds the list of actual function objects associated with the registered_names_list by using the appropriate dictionary of registered functions depending on whether are_global is True or not.

        registered_names_list: list<str> - a list of function names to be used to fetch the appropriate functions
        are_global: bool - If True, the registered_global_computation_function_dict is used instead of the registered_computation_function_dict
        names_list_is_excludelist: bool - if True, registered_names_list is treated as a excludelist, and all functions are returned EXCEPT those that are in registered_names_list

        Usage:
            active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, are_global=are_global)
        '''
        # We want to reload the new/modified versions of the functions:
        self.reload_default_computation_functions()

        if search_mode.name == FunctionsSearchMode.GLOBAL_ONLY.name:
            active_registered_computation_function_dict = self.registered_global_computation_function_dict
        elif search_mode.name == FunctionsSearchMode.NON_GLOBAL_ONLY.name:
            active_registered_computation_function_dict = self.registered_computation_function_dict
        elif search_mode.name == FunctionsSearchMode.ANY.name:
            # build a merged function dictionary containing both global and non-global functions:
            active_registered_computation_function_dict = self.registered_merged_computation_function_dict


        else:
            raise NotImplementedError

        if names_list_is_excludelist:
            # excludelist-style operation: treat the registered_names_list as a excludelist and return all registered functions EXCEPT those that are in registered_names_list
            active_computation_function_dict = {a_computation_fn_name:a_computation_fn for (a_computation_fn_name, a_computation_fn) in active_registered_computation_function_dict.items() if ((a_computation_fn_name not in registered_names_list) and (getattr(a_computation_fn, 'short_name', a_computation_fn.__name__) not in registered_names_list))}
        else:
            # default includelist-style operation:
            active_computation_function_dict = {a_computation_fn_name:a_computation_fn for (a_computation_fn_name, a_computation_fn) in active_registered_computation_function_dict.items() if ((a_computation_fn_name in registered_names_list) or (getattr(a_computation_fn, 'short_name', a_computation_fn.__name__) in registered_names_list))}

        return list(active_computation_function_dict.values())
        

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['registered_load_function_dict']
        del state['registered_computation_function_dict']
        del state['registered_global_computation_function_dict']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if 'identity' not in state:
            print(f'unpickling from old NeuropyPipelineStage')
            state['identity'] = None
            state['identity'] = type(self).get_stage_identity()

        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(LoadedPipelineStage, self).__init__() # from 

        self.registered_load_function_dict = {}
        self.register_default_known_load_functions() # registers the default load functions
        
        self.registered_computation_function_dict = OrderedDict()
        self.registered_global_computation_function_dict = OrderedDict()
        self.reload_default_computation_functions() # registers the default


        

    # ==================================================================================================================== #
    # Specific Context Computation Helpers                                                                                 #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['compute'], input_requires=[], output_provides=[], uses=['find_registered_computation_functions', 'cls._execute_computation_functions'], used_by=['perform_action_for_all_contexts'], creation_date='2024-10-07 15:07', related_items=[])
    def perform_registered_computations_single_context(self, previous_computation_result=None, computation_functions_name_includelist=None, computation_functions_name_excludelist=None, fail_on_exception:bool=False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ Executes all registered computations for a single filter
        
        The return value should be set to the self.computation_results[a_select_config_name]
        """
        search_mode=FunctionsSearchMode.ANY
        # can also do:
        # search_mode=FunctionsSearchMode.initFromIsGlobal(are_global)

        # Need to exclude any computation functions specified in omitted_computation_functions_dict
        if computation_functions_name_includelist is not None:
            active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, search_mode=search_mode, names_list_is_excludelist=False)
            print(f'due to includelist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')

        elif computation_functions_name_excludelist is not None:
            active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, search_mode=search_mode, names_list_is_excludelist=True)
            print(f'due to excludelist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')
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

    @function_attributes(short_name=None, tags=['computation', 'specific'], input_requires=[], output_provides=[], uses=['ComputedPipelineStage._execute_computation_functions'], used_by=[], creation_date='2023-07-21 18:25', related_items=[])
    def run_specific_computations_single_context(self, previous_computation_result, computation_functions_name_includelist, computation_kwargs_list=None, fail_on_exception:bool=False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ re-runs just a specific computation provided by computation_functions_name_includelist """
        active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, search_mode=FunctionsSearchMode.initFromIsGlobal(are_global))
        if progress_logger_callback is not None:
            progress_logger_callback(f'\trun_specific_computations_single_context(including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions): active_computation_functions: {active_computation_functions}...')
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

    @function_attributes(short_name=None, tags=['times', 'computation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-21 18:26', related_items=[])
    def get_computation_times(self, debug_print=False):
        """ gets the latest computation_times from `curr_active_pipeline.computation_results`
        
        Usage:
            any_most_recent_computation_time, each_epoch_latest_computation_time, each_epoch_each_result_computation_completion_times, (global_computations_latest_computation_time, global_computation_completion_times) = curr_active_pipeline.stage.get_computation_times()
            each_epoch_latest_computation_time

            # If you want to simplify the names:
            # Get the names of the global and non-global computations:
            all_validators_dict = curr_active_pipeline.get_merged_computation_function_validators()
            global_only_validators_dict = {k:v for k, v in all_validators_dict.items() if v.is_global}
            non_global_only_validators_dict = {k:v for k, v in all_validators_dict.items() if (not v.is_global)}
            non_global_comp_names: List[str] = [v.short_name for k, v in non_global_only_validators_dict.items() if (not v.short_name.startswith('_DEP'))] # ['firing_rate_trends', 'spike_burst_detection', 'pf_dt_sequential_surprise', 'extended_stats', 'placefield_overlap', 'ratemap_peaks_prominence2d', 'velocity_vs_pf_simplified_count_density', 'EloyAnalysis', '_perform_specific_epochs_decoding', 'recursive_latent_pf_decoding', 'position_decoding_two_step', 'position_decoding', 'lap_direction_determination', 'pfdt_computation', 'pf_computation']
            global_comp_names: List[str] = [v.short_name for k, v in global_only_validators_dict.items() if (not v.short_name.startswith('_DEP'))] # ['long_short_endcap_analysis', 'long_short_inst_spike_rate_groups', 'long_short_post_decoding', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_decoding_analyses', 'PBE_stats', 'rank_order_shuffle_analysis', 'directional_decoders_epoch_heuristic_scoring', 'directional_decoders_evaluate_epochs', 'directional_decoders_decode_continuous', 'merged_directional_placefields', 'split_to_directional_laps']

            # mappings between the long computation function names and their short names:
            non_global_comp_names_map: Dict[str, str] = {v.computation_fn_name:v.short_name for k, v in non_global_only_validators_dict.items() if (not v.short_name.startswith('_DEP'))}
            global_comp_names_map: Dict[str, str] = {v.computation_fn_name:v.short_name for k, v in global_only_validators_dict.items() if (not v.short_name.startswith('_DEP'))} # '_perform_long_short_endcap_analysis': 'long_short_endcap_analysis', '_perform_long_short_instantaneous_spike_rate_groups_analysis': 'long_short_inst_spike_rate_groups', ...}


        """
        
        # inverse_computation_times_key_fn = lambda fn_key: str(fn_key.__name__) # to be used if raw-function references are used.
        inverse_computation_times_key_fn = lambda fn_key: fn_key # Use only the functions name. I think this makes the .computation_times field picklable
        
        each_epoch_each_result_computation_completion_times = {}
        each_epoch_latest_computation_time = {} # the most recent computation for each of the epochs
        # find update time of latest function:
        for k, v in self.computation_results.items():
            extracted_computation_times_dict = v.computation_times
            each_epoch_each_result_computation_completion_times[k] = {inverse_computation_times_key_fn(k):v for k,v in extracted_computation_times_dict.items()}
            each_epoch_latest_computation_time[k] = max(list(each_epoch_each_result_computation_completion_times[k].values()), default=datetime.min)

        non_global_any_most_recent_computation_time: datetime = max(list(each_epoch_latest_computation_time.values()), default=datetime.min) # newest computation out of any of the epochs

        ## Global computations:
        global_computation_completion_times = {inverse_computation_times_key_fn(k):v for k,v in self.global_computation_results.computation_times.items()}
        global_computations_latest_computation_time: datetime = max(list(global_computation_completion_times.values()), default=datetime.min)

        ## Any (global or non-global) computation most recent time):
        any_most_recent_computation_time: datetime = max(non_global_any_most_recent_computation_time, global_computations_latest_computation_time, datetime.min) # returns `datetime.min` if the first arguments are empty

        if debug_print:
            print(f'any_most_recent_computation_time: {any_most_recent_computation_time}')
            print(f'each_epoch_latest_computation_time: {each_epoch_latest_computation_time}')
            print(f'global_computation_completion_times: {global_computation_completion_times}')
        return any_most_recent_computation_time, each_epoch_latest_computation_time, each_epoch_each_result_computation_completion_times, (global_computations_latest_computation_time, global_computation_completion_times)
        

    def get_time_since_last_computation(self, debug_print_timedelta:bool=False) -> timedelta:
        ## Successfully prints the time since the last calculation was performed:
        run_time = datetime.now()
        any_most_recent_computation_time, each_epoch_latest_computation_time, each_epoch_each_result_computation_completion_times, (global_computations_latest_computation_time, global_computation_completion_times) = self.get_computation_times()
        
        delta = run_time - any_most_recent_computation_time
        if debug_print_timedelta:
            # Strangely gives days and then seconds in the remaining day (the days aren't included in the seconds)
            days = delta.days
            total_seconds = delta.seconds
            h, m, s, fractional_seconds, formatted_timestamp_str = format_seconds_human_readable(total_seconds)
            print(f"time since latest calculation performed (days::hours:minutes:seconds): {days}::{formatted_timestamp_str}")
        return delta
        # run_time - any_most_recent_computation_time # datetime.timedelta(days=1, seconds=72179, microseconds=316519)
        # {k:(run_time - v) for k, v in each_epoch_latest_computation_time.items()}


    def find_LongShortGlobal_epoch_names(self):
        """ Helper function to returns the [long, short, global] epoch names. They must exist.
        Usage:
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            long_results = curr_active_pipeline.computation_results[long_epoch_name]['computed_data']
            short_results = curr_active_pipeline.computation_results[short_epoch_name]['computed_data']
            global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

        """
        include_includelist = self.active_completed_computation_result_names # ['maze', 'sprinkle']
        assert (len(include_includelist) >= 3), "Must have at least 3 completed computation results to find the long, short, and global epoch names."
        if (len(include_includelist) > 3):
            """ more than three computed epochs, probably split by laps. Figure out the correct ones. """
            # include_includelist = self.filtered_epochs
            non_global_epoch_names = list(self.sess.paradigm.get_unique_labels()) # ['maze1', 'maze2']
            global_epoch_name: str = 'maze'
            known_epoch_names = [*non_global_epoch_names, global_epoch_name] # a list of names
            
        else:
            # Old method of unwrapping based on hard-coded values:
            assert len(include_includelist) == 3, f"Must have exactly 3: {include_includelist}"
            # long_epoch_name = include_includelist[0] # 'maze1_PYR'
            # short_epoch_name = include_includelist[1] # 'maze2_PYR'
            # they must all have the same suffix:
            long_epoch_name = include_includelist[-3] # 'maze1_PYR'
            short_epoch_name = include_includelist[-2] # 'maze2_PYR'
            global_epoch_name = include_includelist[-1] # 'maze_PYR'
            known_epoch_names = [long_epoch_name, short_epoch_name, global_epoch_name]


        assert len(known_epoch_names) == 3, f"Must have exactly 3: {known_epoch_names}"
        # If defaults are all missing, a renaming has probably been done.
        if np.all(np.logical_not(np.isin(known_epoch_names, include_includelist))):
            # if defaults are all missing, a filtering has probably been done.
            known_epoch_names = [f'{a_name}_any' for a_name in known_epoch_names]

        assert np.all(np.isin(known_epoch_names, include_includelist)), f"all long/short/global epochs must exist in include_includelist! known_epoch_names: {known_epoch_names}, include_includelist: {include_includelist}, np.isin(known_epoch_names, include_includelist): {np.isin(known_epoch_names, include_includelist)}"
        long_epoch_name, short_epoch_name, global_epoch_name = known_epoch_names # unwrap

        return long_epoch_name, short_epoch_name, global_epoch_name


    def find_LongShortDelta_times(self) -> Tuple[float, float, float]:
        """ Helper function to returns the [t_start, t_delta, t_end] session times. They must exist.
        Usage:
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        """
        long_epoch_name, short_epoch_name, global_epoch_name = self.find_LongShortGlobal_epoch_names()
        long_epoch_obj, short_epoch_obj = [Epoch(self.sess.epochs.to_dataframe().epochs.label_slice(an_epoch_name.removesuffix('_any'))) for an_epoch_name in [long_epoch_name, short_epoch_name]] #TODO 2023-11-10 20:41: - [ ] Issue with getting actual Epochs from sess.epochs for directional laps: emerges because long_epoch_name: 'maze1_any' and the actual epoch label in curr_active_pipeline.sess.epochs is 'maze1' without the '_any' part.
        if ((short_epoch_obj.n_epochs == 0) or (short_epoch_obj.n_epochs == 0)):
            ## Epoch names now are 'long' and 'short'
            long_epoch_obj, short_epoch_obj = [self.filtered_epochs[an_epoch_name].to_Epoch() for an_epoch_name in [long_epoch_name, short_epoch_name]]

        
        assert short_epoch_obj.n_epochs > 0, f'long_epoch_obj: {long_epoch_obj}, short_epoch_obj: {short_epoch_obj}'
        assert long_epoch_obj.n_epochs > 0, f'long_epoch_obj: {long_epoch_obj}, short_epoch_obj: {short_epoch_obj}'
        # Uses: long_epoch_obj, short_epoch_obj
        t_start = long_epoch_obj.t_start
        t_delta = short_epoch_obj.t_start
        t_end = short_epoch_obj.t_stop
        return t_start, t_delta, t_end



    def get_failed_computations(self, enabled_filter_names=None):
        """ gets a dictionary of the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result
        
        """
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified
        all_accumulated_errors = {}
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                # print(f'Performing rerun_failed_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                previous_computation_result = self.computation_results[a_select_config_name]
                # curr_active_pipeline.computation_results[a_select_config_name] = curr_active_pipeline.rerun_failed_computations_single_context(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)    
                active_computation_errors = previous_computation_result.accumulated_errors
                if len(active_computation_errors) > 0:
                    # all_accumulated_errors[a_select_config_name] = active_computation_errors
                    all_accumulated_errors[a_select_config_name] = {}
                    for failed_computation_fn, error in active_computation_errors.items():
                        a_failed_fn_name: str = failed_computation_fn.__name__
                        all_accumulated_errors[a_select_config_name][a_failed_fn_name] = error

        return all_accumulated_errors


    def rerun_failed_computations(self, enabled_filter_names=None, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result
        
        TODO: Parallelization opportunity
        
        """
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                print(f'Performing rerun_failed_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                previous_computation_result = self.computation_results[a_select_config_name]
                self.computation_results[a_select_config_name] = self.rerun_failed_computations_single_context(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)    


    @function_attributes(short_name=None, tags=['action', 'computation'], input_requires=[], output_provides=[], uses=['perform_registered_computations_single_context', 'cls._build_initial_computationResult'], used_by=['perform_computations'], creation_date='2023-07-21 18:22', related_items=[])
    def perform_action_for_all_contexts(self, action: EvaluationActions, enabled_filter_names=None, active_computation_params: Optional[DynamicParameters]=None, overwrite_extant_results=False, computation_functions_name_includelist=None, computation_functions_name_excludelist=None,
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

        ## Check for duplicated configs:
        
        # MemoryManagement.has_duplicated_memory_references(list(active_computation_results.values()))
        # MemoryManagement.has_duplicated_memory_references([v.computation_config for v in active_computation_results.values()])

        # MemoryManagement.has_duplicated_memory_references([v.computation_config.pf_params for v in active_computation_results.values()])
        # MemoryManagement.has_duplicated_memory_references([v.computation_config.pf_params.computation_epochs for v in active_computation_results.values()])

        ## Here's where we loop through all possible configs:
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                if debug_print:
                    print(f'Performing perform_action_for_all_contexts with action {action} on filtered_session with filter named "{a_select_config_name}"...')
                if progress_logger_callback is not None:
                    progress_logger_callback(f'Performing perform_action_for_all_contexts with action {action} on filtered_session with filter named "{a_select_config_name}"...')
                
                # TODO 2023-01-15 - ASSUMES 1:1 correspondence between self.filtered_sessions's config names and computation_configs:
                computation_parameters_need_update: bool = True
                if active_computation_params is None:
                    curr_active_computation_params = deepcopy(self.active_configs[a_select_config_name].computation_config) # get the previously set computation configs
                else:
                    # Make sure to duplicate an guarnateed independent copy first:
                    curr_active_computation_params = deepcopy(active_computation_params)

                   
                

                # # ensure config is filtered:
                # # 2023-11-10 - Note hardcoded directional lap/pf suffixes used here.
                # a_base_filter_name: str = a_select_config_name.removesuffix('_any').removesuffix('_odd').removesuffix('_even')
                # a_select_epoch = Epoch(a_filtered_session.epochs.to_dataframe().epochs.label_slice(a_base_filter_name))
                # assert a_select_epoch.n_epochs > 0
                # a_filtered_session.epochs = a_select_epoch # replace the epochs object
                # active_computation_params.pf_params.computation_epochs = active_computation_params.pf_params.computation_epochs.time_slice(a_select_epoch.t_start, a_select_epoch.t_stop)
                # # set/update the computation configs:
                # self.active_configs[a_select_config_name].computation_config = deepcopy(active_computation_params)

                # 'maze1_odd', active_computation_params - active_computation_params.is_directional == False?

                # `a_filtered_session` seems to be working. a_filtered_session.laps is correctly filtered.
                # for maze2: a_filtered_session.t_start, a_filtered_session.t_stop
                curr_active_computation_params.pf_params.computation_epochs = curr_active_computation_params.pf_params.computation_epochs.time_slice(a_filtered_session.t_start, a_filtered_session.t_stop)
                if debug_print:
                    print(f'curr_active_computation_params.pf_params.computation_epochs: {curr_active_computation_params.pf_params.computation_epochs}')


                if computation_parameters_need_update:
                    # set/update the computation configs:
                    self.active_configs[a_select_config_name].computation_config = deepcopy(curr_active_computation_params) #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.

                if action.name == EvaluationActions.EVALUATE_COMPUTATIONS.name:
                    # active_function = self.perform_registered_computations_single_context
                    skip_computations_for_this_result = False
                    if overwrite_extant_results or (active_computation_results.get(a_select_config_name, None) is None):
                        # If we're supposed to overwrite the previous result OR the previous result is already empty/not yet calculated, initialize a new one:
                        active_computation_results[a_select_config_name] = ComputedPipelineStage._build_initial_computationResult(a_filtered_session, curr_active_computation_params) # returns a computation result. This stores the computation config used to compute it.
                        skip_computations_for_this_result = False # need to compute the result
                    else:
                        # Otherwise it already exists and is not None, so don't overwrite it:
                        if progress_logger_callback is not None:
                            progress_logger_callback(f'WARNING: skipping computation because overwrite_extant_results={overwrite_extant_results} and `active_computation_results[{a_select_config_name}]` already exists and is non-None')
                            progress_logger_callback('\t TODO: this will prevent recomputation even when the excludelist/includelist or computation function definitions change. Rework so that this is smarter.')
                        
                        print(f'WARNING: skipping computation because overwrite_extant_results={overwrite_extant_results} and `active_computation_results[{a_select_config_name}]` already exists and is non-None')
                        print('\t TODO: this will prevent recomputation even when the excludelist/includelist or computation function definitions change. Rework so that this is smarter.')
                        # active_computation_results.setdefault(a_select_config_name, ComputedPipelineStage._build_initial_computationResult(a_filtered_session, curr_active_computation_params)) # returns a computation result. This stores the computation config used to compute it.
                        skip_computations_for_this_result = True

                    if not skip_computations_for_this_result:
                        # call to perform any registered computations:
                        active_computation_results[a_select_config_name] = self.perform_registered_computations_single_context(active_computation_results[a_select_config_name],
                            computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

                elif action.name == EvaluationActions.RUN_SPECIFIC.name:
                    print(f'Performing run_specific_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                    # active_function = self.run_specific_computations_single_context
                    previous_computation_result = active_computation_results[a_select_config_name]
                    active_computation_results[a_select_config_name] = self.run_specific_computations_single_context(previous_computation_result, computation_functions_name_includelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

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
                    #     computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)
            else:
                # this filter is excluded from the enabled list, no computations will we performed on it
                if overwrite_extant_results:
                    active_computation_results.pop(a_select_config_name, None) # remove the computation results from previous runs from the dictionary to indicate that it hasn't been computed
                else:
                    # no *additional* computations will be performed on it, but it will be pass through and not removed form the active_computation_results
                    pass

            # Re-apply changes when done:
            if debug_print:
                print(f'updating computation_results...')
            if are_global:
                self.global_computation_results = active_computation_results
            else:
                self.computation_results = active_computation_results
            if debug_print:
                print(f'done.')


    @function_attributes(short_name=None, tags=['computation', 'specific', 'parallel', 'embarassingly-paralell'], input_requires=[], output_provides=[], uses=['run_specific_computations_single_context', 'cls._build_initial_computationResult'], used_by=[], creation_date='2023-07-21 18:21', related_items=[])
    def perform_specific_computation(self, active_computation_params=None, enabled_filter_names=None, computation_functions_name_includelist=None, computation_kwargs_list=None, fail_on_exception:bool=False, debug_print=False, progress_logger_callback=None, enable_parallel: bool=False):
        """ perform a specific computation (specified in computation_functions_name_includelist) in a minimally destructive manner using the previously recomputed results:
        Ideally would already have access to the:
        - Previous computation result
        - Previous computation config (the input parameters)


        computation_kwargs_list: Optional<list>: a list of kwargs corresponding to each function name in computation_functions_name_includelist

        Internally calls: `run_specific_computations_single_context`.

        Updates:
            curr_active_pipeline.computation_results
            curr_active_pipeline.global_computation_results
        """
        if progress_logger_callback is None:
            progress_logger_callback = print

        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified

        has_custom_kwargs_list: bool = False # indicates whether user provided a kwargs list
        if computation_kwargs_list is None:
            computation_kwargs_list = [{} for _ in computation_functions_name_includelist]
        else:
            has_custom_kwargs_list = np.any([len(x)>0 for x in computation_kwargs_list])
            # has_custom_kwargs_list = True            

        assert isinstance(computation_kwargs_list, List), f"computation_kwargs_list: Optional<list>: is supposed to be a list of kwargs corresponding to each function name in computation_functions_name_includelist but instead is of type:\n\ttype(computation_kwargs_list): {type(computation_kwargs_list)}"
        assert len(computation_kwargs_list) == len(computation_functions_name_includelist)


        active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, search_mode=FunctionsSearchMode.ANY) # find_registered_computation_functions is a pipeline.stage property
        contains_any_global_functions = np.any([v.is_global for v in active_computation_functions])
        if contains_any_global_functions:
            assert np.all([v.is_global for v in active_computation_functions]), 'ERROR: cannot mix global and non-global functions in a single call to perform_specific_computation'

            if (self.global_computation_results is None) or (not isinstance(self.global_computation_results, ComputationResult)):
                print(f'global_computation_results is None. Building initial global_computation_results...')
                self.global_computation_results = None # clear existing results
                self.global_computation_results = self.build_initial_global_computationResult(active_computation_params)
                assert self.global_computation_results.computation_config is not None
            ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
            if self.global_computation_results.computation_config is None:
                print('\tglobal_computation_results.computation_config is None! Making new one!')
                if active_computation_params is not None:
                    print(f'WARNING: does this overwrite the custom computation_params with the pipeline defaults???\n\tactive_computation_params: {active_computation_params}')
                    self.global_computation_results.computation_config = deepcopy(active_computation_params)
                else:
                    #     active_computation_params
                    # curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
                    # output_result.computation_config = curr_global_param_typed_parameters
                    self.global_computation_results.update_config_from_pipeline(curr_active_pipeline=self) ## NOTE: self is not a curr_active_pipeline, it's a ComputedPipelineStage

                print(f'\t\tdone. Pipeline needs resave!')


        if contains_any_global_functions:
            # global computation functions:
            if (self.global_computation_results is None) or (not isinstance(self.global_computation_results, ComputationResult)):
                print(f'global_computation_results is None or not a `ComputationResult` object. Building initial global_computation_results...') #TODO 2024-01-10 15:12: - [ ] Check that `self.global_computation_results.keys()` are empty
                self.global_computation_results = None # clear existing results\
                self.global_computation_results = self.build_initial_global_computationResult(active_computation_params)
                assert self.global_computation_results.computation_config is not None
            ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
            if self.global_computation_results.computation_config is None:
                print('\tglobal_computation_results.computation_config is None! Making new one!')
                if active_computation_params is not None:
                    print(f'WARNING: does this overwrite the custom computation_params with the pipeline defaults???\n\tactive_computation_params: {active_computation_params}')
                    self.global_computation_results.computation_config = deepcopy(active_computation_params)
                else:
                    self.global_computation_results.update_config_from_pipeline(curr_active_pipeline=self)

                print(f'\t\tdone. Pipeline needs resave!')

                # ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
                # if self.global_computation_results.computation_config is None:
                #     print('global_computation_results.computation_config is None! Making new one!')
                #     curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=self)
                #     self.global_computation_results.computation_config = curr_global_param_typed_parameters
                #     print(f'\tdone. Pipeline needs resave!')
                # else:
                #     curr_global_param_typed_parameters: ComputationKWargParameters = self.global_computation_results.computation_config
                    


            ## TODO: what is this about?
            previous_computation_result = self.global_computation_results

            ## TODO: ERROR: `owning_pipeline_reference=self` is not CORRECT as self is of type `ComputedPipelineStage` (or `DisplayPipelineStage`) and not `NeuropyPipeline`
                # this has been fine for all the global functions so far because the majority of the properties are defined on the stage anyway, but any pipeline properties will be missing! 
            global_kwargs = dict(owning_pipeline_reference=self, global_computation_results=previous_computation_result, computation_results=self.computation_results, active_configs=self.active_configs, include_includelist=enabled_filter_names, debug_print=debug_print)
            print(f'for global computations: Performing run_specific_computations_single_context(..., computation_functions_name_includelist={computation_functions_name_includelist}, ...)...')
            self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
        else:
            # Non-global functions:
            if not enable_parallel:
                ## enable_parallel == False
                for a_select_config_name, a_filtered_session in self.filtered_sessions.items():
                    if a_select_config_name in enabled_filter_names:
                        print(f'===>|> for filtered_session with filter named "{a_select_config_name}": Performing run_specific_computations_single_context(..., computation_functions_name_includelist={computation_functions_name_includelist})...')
                        if active_computation_params is None:
                            curr_active_computation_params = self.active_configs[a_select_config_name].computation_config
                        else:
                            curr_active_computation_params = active_computation_params
                            self.active_configs[a_select_config_name].computation_config = curr_active_computation_params
                        previous_computation_result = self.computation_results[a_select_config_name]
                        self.computation_results[a_select_config_name] = self.run_specific_computations_single_context(previous_computation_result, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=False, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback)
            else:
                ## enable_parallel == True
                import concurrent.futures ## used for optional paralell computations in `perform_specific_computation`
                print("Running non-global computations in parallel...")

                def _compute_for_one_session(a_select_config_name: str, active_computation_params, previous_result, config_computation_config, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, debug_print):
                    if active_computation_params is not None: config_computation_config = active_computation_params
                    updated_result = self.run_specific_computations_single_context(previous_result, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=False, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback)
                    return (a_select_config_name, updated_result, config_computation_config)

                futures = {}
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for a_select_config_name, a_filtered_session in self.filtered_sessions.items():
                        if a_select_config_name in enabled_filter_names:
                            previous_result = self.computation_results[a_select_config_name]
                            config_computation_config = self.active_configs[a_select_config_name].computation_config
                            future = executor.submit(_compute_for_one_session, a_select_config_name, active_computation_params, previous_result, config_computation_config, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, debug_print)
                            futures[future] = a_select_config_name

                    for future in concurrent.futures.as_completed(futures):
                        a_select_config_name = futures[future]
                        try:
                            (name, updated_result, updated_config) = future.result()
                            self.computation_results[name] = updated_result
                            if active_computation_params is not None: self.active_configs[name].computation_config = updated_config
                        except Exception as e:
                            print(f"Exception for filter {a_select_config_name}: {e}")
                            if fail_on_exception:
                                raise
        return



    ## Computation Helpers: 
    # perform_computations: The main computation function for the computation stage
    @function_attributes(short_name=None, tags=['main', 'computation'], input_requires=[], output_provides=[], uses=['perform_action_for_all_contexts'], used_by=[], creation_date='2023-10-25 12:26', related_items=[])
    def perform_computations(self, active_computation_params: Optional[DynamicParameters]=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_includelist=None, computation_functions_name_excludelist=None, fail_on_exception:bool=False, debug_print=False, progress_logger_callback=None):
        """The main computation function for the pipeline.

        Wraps `perform_action_for_all_contexts`
        
        Internally updates the
            .computation_results


        Args:
            active_computation_params (Optional[DynamicParameters], optional): _description_. Defaults to None.
            enabled_filter_names (_type_, optional): _description_. Defaults to None.
            overwrite_extant_results (bool, optional): _description_. Defaults to False.
            computation_functions_name_includelist (_type_, optional): _description_. Defaults to None.
            computation_functions_name_excludelist (_type_, optional): _description_. Defaults to None.
            fail_on_exception (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        History:
            factored out of `NeuropyPipeline` for use in GlobalComputationFunctions
        """
        assert (self.can_compute), "Current stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
            computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)


    # ==================================================================================================================== #
    # CLASS/STATIC METHODS                                                                                                 #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['computationResult'], input_requires=[], output_provides=[], uses=[], used_by=['cls.continue_computations_if_needed', 'perform_specific_computation', 'perform_action_for_all_contexts'], creation_date='2024-10-07 15:12', related_items=[])
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

        #TODO 2023-11-13 14:43: - [ ] This should require the computation_config.pf_params.computation_epochs is limited to .t_start, .t_stop of `active_session`:

        # active_session.t_start, active_session.t_stop
        # if hasattr(computation_config, 'pf_params')
        # computation_config.pf_params.computation_epochs

        output_result = ComputationResult(active_session, computation_config, computed_data=DynamicParameters(), accumulated_errors=DynamicParameters(), computation_times=DynamicParameters()) # Note that this active_session should be correctly filtered
        
        return output_result
    

    @function_attributes(short_name=None, tags=['computationResult'], input_requires=[], output_provides=[], uses=[], used_by=['build_initial_global_computationResult', 'cls.continue_computations_if_needed', 'perform_specific_computation', 'perform_action_for_all_contexts'], creation_date='2024-10-07 15:12', related_items=[])
    @classmethod
    def _build_initial_global_computationResult(cls, curr_active_pipeline, active_session, computation_config=None) -> ComputationResult:
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): this is the filtered data session
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters # merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, _perform_specific_epochs_decoding_Parameters, _DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

        output_result = ComputationResult(active_session, computation_config=computation_config, computed_data=DynamicParameters(), accumulated_errors=DynamicParameters(), computation_times=DynamicParameters()) # Note that this active_session should be correctly filtered
        ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
        if output_result.computation_config is None:
            print('._build_initial_global_computationResult(...): global_computation_results.computation_config is None! Making new one!')
            # curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            # output_result.computation_config = curr_global_param_typed_parameters
            output_result.update_config_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            print(f'\tdone. Pipeline needs resave!')
        else:
            # curr_global_param_typed_parameters: ComputationKWargParameters = output_result.computation_config
            pass
        
        return output_result
    

    @function_attributes(short_name=None, tags=['computationResult', 'global', 'instance', 'USEFUL'], input_requires=[], output_provides=[], uses=['cls._build_initial_global_computationResult'], used_by=[], creation_date='2025-01-07 12:13', related_items=[])
    def build_initial_global_computationResult(self, computation_config=None) -> ComputationResult:
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): this is the filtered data session
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        if computation_config is None:
            from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters # merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, _perform_specific_epochs_decoding_Parameters, _DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

            curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=self)
            computation_config = curr_global_param_typed_parameters
            
        return self._build_initial_global_computationResult(curr_active_pipeline=self, active_session=self.sess, computation_config=computation_config)



    @function_attributes(short_name=None, tags=['compute', 'main'], input_requires=[], output_provides=[],
                          uses=[], used_by=['perform_registered_computations_single_context', 'rerun_failed_computations_single_context', 'run_specific_computations_single_context'], creation_date='2024-10-07 15:08', related_items=[])
    @staticmethod
    def _execute_computation_functions(active_computation_functions, previous_computation_result=None, computation_kwargs_list=None, fail_on_exception:bool = False, progress_logger_callback=None, are_global:bool=False, debug_print=False) -> ComputationResult:
        """ actually performs the provided computations in active_computation_functions """
        if computation_kwargs_list is None:
            computation_kwargs_list = [{} for _ in active_computation_functions]
        assert len(computation_kwargs_list) == len(active_computation_functions)

        # computation_times_key_fn = lambda fn: fn
        computation_times_key_fn = lambda fn: str(fn.__name__) # Use only the functions name. I think this makes the .computation_times field picklable
        
        if (len(active_computation_functions) > 0):
            if debug_print:
                print(f'Performing _execute_computation_functions(...) with {len(active_computation_functions)} registered_computation_functions...')
            if progress_logger_callback is not None:
                progress_logger_callback(f'Performing _execute_computation_functions(...) with {len(active_computation_functions)} registered_computation_functions...')


            if are_global:
                assert isinstance(previous_computation_result, (dict, DynamicParameters)), 'ERROR: previous_computation_result must be a dict or DynamicParameters object when are_global=True'
                # global_kwargs = dict(owning_pipeline_reference=self, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False)
                previous_computation_result = list(previous_computation_result.values()) # get the list of values since the global computation functions expects positional arguments
                # Wrap the active functions in the wrapper that extracts their arguments:
                active_computation_functions = [_wrap_multi_context_computation_function(a_global_fcn) for a_global_fcn in active_computation_functions]
            
            computation_times = dict() # empty list to keep track of when the computations were completed
            
            if fail_on_exception:
                ## normal version that fails on any exception:
                total_num_funcs = len(active_computation_functions)
                for i, f in enumerate(active_computation_functions):
                    if progress_logger_callback is not None:
                        progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
                    previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here
                    # Log the computation copmlete time:
                    computation_times[computation_times_key_fn(f)] = datetime.now()
                
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
                for i, f in enumerate(active_computation_functions):
                    if progress_logger_callback is not None:
                        progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
                    try:
                        # evaluate the function 'f' using the result provided from the previous output or the initial input
                        temp_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here
                    except (TypeError, ValueError, NameError, AttributeError, KeyError, NotImplementedError) as e:
                        exception_info = sys.exc_info()
                        accumulated_errors[f] = CapturedException(e, exception_info, previous_computation_result)
                        # accumulated_errors.append(e) # add the error to the accumulated error array
                        temp_result = previous_computation_result # restore the result from prior to the calculations?
                        # result shouldn't be updated unless there wasn't an error, so it should be fine to move on to the next function
                        if error_logger is not None:
                            error_logger(f'\t Encountered error: {accumulated_errors[f]} continuing.')
                    except Exception as e:
                        print(f'UNHANDLED EXCEPTION: {e}')
                        raise
                    else:
                        # only if no error occured do we commit the temp_result to result
                        previous_computation_result = temp_result
                        if progress_logger_callback is not None:
                            progress_logger_callback('\t done.')
                        # Log the computation complete time:
                        computation_times[computation_times_key_fn(f)] = datetime.now()

            
            if debug_print:
                print(f'_execute_computation_functions(...): \n\taccumulated_errors: {accumulated_errors}\n\tcomputation_times: {computation_times}')
            

            if are_global:
                # Extract the global_computation_results from the returned list for global computations:
                # owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False
                assert isinstance(previous_computation_result, list)
                previous_computation_result = previous_computation_result[1] # get the global_computation_results object
                assert isinstance(previous_computation_result, ComputationResult)

            # Add the computation_time to the computation result:
            previous_computation_result.computation_times |= (computation_times or {}) # are they not being set?
            # for k,v in (computation_times or {}).items():
            #     previous_computation_result.computation_times[k] = v
        
            # previous_computation_result.computation_times.update(DynamicParameters.init_from_dict(computation_times))
            # Add the accumulated_errors to the computation result:
            # Used to just replace `previous_computation_result.accumulated_errors`:
            # previous_computation_result.accumulated_errors = (accumulated_errors or {})
            previous_computation_result.accumulated_errors |= (accumulated_errors or {})
            # for k,v in (accumulated_errors or {}).items():
            #     previous_computation_result.accumulated_errors[k] = v
            if len(accumulated_errors or {}) > 0:
                if progress_logger_callback is not None:
                    progress_logger_callback(f'WARNING: there were {len(accumulated_errors)} errors that occurred during computation. Check these out by looking at computation_result.accumulated_errors.')
                    
                warn(f'WARNING: there were {len(accumulated_errors)} errors that occurred during computation. Check these out by looking at computation_result.accumulated_errors.')
                error_desc_str = f'{len(accumulated_errors or {})} errors.'
            else:
                error_desc_str = f'no errors!'
            
            if progress_logger_callback is not None:
                progress_logger_callback(f'\t all computations complete! (Computed {len(active_computation_functions)} with {error_desc_str}.')
                                
            return previous_computation_result
            
        else:
            ## empty active_computation_functions list: nothing to compute!
            if progress_logger_callback is not None:
                progress_logger_callback(f'No registered_computation_functions, skipping extended computations.')
                
            if debug_print:
                print(f'No registered_computation_functions, skipping extended computations.')
            return previous_computation_result # just return the unaltered result

    @function_attributes(short_name=None, tags=['compute'], input_requires=[], output_provides=[], uses=['cls._build_initial_computationResult'], used_by=[], creation_date='2024-10-07 15:11', related_items=[])
    @classmethod    
    def continue_computations_if_needed(cls, curr_active_pipeline, active_computation_params=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_includelist=None, computation_functions_name_excludelist=None, fail_on_exception:bool=False, debug_print=False):
        """ continues computations for a pipeline 

            NOTE: TODO: this is not yet implemented.
            Calls perform_specific_context_registered_computations(...) to do the actual comptuations

        
            TODO: the rest of the system can't work until we have a way of associating the previously computed results with the functions that compute them. As it stands we don't know anything about whether a new function was registered after the computations were complete, etc.
                DESIGN GOAL: don't make this too complicated.
        
        Usage:
            continue_computations_if_needed(curr_active_pipeline, active_session_computation_configs[0], overwrite_extant_results=False, computation_functions_name_excludelist=['_perform_spike_burst_detection_computation'], debug_print=True)

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
            curr_active_pipeline.computation_results[an_incomplete_config_name] = curr_active_pipeline.perform_specific_context_registered_computations(curr_active_pipeline.computation_results[an_incomplete_config_name], computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, debug_print=debug_print)

            ## TODO: initially compute incomplete_computed_config_dict items...

            ## Next look for previously failed computation results:

            ## Next look for previously complete computation results that lack computations for functions explicitly specified in the includelist (if provided):

            ## Then look for previously complete computation results that are missing computations that have been registered after they were computed, or that were previously part of the excludelist but now are not:


    @function_attributes(short_name=None, tags=['valid_track_times'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-05 16:18', related_items=[])
    @classmethod
    def perform_find_first_and_last_valid_position_times(cls, pos_df, loaded_track_limits):
        """ uses the positions and the loaded_track_limits to determine the first and last valid times for each session. 
        
        Usage:
            pos_df: pd.DataFrame = deepcopy(global_session.position.to_dataframe())
            active_sess_config = deepcopy(curr_active_pipeline.active_sess_config)
            # absolute_start_timestamp: float = active_sess_config.absolute_start_timestamp
            loaded_track_limits = active_sess_config.loaded_track_limits # x_midpoint, 

            (first_valid_pos_time, last_valid_pos_time) = _find_first_and_last_valid_position_times(pos_df, loaded_track_limits)
            (first_valid_pos_time, last_valid_pos_time)
            
        """
        # loaded_track_limits
        xlim_min, xlim_max = loaded_track_limits['long_xlim'] # [138.393 146.947]
        ylim_min, ylim_max = loaded_track_limits['long_ylim']

        pos_df['is_outside_xlim'] = np.logical_or((pos_df['x'] < xlim_min), (pos_df['x'] > xlim_max))
        pos_df['is_outside_ylim'] = np.logical_or((pos_df['y'] < ylim_min), (pos_df['y'] > ylim_max))
        pos_df['is_outside_bounds'] = np.logical_or(pos_df['is_outside_xlim'], pos_df['is_outside_ylim'])

        valid_pos_df = pos_df[np.logical_not(pos_df['is_outside_bounds'])]
        first_valid_pos_time: float = valid_pos_df['t'].min()
        last_valid_pos_time: float = valid_pos_df['t'].max()
        return (first_valid_pos_time, last_valid_pos_time)

    @function_attributes(short_name=None, tags=['valid_track_times'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-05 16:18', related_items=[])
    def find_first_and_last_valid_position_times(self):
        """ uses the positions and the loaded_track_limits to determine the first and last valid times for each session. 
        
        Usage:
        
            active_sess_config = deepcopy(curr_active_pipeline.active_sess_config)
            # absolute_start_timestamp: float = active_sess_config.absolute_start_timestamp
            loaded_track_limits = active_sess_config.loaded_track_limits # x_midpoint, 

            (first_valid_pos_time, last_valid_pos_time) = curr_active_pipeline.find_first_and_last_valid_position_times()
            (first_valid_pos_time, last_valid_pos_time)
            
        """
        # pos_df: pd.DataFrame = deepcopy(self.global_session.position.to_dataframe())
        pos_df: pd.DataFrame = deepcopy(self.sess.position.to_dataframe())
        active_sess_config = deepcopy(self.active_sess_config)
        # absolute_start_timestamp: float = active_sess_config.absolute_start_timestamp
        loaded_track_limits = active_sess_config.loaded_track_limits # x_midpoint, 
        # active_sess_config.
        # curr_config.pf_params.grid_bin_bounds = grid_bin_bounds # same bounds for all
        # if override_dict.get('track_start_t', None) is not None:
        # 	track_start_t = override_dict['track_start_t']
        # 	curr_config.pf_params.track_start_t = track_start_t
        # else:
        # 	curr_config.pf_params.track_start_t = None

        # if override_dict.get('track_end_t', None) is not None:
        # 	track_end_t = override_dict['track_end_t']
        # 	curr_config.pf_params.track_end_t = track_end_t
        # else:
        # 	curr_config.pf_params.track_end_t = None
        (first_valid_pos_time, last_valid_pos_time) = self.perform_find_first_and_last_valid_position_times(pos_df, loaded_track_limits)
        return (first_valid_pos_time, last_valid_pos_time)
        


        

# self


# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
class PipelineWithComputedPipelineStageMixin:
    """ To be added to the pipeline to enable conveninece access ot its pipeline stage post Computed stage. """
    # @property
    # def stage(self) -> ComputedPipelineStage:
    #     """The stage property."""
    #     return self._stage
    

    ## Computed Properties:
    @property
    def is_computed(self) -> bool:
        """The is_computed property. TODO: Needs validation/Testing """
        return (self.can_compute and (self.computation_results is not None) and (len(self.computation_results) > 0))
        # return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage) and (self.computation_results is not None) and (len(self.computation_results) > 0))

    @property
    def can_compute(self) -> bool:
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
    def registered_computation_functions(self) -> List[Callable]:
        """The registered_computation_functions property."""
        return self.stage.registered_computation_functions
        
    @property
    def registered_computation_function_names(self) -> List[str]:
        """The registered_computation_function_names property."""
        return self.stage.registered_computation_function_names
    
    @property
    def registered_computation_function_dict(self) -> Dict[str, Callable]:
        """The registered_computation_function_dict property can be used to get the corresponding function from the string name."""
        return self.stage.registered_computation_function_dict
    
    @property
    def registered_computation_function_docs_dict(self) -> Dict[str, str]:
        """Returns the doc strings for each registered computation function. This is taken from their docstring at the start of the function defn, and provides an overview into what the function will do."""
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_computation_function_dict.items()}

    # Global Computation Properties ______________________________________________________________________________________ #
    @property
    def global_computation_results(self) -> Optional[ComputationResult]:
        """The global_computation_results property, accessed through the stage."""
        return self.stage.global_computation_results
    
    @property
    def global_computation_config(self):
        """The global_computation_results.config property, accessed through the stage."""
        return self.stage.global_computation_results.computation_config



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
    def registered_global_computation_functions(self) -> List[Callable]:
        """The registered_global_computation_functions property."""
        return self.stage.registered_global_computation_functions
        
    @property
    def registered_global_computation_function_names(self) -> List[str]:
        """The registered_global_computation_function_names property."""
        return self.stage.registered_global_computation_function_names
    
    @property
    def registered_global_computation_function_dict(self) -> Dict[str, Callable]:
        """The registered_global_computation_function_dict property can be used to get the corresponding function from the string name."""
        return self.stage.registered_global_computation_function_dict
    
    @property
    def registered_global_computation_function_docs_dict(self) -> Dict[str, str]:
        """Returns the doc strings for each registered computation function. This is taken from their docstring at the start of the function defn, and provides an overview into what the function will do."""
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_global_computation_function_dict.items()}
    

    # 'merged' refers to the fact that both global and non-global computation functions are included _____________________ #
    @property
    def registered_merged_computation_function_dict(self) -> Dict[str, Callable]:
        """build a merged function dictionary containing both global and non-global functions:"""
        return self.stage.registered_merged_computation_function_dict
    @property
    def registered_merged_computation_functions(self) -> List[Callable]:
        return self.stage.registered_merged_computation_functions
    @property
    def registered_merged_computation_function_names(self) -> List[str]:
        return self.stage.registered_merged_computation_function_names

    def get_merged_computation_function_validators(self) -> Dict[str, SpecificComputationValidator]:
        ## From the registered computation functions, gather any validators and build the SpecificComputationValidator for them, then append them to `_comp_specifiers`:
        return {k:SpecificComputationValidator.init_from_decorated_fn(v) for k,v in self.registered_merged_computation_function_dict.items() if hasattr(v, 'validate_computation_test') and (v.validate_computation_test is not None)}


    # ==================================================================================================================== #
    # Dependency Parsing/Determination                                                                                     #
    # ==================================================================================================================== #
    def find_matching_validators(self, probe_fn_names: List[str], skip_reload_computation_fcns: bool = False, debug_print=False):
        """
        Usage:
            remaining_comp_specifiers_dict, found_matching_validators, provided_global_keys = curr_active_pipeline.find_matching_validators(probe_fn_names=['long_short_decoding_analyses','long_short_fr_indicies_analyses'])
            provided_global_keys
        """
        if not skip_reload_computation_fcns:
            self.reload_default_computation_functions()
        return SpecificComputationValidator.find_matching_validators(remaining_comp_specifiers_dict=deepcopy(self.get_merged_computation_function_validators()), probe_fn_names=probe_fn_names)

    @function_attributes(short_name=None, tags=['dependencies'], input_requires=[], output_provides=[], uses=[], used_by=['self.find_downstream_dependencies'], creation_date='2025-01-13 12:32', related_items=[])
    def find_immediate_dependencies(self, provided_global_keys: List[str], provided_local_keys: Optional[List[str]] = None, skip_reload_computation_fcns: bool = False, debug_print=False):
        """
        Usage:

        remaining_comp_specifiers_dict, dependent_validators, (provided_global_keys, provided_local_keys)  = SpecificComputationValidator.find_immediate_dependencies(remaining_comp_specifiers_dict=remaining_comp_specifiers_dict, provided_global_keys=provided_global_keys, provided_local_keys=provided_local_keys)
        provided_global_keys

        """
        if not skip_reload_computation_fcns:
            self.reload_default_computation_functions()
        return SpecificComputationValidator.find_immediate_dependencies(remaining_comp_specifiers_dict=deepcopy(self.get_merged_computation_function_validators()), provided_global_keys=provided_global_keys, provided_local_keys=provided_local_keys)

    @function_attributes(short_name=None, tags=['dependencies', 'downstream'], input_requires=[], output_provides=[], uses=['self.find_immediate_dependencies'], used_by=[], creation_date='2025-01-13 12:32', related_items=[])
    def find_downstream_dependencies(self, provided_local_keys: List[str]=None, provided_global_keys: List[str]=None, skip_reload_computation_fcns: bool = False, debug_print=False):
        """
        Usage:

        dependent_validators, (provided_global_keys, provided_local_keys) = curr_active_pipeline.find_downstream_dependencies(provided_global_keys=provided_global_keys)
        provided_global_keys
        provided_local_keys
        """
        from neuropy.utils.indexing_helpers import flatten
        if skip_reload_computation_fcns:
            self.reload_default_computation_functions()
            
        _comp_specifiers_dict: Dict[str, SpecificComputationValidator] = self.get_merged_computation_function_validators()
        validators = deepcopy(_comp_specifiers_dict) # { ... }  # Your validators here
        dependent_validators = {}
        for a_name, a_validator in validators.items():
            if provided_global_keys is not None:
                for a_changed_key in provided_global_keys:
                    if (a_changed_key in a_validator.results_specification.requires_global_keys):
                        dependent_validators[a_name] = a_validator
            if provided_local_keys is not None:
                for a_changed_key in provided_local_keys:
                    if (a_changed_key in a_validator.results_specification.requires_local_keys):
                        dependent_validators[a_name] = a_validator

        # dependent_validators_names = [k for k, v in dependent_validators.items()]
        provided_global_keys = list(set(flatten([v.provides_global_keys for v in dependent_validators.values()]))) # ['DirectionalMergedDecoders', 'DirectionalDecodersEpochsEvaluations', 'TrainTestSplit', 'TrialByTrialActivity']
        provided_local_keys = list(set(flatten([v.provides_local_keys for v in dependent_validators.values()])))
        
        ## OUTPUT: dependent_validators_provides, dependent_validators

        ## loop until no changes
        max_num_iterations: int = 5
        curr_iter: int = 0
        _prev_provided_global_keys = []
        _prev_provided_local_keys = []
        while ((curr_iter < max_num_iterations) and ((provided_global_keys != _prev_provided_global_keys) or (provided_local_keys != _prev_provided_local_keys))):
            _prev_provided_global_keys = provided_global_keys
            _prev_provided_local_keys = provided_local_keys
            curr_order_remaining_comp_specifiers_dict, curr_order_dependent_validators, (curr_order_provided_global_keys, curr_order_provided_local_keys) = self.find_immediate_dependencies(provided_global_keys=provided_global_keys, provided_local_keys=provided_local_keys, skip_reload_computation_fcns=True, debug_print=debug_print)
            dependent_validators.update(curr_order_dependent_validators)
            provided_global_keys = list(set(provided_global_keys + curr_order_provided_global_keys))
            provided_local_keys = list(set(provided_local_keys + curr_order_provided_local_keys))
            curr_iter = curr_iter + 1

        return dependent_validators, (provided_global_keys, provided_local_keys)


    def find_provided_result_keys(self, probe_fn_names: List[str]) -> List[str]:
        """ returns a list of computed properties that the specified functions provide. 
        
        (provided_global_keys, provided_local_keys) = curr_active_pipeline.find_provided_result_keys(probe_fn_names=['perform_wcorr_shuffle_analysis',  'merged_directional_placefields', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring'])
        provided_global_keys
        provided_local_keys
        """
        self.reload_default_computation_functions()
        return SpecificComputationValidator.find_provided_result_keys(remaining_comp_specifiers_dict=deepcopy(self.get_merged_computation_function_validators()), probe_fn_names=probe_fn_names)
    

    def find_validators_providing_results(self, probe_provided_result_keys: List[str], return_flat_list:bool=True) -> List[str]:
        """ returns a list of computed properties that the specified functions provide. 
        
            found_validators_dict = curr_active_pipeline.find_validators_providing_results(probe_provided_result_keys=['DirectionalMergedDecoders', 'DirectionalDecodersEpochsEvaluations', 'SequenceBased'])
            [v.computation_fn_name for v in found_validators_dict]

        """
        self.reload_default_computation_functions()
        return SpecificComputationValidator.find_validators_providing_results(remaining_comp_specifiers_dict=deepcopy(self.get_merged_computation_function_validators()), probe_provided_result_keys=probe_provided_result_keys, return_flat_list=return_flat_list)


    @function_attributes(short_name=None, tags=['requirements'], input_requires=[], output_provides=[], uses=['SpecificComputationValidator.find_immediate_requirements'], used_by=['find_upstream_requirements'], creation_date='2025-01-13 09:16', related_items=[])
    def find_immediate_requirements(self, required_global_keys: List[str], required_local_keys: Optional[List[str]] = None, skip_reload_computation_fcns: bool = False, debug_print=False):
        """
        Identifies the immediate requirements for the given global or local keys.
        
        Parameters:
            required_global_keys (List[str]): The global keys for which requirements are checked.
            required_local_keys (List[str]): The local keys for which requirements are checked (optional).
            debug_print (bool): If True, enables debug prints for detailed output.
            
        Returns:
            Dict[str, SpecificComputationValidator]: Validators that have the provided keys as their requirements.
            
            
        Usage:
        
            _curr_remaining_comp_specifiers_dict, _curr_required_validators, _curr_required_global_keys = curr_active_pipeline.find_immediate_requirements(required_global_keys=list(remaining_required_global_result_keys))
            print(f"_curr_required_global_keys: {_curr_required_global_keys}")

        """
        if not skip_reload_computation_fcns:
            self.reload_default_computation_functions()
        # assert required_local_keys is None, f"required_local_keys: {required_local_keys} but these are not currently supported!"
        if (required_local_keys is None):
            required_local_keys = [] ## empty list
        return SpecificComputationValidator.find_immediate_requirements(remaining_comp_specifiers_dict=deepcopy(self.get_merged_computation_function_validators()), required_global_keys=required_global_keys, required_local_keys=required_local_keys)


    @function_attributes(short_name=None, tags=['requirements'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-13 09:17', related_items=[])
    def find_upstream_requirements(self, required_global_keys: List[str], required_local_keys: Optional[List[str]] = None, skip_reload_computation_fcns: bool = False, debug_print=False):
        """
        Identifies all upstream requirements recursively for the given global keys.
        
        Parameters:
            required_global_keys (List[str]): The global keys for which upstream requirements are traced.
            debug_print (bool): If True, enables debug prints for detailed output.
            
        Returns:
            Tuple[Dict[str, SpecificComputationValidator], List[str]]: 
                - Validators involved in fulfilling the requirements.
                - Global keys that these validators provide.
                
                
        Usage:
        
            upstream_required_validators, required_global_keys = curr_active_pipeline.find_upstream_requirements(required_global_keys=['DirectionalDecodersDecoded'])
            required_global_keys # ['DirectionalLaps', 'DirectionalMergedDecoders', 'DirectionalDecodersDecoded']

        """
        if not skip_reload_computation_fcns:
            self.reload_default_computation_functions()
        
        # assert (required_local_keys is None), f"required_local_keys: {required_local_keys} but these are not currently supported!"
        if required_local_keys is None:
            required_local_keys = []
            
        ## INPUTS: debug_print, remaining_required_global_result_keys
        remaining_required_global_result_keys = set(required_global_keys) ## convert to a set to ensure uniqueness
        remaining_required_local_result_keys = set(required_local_keys) ## convert to a set to ensure uniqueness

        upstream_required_validators = {}
        
        ## first iteration:
        _curr_remaining_comp_specifiers_dict, _curr_required_validators, (_curr_required_global_keys, _curr_required_local_keys) = self.find_immediate_requirements(required_global_keys=list(remaining_required_global_result_keys), required_local_keys=list(remaining_required_local_result_keys))
        if debug_print:
            print(f"_curr_required_global_keys: {_curr_required_global_keys}")
            print(f"_curr_required_local_keys: {_curr_required_local_keys}")
        remaining_required_global_result_keys = remaining_required_global_result_keys.union(_curr_required_global_keys)
        remaining_required_local_result_keys = remaining_required_local_result_keys.union(_curr_required_local_keys)
        upstream_required_validators.update(_curr_required_validators)
        if debug_print:
            print(f"remaining_required_global_result_keys: {remaining_required_global_result_keys}")
            print(f"remaining_required_local_result_keys: {remaining_required_local_result_keys}")
        ## OUTPUTS: remaining_required_global_result_keys, upstream_required_validators

        max_num_iterations: int = 5
        curr_iter: int = 0
        _prev_provided_global_keys = set([]) ## empty set
        _prev_provided_local_keys = set([]) ## empty set
        
        while (curr_iter < max_num_iterations) and ((remaining_required_global_result_keys != _prev_provided_global_keys) or (remaining_required_local_result_keys != _prev_provided_local_keys)):
            _prev_provided_global_keys = remaining_required_global_result_keys
            _prev_provided_local_keys = remaining_required_local_result_keys
            
            _curr_remaining_comp_specifiers_dict, _curr_required_validators, (_curr_required_global_keys, _curr_required_local_keys) = self.find_immediate_requirements(required_global_keys=list(remaining_required_global_result_keys), required_local_keys=list(remaining_required_local_result_keys))
            if debug_print:
                print(f"curr_iter: {curr_iter}/{max_num_iterations}: _curr_required_global_keys: {_curr_required_global_keys}, _curr_required_local_keys: {_curr_required_local_keys}")
            remaining_required_global_result_keys = remaining_required_global_result_keys.union(_curr_required_global_keys)
            remaining_required_local_result_keys = remaining_required_local_result_keys.union(_curr_required_local_keys)
            upstream_required_validators.update(_curr_required_validators)
            if debug_print:
                print(f"remaining_required_global_result_keys: {remaining_required_global_result_keys}")
                print(f"remaining_required_local_result_keys: {remaining_required_local_result_keys}")
            curr_iter += 1
            

        ## END while...
        required_global_keys = list(remaining_required_global_result_keys) ## convert to a list
        required_local_keys = list(remaining_required_local_result_keys) ## convert to a list
        return upstream_required_validators, (required_global_keys, required_local_keys)



    @function_attributes(short_name=None, tags=['parameters', 'update'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-28 18:45', related_items=[])
    def apply_changed_parameters(self, minimum_inclusion_fr_Hz=5.0, included_qclu_values = [1, 2, 4, 9], is_dry_run: bool=True):
        """ Applies the changed parameters to the pipeline and recomputes as needed.

        Usage:
            minimum_inclusion_fr_Hz = 5.0
            included_qclu_values = [1, 2, 4, 9]

            (dependent_validators, provided_global_keys), old_new_values_change_dict = curr_active_pipeline.apply_changed_parameters(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values,
                                                                                                                                    is_dry_run=False)
            old_new_values_change_dict

        """
        original_minimum_inclusion_fr_Hz: float = deepcopy(self.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz)
        original_included_qclu_values: List[int] = deepcopy(self.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values)


        ## Determine which computations need to be re-ran after changing a config properity:
        did_change_list = []
        old_new_values_change_dict = {}
        if minimum_inclusion_fr_Hz != original_minimum_inclusion_fr_Hz:
            did_change_list.append('global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz')
            old_new_values_change_dict['global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz'] = {'old': original_minimum_inclusion_fr_Hz, 'new': minimum_inclusion_fr_Hz}
            
        if included_qclu_values != original_included_qclu_values:
            did_change_list.append('global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values')
            old_new_values_change_dict['global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values'] = {'old': original_included_qclu_values, 'new': included_qclu_values}    
        
        did_any_change: bool = (len(did_change_list) > 0)
        if is_dry_run:
            print(f'did_change_list: {did_change_list}')
            print(f'did_any_change: {did_any_change}')
        
        dependent_validators, provided_global_keys = self.find_downstream_dependencies(provided_local_keys=did_change_list, provided_global_keys=None)
        # provided_global_keys: ['SequenceBased', 'TrainTestSplit', 'TrialByTrialActivity', 'DirectionalDecodersEpochsEvaluations', 'DirectionalDecodersDecoded', 'DirectionalMergedDecoders', 'RankOrder']
        
        if (not is_dry_run):
            ## Apply the changes:
            self.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz = minimum_inclusion_fr_Hz
            self.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values = included_qclu_values

            ## recompute:
            for k, v in dependent_validators.items():
                v.try_remove_provided_keys(curr_active_pipeline=self)
                v.try_computation_if_needed(curr_active_pipeline=self, computation_filter_name=None)
                # remaining_comp_specifiers_dict, dependent_validators, provided_global_keys = SpecificComputationValidator.find_immediate_dependencies(remaining_comp_specifiers_dict=v, provided_global_keys=provided_global_keys)
                # provided_global_keys
        else:
            ## dry_run:
            print(f'provided_global_keys: {provided_global_keys}')
            print(f'is_dry_run == True mode, so recomputations will not be performed.')

        return (dependent_validators, provided_global_keys), old_new_values_change_dict


    def reload_default_computation_functions(self):
        """ reloads/re-registers the default display functions after adding a new one """
        self.stage.reload_default_computation_functions()
        
    def register_computation(self, registered_name, computation_function, is_global:bool):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_computation(registered_name, computation_function, is_global)

        
    ## Computation Helpers: 
    # perform_computations: The main computation function for the pipeline
    def perform_computations(self, active_computation_params: Optional[DynamicParameters]=None, enabled_filter_names=None, overwrite_extant_results=False, computation_functions_name_includelist=None, computation_functions_name_excludelist=None, fail_on_exception:bool=False, debug_print=False):
        """The main computation function for the pipeline.

        Internally updates the
            .computation_results


        Args:
            active_computation_params (Optional[DynamicParameters], optional): _description_. Defaults to None.
            enabled_filter_names (_type_, optional): _description_. Defaults to None.
            overwrite_extant_results (bool, optional): _description_. Defaults to False.
            computation_functions_name_includelist (_type_, optional): _description_. Defaults to None.
            computation_functions_name_excludelist (_type_, optional): _description_. Defaults to None.
            fail_on_exception (bool, optional): _description_. Defaults to False.
            debug_print (bool, optional): _description_. Defaults to False.

        History:
            perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS
        """
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        progress_logger_callback=(lambda x: self.logger.info(x))
        # self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
        #     computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)

        # Calls self.stage's version:
        self.stage.perform_computations(enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
            computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)
        
        # Global MultiContext computations will be done here:
        if progress_logger_callback is not None:
            progress_logger_callback(f'Performing global computations...')

        ## TODO: BUG: WHY IS THIS CALLED TWICE? Was this supposed to be the global implementation or something?
        # self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
        #     computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=(lambda x: self.logger.info(x)), debug_print=debug_print)
        # self.stage.evaluate_computations_for_single_params(active_computation_params, enabled_filter_names=enabled_filter_names, overwrite_extant_results=overwrite_extant_results,
            # computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=(lambda x: self.logger.info(x)), debug_print=debug_print)


    def rerun_failed_computations(self, previous_computation_result, fail_on_exception:bool=False, debug_print=False):
        """ retries the computation functions that previously failed and resulted in accumulated_errors in the previous_computation_result """
        # return self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, ... # TODO: refactor to use new layout
        return self.stage.rerun_failed_computations(previous_computation_result, fail_on_exception=fail_on_exception, debug_print=debug_print)
    

    def perform_specific_computation(self, active_computation_params=None, enabled_filter_names=None, computation_functions_name_includelist=None, computation_kwargs_list=None, fail_on_exception:bool=False, debug_print=False):
        """ perform a specific computation (specified in computation_functions_name_includelist) in a minimally destructive manner using the previously recomputed results:
        Passthrough wrapper to self.stage.perform_specific_computation(...) with the same arguments.

        Updates:
            curr_active_pipeline.computation_results
        """
        # self.stage is of type ComputedPipelineStage
        return self.stage.perform_specific_computation(active_computation_params=active_computation_params, enabled_filter_names=enabled_filter_names, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, fail_on_exception=fail_on_exception, debug_print=debug_print)
    

    # Utility/Debugging Functions:
    def perform_drop_entire_computed_config(self, config_names_to_drop = ['maze1_rippleOnly', 'maze2_rippleOnly']):
        """ Loops through all the configs and drops all results of the specified configs
        2023-10-25 - Seems to work to drop ALL of the computed items for a specified set of configs/contexts (not a specific computed item across configs/contexts)

        Usage:
          curr_active_pipeline.perform_drop_entire_computed_config(config_names_to_drop=['maze1_odd_laps', 'maze1_even_laps', 'maze2_odd_laps', 'maze2_even_laps'])

        """
        # config_names_to_drop
        print(f'_drop_computed_items(config_names_to_drop: {config_names_to_drop}):\n\tpre keys: {list(self.active_configs.keys())}')
        
        for a_config_name in config_names_to_drop:
            a_config_to_drop = self.active_configs.pop(a_config_name, None)
            if a_config_to_drop is not None:
                print(f'\tpreparing to drop: {a_config_name}')
                
                _dropped_computation_results = self.computation_results.pop(a_config_name, None)
                a_filter_context_to_drop = self.filtered_contexts.pop(a_config_name, None)
                if a_filter_context_to_drop is not None:
                    _dropped_display_items = self.display_output.pop(a_filter_context_to_drop, None)
                ## filtered_sessions, filtered_epochs
                a_filter_epoch_to_drop = self.filtered_epochs.pop(a_config_name, None)
                a_filter_session_to_drop = self.filtered_sessions.pop(a_config_name, None)

            print(f'\t dropped.')
            
        print(f'\tpost keys: {list(self.active_configs.keys())}')

    def perform_drop_computed_result(self, computed_data_keys_to_drop, config_names_includelist=None, debug_print=False):
        """ Loops through all computed items and drops a specific result across all configs/contexts  
        Inputs:
            computed_data_keys_to_drop: list of specific results to drop for each context
            config_names_includelist: optional list of names to operate on. No changes will be made to results for configs not in the includelist
        """
        # config_names_to_drop
        if debug_print:
            print(f'perform_drop_computed_result(computed_data_keys_to_drop: {computed_data_keys_to_drop}, config_names_includelist: {config_names_includelist})')

        if isinstance(computed_data_keys_to_drop, str):
            computed_data_keys_to_drop = [computed_data_keys_to_drop] # wrap in a list

        if config_names_includelist is None:
            # if no includelist specified, get all computed keys:
            config_names_includelist = self.active_completed_computation_result_names # ['maze1_PYR', 'maze2_PYR', 'maze_PYR']
        

        ## Global:
        global_dropped_keys = []
        global_not_found_keys_to_drop = []

        for a_key_to_drop in computed_data_keys_to_drop:
            curr_global_computation_results = self.global_computation_results
            curr_global_computed_data = curr_global_computation_results.computed_data
            # curr_global_computed_data
            a_result_to_drop = curr_global_computed_data.pop(a_key_to_drop, None) # AttributeError: 'ComputationResult' object has no attribute 'pop'
            ## TODO: Should we drop from curr_computed_results.accumulated_errors in addition to curr_computed_results.computed_data? Probably fine not to.
            if a_result_to_drop is not None:
                # Successfully dropped
                print(f"\t Dropped global_computation_results.computed_data['{a_key_to_drop}'].")
                global_dropped_keys.append(a_key_to_drop)
                # remove from .computation_times, .accumulated_errors as well
                curr_global_computation_results.computation_times.pop(a_key_to_drop, None)
                curr_global_computation_results.accumulated_errors.pop(a_key_to_drop, None)
                
            else:
                print(f"\t global_computation_results.computed_data['{a_key_to_drop}'] did not exist.")
                global_not_found_keys_to_drop.append(a_key_to_drop) # key might be local


        # only check locals for global_not_found_keys
        
        
        local_dropped_keys = []
        local_not_found_keys_to_drop = []
        
        ## Loop across all computed contexts
        for a_config_name, curr_computed_results in self.computation_results.items():
            if a_config_name in config_names_includelist:            
                # remove the results from this config
                for a_key_to_drop in global_not_found_keys_to_drop:
                    try:
                        a_result_to_drop = curr_computed_results.pop(a_key_to_drop, []) 
                    except AttributeError as e:
                        # this seems like it would always be the case because `# AttributeError: 'ComputationResult' object has no attribute 'pop'`
                        a_result_to_drop = curr_computed_results.computed_data.pop(a_key_to_drop, [])

                    ## TODO: Should we drop from curr_computed_results.accumulated_errors in addition to curr_computed_results.computed_data? Probably fine not to.
                    if a_result_to_drop is not []:
                        # Successfully dropped
                        print(f"\t Dropped computation_results['{a_config_name}'].computed_data['{a_key_to_drop}'].")
                        local_dropped_keys.append(a_key_to_drop)
                    else:
                        print(f"\t computation_results['{a_config_name}'].computed_data['{a_key_to_drop}'] did not exist.")
                        local_not_found_keys_to_drop.append(a_key_to_drop)
            else:
                # Otherwise skip it if it isn't in the includelist
                if debug_print:
                    print(f'skipping {a_config_name} because it is not in the context includelist.')
                local_not_found_keys_to_drop.append(a_key_to_drop)

        return global_dropped_keys, local_dropped_keys

    def find_LongShortGlobal_epoch_names(self):
        """ Returns the [long, short, global] epoch names. They must exist.
        Usage:
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            long_results = curr_active_pipeline.computation_results[long_epoch_name]['computed_data']
            short_results = curr_active_pipeline.computation_results[short_epoch_name]['computed_data']
            global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

        """
        return self.stage.find_LongShortGlobal_epoch_names()


    def find_LongShortDelta_times(self) -> Tuple[float, float, float]:
        """ Helper function to returns the [t_start, t_delta, t_end] session times. They must exist.
        Usage:
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        """
        return self.stage.find_LongShortDelta_times()


    def get_output_path(self) -> Path:
        """ returns the appropriate output path to store the outputs for this session. Usually '$session_folder/outputs/' """
        return self.sess.get_output_path()


    def get_session_context(self) -> IdentifyingContext:
        """ returns the context of the unfiltered session (self.sess) """
        return self.sess.get_context()


    def get_session_unique_aclu_information(self) -> pd.DataFrame:
        """  Get the aclu information for each aclu in the dataframe. Adds the ['aclu', 'shank', 'cluster', 'qclu', 'neuron_type'] columns """
        return self.sess.spikes_df.spikes.extract_unique_neuron_identities()



    # @property
    def get_output_manager(self) -> FileOutputManager:
        """ returns the FileOutputManager that specifies where outputs are stored. """
        # return FileOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE)
        return FileOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.HIERARCHY_UNIQUE)

    def get_computation_times(self, debug_print=False):
        return self.stage.get_computation_times(debug_print=debug_print)
    
    def get_time_since_last_computation(self, debug_print_timedelta:bool=False) -> timedelta:
        ## Successfully prints the time since the last calculation was performed:
        return self.stage.get_time_since_last_computation(debug_print_timedelta=debug_print_timedelta)

    """ Global Computation Results Persistance: Loads/Saves out the `global_computation_results` which are not currently saved with the pipeline
    
    `self.global_computation_results_pickle_path`
    `save_global_computation_results()`
    `load_pickled_global_computation_results(self, override_global_computation_results_pickle_path=None)`
    
    """
    
    @property
    def special_pickle_designator_suffix(self) -> str:
        # Get special suffix if specified and use that for global result too: 'loadedSessPickle_withDirectionalLaps.pkl'
        if self.pickle_path is not None:
            local_pickle_filename = self.pickle_path.name
            special_designator_suffix = local_pickle_filename.removeprefix('loadedSessPickle').removesuffix('.pkl')
        else:
            special_designator_suffix = ""
        return special_designator_suffix

    @property
    def global_computation_results_pickle_path(self) -> Path:
        """ The path to pickle the global_computation_results 
        Looks in the `output/global_computation_results.pkl` folder first.
        
        """
        special_designator_suffix = self.special_pickle_designator_suffix
        
        # if len(special_designator_suffix) > 0:
        desired_global_pickle_filename = f"global_computation_results{special_designator_suffix}.pkl" # 'global_computation_results_withDirectionalLaps.pkl'
        # desired_global_pickle_filename = f'global_computation_results.pkl' # old way
        return self.get_output_path().joinpath(desired_global_pickle_filename).resolve()

    ## Global Computation Result Persistance Hacks:
    @function_attributes(short_name=None, tags=['save'], input_requires=[], output_provides=[], uses=['saveData'], used_by=[], creation_date='2024-05-29 08:20', related_items=[])
    def save_global_computation_results(self, override_global_pickle_path: Optional[Path]=None, override_global_pickle_filename:Optional[str]=None):
        """Save out the `global_computation_results` which are not currently saved with the pipeline
        Usage:
            curr_active_pipeline.save_global_computation_results()
        """
        'global_computation_results.pkl'
        ## Case 1. `override_global_pickle_path` is provided:
        if override_global_pickle_path is not None:
            ## override_global_pickle_path is provided:
            if not isinstance(override_global_pickle_path, Path):
                override_global_pickle_path = Path(override_global_pickle_path).resolve()
            # Case 1a: `override_global_pickle_path` is a complete file path
            if not override_global_pickle_path.is_dir():
                # a full filepath, just use that directly
                global_computation_results_pickle_path = override_global_pickle_path.resolve()
            else:
                # default case, assumed to be a directory and we'll use the normal filename.
                active_global_pickle_filename: str = (override_global_pickle_filename or self.global_computation_results_pickle_path or "global_computation_results.pkl")
                global_computation_results_pickle_path = override_global_pickle_path.joinpath(active_global_pickle_filename).resolve()

        else:
            # No override path provided
            if override_global_pickle_filename is None:
                # no filename provided either, use default global pickle path:
                global_computation_results_pickle_path = self.global_computation_results_pickle_path
            else:
                # Otherwise use default output path but specified override_global_pickle_filename:
                global_computation_results_pickle_path = self.get_output_path().joinpath(override_global_pickle_filename).resolve() 

        print(f'global_computation_results_pickle_path: {global_computation_results_pickle_path}')
        saveData(global_computation_results_pickle_path, (self.global_computation_results.to_dict())) # AttributeError: 'directional_decoders_decode_continuous_Parameters' object has no attribute 'should_disable_cache'
        return global_computation_results_pickle_path


    @function_attributes(short_name=None, tags=['save', 'pickle', 'split'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-11 08:11', related_items=['load_split_pickled_global_computation_results'])
    def save_split_global_computation_results(self, override_global_pickle_path: Optional[Path]=None, override_global_pickle_filename:Optional[str]=None,
                                              include_includelist=None, continue_after_pickling_errors: bool=True, debug_print:bool=True):
        """Save out the `global_computation_results` which are not currently saved with the pipeline

        Reciprocal:
            load_pickled_global_computation_results

        Usage:
            split_save_folder, split_save_paths, split_save_output_types, failed_keys = curr_active_pipeline.save_split_global_computation_results(debug_print=True)
            
        #TODO 2023-11-22 18:54: - [ ] One major issue is that the types are lost upon reloading, so I think we'll need to save them somewhere. They can be fixed post-hoc like:
        # Update result with correct type:
        curr_active_pipeline.global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(**curr_active_pipeline.global_computation_results.computed_data['RankOrder'])

        """
        from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage
        

        ## Case 1. `override_global_pickle_path` is provided:
        if override_global_pickle_path is not None:
            ## override_global_pickle_path is provided:
            if not isinstance(override_global_pickle_path, Path):
                override_global_pickle_path = Path(override_global_pickle_path).resolve()
            # Case 1a: `override_global_pickle_path` is a complete file path
            if not override_global_pickle_path.is_dir():
                # a full filepath, just use that directly
                global_computation_results_pickle_path = override_global_pickle_path.resolve()
            else:
                # default case, assumed to be a directory and we'll use the normal filename.
                active_global_pickle_filename: str = (override_global_pickle_filename or self.global_computation_results_pickle_path or "global_computation_results.pkl")
                global_computation_results_pickle_path = override_global_pickle_path.joinpath(active_global_pickle_filename).resolve()

        else:
            # No override path provided
            if override_global_pickle_filename is None:
                # no filename provided either, use default global pickle path:
                global_computation_results_pickle_path = self.global_computation_results_pickle_path
            else:
                # Otherwise use default output path but specified override_global_pickle_filename:
                global_computation_results_pickle_path = self.get_output_path().joinpath(override_global_pickle_filename).resolve() 

        if debug_print:
            print(f'global_computation_results_pickle_path: {global_computation_results_pickle_path}')
        
        ## In split save, we save each result separately in a folder
        split_save_folder_name: str = f'{global_computation_results_pickle_path.stem}_split'
        split_save_folder: Path = global_computation_results_pickle_path.parent.joinpath(split_save_folder_name).resolve()
        if debug_print:
            print(f'split_save_folder: {split_save_folder}')
        # make if doesn't exist
        split_save_folder.mkdir(exist_ok=True)
        
        if include_includelist is None:
            ## include all keys if none are specified
            include_includelist = list(self.global_computation_results.computed_data.keys())

        ## only saves out the `global_computation_results` data:
        global_computed_data = self.global_computation_results.computed_data
        split_save_paths = {}
        split_save_output_types = {}
        failed_keys = []
        skipped_keys = []
        for k, v in global_computed_data.items():
            if k in include_includelist:
                curr_split_result_pickle_path = split_save_folder.joinpath(f'Split_{k}.pkl').resolve()
                if debug_print:
                    print(f'k: {k} -- size_MB: {print_object_memory_usage(v, enable_print=False)}')
                    print(f'\tcurr_split_result_pickle_path: {curr_split_result_pickle_path}')
                was_save_success = False
                curr_item_type = type(v)
                try:
                    ## try get as dict                
                    v_dict = v.__dict__ #__getstate__()
                    # saveData(curr_split_result_pickle_path, (v_dict))
                    saveData(curr_split_result_pickle_path, (v_dict, str(curr_item_type.__module__), str(curr_item_type.__name__)))    
                    was_save_success = True
                except KeyError as e:
                    print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                    pass
                except PicklingError as e:
                    if not continue_after_pickling_errors:
                        raise
                    else:
                        print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                        pass
                    
                if was_save_success:
                    split_save_paths[k] = curr_split_result_pickle_path
                    split_save_output_types[k] = curr_item_type
                    if debug_print:
                        print(f'\tfile_size_MB: {print_filesystem_file_size(curr_split_result_pickle_path, enable_print=False)} MB')
                else:
                    failed_keys.append(k)
            else:
                if debug_print:
                    print(f'\tskipping key "{k}" because it is not included in include_includelist: {include_includelist}')
                skipped_keys.append(k)
                
        if len(failed_keys) > 0:
            print(f'WARNING: failed_keys: {failed_keys} did not save for global results! They HAVE NOT BEEN SAVED!')
        return split_save_folder, split_save_paths, split_save_output_types, failed_keys


    @function_attributes(short_name=None, tags=['fixup', 'deserialization', 'filesystem', 'post-load', 'cross-platform'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-14 09:41', related_items=[])
    def post_load_fixup_sess_basedirs(self, updated_session_basepath: Path):
        """ after loading from pickle from another computer, fixes up the session's basepaths so they actually exist.
        
        Updates:
            self.sess.config.basepath
            self.filtered_sessions[an_epoch_name].config.basepath
            
        """
        did_fixup_any_missing_basepath: bool = False
        Assert.path_exists(updated_session_basepath)
        is_missing_basepath = (not self.sess.basepath.exists())
        if is_missing_basepath:
            self.sess.config.basepath = deepcopy(updated_session_basepath)
            did_fixup_any_missing_basepath = True
            
        for a_name, a_sess in self.filtered_sessions.items():
            is_missing_basepath = (not a_sess.basepath.exists())
            if is_missing_basepath:
                print(f"sess[{a_name}] is missing basepath: {a_sess.basepath}. updating.")
                a_sess.config.basepath = deepcopy(updated_session_basepath)
                did_fixup_any_missing_basepath = True
        ## END for a_name, a_se...
        
        return did_fixup_any_missing_basepath               


    
    def load_pickled_global_computation_results(self, override_global_computation_results_pickle_path=None, allow_overwrite_existing:bool=False, allow_overwrite_existing_allow_keys: Optional[List[str]]=None, debug_print=True):
        """ loads the previously pickled `global_computation_results` into `self.global_computation_results`, replacing the current values.
        TODO: shouldn't replace newer values without prompting, especially if the loaded value doesn't have that computed property but the current results do
        
         - [X] TODO 2023-05-19 - Implemented merging into existing `self.global_computation_results` results instead of replacing with loaded values.
                If `allow_overwrite_existing=True` all existing values in `self.global_computation_results` will be replaced with their versions in the loaded file, unless they only exist in the variable.
                Otherwise, specific keys can be specified using `allow_overwrite_existing_allow_keys=[]` to replace the variable versions of those keys with the ones loaded from file. 
                ?? Requires checking parameters.
         
        
        Usage:
            curr_active_pipeline.load_pickled_global_computation_results()
        """
        allow_overwrite_existing_allow_keys = allow_overwrite_existing_allow_keys or []

        if override_global_computation_results_pickle_path is None:
            # Use the default if no override is provided.
            global_computation_results_pickle_path = self.global_computation_results_pickle_path
        else:
            global_computation_results_pickle_path = override_global_computation_results_pickle_path


        loaded_global_computation_results = loadData(global_computation_results_pickle_path) # returns a dict
        if ((self.global_computation_results is None) or (self.global_computation_results.computed_data is None)):
            """ only if no previous global result at all """
            loaded_global_computation_results = ComputationResult(**loaded_global_computation_results) # convert to proper object type.
            self.stage.global_computation_results = loaded_global_computation_results # TODO 2023-05-19 - Merge results instead of replacing. Requires checking parameters.
        else:
            # Have extant global result of some kind:
            
            loaded_global_computation_result_dict = loaded_global_computation_results['computed_data']

            # successfully_loaded_keys = list(loaded_global_computation_result_dict.keys())
            successfully_loaded_keys = list(loaded_global_computation_result_dict.keys())
            sucessfully_updated_keys = []
            
            ## append them to the extant global_computations (`curr_active_pipeline.global_computation_results.computed_data`)
            for curr_result_key, loaded_value in loaded_global_computation_result_dict.items():
                should_apply: bool = False
                if curr_result_key in self.global_computation_results.computed_data:
                    if self.global_computation_results.computed_data[curr_result_key] is None:
                        # it exists, but is None. Overwrite the None value.
                        print(f'WARN: key "{curr_result_key}" already exists but is None! It will be overwritten with the loaded value.')
                        should_apply = True
                    else:
                        # key already exists, and is non-None. overwrite it?
                        if not allow_overwrite_existing:
                            if (curr_result_key in allow_overwrite_existing_allow_keys):
                                should_apply = True
                            else:
                                print(f'WARN: key "{curr_result_key}" already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?')
                                # Error:
                                # WARN: key sess already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?
                                # WARN: key computation_config already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?
                                # WARN: key computed_data already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?
                                # WARN: key accumulated_errors already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?
                                # WARN: key computation_times already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?
                        else:
                            # allow_overwrite_existing: means always overwrite existing results with loaded ones
                            should_apply = True
                else:
                    ## doesn't exist, add it
                    should_apply = True
                
                if should_apply:
                    # apply the loaded result to the computed_data.
                    sucessfully_updated_keys.append(curr_result_key)
                    self.global_computation_results.computed_data[curr_result_key] = loaded_value

            return sucessfully_updated_keys, successfully_loaded_keys
        

    @classmethod
    def try_load_split_pickled_global_computation_results(cls, global_computation_results_pickle_path, debug_print=True):
        """ just tries to load the previously pickled `global_computation_results` into `self.global_computation_results`, replacing the current values.
        Reciprocal: `save_split_global_computation_results`
        
        allow_overwrite_existing_allow_keys: keys allowed to be overwritten in the current computation_results if they can be loaded from disk.
        
        Usage:
            sucessfully_updated_keys, successfully_loaded_keys, found_split_paths = curr_active_pipeline.load_split_pickled_global_computation_results(allow_overwrite_existing_allow_keys=['DirectionalLaps', 'RankOrder'])

            
        global_computation_results_pickle_path = override_global_computation_results_pickle_path or self.global_computation_results_pickle_path


        """
        assert global_computation_results_pickle_path is not None
        if not isinstance(global_computation_results_pickle_path, Path):
            global_computation_results_pickle_path = Path(global_computation_results_pickle_path).resolve()
            
        if not global_computation_results_pickle_path.is_dir():	
            split_save_folder_name: str = f'{global_computation_results_pickle_path.stem}_split'
            split_save_folder: Path = global_computation_results_pickle_path.parent.joinpath(split_save_folder_name).resolve()
        else:
            split_save_folder: Path = global_computation_results_pickle_path.resolve()
            
        if debug_print:
            print(f'split_save_folder to load from: {split_save_folder}')
            
        assert split_save_folder.exists()
        assert split_save_folder.is_dir()
        loaded_global_computation_results = {}
        
        found_split_paths = []
        successfully_loaded_keys = {}
        failed_loaded_keys = {}


        for p in split_save_folder.rglob('Split_*.pkl'):
            if debug_print:
                print(f'p: {p}')
            found_split_paths.append(p)
            curr_result_key: str = p.stem.removeprefix('Split_') # the key name into global_computation_results, parsed back from the file name
            
            # Actually do the loading:
            try:
                loaded_value = loadData(p)
                loaded_global_computation_results[curr_result_key] = loaded_value
                successfully_loaded_keys[curr_result_key] = p

            except BaseException as e:
                print(f'Error loading {curr_result_key} from "{p}": {e}')
                failed_loaded_keys[curr_result_key] = p

        return loaded_global_computation_results, successfully_loaded_keys, failed_loaded_keys, found_split_paths
    
    @function_attributes(short_name=None, tags=['load', 'split'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-05-29 08:18', related_items=['save_split_global_computation_results'])
    def load_split_pickled_global_computation_results(self, override_global_computation_results_pickle_path=None, allow_overwrite_existing:bool=False, allow_overwrite_existing_allow_keys: Optional[List[str]]=None, debug_print=True):
        """ loads the previously pickled `global_computation_results` into `self.global_computation_results`, replacing the current values.
        Reciprocal: `save_split_global_computation_results`
        
        allow_overwrite_existing_allow_keys: keys allowed to be overwritten in the current computation_results if they can be loaded from disk.
        
        Usage:
            sucessfully_updated_keys, successfully_loaded_keys, failed_loaded_keys, found_split_paths = curr_active_pipeline.load_split_pickled_global_computation_results(allow_overwrite_existing_allow_keys=['DirectionalLaps', 'RankOrder'])
        """
        allow_overwrite_existing_allow_keys = allow_overwrite_existing_allow_keys or []
        loaded_global_computation_results, successfully_loaded_keys, failed_loaded_keys, found_split_paths = self.try_load_split_pickled_global_computation_results(global_computation_results_pickle_path=(override_global_computation_results_pickle_path or self.global_computation_results_pickle_path),
                                                                                                                                                                    debug_print=debug_print)
        
        sucessfully_updated_keys = []
        ## append them to the extant global_computations (`curr_active_pipeline.global_computation_results.computed_data`)
        for curr_result_key, loaded_value in loaded_global_computation_results.items():
            should_apply: bool = False
            if curr_result_key in self.global_computation_results.computed_data:
                # key already exists, overwrite it?
                
                if (not allow_overwrite_existing):
                    if (curr_result_key in allow_overwrite_existing_allow_keys):
                        should_apply = True
                    else:
                        print(f'WARN: key {curr_result_key} already exists in `curr_active_pipeline.global_computation_results.computed_data`. Overwrite it?')
                else:
                    should_apply = True
            else:
                ## add it
                should_apply = True
            
            if should_apply:
                # apply the loaded result to the computed_data.
                sucessfully_updated_keys.append(curr_result_key)
                if isinstance(loaded_value, dict):
                    loaded_result_dict = loaded_value
                elif isinstance(loaded_value, tuple):
                    assert len(loaded_value) == 3
                    # saved with 2024-01-24 - (v_dict, str(curr_item_type.__module__), str(curr_item_type.__name__)
                    loaded_result_dict, curr_item_type_module, curr_item_type_name = loaded_value
                    print(f'curr_item_type_module: {curr_item_type_module}, curr_item_type_name: {curr_item_type_name}')
                    # TODO: use thse to unarchive the object into the correct format:

                self.global_computation_results.computed_data[curr_result_key] = loaded_result_dict

        return sucessfully_updated_keys, successfully_loaded_keys, failed_loaded_keys, found_split_paths
    

    # ==================================================================================================================== #
    # Split Save General                                                                                                   #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['save', 'pickle', 'split'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-11 08:11', related_items=['load_split_pickled_global_computation_results'])
    def save_split_custom_results(self, override_global_pickle_path: Optional[Path]=None, override_global_pickle_filename:Optional[str]=None,
                                              include_includelist=None, continue_after_pickling_errors: bool=True, debug_print:bool=True):
        """Save out the `global_computation_results` which are not currently saved with the pipeline

        Reciprocal:
            load_pickled_global_computation_results

        Usage:
            split_save_folder, split_save_paths, split_save_output_types, failed_keys = curr_active_pipeline.save_split_global_computation_results(debug_print=True)
            
        #TODO 2023-11-22 18:54: - [ ] One major issue is that the types are lost upon reloading, so I think we'll need to save them somewhere. They can be fixed post-hoc like:
        # Update result with correct type:
        curr_active_pipeline.global_computation_results.computed_data['RankOrder'] = RankOrderComputationsContainer(**curr_active_pipeline.global_computation_results.computed_data['RankOrder'])

        """
        from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage
        

        ## Case 1. `override_global_pickle_path` is provided:
        if override_global_pickle_path is not None:
            ## override_global_pickle_path is provided:
            if not isinstance(override_global_pickle_path, Path):
                override_global_pickle_path = Path(override_global_pickle_path).resolve()
            # Case 1a: `override_global_pickle_path` is a complete file path
            if not override_global_pickle_path.is_dir():
                # a full filepath, just use that directly
                global_computation_results_pickle_path = override_global_pickle_path.resolve()
            else:
                # default case, assumed to be a directory and we'll use the normal filename.
                active_global_pickle_filename: str = (override_global_pickle_filename or self.global_computation_results_pickle_path or "global_computation_results.pkl")
                global_computation_results_pickle_path = override_global_pickle_path.joinpath(active_global_pickle_filename).resolve()

        else:
            # No override path provided
            if override_global_pickle_filename is None:
                # no filename provided either, use default global pickle path:
                global_computation_results_pickle_path = self.global_computation_results_pickle_path
            else:
                # Otherwise use default output path but specified override_global_pickle_filename:
                global_computation_results_pickle_path = self.get_output_path().joinpath(override_global_pickle_filename).resolve() 

        if debug_print:
            print(f'global_computation_results_pickle_path: {global_computation_results_pickle_path}')
        
        ## In split save, we save each result separately in a folder
        split_save_folder_name: str = f'{global_computation_results_pickle_path.stem}_split'
        split_save_folder: Path = global_computation_results_pickle_path.parent.joinpath(split_save_folder_name).resolve()
        if debug_print:
            print(f'split_save_folder: {split_save_folder}')
        # make if doesn't exist
        split_save_folder.mkdir(exist_ok=True)
        
        if include_includelist is None:
            ## include all keys if none are specified
            include_includelist = list(self.global_computation_results.computed_data.keys())

        ## only saves out the `global_computation_results` data:
        global_computed_data = self.global_computation_results.computed_data
        split_save_paths = {}
        split_save_output_types = {}
        failed_keys = []
        skipped_keys = []
        for k, v in global_computed_data.items():
            if k in include_includelist:
                curr_split_result_pickle_path = split_save_folder.joinpath(f'Split_{k}.pkl').resolve()
                if debug_print:
                    print(f'k: {k} -- size_MB: {print_object_memory_usage(v, enable_print=False)}')
                    print(f'\tcurr_split_result_pickle_path: {curr_split_result_pickle_path}')
                was_save_success = False
                curr_item_type = type(v)
                try:
                    ## try get as dict                
                    v_dict = v.__dict__ #__getstate__()
                    # saveData(curr_split_result_pickle_path, (v_dict))
                    saveData(curr_split_result_pickle_path, (v_dict, str(curr_item_type.__module__), str(curr_item_type.__name__)))    
                    was_save_success = True
                except KeyError as e:
                    print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                    pass
                except PicklingError as e:
                    if not continue_after_pickling_errors:
                        raise
                    else:
                        print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                        pass
                    
                if was_save_success:
                    split_save_paths[k] = curr_split_result_pickle_path
                    split_save_output_types[k] = curr_item_type
                    if debug_print:
                        print(f'\tfile_size_MB: {print_filesystem_file_size(curr_split_result_pickle_path, enable_print=False)} MB')
                else:
                    failed_keys.append(k)
            else:
                if debug_print:
                    print(f'\tskipping key "{k}" because it is not included in include_includelist: {include_includelist}')
                skipped_keys.append(k)
                
        if len(failed_keys) > 0:
            print(f'WARNING: failed_keys: {failed_keys} did not save for global results! They HAVE NOT BEEN SAVED!')
            

        return split_save_folder, split_save_paths, split_save_output_types, failed_keys




    # ==================================================================================================================== #
    # Parameters                                                                                                           #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['parameters', 'computaton'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-23 06:29', related_items=[])
    def get_all_parameters(self, allow_update_global_computation_config:bool=True) -> Dict:
        """ gets all user-parameters from the pipeline
        
        Actually updates `self.global_computation_results.computation_config`
        
        """
        from benedict import benedict
        from neuropy.core.parameters import ParametersContainer
        from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
        from pyphoplacecellanalysis.General.PipelineParameterClassTemplating import GlobalComputationParametersAttrsClassTemplating
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters, merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, perform_specific_epochs_decoding_Parameters, DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

        preprocessing_parameters: ParametersContainer = deepcopy(self.active_sess_config)

        ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
        if self.global_computation_results.computation_config is None:
            curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=self)
            if allow_update_global_computation_config:
                print('global_computation_results.computation_config is None! Making new one!')
                self.global_computation_results.computation_config = curr_global_param_typed_parameters
                print(f'\tdone. Pipeline needs resave!')
        else:
            curr_global_param_typed_parameters: ComputationKWargParameters = self.global_computation_results.computation_config
            
        ## Ensured that we have a valid `curr_global_param_typed_parameters` that was created with the kwarg defaults if it didn't exist.
        #TODO 2024-10-23 06:45: - [ ] What about when a config was created and then later new kwarg values were added to a computation function, or the default values were updated?
        _master_params_dict = {}
        _master_params_dict['preprocessing'] = preprocessing_parameters.to_dict()
        _master_params_dict.update(curr_global_param_typed_parameters.to_dict())
        
        # if self.global_computation_results.computation_config is not None:
        #     curr_global_param_typed_parameters: ComputationKWargParameters = deepcopy(self.global_computation_results.computation_config)
        #     _master_params_dict.update(curr_global_param_typed_parameters.to_dict())
        #     ## TODO: are we sure we have all the parameters just from a global config? do we need to capture the default kwarg values that haven't been assigned or something?
        # else:
        #     print(f'WARNING: no global config so using kwarg defaults...')
        #     ## only the default kwarg values:
        #     registered_merged_computation_function_default_kwargs_dict, code_str, nested_classes_dict, (imports_dict, imports_list, imports_string) = GlobalComputationParametersAttrsClassTemplating.main_generate_params_classes(curr_active_pipeline=self)
        #     # registered_merged_computation_function_default_kwargs_dict
        #     _master_params_dict.update(registered_merged_computation_function_default_kwargs_dict)

        # _master_params_dict
        # {'merged_directional_placefields': {'laps_decoding_time_bin_size': 0.25, 'ripple_decoding_time_bin_size': 0.025, 'should_validate_lap_decoding_performance': False},
        #  'rank_order_shuffle_analysis': {'num_shuffles': 500, 'minimum_inclusion_fr_Hz': 5.0, 'included_qclu_values': [1, 2], 'skip_laps': False},
        #  'directional_decoders_decode_continuous': {'time_bin_size': None},
        #  'directional_decoders_evaluate_epochs': {'should_skip_radon_transform': False},
        #  'directional_train_test_split': {'training_data_portion': 0.8333333333333334, 'debug_output_hdf5_file_path': None},
        #  'long_short_decoding_analyses': {'decoding_time_bin_size': None, 'perform_cache_load': False, 'always_recompute_replays': False, 'override_long_epoch_name': None, 'override_short_epoch_name': None},
        #  'long_short_rate_remapping': {'decoding_time_bin_size': None, 'perform_cache_load': False, 'always_recompute_replays': False},
        #  'long_short_inst_spike_rate_groups': {'instantaneous_time_bin_size_seconds': 0.01},
        #  'wcorr_shuffle_analysis': {'num_shuffles': 1024, 'drop_previous_result_and_compute_fresh': False},
        #  '_perform_specific_epochs_decoding': {'decoder_ndim': 2, 'filter_epochs': 'ripple', 'decoding_time_bin_size': 0.02},
        #  '_DEP_ratemap_peaks': {'peak_score_inclusion_percent_threshold': 0.25},
        #  'ratemap_peaks_prominence2d': {'step': 0.01, 'peak_height_multiplier_probe_levels': (0.5, 0.9), 'minimum_included_peak_height': 0.2, 'uniform_blur_size': 3, 'gaussian_blur_sigma': 3}}

        return benedict(_master_params_dict)

        # ## OUTPUTS: param_typed_parameters
        # return {
        #     'preprocessing_parameters': preprocessing_parameters,
        #     'curr_global_param_typed_parameters': curr_global_param_typed_parameters,
        #     'param_typed_parameters': param_typed_parameters,
        # }

    def update_parameters(self, override_parameters_flat_keypaths_dict: Dict[str, Any]=None) -> None:
        """ updates any of the user-parameters by keypaths for the pipeline
        
        """
        from neuropy.core.parameters import ParametersContainer
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters

        if override_parameters_flat_keypaths_dict is None:
            return
        else:
            if self.is_computed:
                ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
                if self.global_computation_results.computation_config is None:
                    print('global_computation_results.computation_config is None! Making new one!')
                    curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=self)
                    self.global_computation_results.computation_config = curr_global_param_typed_parameters
                    print(f'\tdone. Pipeline needs resave!')
                else:
                    curr_global_param_typed_parameters: ComputationKWargParameters = self.global_computation_results.computation_config
                    

                for k, v in override_parameters_flat_keypaths_dict.items():
                    if k.startswith('preprocessing'):
                        raise NotImplementedError("Updating preprocessing parameters is not yet implemented!")            
                        preprocessing_parameters: ParametersContainer = deepcopy(self.active_sess_config)
                    else:                
                        # Set a value using keypath (e.g. 'directional_train_test_split.training_data_portion')
                        curr_global_param_typed_parameters.set_by_keypath(k, v)

                self.global_computation_results.computation_config = curr_global_param_typed_parameters
                # return self.global_computation_results.computation_config # return the updated parameters
            else:
                print(f'too early to set the computation_config override_parameters, not yet at the computation stage!!')
                pass


    @function_attributes(short_name=None, tags=['UNFINSHED', 'context', 'custom', 'parameters'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-31 19:46', related_items=[])
    def get_session_additional_parameters_context(self, parts_separator:str='-') -> DisplaySpecifyingIdentifyingContext:
        """ gets the entire session context, including the noteworthy computation parameters that would be needed for determing which filename to save under .
        
        Usage:
            active_context, session_ctxt_key, CURR_BATCH_OUTPUT_PREFIX, additional_session_context = curr_active_pipeline.get_complete_session_context(BATCH_DATE_TO_USE=self.BATCH_DATE_TO_USE)
        
        """
        to_filename_conversion_dict = {'compute_diba_quiescent_style_replay_events':'_withNewComputedReplays', 'diba_evt_file':'_withNewKamranExportedReplays', 'initial_loaded': '_withOldestImportedReplays', 'normal_computed': '_withNormalComputedReplays'}

        all_params_dict = self.get_all_parameters()

        # preprocessing_parameters = all_params_dict['preprocessing']
        rank_order_shuffle_analysis_parameters = all_params_dict['rank_order_shuffle_analysis']
        included_qclu_values = deepcopy(rank_order_shuffle_analysis_parameters['included_qclu_values']) # [1, 2, 4, 6, 7, 9]
        minimum_inclusion_fr_Hz = deepcopy(rank_order_shuffle_analysis_parameters['minimum_inclusion_fr_Hz']) # 5.0
        
        
        ## TODO: Ideally would use the value passed in self.get_all_parameters():
        active_replay_epoch_parameters = deepcopy(self.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays)
        epochs_source: str = active_replay_epoch_parameters.get('epochs_source', 'normal_computed')
        
        _filename_formatting_fn = partial(
            session_context_filename_formatting_fn,
            parts_separator=parts_separator,
        )

        additional_session_context: DisplaySpecifyingIdentifyingContext = DisplaySpecifyingIdentifyingContext(epochs_source=epochs_source, included_qclu_values=included_qclu_values, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz,
            specific_purpose_display_dict={'filename_formatting': _filename_formatting_fn, 

        }, display_dict={'epochs_source': lambda k, v: to_filename_conversion_dict[v],
                'included_qclu_values': lambda k, v: f"qclu_{v}",
                'minimum_inclusion_fr_Hz': lambda k, v: f"frateThresh_{v:.1f}",
        })
        return additional_session_context
    

    @function_attributes(short_name=None, tags=['parameters', 'filenames', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-28 16:10', related_items=[])
    def get_custom_pipeline_filenames_from_parameters(self, parts_separator:str='-') -> Tuple:
        """ gets the custom suffix from the pipeline's parameters 
        
        custom_save_filepaths, custom_save_filenames, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters()
        
        """
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import _get_custom_filenames_from_computation_metadata
        
        all_params_dict = self.get_all_parameters()

        # preprocessing_parameters = all_params_dict['preprocessing']
        rank_order_shuffle_analysis_parameters = all_params_dict['rank_order_shuffle_analysis']
        included_qclu_values = deepcopy(rank_order_shuffle_analysis_parameters['included_qclu_values']) # [1, 2, 4, 6, 7, 9]
        minimum_inclusion_fr_Hz = deepcopy(rank_order_shuffle_analysis_parameters['minimum_inclusion_fr_Hz']) # 5.0
        
        ## TODO: Ideally would use the value passed in self.get_all_parameters():
        active_replay_epoch_parameters = deepcopy(self.sess.config.preprocessing_parameters.epoch_estimation_parameters.replays)
        epochs_source: str = active_replay_epoch_parameters.get('epochs_source', 'normal_computed')
        custom_suffix: str = epochs_source
        # custom_suffix += _get_custom_suffix_for_filename_from_computation_metadata(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
        custom_save_filepaths, custom_save_filenames, custom_suffix = _get_custom_filenames_from_computation_metadata(epochs_source=epochs_source, included_qclu_values=included_qclu_values, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, parts_separator=parts_separator)
        # print(f'custom_save_filenames: {custom_save_filenames}')
        # print(f'custom_suffix: "{custom_suffix}"')
        
        return custom_save_filepaths, custom_save_filenames, custom_suffix
    

    @function_attributes(short_name=None, tags=['parameters', 'filenames', 'export'], input_requires=[], output_provides=[], uses=['get_custom_pipeline_filenames_from_parameters'], used_by=[], creation_date='2024-11-08 10:36', related_items=[])
    def get_complete_session_identifier_string(self, parts_separator:str='_', custom_parameter_keyvalue_parts_separator:str='-', session_identity_parts_separator:str='_') -> str:
        """ returns a string like 'kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]', with the session context and the parameters
        complete_session_identifier_string: str = curr_active_pipeline.get_complete_session_identifier_string()
    
        Used to be `parts_separator:str='_'`
        """
        custom_save_filepaths, custom_save_filenames, custom_suffix = self.get_custom_pipeline_filenames_from_parameters(parts_separator=custom_parameter_keyvalue_parts_separator) # 'normal_computed-frateThresh_5.0-qclu_[1, 2]'
        complete_session_identifier_string: str = parts_separator.join([self.get_session_context().get_description(separator=session_identity_parts_separator), custom_suffix]) # 'kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]'
        return complete_session_identifier_string


    @function_attributes(short_name=None, tags=['parameters', 'filenames', 'export'], input_requires=[], output_provides=[], uses=['get_complete_session_identifier_string'], used_by=[], creation_date='2024-11-19 01:19', related_items=[])
    def build_complete_session_identifier_filename_string(self, data_identifier_str: str, parent_output_path: Optional[Path]=None, extra_parts: Optional[List[str]]=None, out_extension: Optional[str]='.csv', suffix_string: Optional[str]=None,
            output_date_str: Optional[str]=None, parts_separator:str='_', custom_parameter_keyvalue_parts_separator:str='-', session_identity_parts_separator:str='_', ensure_no_duplicate_parts: bool = True) -> Tuple[Path, str, str]:
        """ returns a string like 'kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]', with the session context and the parameters
        complete_session_identifier_string: str = curr_active_pipeline.get_complete_session_identifier_string()
    
        Used to be `parts_separator:str='_'`
        
        Usage:
        
        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(output_date_str=None, data_identifier_str="(ripple_WCorrShuffle_df)", parent_output_path=None, out_extension='.csv', extra_parts=None, ensure_no_duplicate_parts=False)
        out_filename # '2024-11-19_0148AM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_WCorrShuffle_df).csv'

        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(data_identifier_str="(ripple_WCorrShuffle_df)", parent_output_path=None, out_extension='.csv', extra_parts=['tbin-0.025'])
        out_filename  # '2024-11-19_0148AM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_WCorrShuffle_df)-tbin-0.025.csv'

        out_path, out_filename, out_basename = curr_active_pipeline.build_complete_session_identifier_filename_string(data_identifier_str="(ripple_WCorrShuffle_df)", parent_output_path=None, out_extension='.csv', suffix_string='_tbin-0.025')
        out_filename  # '2024-11-19_0148AM-kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0-(ripple_WCorrShuffle_df)_tbin-0.025.csv'

        
        """
        from pyphocorehelpers.print_helpers import get_now_day_str, get_now_rounded_time_str

        session_identifier_str: str = self.get_complete_session_identifier_string(parts_separator=parts_separator, custom_parameter_keyvalue_parts_separator=custom_parameter_keyvalue_parts_separator, session_identity_parts_separator=session_identity_parts_separator)

        # custom_save_filepaths, custom_save_filenames, custom_suffix = self.get_custom_pipeline_filenames_from_parameters(parts_separator=sub_parts_separator) # 'normal_computed-frateThresh_5.0-qclu_[1, 2]'
        # complete_session_identifier_string: str = parts_separator.join([self.get_session_context().get_description(separator=session_identity_parts_separator), custom_suffix]) # 'kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]'
        if output_date_str is None:
            output_date_str = get_now_rounded_time_str()
            # output_date_str = get_now_day_str()

        _all_parts = []

        # _all_parts = [output_date_str, session_identifier_str, data_identifier_str]

        if (output_date_str is not None) and (len(output_date_str) > 0):
            _all_parts.append(output_date_str)
            
        if (session_identifier_str is not None) and (len(session_identifier_str) > 0):
            _all_parts.append(session_identifier_str)
            
        if (data_identifier_str is not None) and (len(data_identifier_str) > 0):
            _all_parts.append(data_identifier_str)

        # assert output_date_str is not None
        if extra_parts is not None:
            for a_part in extra_parts:
                if (a_part is not None) and (len(a_part) > 0):
                    _all_parts.append(a_part)

            # _all_parts.extend(extra_parts)

        
        if ensure_no_duplicate_parts:
            # _all_parts = list(set(_all_parts)) ## drop duplicate parts
            # _all_parts = np.unique(_all_parts).tolist()
            _all_parts = list(dict.fromkeys(_all_parts))
            
            
        # out_basename: str = '-'.join([output_date_str, session_identifier_str, data_identifier_str]) # '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
        out_basename: str = custom_parameter_keyvalue_parts_separator.join(_all_parts) # '2024-01-04-kdiba_gor01_one_2006-6-09_1-22-43|(laps_marginals_df).csv'
        if (suffix_string is not None) and (len(suffix_string) > 0):
            if ensure_no_duplicate_parts:
                assert (not out_basename.endswith(suffix_string)), f"out_basename: '{out_basename}', suffix_string: '{suffix_string}'"
            out_basename = f"{out_basename}{suffix_string}" ## append suffix string before extension
        
        if out_extension is None:
            out_extension = ''
        out_filename: str = f"{out_basename}{out_extension}"
        if parent_output_path is not None:
            out_path: Path = parent_output_path.joinpath(out_filename).resolve()
        else:
            out_path: Path = Path(out_filename)
        return out_path, out_filename, out_basename





    @function_attributes(short_name=None, tags=['context', 'custom', 'parameters'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-01 00:00', related_items=[])
    def get_complete_session_context(self, parts_separator:str='_') -> Tuple[DisplaySpecifyingIdentifyingContext, Tuple[DisplaySpecifyingIdentifyingContext]]:
        """ gets the entire session context, including the noteworthy computation parameters that would be needed for determing which filename to save under .
        
        Usage:
            complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
        
        """
        _filename_formatting_fn = partial(
            session_context_filename_formatting_fn,
            parts_separator=parts_separator,
        )

        curr_session_context: DisplaySpecifyingIdentifyingContext = DisplaySpecifyingIdentifyingContext.init_from_context(a_context=self.get_session_context(),
        specific_purpose_display_dict={'filename_formatting': _filename_formatting_fn,},
        # display_dict={'epochs_source': lambda k, v: to_filename_conversion_dict[v],
        #         'included_qclu_values': lambda k, v: f"qclu_{v}",
        #         'minimum_inclusion_fr_Hz': lambda k, v: f"frateThresh_{v:.1f}",
        # },
        ) # **_obj.to_dict(),
        additional_session_context: DisplaySpecifyingIdentifyingContext = self.get_session_additional_parameters_context(parts_separator=parts_separator)
        # complete_session_context: DisplaySpecifyingIdentifyingContext = curr_session_context | additional_session_context # hoping this merger works
        complete_session_context: DisplaySpecifyingIdentifyingContext = curr_session_context.adding_context(collision_prefix='_additional', **additional_session_context.to_dict()) # hoping this merger works
        
        return complete_session_context, (curr_session_context,  additional_session_context)

    @function_attributes(short_name=None, tags=['valid_track_times'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-05 16:18', related_items=[])
    def find_first_and_last_valid_position_times(self):
        """ uses the positions and the loaded_track_limits to determine the first and last valid times for each session. 
        
        Usage:
        
            active_sess_config = deepcopy(curr_active_pipeline.active_sess_config)
            # absolute_start_timestamp: float = active_sess_config.absolute_start_timestamp
            loaded_track_limits = active_sess_config.loaded_track_limits # x_midpoint, 

            (first_valid_pos_time, last_valid_pos_time) = curr_active_pipeline.find_first_and_last_valid_position_times()
            (first_valid_pos_time, last_valid_pos_time)
            
        """
        return self.stage.find_first_and_last_valid_position_times()
        

    @function_attributes(short_name=None, tags=['computationResult', 'global', 'instance', 'USEFUL'], input_requires=[], output_provides=[], uses=['cls._build_initial_global_computationResult'], used_by=[], creation_date='2025-01-07 12:13', related_items=[])
    def build_initial_global_computationResult(self) -> ComputationResult:
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): this is the filtered data session
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters # merged_directional_placefields_Parameters, rank_order_shuffle_analysis_Parameters, directional_decoders_decode_continuous_Parameters, directional_decoders_evaluate_epochs_Parameters, directional_train_test_split_Parameters, long_short_decoding_analyses_Parameters, long_short_rate_remapping_Parameters, long_short_inst_spike_rate_groups_Parameters, wcorr_shuffle_analysis_Parameters, _perform_specific_epochs_decoding_Parameters, _DEP_ratemap_peaks_Parameters, ratemap_peaks_prominence2d_Parameters

        curr_global_param_typed_parameters: ComputationKWargParameters = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=self)
        
        return self.stage.build_initial_global_computationResult(computation_config=curr_global_param_typed_parameters)
    




    # @function_attributes(short_name=None, tags=['UNFINSHED', 'context', 'custom', 'parameters'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-31 19:46', related_items=[])
    # def get_complete_session_context_description_str(self, BATCH_DATE_TO_USE: Optional[str]=None):
    # 	""" gets the entire session context, including the noteworthy computation parameters that would be needed for determing which filename to save under .
        
    # 	Usage:
    # 		# active_context, session_ctxt_key, CURR_BATCH_OUTPUT_PREFIX, additional_session_context = curr_active_pipeline.get_complete_session_context(BATCH_DATE_TO_USE=self.BATCH_DATE_TO_USE)
        
    # 	"""
    # 	# curr_session_name: str = self.session_name # '2006-6-08_14-26-15'
    # 	# _, _, custom_suffix = self.get_custom_pipeline_filenames_from_parameters()
        
    # 	# additional_session_context


    # 	# if len(custom_suffix) > 0:
    # 	# 	if additional_session_context is not None:
    # 	# 		if isinstance(additional_session_context, dict):
    # 	# 			additional_session_context = IdentifyingContext(**additional_session_context)

    # 	# 		## easiest to update as dict:	
    # 	# 		additional_session_context = additional_session_context.to_dict()
    # 	# 		additional_session_context['custom_suffix'] = (additional_session_context.get('custom_suffix', '') or '') + custom_suffix
    # 	# 		additional_session_context = IdentifyingContext(**additional_session_context)
                
    # 	# 	else:
    # 	# 		additional_session_context = IdentifyingContext(custom_suffix=custom_suffix)
        
    # 	# assert (additional_session_context is not None), f"perform_sweep_decoding_time_bin_sizes_marginals_dfs_completion_function: additional_session_context is None even after trying to add the computation params as additional_session_context"
    # 	# # active_context = curr_active_pipeline.get_session_context()
    # 	# if additional_session_context is not None:
    # 	# 	if isinstance(additional_session_context, dict):
    # 	# 		additional_session_context = IdentifyingContext(**additional_session_context)
    # 	# 	active_context: IdentifyingContext = (self.get_session_context() | additional_session_context)
    # 	# 	# if len(custom_suffix) == 0:
    # 	# 	session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=(IdentifyingContext._get_session_context_keys() + list(additional_session_context.keys())))
    # 	# 	CURR_BATCH_OUTPUT_PREFIX: str = f"{curr_session_name}-{additional_session_context.get_description()}"
    # 	# 	# else:
    # 	# 	# 	session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=(IdentifyingContext._get_session_context_keys() + list(additional_session_context.keys()))) + f'|{custom_suffix}'
    # 	# 	# 	CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{additional_session_context.get_description()}-{custom_suffix}"
    # 	# else:
    # 	# 	active_context: IdentifyingContext = self.get_session_context()
    # 	# 	session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
    # 	# 	# if len(custom_suffix) == 0:
    # 	# 	CURR_BATCH_OUTPUT_PREFIX: str = f"{curr_session_name}"
    # 	# 	# else:
    # 	# 	# 	session_ctxt_key:str = session_ctxt_key + custom_suffix
    # 	# 	# 	CURR_BATCH_OUTPUT_PREFIX: str = f"{self.BATCH_DATE_TO_USE}-{curr_session_name}-{custom_suffix}"
            
    # 	# if BATCH_DATE_TO_USE is not None and len(BATCH_DATE_TO_USE) > 0:
    # 	# 	CURR_BATCH_OUTPUT_PREFIX = f"{BATCH_DATE_TO_USE}-{CURR_BATCH_OUTPUT_PREFIX}"

    # 	# print(f'\tactive_context: {active_context}')
        
    # 	# return active_context, session_ctxt_key, CURR_BATCH_OUTPUT_PREFIX, additional_session_context
    