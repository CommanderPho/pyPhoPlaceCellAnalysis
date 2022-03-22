from collections import OrderedDict
import sys
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields

from pyphocorehelpers.function_helpers import compose_functions

from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import LoadableInput, LoadableSessionInput, LoadedPipelineStage    
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import DefaultComputationFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats import ExtendedStatsComputations
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeAnalysisComputations


class ComputablePipelineStage:
    """ Designates that a pipeline stage is computable. """
        
    @classmethod
    def _perform_single_computation(cls, active_session, computation_config):
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): this is the filtered data session
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        # only requires that active_session has the .spikes_df and .position  properties
        output_result = ComputationResult(active_session, computation_config, computed_data=dict()) # Note that this active_session should be correctly filtered
        output_result.computed_data['pf1D'], output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=computation_config.computation_epochs, should_force_recompute_placefields=True)

        return output_result

    def single_computation(self, active_computation_params: PlacefieldComputationParameters=None, enabled_filter_names=None):
        """ Takes its filtered_session and applies the provided active_computation_params to it. The results are stored in self.computation_results under the same key as the filtered session. """
        assert (len(self.filtered_sessions.keys()) > 0), "Must have at least one filtered session before calling single_computation(...). Call self.select_filters(...) first."
        # self.active_computation_results = dict()
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified

        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
            if a_select_config_name in enabled_filter_names:
                print(f'Performing single_computation on filtered_session with filter named "{a_select_config_name}"...')
                if active_computation_params is None:
                    active_computation_params = self.active_configs[a_select_config_name].computation_config # get the previously set computation configs
                else:
                    # set/update the computation configs:
                    self.active_configs[a_select_config_name].computation_config = active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.
                self.computation_results[a_select_config_name] = ComputablePipelineStage._perform_single_computation(a_filtered_session, active_computation_params) # returns a computation result. Does this store the computation config used to compute it?
                # call to perform any registered computations:
                self.computation_results[a_select_config_name] = self.perform_registered_computations(self.computation_results[a_select_config_name], debug_print=True)
            else:
                # this filter is excluded from the enabled list, no computations will we performed on it
                self.computation_results.pop(a_select_config_name, None) # remove the computation results from previous runs from the dictionary to indicate that it hasn't been computed


class DefaultRegisteredComputations:
    """ Simply enables specifying the default computation functions that will be defined in this file and automatically registered. """
    def register_default_known_computation_functions(self):
        # TODO: Note that order matters for the computation functions, unlike the display functions, so they need to be enumerated in the correct order and not sorted alphabetically
        
        # Register the neuronal firing analysis computation functions:
        for (a_computation_fn_name, a_computation_fn) in reversed(SpikeAnalysisComputations.get_all_functions(use_definition_order=True)):
            self.register_computation(a_computation_fn_name, a_computation_fn)
            
        # Register the Ratemap/Placemap computation functions: 
        for (a_computation_fn_name, a_computation_fn) in reversed(ExtendedStatsComputations.get_all_functions(use_definition_order=True)):
            self.register_computation(a_computation_fn_name, a_computation_fn)
            
        for (a_computation_fn_name, a_computation_fn) in reversed(DefaultComputationFunctions.get_all_functions(use_definition_order=True)):
            self.register_computation(a_computation_fn_name, a_computation_fn)
            
        # # old way:
        # self.register_computation(ExtendedStatsComputations._perform_placefield_overlap_computation)
        # self.register_computation(ExtendedStatsComputations._perform_firing_rate_trends_computation)
        # self.register_computation(ExtendedStatsComputations._perform_extended_statistics_computation)
        # # self.register_computation(DefaultComputationFunctions._perform_extended_statistics_computation)
        # self.register_computation(DefaultComputationFunctions._perform_two_step_position_decoding_computation)
        # self.register_computation(DefaultComputationFunctions._perform_position_decoding_computation)
        
        
        
        
    


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
    def registered_computation_functions(self):
        """The registered_computation_functions property."""
        return self.stage.registered_computation_functions
        
    @property
    def registered_computation_function_names(self):
        """The registered_computation_function_names property."""
        return self.stage.registered_computation_function_names
    
    
    ## Computation Helpers: 
    def perform_computations(self, active_computation_params: PlacefieldComputationParameters=None, enabled_filter_names=None):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.single_computation(active_computation_params, enabled_filter_names=enabled_filter_names)
        
    def register_computation(self, registered_name, computation_function):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_computation(registered_name, computation_function)

    def perform_registered_computations(self, previous_computation_result=None, debug_print=False):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.perform_computations to reach this step."
        self.stage.perform_registered_computations(previous_computation_result, debug_print=debug_print)
    
    
    
    

class ComputedPipelineStage(LoadableInput, LoadableSessionInput, FilterablePipelineStage, DefaultRegisteredComputations, ComputablePipelineStage, BaseNeuropyPipelineStage):
    """Docstring for ComputedPipelineStage."""
    identity: PipelineStage = PipelineStage.Computed
    filtered_sessions: dict = None
    filtered_epochs: dict = None
    active_configs: dict = None
    computation_results: dict = None
    
    def __init__(self, loaded_stage: LoadedPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = loaded_stage.stage_name
        self.basedir = loaded_stage.basedir
        self.loaded_data = loaded_stage.loaded_data

        # Initialize custom fields:
        self.filtered_sessions = dict()
        self.filtered_epochs = dict()
        self.active_configs = dict() # active_config corresponding to each filtered session/epoch
        self.computation_results = dict()
        
        self.registered_computation_function_dict = OrderedDict()
        # self.registered_computation_functions = list()
        self.register_default_known_computation_functions() # registers the default
        
    @property
    def registered_computation_functions(self):
        """The registered_computation_functions property."""
        return list(self.registered_computation_function_dict.values())

    @property
    def registered_computation_function_names(self):
        """The registered_computation_function_names property."""
        return list(self.registered_computation_function_dict.keys()) 
    
    
    def register_computation(self, registered_name, computation_function):
        self.registered_computation_function_dict[registered_name] = computation_function
        # self.registered_computation_functions.append(computation_function)
        
    def perform_registered_computations(self, previous_computation_result=None, debug_print=False):
        """ Called after load is complete to post-process the data """
        if (len(self.registered_computation_functions) > 0):
            if debug_print:
                print(f'Performing perform_registered_computations(...) with {len(self.registered_computation_functions)} registered_computation_functions...')
            composed_registered_computations_function = compose_functions(*self.registered_computation_functions) # functions are composed left-to-right
            # if previous_computation_result is None:
            #     assert (self.computation_results is not None), "if no previous_computation_result is passed, one should have been computed previously."
            #     previous_computation_result = self.computation_results # Get the previously computed computation results. Note that if this function is called multiple times and assumes the results are coming in fresh, this can be an error.
            
            previous_computation_result = composed_registered_computations_function(previous_computation_result)
            return previous_computation_result
            
        else:
            if debug_print:
                print(f'No registered_computation_functions, skipping extended computations.')
            return previous_computation_result # just return the unaltered result
    