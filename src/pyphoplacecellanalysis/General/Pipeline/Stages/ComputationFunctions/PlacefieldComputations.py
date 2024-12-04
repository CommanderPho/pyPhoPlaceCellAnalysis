import numpy as np
import pandas as pd
import itertools

from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from copy import deepcopy
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent, perform_compute_time_dependent_placefields

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder, computation_precidence_specifying_function, global_function
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult


class PlacefieldComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 0 # must be done first.
    _is_global = False

    @function_attributes(short_name='pf_computation', tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-30 19:50', related_items=[],
                        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf1D'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf2D']))
    def _perform_baseline_placefield_computation(computation_result: ComputationResult, debug_print=False):
        """ Builds the initial 1D and 2D placefields 
        
        Provides 
        
        """
        def _initial_placefield_computation(active_session, pf_computation_config, prev_output_result: ComputationResult):
            prev_output_result.computed_data['pf1D'], prev_output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, pf_computation_config, None, None, included_epochs=pf_computation_config.computation_epochs, should_force_recompute_placefields=True)
            return prev_output_result
        
        """ 
        Access via:
        ['pf1D']
        ['pf2D']
        
        Example:
            active_pf_1D = curr_active_pipeline.computation_results['maze1'].computed_data['pf1D']
            active_pf_2D = curr_active_pipeline.computation_results['maze1'].computed_data['pf2D']
            
            active_pf_2D
        """
        return _initial_placefield_computation(computation_result.sess, computation_result.computation_config.pf_params, computation_result)
    
    
    @computation_precidence_specifying_function(overriden_computation_precidence=9)
    @function_attributes(short_name='pfdt_computation', tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-08-30 19:58', related_items=[],
                         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf1D_dt'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf2D_dt']))
    def _perform_time_dependent_placefield_computation(computation_result: ComputationResult, debug_print=False):
        """ Builds the time-dependent 2D placefields 
        
        
        perform_compute_time_dependent_placefields
        
        
        Provides 
        
        """
        def _initial_time_dependent_placefield_computation(active_session, pf_computation_config, prev_output_result: ComputationResult):
            prev_output_result.computed_data['pf1D_dt'], prev_output_result.computed_data['pf2D_dt'] = perform_compute_time_dependent_placefields(active_session.spikes_df, active_session.position, pf_computation_config, None, None, included_epochs=pf_computation_config.computation_epochs, should_force_recompute_placefields=True)
            return prev_output_result
        """ 
        Access via:
        ['pf1D_dt']
        ['pf2D_dt']
        
        Example:
            active_pf_1D_dt = curr_active_pipeline.computation_results['maze1'].computed_data['pf1D_dt']
            active_pf_2D_dt = curr_active_pipeline.computation_results['maze1'].computed_data['pf2D_dt']
            
            active_pf_2D
        """
        return _initial_time_dependent_placefield_computation(computation_result.sess, computation_result.computation_config.pf_params, computation_result)
