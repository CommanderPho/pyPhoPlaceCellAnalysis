import numpy as np
import pandas as pd
import itertools

from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from copy import deepcopy
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent, perform_compute_time_dependent_placefields

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult


class PlacefieldComputations(AllFunctionEnumeratingMixin):
    
    def _perform_baseline_placefield_computation(computation_result: ComputationResult, debug_print=False):
        """ Builds the initial 1D and 2D placefields 
        
        Provides 
        
        """
        def initial_placefield_computation(active_session, computation_config, prev_output_result: ComputationResult):
            prev_output_result.computed_data['pf1D'], prev_output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=computation_config.computation_epochs, should_force_recompute_placefields=True)
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
        return initial_placefield_computation(computation_result.sess, computation_result.computation_config, computation_result)
    
    


    def _perform_time_dependent_placefield_computation(computation_result: ComputationResult, debug_print=False):
        """ Builds the time-dependent 2D placefields 
        
        
        perform_compute_time_dependent_placefields
        
        
        Provides 
        
        """
        def initial_time_dependent_placefield_computation(active_session, computation_config, prev_output_result: ComputationResult):
            prev_output_result.computed_data['pf1D_dt'], prev_output_result.computed_data['pf2D_dt'] = perform_compute_time_dependent_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=computation_config.computation_epochs, should_force_recompute_placefields=True)
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
        return initial_time_dependent_placefield_computation(computation_result.sess, computation_result.computation_config, computation_result)
