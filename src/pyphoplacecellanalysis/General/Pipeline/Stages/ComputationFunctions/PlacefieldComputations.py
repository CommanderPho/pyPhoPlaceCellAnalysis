import numpy as np
import pandas as pd
import itertools

from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from copy import deepcopy
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent, perform_compute_time_dependent_placefields

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult


class PlacefieldComputations(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 0 # must be done first.
    _is_global = False


    def _perform_estimated_epochs_computation(computation_result: ComputationResult, **kwargs):
        """ Discovers the relevant epochs for computation, such as PBEs, Laps, Replays, Ripples, etc.
            
        Needs to be ran prior to any placefield computation fucntions.
        
        TODO: this needs to be mirgrated into a "post-load function" instead of a computation function whenever that functionality gets implemented.
            1. It is more efficient to do this on the pre-filtered session
            2. The laps for example are needed for setting the computation configs, although I suppose they could be updated later.
        
        """
        # placefield_computation_config = computation_result.computation_config.pf_params # should be a PlacefieldComputationParameters

        active_epoch_estimation_parameters = computation_result.computation_config['epoch_estimation_parameters']
        assert active_epoch_estimation_parameters is not None, f"TODO 2023-05-22 - Only for KDIBA-style sessions for now since that's the only place these estimations are set."

        # ## TODO: fall-back to these when needed.
        # lap_estimation_parameters = DynamicParameters(N=20, should_backup_extant_laps_obj=True) # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`
        # PBE_estimation_parameters = DynamicParameters(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.300) # NewPaper's Parameters
        # replay_estimation_parameters = DynamicParameters(require_intersecting_epoch=None, min_epoch_included_duration=0.06, max_epoch_included_duration=None, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3)
        
        # # Write the parameters:
        # computation_result.computation_config['epoch_estimation_parameters'] = DynamicParameters.init_from_dict({
        #     'laps': lap_estimation_parameters,
        #     'PBEs': PBE_estimation_parameters,
        #     'replays': replay_estimation_parameters
        # })

        lap_estimation_parameters = active_epoch_estimation_parameters.laps
        PBE_estimation_parameters = active_epoch_estimation_parameters.PBEs
        replay_estimation_parameters = active_epoch_estimation_parameters.replays

        ## Allow saving the input/output results:
        # prev_output_result.computed_data['computed_epochs'] = 

        # 2023-05-16 - Laps conformance function (TODO 2023-05-16 - factor out?)
        try:
            computation_result.sess.replace_session_laps_with_estimates(**lap_estimation_parameters, should_plot_laps_2d=False) # , time_variable_name=None
        except AssertionError as e:
            print(f'RAISE - Laps Computation Assertion Error: {e}')
            raise e
        except Exception as e:
            raise e
        # filtered_laps = Epoch.filter_epochs(session.laps.as_epoch_obj(), pos_df=session.position.to_dataframe(), spikes_df=session.spikes_df, min_epoch_included_duration=1.0, max_epoch_included_duration=30.0, maximum_speed_thresh=None, min_num_unique_aclu_inclusions=3)
        ## Apply the laps as the limiting computation epochs:
        computation_result.computation_config.pf_params.computation_epochs = computation_result.sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0)
        


        # ## TODO 2023-05-19 - FIX SLOPPY PBE HANDLING
        # ## Get PBEs first:

        # num_pre_epochs = computation_result.sess.pbe

        new_pbe_epochs = computation_result.sess.compute_pbe_epochs(computation_result.sess, active_parameters=PBE_estimation_parameters)
        computation_result.sess.pbe = new_pbe_epochs
        updated_spk_df = computation_result.sess.compute_spikes_PBEs()
        

        # 2023-05-16 - Replace loaded replays (which are bad) with estimated ones:
        # num_pre = session.replay.

        computation_result.sess.replace_session_replays_with_estimates(**replay_estimation_parameters) # TODO: set requirements here?
        
        return computation_result



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
