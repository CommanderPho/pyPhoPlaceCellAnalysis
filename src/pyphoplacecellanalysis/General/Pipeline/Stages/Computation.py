import sys
import numpy as np
import pandas as pd
from pyphocorehelpers.function_helpers import compose_functions

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields


from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage, PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import LoadableInput, LoadableSessionInput, LoadedPipelineStage    
from pyphoplacecellanalysis.General.ComputationResults import ComputationResult
from pyphoplacecellanalysis.General.Decoder.decoder_result import build_position_df_discretized_binned_positions, build_position_df_resampled_to_time_windows


class ComputablePipelineStage:
    """ Designates that a pipeline stage is computable. """
        
    @classmethod
    def _perform_single_computation(cls, active_session, computation_config):
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): [description]
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        # only requires that active_session has the .spikes_df and .position  properties
        output_result = ComputationResult(active_session, computation_config, computed_data=dict())        
        output_result.computed_data['pf1D'], output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=computation_config.computation_epochs, should_force_recompute_placefields=True)

        return output_result

    def single_computation(self, active_computation_params: PlacefieldComputationParameters=None):
        """ Takes its filtered_session and applies the provided active_computation_params to it. The results are stored in self.computation_results under the same key as the filtered session. """
        assert (len(self.filtered_sessions.keys()) > 0), "Must have at least one filtered session before calling single_computation(...). Call self.select_filters(...) first."
        # self.active_computation_results = dict()
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():
            print(f'Performing single_computation on filtered_session with filter named "{a_select_config_name}"...')
            if active_computation_params is None:
                active_computation_params = self.active_configs[a_select_config_name].computation_config # get the previously set computation configs
            else:
                # set/update the computation configs:
                self.active_configs[a_select_config_name].computation_config = active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.
            self.computation_results[a_select_config_name] = ComputablePipelineStage._perform_single_computation(a_filtered_session, active_computation_params) # returns a computation result. Does this store the computation config used to compute it?
        
            # call to perform any registered computations:
            self.computation_results[a_select_config_name] = self.perform_registered_computations(self.computation_results[a_select_config_name], debug_print=True)


"""-------------- Specific Computation Functions to be registered --------------"""

from pyphoplacecellanalysis.Analysis.reconstruction import BayesianPlacemapPositionDecoder, Zhang_Two_Step

class DefaultComputationFunctions:
    def _perform_position_decoding_computation(computation_result: ComputationResult):
        """ Builds the 2D Placefield Decoder """
        def position_decoding_computation(active_session, computation_config, prev_output_result: ComputationResult):
            prev_output_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(computation_config.time_bin_size, prev_output_result.computed_data['pf2D'], active_session.spikes_df.copy(), debug_print=False)
            # %timeit pho_custom_decoder.compute_all():  18.8 s ± 149 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            prev_output_result.computed_data['pf2D_Decoder'].compute_all() #  --> n = self.
            return prev_output_result

        return position_decoding_computation(computation_result.sess, computation_result.computation_config, computation_result)
    
    
    def _perform_extended_statistics_computation(computation_result: ComputationResult, debug_print=False):
        """ Computes extended statistics regarding firing rates and such from the various dataframes. """
        time_binned_position_resampler = build_position_df_resampled_to_time_windows(computation_result.sess.position.to_dataframe(), time_bin_size=computation_result.computation_config.time_bin_size) # TimedeltaIndexResampler
        time_binned_position_df = time_binned_position_resampler.nearest() # an actual dataframe
        computation_result.computed_data['extended_stats'] = {
         'time_binned_positioned_resampler': time_binned_position_resampler,
         'time_binned_position_df': time_binned_position_df,
         'time_binned_position_mean': time_binned_position_df.resample("1min").mean(), # 3 minutes
         'time_binned_position_covariance': time_binned_position_df.cov(min_periods=12)
        }
        """ 
        Access via ['extended_stats']['time_binned_position_df']
        Example:
            active_extended_stats = curr_active_pipeline.computation_results['maze1'].computed_data['extended_stats']
            time_binned_pos_df = active_extended_stats['time_binned_position_df']
            time_binned_pos_df
        """
        return computation_result
    
    def _perform_two_step_position_decoding_computation(computation_result: ComputationResult, debug_print=False):
        """ Builds the Zhang Velocity/Position For 2-step Bayesian Decoder for 2D Placefields """
        def _compute_avg_speed_at_each_position_bin(active_position_df, active_computation_config, xbin, ybin, show_plots=False, debug_print=False):
            """ compute the average speed at each position x: """
            ## Non-working attempt to use edges instead of bins:
            # xbin_edges = xbin + [(xbin[-1] + (xbin[1] - xbin[0]))] # add an additional (right) edge to the end of the xbin array for use with pd.cut
            # ybin_edges = ybin + [(ybin[-1] + (ybin[1] - ybin[0]))] # add an additional (right) edge to the end of the ybin array for use with pd.cut
            # print(f'xbin_edges: {np.shape(xbin_edges)}\n ybin_edges: {np.shape(ybin_edges)}, np.shape(np.arange(len(xbin_edges))): {np.shape(np.arange(len(xbin_edges)))}')
            # active_position_df['binned_x'] = pd.cut(active_position_df['x'].to_numpy(), bins=xbin_edges, include_lowest=True, labels=np.arange(start=1, stop=len(xbin_edges))) # same shape as the input data 
            # active_position_df['binned_y'] = pd.cut(active_position_df['y'].to_numpy(), bins=ybin_edges, include_lowest=True, labels=np.arange(start=1, stop=len(ybin_edges))) # same shape as the input data 

            def _compute_group_stats_for_var(active_position_df, xbin, ybin, variable_name:str = 'speed', debug_print=False):
                # For each unique binned_x and binned_y value, what is the average velocity_x at that point?
                position_bin_dependent_specific_average_velocities = active_position_df.groupby(['binned_x','binned_y'])[variable_name].agg([np.nansum, np.nanmean, np.nanmin, np.nanmax]).reset_index() #.apply(lambda g: g.mean(skipna=True)) #.agg((lambda x: x.mean(skipna=False)))
                if debug_print:
                    print(f'np.shape(position_bin_dependent_specific_average_velocities): {np.shape(position_bin_dependent_specific_average_velocities)}')
                # position_bin_dependent_specific_average_velocities # 1856 rows
                output = np.zeros((len(xbin), len(ybin))) # (65, 30)
                if debug_print:
                    print(f'np.shape(output): {np.shape(output)}')
                    print(f"np.shape(position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()): {np.shape(position_bin_dependent_specific_average_velocities['binned_x'])}")
                
                if debug_print:
                    max_binned_x_index = np.max(position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1) # 63
                    max_binned_y_index = np.max(position_bin_dependent_specific_average_velocities['binned_y'].to_numpy()-1) # 28
                

                # (len(xbin), len(ybin)): (59, 21)
                output[position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1, position_bin_dependent_specific_average_velocities['binned_y'].to_numpy()-1] = position_bin_dependent_specific_average_velocities['nanmean'].to_numpy() # ValueError: shape mismatch: value array of shape (1856,) could not be broadcast to indexing result of shape (1856,2,30)
                return output

            outputs = dict()
            outputs['speed'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'speed')
            # outputs['velocity_x'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'velocity_x')
            # outputs['acceleration_x'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'acceleration_x')
            return outputs['speed']


        prev_one_step_bayesian_decoder = computation_result.computed_data['pf2D_Decoder']
        
        # makes sure to use the xbin_center and ybin_center from the previous one_step decoder to bin the positions:
        computation_result.sess.position.df, xbin, ybin, bin_info = build_position_df_discretized_binned_positions(computation_result.sess.position.df, computation_result.computation_config, xbin_values=prev_one_step_bayesian_decoder.xbin_centers, ybin_values=prev_one_step_bayesian_decoder.ybin_centers, debug_print=debug_print) # update the session's position dataframe with the new columns.
        
        active_xbins = xbin
        active_ybins = ybin      
        # active_xbins = prev_one_step_bayesian_decoder.xbin_centers
        # active_ybins = prev_one_step_bayesian_decoder.ybin_centers
        
        avg_speed_per_pos = _compute_avg_speed_at_each_position_bin(computation_result.sess.position.to_dataframe(), computation_result.computation_config, prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers, debug_print=debug_print)
                
        if debug_print:
            print(f'np.shape(avg_speed_per_pos): {np.shape(avg_speed_per_pos)}')
        
        max_speed = np.nanmax(avg_speed_per_pos)
        # max_speed # 73.80995983236636
        min_speed = np.nanmin(avg_speed_per_pos)
        # min_speed # 0.0
        K_over_V = 60.0 / max_speed # K_over_V = 0.8128984236852197
    
        # K = 1.0
        K = K_over_V
        V = 1.0
        sigma_t_all = Zhang_Two_Step.sigma_t(avg_speed_per_pos, K, V, d=1.0) # np.shape(sigma_t_all): (64, 29)
        if debug_print:
            print(f'np.shape(sigma_t_all): {np.shape(sigma_t_all)}')
        
        # normalize sigma_t_all:
        computation_result.computed_data['pf2D_TwoStepDecoder'] = {'xbin':active_xbins, 'ybin':active_ybins,
                                                                   'avg_speed_per_pos': avg_speed_per_pos,
                                                                   'K':K, 'V':V,
                                                                   'sigma_t_all':sigma_t_all, 'flat_sigma_t_all': np.squeeze(np.reshape(sigma_t_all, (-1, 1)))
        }
        
        computation_result.computed_data['pf2D_TwoStepDecoder']['C'] = 1.0
        computation_result.computed_data['pf2D_TwoStepDecoder']['k'] = 1.0
        # computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev_fn'] = lambda x_prev, all_x: Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep(prev_one_step_bayesian_decoder.p_x_given_n, x_prev, all_x, 
        #                                                                             computation_result.computed_data['pf2D_TwoStepDecoder']['sigma_t_all'], 
        #                                                                             computation_result.computed_data['pf2D_TwoStepDecoder']['C'], computation_result.computed_data['pf2D_TwoStepDecoder']['k'])
        
        # ValueError: operands could not be broadcast together with shapes (64,29,3434) (65,30)

        # pre-allocate outputs:
        # np.vstack((self.xbin_centers[self.most_likely_position_indicies[0,:]], self.ybin_centers[self.most_likely_position_indicies[1,:]])).T
        # twoDimGrid_x, twoDimGrid_y = np.meshgrid(prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers)
        
        # computation_result.computed_data['pf2D_TwoStepDecoder']['all_x'] = cartesian_product((active_xbins, active_ybins)) # (1856, 2)
        
        computation_result.computed_data['pf2D_TwoStepDecoder']['all_x'], computation_result.computed_data['pf2D_TwoStepDecoder']['flat_all_x'], original_data_shape = Zhang_Two_Step.build_all_positions_matrix(prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers) # all_x: (64, 29, 2), flat_all_x: (1856, 2)
        computation_result.computed_data['pf2D_TwoStepDecoder']['original_all_x_shape'] = original_data_shape # add the original data shape to the computed data
  
        # Pre-allocate output:
        computation_result.computed_data['pf2D_TwoStepDecoder']['flat_p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.flat_p_x_given_n, 9.0) # fill with NaNs. 
        computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.p_x_given_n, 9.0) # fill with NaNs. Pre-allocate output
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'] = np.zeros((2, prev_one_step_bayesian_decoder.num_time_windows), dtype=int) # (2, 85841)
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'] = np.zeros((2, prev_one_step_bayesian_decoder.num_time_windows)) # (2, 85841)
                
        if debug_print:
            print(f'np.shape(prev_one_step_bayesian_decoder.p_x_given_n): {np.shape(prev_one_step_bayesian_decoder.p_x_given_n)}')
        
        computation_result.computed_data['pf2D_TwoStepDecoder']['all_scaling_factors_k'] = Zhang_Two_Step.compute_scaling_factor_k(prev_one_step_bayesian_decoder.flat_p_x_given_n)
        
        # TODO: Efficiency: This will be inefficient, but do a slow iteration. 
        for time_window_bin_idx in np.arange(prev_one_step_bayesian_decoder.num_time_windows):
            flat_p_x_given_n = prev_one_step_bayesian_decoder.flat_p_x_given_n[:, time_window_bin_idx] # this gets the specific n_t for this time window
            curr_p_x_given_n = prev_one_step_bayesian_decoder.p_x_given_n[:, :, time_window_bin_idx]
            # also have p_x_given_n = prev_one_step_bayesian_decoder.p_x_given_n if we'd prefer

            # TODO: as for prev_x_position: this should actually be the computed two-step position, not the one_step position.
            
            # previous positions as determined by the one_step decoder:
            # prev_x_flat_index = prev_one_step_bayesian_decoder.most_likely_position_flat_indicies[time_window_bin_idx-1] # this is the most likely position (represented as the flattened position bin index) at the last dataframe
            # prev_x_position = prev_one_step_bayesian_decoder.most_likely_positions[time_window_bin_idx-1, :] # (85844, 2)
            
            # previous positions as determined by the two-step decoder: this uses the two_step previous position instead of the one_step previous position:
            prev_x_position = computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][:, time_window_bin_idx-1]
            
            active_k = computation_result.computed_data['pf2D_TwoStepDecoder']['all_scaling_factors_k'][time_window_bin_idx] # get the specific k value
            # active_k = computation_result.computed_data['pf2D_TwoStepDecoder']['k']
            if debug_print:
                print(f'np.shape(curr_p_x_given_n): {np.shape(curr_p_x_given_n)}')
                print(f'np.shape(prev_x_position): {np.shape(prev_x_position)}')
                        
            # Flat version:
            computation_result.computed_data['pf2D_TwoStepDecoder']['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx] = Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep(flat_p_x_given_n, prev_x_position, computation_result.computed_data['pf2D_TwoStepDecoder']['flat_all_x'], computation_result.computed_data['pf2D_TwoStepDecoder']['flat_sigma_t_all'], computation_result.computed_data['pf2D_TwoStepDecoder']['C'], active_k) # output shape (1856, )
            
            computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'][:,:,time_window_bin_idx] = np.reshape(computation_result.computed_data['pf2D_TwoStepDecoder']['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx], (original_data_shape[0], original_data_shape[1]))
            
            # Compute the most-likely positions from the p_x_given_n_and_x_prev:
            active_argmax_idx = np.argmax(computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'][:,:,time_window_bin_idx], axis=None)
            # active_unreaveled_argmax_idx = np.array(np.unravel_index(active_argmax_idx, computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'].shape))
            active_unreaveled_argmax_idx = np.array(np.unravel_index(active_argmax_idx, prev_one_step_bayesian_decoder.original_position_data_shape))
            # print(f'active_argmax_idx: {active_argmax_idx}, active_unreaveled_argmax_idx: {active_unreaveled_argmax_idx}')
            
            computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][:, time_window_bin_idx] = active_unreaveled_argmax_idx # build the multi-dimensional maximum index for the position (not using the flat notation used in the other class)            
            computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][0, time_window_bin_idx] = prev_one_step_bayesian_decoder.xbin_centers[int(computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][0, time_window_bin_idx])]
            computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][1, time_window_bin_idx] = prev_one_step_bayesian_decoder.ybin_centers[int(computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][1, time_window_bin_idx])]
            

        # # POST-hoc most-likely computations: Compute the most-likely positions from the p_x_given_n_and_x_prev:
        # computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'] = np.array(np.unravel_index(np.argmax(computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'], axis=None), computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'].shape)) # build the multi-dimensional maximum index for the position (not using the flat notation used in the other class)
        # # np.shape(self.most_likely_position_indicies) # (2, 85841)
        
        
        # computation_result.computed_data['pf2D_TwoStepDecoder']['sigma_t_all'] = sigma_t_all # set sigma_t_all                
        # return position_decoding_second_order_computation(computation_result.sess, computation_result.computation_config, computation_result)
        return computation_result



class DefaultRegisteredComputations:
    """ Simply enables specifying the default computation functions that will be defined in this file and automatically registered. """
    def register_default_known_computation_functions(self):
        self.register_computation(DefaultComputationFunctions._perform_extended_statistics_computation)
        self.register_computation(DefaultComputationFunctions._perform_two_step_position_decoding_computation)
        self.register_computation(DefaultComputationFunctions._perform_position_decoding_computation)
        
    


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
    
    ## Computation Helpers: 
    def perform_computations(self, active_computation_params: PlacefieldComputationParameters=None):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.single_computation(active_computation_params)
        
    def register_computation(self, computation_function):
        assert (self.can_compute), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_computation(computation_function)

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
        self.registered_computation_functions = list()
        self.register_default_known_computation_functions() # registers the default
        
    def register_computation(self, computation_function):
        self.registered_computation_functions.append(computation_function)
        
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
    