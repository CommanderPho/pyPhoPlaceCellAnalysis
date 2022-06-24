import sys
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult


from pyphoplacecellanalysis.General.Decoder.decoder_result import build_position_df_discretized_binned_positions, build_position_df_resampled_to_time_windows
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.Analysis.reconstruction import BayesianPlacemapPositionDecoder, Zhang_Two_Step

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


"""-------------- Specific Computation Functions to be registered --------------"""

class DefaultComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 1 # must be done after PlacefieldComputations

    def _perform_position_decoding_computation(computation_result: ComputationResult):
        """ Builds the 2D Placefield Decoder """
        def position_decoding_computation(active_session, pf_computation_config, prev_output_result: ComputationResult):
            prev_output_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, prev_output_result.computed_data['pf2D'], active_session.spikes_df.copy(), debug_print=False)
            # %timeit pho_custom_decoder.compute_all():  18.8 s ± 149 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            prev_output_result.computed_data['pf2D_Decoder'].compute_all() #  --> n = self.
            return prev_output_result

        placefield_computation_config = computation_result.computation_config.pf_params # should be a PlacefieldComputationParameters
        return position_decoding_computation(computation_result.sess, placefield_computation_config, computation_result)
    
    
    def _perform_two_step_position_decoding_computation(computation_result: ComputationResult, debug_print=False):
        """ Builds the Zhang Velocity/Position For 2-step Bayesian Decoder for 2D Placefields """
        def _compute_avg_speed_at_each_position_bin(active_position_df, active_pf_computation_config, xbin, ybin, show_plots=False, debug_print=False):
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
        computation_result.sess.position.df, xbin, ybin, bin_info = build_position_df_discretized_binned_positions(computation_result.sess.position.df, computation_result.computation_config.pf_params, xbin_values=prev_one_step_bayesian_decoder.xbin_centers, ybin_values=prev_one_step_bayesian_decoder.ybin_centers, debug_print=debug_print) # update the session's position dataframe with the new columns.
        
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
        computation_result.computed_data['pf2D_TwoStepDecoder']['flat_p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.flat_p_x_given_n, 0.0) # fill with NaNs. 
        computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.p_x_given_n, 0.0) # fill with NaNs. Pre-allocate output
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
            # active_argmax_idx = np.argmax(computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'][:,:,time_window_bin_idx], axis=None)
            # active_unreaveled_argmax_idx = np.array(np.unravel_index(active_argmax_idx, computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'].shape))
            # active_unreaveled_argmax_idx = np.array(np.unravel_index(active_argmax_idx, prev_one_step_bayesian_decoder.original_position_data_shape))
            # print(f'active_argmax_idx: {active_argmax_idx}, active_unreaveled_argmax_idx: {active_unreaveled_argmax_idx}')
            # computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][:, time_window_bin_idx] = active_unreaveled_argmax_idx # build the multi-dimensional maximum index for the position (not using the flat notation used in the other class)            
            # computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][0, time_window_bin_idx] = prev_one_step_bayesian_decoder.xbin_centers[int(computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][0, time_window_bin_idx])]
            # computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][1, time_window_bin_idx] = prev_one_step_bayesian_decoder.ybin_centers[int(computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][1, time_window_bin_idx])]
            

        # POST-hoc most-likely computations: Compute the most-likely positions from the p_x_given_n_and_x_prev:
        # computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'] = np.array(np.unravel_index(np.argmax(computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'], axis=None), computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'].shape)) # build the multi-dimensional maximum index for the position (not using the flat notation used in the other class)
        # # np.shape(self.most_likely_position_indicies) # (2, 85841)
        """ Computes the most likely positions at each timestep from flat_p_x_given_n_and_x_prev """
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_flat_indicies'] = np.argmax(computation_result.computed_data['pf2D_TwoStepDecoder']['flat_p_x_given_n_and_x_prev'], axis=0)
        # np.shape(self.most_likely_position_flat_indicies) # (85841,)
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'] = np.array(np.unravel_index(computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_flat_indicies'], prev_one_step_bayesian_decoder.original_position_data_shape)) # convert back to an array
        # np.shape(self.most_likely_position_indicies) # (2, 85841)
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][0, :] = prev_one_step_bayesian_decoder.xbin_centers[computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][0, :]]
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'][1, :] = prev_one_step_bayesian_decoder.ybin_centers[computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_position_indicies'][1, :]]
        # computation_result.computed_data['pf2D_TwoStepDecoder']['sigma_t_all'] = sigma_t_all # set sigma_t_all                
        return computation_result
