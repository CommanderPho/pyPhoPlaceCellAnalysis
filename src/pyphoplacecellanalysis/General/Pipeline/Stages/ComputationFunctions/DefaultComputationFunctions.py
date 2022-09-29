from copy import deepcopy
import sys
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult


# from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import build_position_df_discretized_binned_positions # old weird re-implementation
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.utils.dynamic_container import DynamicContainer # for _perform_two_step_position_decoding_computation

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, Zhang_Two_Step

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # For _perform_new_position_decoding_computation
from pyphocorehelpers.indexing_helpers import BinningInfo, compute_spanning_bins, get_bin_centers, get_bin_edges, debug_print_1D_bin_infos, interleave_elements # For _perform_new_position_decoding_computation
from pyphocorehelpers.indexing_helpers import build_spanning_grid_matrix # For _perform_new_position_decoding_computation



"""-------------- Specific Computation Functions to be registered --------------"""

class DefaultComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 1 # must be done after PlacefieldComputations

    def _perform_position_decoding_computation(computation_result: ComputationResult, **kwargs):
        """ Builds the 2D Placefield Decoder 
        
            ## - [ ] TODO: IMPORTANT!! POTENTIAL_BUG: Should this passed-in spikes_df actually be the filtered spikes_df that was used to compute the placefields in PfND? That would be `prev_output_result.computed_data['pf2D'].filtered_spikes_df`
                TODO: CORRECTNESS: Consider whether spikes_df or just the spikes_df used to compute the pf2D should be passed. Only the cells used to build the decoder should be used to decode, that much is certain.
                    - 2022-09-15: it appears that the 'filtered_spikes_df version' improves issues with decoder jumpiness and general inaccuracy that was present when using the session spikes_df: previously it was just jumping to random points far from the animal's location and then sticking there for a long time.
        
        """
        def position_decoding_computation(active_session, pf_computation_config, prev_output_result: ComputationResult):
            """ uses the pf2D property of "prev_output_result.computed_data['pf2D'] """
            ## filtered_spikes_df version:
            prev_output_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, prev_output_result.computed_data['pf2D'], prev_output_result.computed_data['pf2D'].filtered_spikes_df.copy(), debug_print=False)
            ## original `active_session.spikes_df` version:
            # prev_output_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, prev_output_result.computed_data['pf2D'], active_session.spikes_df.copy(), debug_print=False)
            # %timeit pho_custom_decoder.compute_all():  18.8 s ± 149 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            prev_output_result.computed_data['pf2D_Decoder'].compute_all() #  --> n = self.
            return prev_output_result

        placefield_computation_config = computation_result.computation_config.pf_params # should be a PlacefieldComputationParameters
        return position_decoding_computation(computation_result.sess, placefield_computation_config, computation_result)
    
    
    def _perform_two_step_position_decoding_computation(computation_result: ComputationResult, debug_print=False, **kwargs):
        """ Builds the Zhang Velocity/Position For 2-step Bayesian Decoder for 2D Placefields """
        def _compute_avg_speed_at_each_position_bin(active_position_df, active_pf_computation_config, xbin, ybin, show_plots=False, debug_print=False):
            """ compute the average speed at each position x: """
            
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
        # active_pos_df: computation_result.sess.position.df
        # xbin_values = prev_one_step_bayesian_decoder.xbin_centers.copy()
        # ybin_values = prev_one_step_bayesian_decoder.ybin_centers.copy()
                
        # ## Old pyphoplacecellanalysis.Analysis.Decoder.decoder_result.build_position_df_discretized_binned_positions(...) version:
        # # makes sure to use the xbin_center and ybin_center from the previous one_step decoder to bin the positions:
        # computation_result.sess.position.df, xbin, ybin, bin_info = build_position_df_discretized_binned_positions(computation_result.sess.position.df, computation_result.computation_config.pf_params, xbin_values=xbin_values, ybin_values=ybin_values, debug_print=debug_print) # update the session's position dataframe with the new columns.
        
        ## New 2022-09-15 direct neuropy.utils.mixins.binning_helpers.build_df_discretized_binned_position_columns version:
        computation_result.sess.position.df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(computation_result.sess.position.df, bin_values=(prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers), active_computation_config=computation_result.computation_config.pf_params, force_recompute=False, debug_print=debug_print)
        # computation_result.sess.position.df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(computation_result.sess.position.df, bin_values=(xbin_values, ybin_values), active_computation_config=computation_result.computation_config.pf_params, force_recompute=False, debug_print=debug_print)
        
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
        computation_result.computed_data['pf2D_TwoStepDecoder'] = DynamicParameters.init_from_dict({'xbin':active_xbins, 'ybin':active_ybins,
                                                                   'avg_speed_per_pos': avg_speed_per_pos,
                                                                   'K':K, 'V':V,
                                                                   'sigma_t_all':sigma_t_all, 'flat_sigma_t_all': np.squeeze(np.reshape(sigma_t_all, (-1, 1)))
        })
        
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

        ## For some reason we set up the two-step decoder's most_likely_positions with the tranposed shape compared to the one-step decoder:
        computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'] = computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'].T
        
        ## Once done, compute marginals for the two-step:
        curr_unit_marginal_x, curr_unit_marginal_y = prev_one_step_bayesian_decoder.perform_build_marginals(computation_result.computed_data['pf2D_TwoStepDecoder']['p_x_given_n_and_x_prev'], computation_result.computed_data['pf2D_TwoStepDecoder']['most_likely_positions'], debug_print=debug_print)
        computation_result.computed_data['pf2D_TwoStepDecoder']['marginal'] = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
        
        
        
        
        return computation_result

