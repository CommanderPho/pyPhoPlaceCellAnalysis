from copy import deepcopy
import sys
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.utils.dynamic_container import DynamicContainer # for _perform_two_step_position_decoding_computation
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # used in _subfn_compute_decoded_epochs to get only the valid (non-overlapping) epochs

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, Zhang_Two_Step

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder


# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # For _perform_new_position_decoding_computation
from pyphocorehelpers.indexing_helpers import BinningInfo, compute_spanning_bins, get_bin_centers, get_bin_edges, debug_print_1D_bin_infos, interleave_elements # For _perform_new_position_decoding_computation
from pyphocorehelpers.indexing_helpers import build_spanning_grid_matrix # For _perform_new_position_decoding_computation

# ### For _perform_recursive_latent_placefield_decoding
# from neuropy.utils import position_util
# from neuropy.core import Position
# from neuropy.analyses.placefields import perform_compute_placefields

"""-------------- Specific Computation Functions to be registered --------------"""

class DefaultComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 1 # must be done after PlacefieldComputations
    _is_global = False

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
        
        ## In this new mode we'll add the two-step properties to the original one-step decoder:
        ## Adds the directly accessible properties to the active_one_step_decoder after they're computed in the active_two_step_decoder so that they can be plotted with the same functions/etc.

        # None initialize two-step properties on the one_step_decoder:
        prev_one_step_bayesian_decoder.p_x_given_n_and_x_prev = None
        prev_one_step_bayesian_decoder.two_step_most_likely_positions = None

        prev_one_step_bayesian_decoder.marginal.x.p_x_given_n_and_x_prev = None
        prev_one_step_bayesian_decoder.marginal.x.two_step_most_likely_positions_1D = None

        prev_one_step_bayesian_decoder.marginal.y.p_x_given_n_and_x_prev = None
        prev_one_step_bayesian_decoder.marginal.y.two_step_most_likely_positions_1D = None

        # Set the two-step properties on the one-step decoder:
        prev_one_step_bayesian_decoder.p_x_given_n_and_x_prev = computation_result.computed_data['pf2D_TwoStepDecoder'].p_x_given_n_and_x_prev.copy()
        prev_one_step_bayesian_decoder.two_step_most_likely_positions = computation_result.computed_data['pf2D_TwoStepDecoder'].most_likely_positions.copy()

        prev_one_step_bayesian_decoder.marginal.x.p_x_given_n_and_x_prev = computation_result.computed_data['pf2D_TwoStepDecoder'].marginal.x.p_x_given_n.copy()
        prev_one_step_bayesian_decoder.marginal.x.two_step_most_likely_positions_1D = computation_result.computed_data['pf2D_TwoStepDecoder'].marginal.x.most_likely_positions_1D.copy()

        if prev_one_step_bayesian_decoder.marginal.y is not None:
            prev_one_step_bayesian_decoder.marginal.y.p_x_given_n_and_x_prev = computation_result.computed_data['pf2D_TwoStepDecoder'].marginal.y.p_x_given_n.copy()
            prev_one_step_bayesian_decoder.marginal.y.two_step_most_likely_positions_1D = computation_result.computed_data['pf2D_TwoStepDecoder'].marginal.y.most_likely_positions_1D.copy()

        return computation_result


    def _perform_recursive_latent_placefield_decoding(computation_result: ComputationResult, **kwargs):
        """ note that currently the pf1D_Decoders are not built or used. 

        """
        ### For _perform_recursive_latent_placefield_decoding
        from neuropy.utils import position_util
        from neuropy.core import Position
        from neuropy.analyses.placefields import perform_compute_placefields

        def _subfn_build_recurrsive_placefields(active_one_step_decoder, next_order_computation_config, spikes_df=None, pos_df=None, pos_linearization_method='isomap'):
            if spikes_df is None:
                spikes_df = active_one_step_decoder.spikes_df
            if pos_df is None:
                pos_df = active_one_step_decoder.pf.filtered_pos_df
            
            def _prepare_pos_df_for_recurrsive_decoding(active_one_step_decoder, pos_df, pos_linearization_method='isomap'):
                """ duplicates pos_df and builds a new pseudo-pos_df for building second-order placefields/decoder 
                pos_df comes in with columns: ['t', 'x', 'y', 'lin_pos', 'speed', 'binned_x', 'binned_y']

                ISSUE/POTENTIAL BUG: 'lin_pos' which is computed for the second-order pos_df seems completely off when compared to the incoming pos_df.
                    - [ ] Actually the 'x' and 'y' values seem pretty off too. Not sure if this is being computed correctly.
                """
                ## Build the new second-order pos_df from the decoded positions:
                active_second_order_pos_df = pd.DataFrame({'t': active_one_step_decoder.active_time_window_centers, 'x': active_one_step_decoder.most_likely_positions[:,0], 'y': active_one_step_decoder.most_likely_positions[:,1]})

                ## Build the linear position for the second-order pos_df:
                _temp_pos_obj = Position(active_second_order_pos_df) # position_util.linearize_position(...) expects a neuropy Position object instead of a raw DataFrame, so build a temporary one to make it happy
                linear_pos = position_util.linearize_position(_temp_pos_obj, method=pos_linearization_method)
                active_second_order_pos_df['lin_pos'] = linear_pos.x
                return active_second_order_pos_df

            def _prepare_spikes_df_for_recurrsive_decoding(active_one_step_decoder, spikes_df):
                """ duplicates spikes_df and builds a new pseudo-spikes_df for building second-order placefields/decoder """
                active_second_order_spikes_df = deepcopy(spikes_df)
                # TODO: figure it out instead of hacking -- Just drop the last time bin because something is off 
                invalid_timestamp = np.nanmax(active_second_order_spikes_df['binned_time'].astype(int).to_numpy()) # 11881
                active_second_order_spikes_df = active_second_order_spikes_df[active_second_order_spikes_df['binned_time'].astype(int) < invalid_timestamp] # drop the last time-bin as a workaround
                # backup measured columns because they will be replaced by the new values:
                active_second_order_spikes_df['x_measured'] = active_second_order_spikes_df['x'].copy()
                active_second_order_spikes_df['y_measured'] = active_second_order_spikes_df['y'].copy()
                spike_binned_time_idx = (active_second_order_spikes_df['binned_time'].astype(int)-1) # subtract one to get to a zero-based index
                # replace the x and y measured positions with the most-likely decoded ones for the next round of decoding
                active_second_order_spikes_df['x'] = active_one_step_decoder.most_likely_positions[spike_binned_time_idx.to_numpy(),0] # x-pos
                active_second_order_spikes_df['y'] = active_one_step_decoder.most_likely_positions[spike_binned_time_idx.to_numpy(),1] # y-pos
                return active_second_order_spikes_df

            def _next_order_decode(active_pf_1D, active_pf_2D, pf_computation_config, manual_time_window_edges=None, manual_time_window_edges_binning_info=None):
                ## 1D Decoder
                new_decoder_pf1D = active_pf_1D
                new_1D_decoder_spikes_df = new_decoder_pf1D.filtered_spikes_df.copy()
                # new_1D_decoder_spikes_df = new_1D_decoder_spikes_df.spikes.add_binned_time_column(manual_time_window_edges, manual_time_window_edges_binning_info, debug_print=False)
                new_1D_decoder = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, new_decoder_pf1D, new_1D_decoder_spikes_df, debug_print=False) # , manual_time_window_edges=manual_time_window_edges, manual_time_window_edges_binning_info=manual_time_window_edges_binning_info
                # new_1D_decoder.compute_all() #  --> n = self.

                ## Custom Manual 2D Decoder:
                new_decoder_pf2D = active_pf_2D # 
                new_decoder_spikes_df = new_decoder_pf2D.filtered_spikes_df.copy()
                # new_decoder_spikes_df = new_decoder_spikes_df.spikes.add_binned_time_column(manual_time_window_edges, manual_time_window_edges_binning_info, debug_print=False)
                new_2D_decoder = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, new_decoder_pf2D, new_decoder_spikes_df, debug_print=False) # , manual_time_window_edges=manual_time_window_edges, manual_time_window_edges_binning_info=manual_time_window_edges_binning_info
                new_2D_decoder.compute_all() #  --> n = self.
                
                return new_1D_decoder, new_2D_decoder

            active_second_order_spikes_df = _prepare_spikes_df_for_recurrsive_decoding(active_one_step_decoder, spikes_df)
            active_second_order_pos_df = _prepare_pos_df_for_recurrsive_decoding(active_one_step_decoder, pos_df)
            active_second_order_pf_1D, active_second_order_pf_2D = perform_compute_placefields(active_second_order_spikes_df, Position(active_second_order_pos_df),
                                                                                            next_order_computation_config.pf_params, None, None, included_epochs=None, should_force_recompute_placefields=True)
            # build the second_order decoders:
            active_second_order_1D_decoder, active_second_order_2D_decoder = _next_order_decode(active_second_order_pf_1D, active_second_order_pf_2D, next_order_computation_config.pf_params) # , manual_time_window_edges=active_one_step_decoder.time_window_edges, manual_time_window_edges_binning_info=active_one_step_decoder.time_window_edges_binning_info
            
            return active_second_order_pf_1D, active_second_order_pf_2D, active_second_order_1D_decoder, active_second_order_2D_decoder

        pos_linearization_method='isomap'
        prev_one_step_bayesian_decoder = computation_result.computed_data['pf2D_Decoder']
        ## Builds a duplicate of the current computation config but sets the speed_thresh to 0.0:

        next_order_computation_config = deepcopy(computation_result.computation_config) # make a deepcopy of the active computation config
        # next_order_computation_config = deepcopy(active_session_computation_configs[0]) # make a deepcopy of the active computation config
        next_order_computation_config.pf_params.speed_thresh = 0.0 # no speed thresholding because the speeds aren't real for the second-order fields

        # Start with empty lists, which will accumulate the different levels of recurrsive depth:
        computation_result.computed_data['pf1D_RecursiveLatent'] = []
        computation_result.computed_data['pf2D_RecursiveLatent'] = []

        # 1st Order:
        computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':computation_result.computed_data['pf1D'], 'pf1D_Decoder':computation_result.computed_data.get('pf1D_Decoder', {})}))
        computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':computation_result.computed_data['pf2D'], 'pf2D_Decoder':computation_result.computed_data['pf2D_Decoder']}))

        # 2nd Order:
        active_second_order_pf_1D, active_second_order_pf_2D, active_second_order_1D_decoder, active_second_order_2D_decoder = _subfn_build_recurrsive_placefields(prev_one_step_bayesian_decoder, next_order_computation_config=next_order_computation_config, spikes_df=prev_one_step_bayesian_decoder.spikes_df, pos_df=prev_one_step_bayesian_decoder.pf.filtered_pos_df, pos_linearization_method=pos_linearization_method)
        computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':active_second_order_pf_1D, 'pf1D_Decoder':active_second_order_1D_decoder}))
        computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':active_second_order_pf_2D, 'pf2D_Decoder':active_second_order_2D_decoder}))

        # # 3rd Order:
        active_third_order_pf_1D, active_third_order_pf_2D, active_third_order_1D_decoder, active_third_order_2D_decoder = _subfn_build_recurrsive_placefields(active_second_order_2D_decoder, next_order_computation_config=next_order_computation_config, spikes_df=active_second_order_2D_decoder.spikes_df, pos_df=active_second_order_2D_decoder.pf.filtered_pos_df, pos_linearization_method=pos_linearization_method)
        computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':active_third_order_pf_1D, 'pf1D_Decoder':active_third_order_1D_decoder}))
        computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':active_third_order_pf_2D, 'pf2D_Decoder':active_third_order_2D_decoder}))

        return computation_result



    def _perform_specific_epochs_decoding(computation_result: ComputationResult, active_config, filter_epochs='ripple', decoding_time_bin_size=0.02, **kwargs):
        """ TODO: meant to be used by `_display_plot_decoded_epoch_slices` but needs a smarter way to cache the computations and etc. 
        Eventually to replace `pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError._compute_specific_decoded_epochs`

        Usage:
            ## Test _perform_specific_epochs_decoding
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import DefaultComputationFunctions
            computation_result = curr_active_pipeline.computation_results['maze1_PYR']
            computation_result = DefaultComputationFunctions._perform_specific_epochs_decoding(computation_result, curr_active_pipeline.active_configs['maze1_PYR'], filter_epochs='ripple', decoding_time_bin_size=0.02)
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = computation_result.computed_data['specific_epochs_decoding'][('Ripples', 0.02)]

        """

        def _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs='ripple', decoding_time_bin_size=0.02):
            """ compuites a plot with the 1D Marginals either (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. 
            
            It determines which epochs are being referred to (enabling specifying them by a simple string identifier, like 'ripple', 'pbe', or 'laps') and then gets the coresponding data that's needed to recompute the decoded data for them.
            This decoding is done by calling:
                active_decoder.decode_specific_epochs(...) which returns a result that can then be plotted.
            
            """    
            default_figure_name = 'stacked_epoch_slices_matplotlib_subplots'
            active_decoder = computation_result.computed_data['pf2D_Decoder']
            
            if isinstance(filter_epochs, str):
                if filter_epochs == 'laps':
                    ## Lap-Epochs Decoding:
                    laps_copy = deepcopy(computation_result.sess.laps)
                    # active_filter_epochs = laps_copy.filtered_by_lap_flat_index(np.arange(6)).as_epoch_obj() # epoch object
                    active_filter_epochs = laps_copy.as_epoch_obj() # epoch object
                    pre_exclude_n_epochs = active_filter_epochs.n_epochs

                    # default_figure_name = f'{default_figure_name}_Laps'
                    default_figure_name = f'Laps'

                    ## HANDLE OVERLAPPING EPOCHS: Note that there is a problem that occurs here with overlapping epochs for laps. Below we remove any overlapping epochs and leave only the valid ones.
                    is_non_overlapping = get_non_overlapping_epochs(active_filter_epochs.to_dataframe()[['start','stop']].to_numpy()) # returns a boolean array of the same length as the number of epochs
                    non_overlapping_labels = active_filter_epochs.labels[is_non_overlapping] # array(['41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74'], dtype=object)
                    # Slice by the valid (non-overlapping) labels to get the new epoch object:
                    active_filter_epochs = active_filter_epochs.label_slice(non_overlapping_labels)
                    post_exclude_n_epochs = active_filter_epochs.n_epochs                    
                    num_excluded_epochs = post_exclude_n_epochs - pre_exclude_n_epochs
                    if num_excluded_epochs > 0:
                        print(f'num_excluded_epochs: {num_excluded_epochs} due to overlap.')

                    # ## Build Epochs:
                    # # epochs = sess.laps.to_dataframe()
                    # epochs = active_filter_epochs.to_dataframe()
                    # epoch_slices = epochs[['start', 'stop']].to_numpy()
                    # epoch_description_list = [f'lap {epoch_tuple.lap_id} (maze: {epoch_tuple.maze_id}, direction: {epoch_tuple.lap_dir})' for epoch_tuple in active_filter_epochs.to_dataframe()[['lap_id','maze_id','lap_dir']].itertuples()] # LONG
                    epoch_description_list = [f'lap[{epoch_tuple.lap_id}]' for epoch_tuple in active_filter_epochs.to_dataframe()[['lap_id']].itertuples()] # Short
                    
                    
                elif filter_epochs == 'pbe':
                    ## PBEs-Epochs Decoding:
                    active_filter_epochs = deepcopy(computation_result.sess.pbe) # epoch object
                    # default_figure_name = f'{default_figure_name}_PBEs'
                    default_figure_name = f'PBEs'
                
                elif filter_epochs == 'ripple':
                    ## Ripple-Epochs Decoding:
                    active_filter_epochs = deepcopy(computation_result.sess.ripple) # epoch object
                    # default_figure_name = f'{default_figure_name}_Ripples'
                    default_figure_name = f'Ripples'
                    active_epoch_df = active_filter_epochs.to_dataframe()
                    # if 'label' not in active_epoch_df.columns:
                    active_epoch_df['label'] = active_epoch_df.index.to_numpy() # integer ripple indexing
                    # epoch_description_list = [f'ripple {epoch_tuple.label} (peakpower: {epoch_tuple.peakpower})' for epoch_tuple in active_filter_epochs.to_dataframe()[['label', 'peakpower']].itertuples()] # LONG
                    epoch_description_list = [f'ripple[{epoch_tuple.label}]' for epoch_tuple in active_epoch_df[['label']].itertuples()] # SHORT
                    
                elif filter_epochs == 'replay':
                    active_filter_epochs = deepcopy(computation_result.sess.replay) # epoch object
                    # active_filter_epochs = active_filter_epochs.drop_duplicates("start") # tries to remove duplicate replays to take care of `AssertionError: Intervals in start_stop_times_arr must be non-overlapping`, but it hasn't worked.

                    # filter_epochs.columns # ['epoch_id', 'rel_id', 'start', 'end', 'replay_r', 'replay_p', 'template_id', 'flat_replay_idx', 'duration']
                    if not 'stop' in active_filter_epochs.columns:
                        # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
                        active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
                    # default_figure_name = f'{default_figure_name}_Replay'
                    default_figure_name = f'Replay'

                    # TODO 2022-10-04 - CORRECTNESS - AssertionError: Intervals in start_stop_times_arr must be non-overlapping. I believe this is due to the stop values overlapping somewhere
                    print(f'active_filter_epochs: {active_filter_epochs}')
                    ## HANDLE OVERLAPPING EPOCHS: Note that there is a problem that occurs here with overlapping epochs for laps. Below we remove any overlapping epochs and leave only the valid ones.
                    is_non_overlapping = get_non_overlapping_epochs(active_filter_epochs[['start','stop']].to_numpy()) # returns a boolean array of the same length as the number of epochs 
                    # Just drop the rows of the dataframe that are overlapping:
                    # active_filter_epochs = active_filter_epochs[is_non_overlapping, :]
                    active_filter_epochs = active_filter_epochs.loc[is_non_overlapping]
                    print(f'active_filter_epochs: {active_filter_epochs}')

                    # epoch_description_list = [f'{default_figure_name} {epoch_tuple.epoch_id}' for epoch_tuple in active_filter_epochs[['epoch_id']].itertuples()]
                    epoch_description_list = [f'{default_figure_name} {epoch_tuple.flat_replay_idx}' for epoch_tuple in active_filter_epochs[['flat_replay_idx']].itertuples()]
                    
                else:
                    raise NotImplementedError
            else:
                # Use it raw, hope it's right
                active_filter_epochs = filter_epochs
                default_figure_name = f'{default_figure_name}_CUSTOM'
                epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs.to_dataframe()[['label']].itertuples()]
                
            filter_epochs_decoder_result = active_decoder.decode_specific_epochs(computation_result.sess.spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
            filter_epochs_decoder_result.epoch_description_list = epoch_description_list
            return filter_epochs_decoder_result, active_filter_epochs, default_figure_name


        curr_result = computation_result.computed_data.get('specific_epochs_decoding', {})
        
        ## Do the computation:
        filter_epochs_decoder_result, active_filter_epochs, default_figure_name = _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size)

        crappy_key = (default_figure_name, decoding_time_bin_size)
        curr_result[crappy_key] = (filter_epochs_decoder_result, active_filter_epochs, default_figure_name)

        computation_result.computed_data['specific_epochs_decoding'] = curr_result
        return computation_result
