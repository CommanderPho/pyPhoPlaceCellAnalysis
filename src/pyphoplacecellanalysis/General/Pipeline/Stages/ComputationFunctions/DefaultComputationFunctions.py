from copy import deepcopy
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance # for Jensen-Shannon distance in `_subfn_compute_leave_one_out_analysis`

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.core.epoch import Epoch
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.utils.dynamic_container import DynamicContainer # for _perform_two_step_position_decoding_computation
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # used in _subfn_compute_decoded_epochs to get only the valid (non-overlapping) epochs

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, Zhang_Two_Step

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder

# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # For _perform_new_position_decoding_computation

# ### For _perform_recursive_latent_placefield_decoding
# from neuropy.utils import position_util
# from neuropy.core import Position
# from neuropy.analyses.placefields import perform_compute_placefields

"""-------------- Specific Computation Functions to be registered --------------"""

class DefaultComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationPrecidence = 1 # must be done after PlacefieldComputations
    _is_global = False

    def _perform_position_decoding_computation(computation_result: ComputationResult, **kwargs):
        """ Builds the 1D & 2D Placefield Decoder 
        
            ## - [ ] TODO: IMPORTANT!! POTENTIAL_BUG: Should this passed-in spikes_df actually be the filtered spikes_df that was used to compute the placefields in PfND? That would be `prev_output_result.computed_data['pf2D'].filtered_spikes_df`
                TODO: CORRECTNESS: Consider whether spikes_df or just the spikes_df used to compute the pf2D should be passed. Only the cells used to build the decoder should be used to decode, that much is certain.
                    - 2022-09-15: it appears that the 'filtered_spikes_df version' improves issues with decoder jumpiness and general inaccuracy that was present when using the session spikes_df: **previously it was just jumping to random points far from the animal's location and then sticking there for a long time**.
        
        """
        def position_decoding_computation(active_session, pf_computation_config, prev_output_result):
            """ uses the pf2D property of "prev_output_result.computed_data['pf2D'] """
            ## filtered_spikes_df version:
            prev_output_result.computed_data['pf1D_Decoder'] = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, prev_output_result.computed_data['pf1D'], prev_output_result.computed_data['pf1D'].filtered_spikes_df.copy(), debug_print=False)
            prev_output_result.computed_data['pf1D_Decoder'].compute_all() #

            prev_output_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, prev_output_result.computed_data['pf2D'], prev_output_result.computed_data['pf2D'].filtered_spikes_df.copy(), debug_print=False)
            prev_output_result.computed_data['pf2D_Decoder'].compute_all() #
            return prev_output_result

        placefield_computation_config = computation_result.computation_config.pf_params # should be a PlacefieldComputationParameters
        return position_decoding_computation(computation_result.sess, placefield_computation_config, computation_result)
    
    def _perform_two_step_position_decoding_computation(computation_result: ComputationResult, debug_print=False, **kwargs):
        """ Builds the Zhang Velocity/Position For 2-step Bayesian Decoder for 2D Placefields
        """

        def _subfn_compute_two_step_decoder(active_xbins, active_ybins, prev_one_step_bayesian_decoder, pos_df, computation_config, debug_print=False):
            """ captures debug_print 

            pos_df = computation_result.sess.position.to_dataframe()
            computation_config = computation_result.computation_config
            """
            
            avg_speed_per_pos = _compute_avg_speed_at_each_position_bin(pos_df, computation_config, prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers, debug_print=debug_print)
                    
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
            two_step_decoder_result = DynamicParameters.init_from_dict({'xbin':active_xbins, 'ybin':active_ybins,
                                                                    'avg_speed_per_pos': avg_speed_per_pos,
                                                                    'K':K, 'V':V,
                                                                    'sigma_t_all':sigma_t_all, 'flat_sigma_t_all': np.squeeze(np.reshape(sigma_t_all, (-1, 1)))
            })
            
            two_step_decoder_result['C'] = 1.0
            two_step_decoder_result['k'] = 1.0

            if (prev_one_step_bayesian_decoder.ndim < 2):
                assert prev_one_step_bayesian_decoder.ndim == 1, f"prev_one_step_bayesian_decoder.ndim must be either 1 or 2, but prev_one_step_bayesian_decoder.ndim: {prev_one_step_bayesian_decoder.ndim}"
                print(f'prev_one_step_bayesian_decoder.ndim == 1, so using [0.0] as active_ybin_centers.')
                active_ybin_centers = np.array([0.0])

            else:
                # 2D Case:
                assert prev_one_step_bayesian_decoder.ndim == 2, f"prev_one_step_bayesian_decoder.ndim must be either 1 or 2, but prev_one_step_bayesian_decoder.ndim: {prev_one_step_bayesian_decoder.ndim}"
                active_ybin_centers = prev_one_step_bayesian_decoder.ybin_centers

            # pre-allocate outputs:
            two_step_decoder_result['all_x'], two_step_decoder_result['flat_all_x'], original_data_shape = Zhang_Two_Step.build_all_positions_matrix(prev_one_step_bayesian_decoder.xbin_centers, active_ybin_centers) # all_x: (64, 29, 2), flat_all_x: (1856, 2)
            two_step_decoder_result['original_all_x_shape'] = original_data_shape # add the original data shape to the computed data
    
            # Pre-allocate output:
            two_step_decoder_result['flat_p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.flat_p_x_given_n, 0.0) # fill with NaNs. 
            two_step_decoder_result['p_x_given_n_and_x_prev'] = np.full_like(prev_one_step_bayesian_decoder.p_x_given_n, 0.0) # fill with NaNs. Pre-allocate output
            two_step_decoder_result['most_likely_position_indicies'] = np.zeros((2, prev_one_step_bayesian_decoder.num_time_windows), dtype=int) # (2, 85841)
            two_step_decoder_result['most_likely_positions'] = np.zeros((2, prev_one_step_bayesian_decoder.num_time_windows)) # (2, 85841)
                    
            if debug_print:
                print(f'np.shape(prev_one_step_bayesian_decoder.p_x_given_n): {np.shape(prev_one_step_bayesian_decoder.p_x_given_n)}')
            
            two_step_decoder_result['all_scaling_factors_k'] = Zhang_Two_Step.compute_scaling_factor_k(prev_one_step_bayesian_decoder.flat_p_x_given_n)
            
            # TODO: Efficiency: This will be inefficient, but do a slow iteration. 
            for time_window_bin_idx in np.arange(prev_one_step_bayesian_decoder.num_time_windows):
                flat_p_x_given_n = prev_one_step_bayesian_decoder.flat_p_x_given_n[:, time_window_bin_idx] # this gets the specific n_t for this time window                
                # previous positions as determined by the two-step decoder: this uses the two_step previous position instead of the one_step previous position:
                prev_x_position = two_step_decoder_result['most_likely_positions'][:, time_window_bin_idx-1] # TODO: is this okay for 1D as well?
                active_k = two_step_decoder_result['all_scaling_factors_k'][time_window_bin_idx] # get the specific k value
                # active_k = two_step_decoder_result['k']
                if debug_print:
                    print(f'np.shape(prev_x_position): {np.shape(prev_x_position)}')
                            
                # Flat version:
                two_step_decoder_result['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx] = Zhang_Two_Step.compute_bayesian_two_step_prob_single_timestep(flat_p_x_given_n, prev_x_position, two_step_decoder_result['flat_all_x'], two_step_decoder_result['flat_sigma_t_all'], two_step_decoder_result['C'], active_k) # output shape (1856, )            
                

                if (prev_one_step_bayesian_decoder.ndim < 2):
                    # 1D case:
                    two_step_decoder_result['p_x_given_n_and_x_prev'][:,time_window_bin_idx] = two_step_decoder_result['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx] # used to be (original_data_shape[0], original_data_shape[1])
                else:
                    # 2D:
                    two_step_decoder_result['p_x_given_n_and_x_prev'][:,:,time_window_bin_idx] = np.reshape(two_step_decoder_result['flat_p_x_given_n_and_x_prev'][:,time_window_bin_idx], (original_data_shape[0], original_data_shape[1]))
                

            # POST-hoc most-likely computations: Compute the most-likely positions from the p_x_given_n_and_x_prev:
            # # np.shape(self.most_likely_position_indicies) # (2, 85841)
            """ Computes the most likely positions at each timestep from flat_p_x_given_n_and_x_prev """
            two_step_decoder_result['most_likely_position_flat_indicies'] = np.argmax(two_step_decoder_result['flat_p_x_given_n_and_x_prev'], axis=0)
            
            # Adds `most_likely_position_flat_max_likelihood_values`:
            # Same as np.amax(x, axis=axis_idx)
            # axis_idx = 0
            # x = two_step_decoder_result['flat_p_x_given_n_and_x_prev']
            # index_array = two_step_decoder_result['most_likely_position_flat_indicies']
            # two_step_decoder_result['most_likely_position_flat_max_likelihood_values'] = np.take_along_axis(x, np.expand_dims(index_array, axis=axis_idx), axis=axis_idx).squeeze(axis=axis_idx)
            two_step_decoder_result['most_likely_position_flat_max_likelihood_values'] = np.take_along_axis(two_step_decoder_result['flat_p_x_given_n_and_x_prev'], np.expand_dims(two_step_decoder_result['most_likely_position_flat_indicies'], axis=0), axis=0).squeeze(axis=0) # get the flat maximum values
            print(f"{two_step_decoder_result['most_likely_position_flat_max_likelihood_values'].shape = }")
            # Reshape the maximum values:
            # two_step_decoder_result['most_likely_position_max_likelihood_values'] = np.array(np.unravel_index(two_step_decoder_result['most_likely_position_flat_max_likelihood_values'], prev_one_step_bayesian_decoder.original_position_data_shape)) # convert back to an array

            # np.shape(self.most_likely_position_flat_indicies) # (85841,)
            two_step_decoder_result['most_likely_position_indicies'] = np.array(np.unravel_index(two_step_decoder_result['most_likely_position_flat_indicies'], prev_one_step_bayesian_decoder.original_position_data_shape)) # convert back to an array
            # np.shape(self.most_likely_position_indicies) # (2, 85841)
            two_step_decoder_result['most_likely_positions'][0, :] = prev_one_step_bayesian_decoder.xbin_centers[two_step_decoder_result['most_likely_position_indicies'][0, :]]

            if prev_one_step_bayesian_decoder.ndim > 1:
                two_step_decoder_result['most_likely_positions'][1, :] = prev_one_step_bayesian_decoder.ybin_centers[two_step_decoder_result['most_likely_position_indicies'][1, :]]
            # two_step_decoder_result['sigma_t_all'] = sigma_t_all # set sigma_t_all                

            ## For some reason we set up the two-step decoder's most_likely_positions with the tranposed shape compared to the one-step decoder:
            two_step_decoder_result['most_likely_positions'] = two_step_decoder_result['most_likely_positions'].T
            
            ## Once done, compute marginals for the two-step:
            curr_unit_marginal_x, curr_unit_marginal_y = prev_one_step_bayesian_decoder.perform_build_marginals(two_step_decoder_result['p_x_given_n_and_x_prev'], two_step_decoder_result['most_likely_positions'], debug_print=debug_print)
            two_step_decoder_result['marginal'] = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
            
            return two_step_decoder_result


        ndim = kwargs.get('ndim', 2)
        if ndim is None:
            ndim = 2 # add the 2D version if no alterantive is passed in.
        if ndim == 1:
            one_step_decoder_key = 'pf1D_Decoder'
            two_step_decoder_key = 'pf1D_TwoStepDecoder'

        elif ndim == 2:
            one_step_decoder_key = 'pf2D_Decoder'
            two_step_decoder_key = 'pf2D_TwoStepDecoder'
        else:
            raise NotImplementedError # dimensionality must be 1 or 2

        # Get the one-step decoder:
        prev_one_step_bayesian_decoder = computation_result.computed_data[one_step_decoder_key]
        ## New 2022-09-15 direct neuropy.utils.mixins.binning_helpers.build_df_discretized_binned_position_columns version:
        computation_result.sess.position.df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(computation_result.sess.position.df, bin_values=(prev_one_step_bayesian_decoder.xbin_centers, prev_one_step_bayesian_decoder.ybin_centers), active_computation_config=computation_result.computation_config.pf_params, force_recompute=False, debug_print=debug_print)
        active_xbins = xbin
        active_ybins = ybin      

        computation_result.computed_data[two_step_decoder_key] = _subfn_compute_two_step_decoder(active_xbins, active_ybins, prev_one_step_bayesian_decoder, computation_result.sess.position.df, computation_config=computation_result.computation_config, debug_print=debug_print)
        ## In this new mode we'll add the two-step properties to the original one-step decoder:
        ## Adds the directly accessible properties to the active_one_step_decoder after they're computed in the active_two_step_decoder so that they can be plotted with the same functions/etc.
        prev_one_step_bayesian_decoder.add_two_step_decoder_results(computation_result.computed_data[two_step_decoder_key])

        return computation_result

    def _perform_recursive_latent_placefield_decoding(computation_result: ComputationResult, **kwargs):
        """ note that currently the pf1D_Decoders are not built or used. 

        """
        ### For _perform_recursive_latent_placefield_decoding
        from neuropy.utils import position_util
        from neuropy.core import Position
        from neuropy.analyses.placefields import perform_compute_placefields

        def _subfn_build_recurrsive_placefields(active_one_step_decoder, next_order_computation_config, spikes_df=None, pos_df=None, pos_linearization_method='isomap'):
            """ This subfunction is called to produce a specific depth level of placefields. 
            """
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
                # TODO: figure it out instead of hacking -- Currently just dropping the last time bin because something is off 
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

            def _next_order_decode(active_pf_1D, active_pf_2D, pf_computation_config):
                """ performs the actual decoding given the"""
                ## 1D Decoder
                new_decoder_pf1D = active_pf_1D
                new_1D_decoder_spikes_df = new_decoder_pf1D.filtered_spikes_df.copy()
                new_1D_decoder = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, new_decoder_pf1D, new_1D_decoder_spikes_df, debug_print=False) 
                new_1D_decoder.compute_all() #  --> TODO: NOTE: 1D .compute_all() has just been recently added due to a previous error in ffill

                ## Custom Manual 2D Decoder:
                new_decoder_pf2D = active_pf_2D # 
                new_decoder_spikes_df = new_decoder_pf2D.filtered_spikes_df.copy()
                new_2D_decoder = BayesianPlacemapPositionDecoder(pf_computation_config.time_bin_size, new_decoder_pf2D, new_decoder_spikes_df, debug_print=False)
                new_2D_decoder.compute_all() #  --> n = self.
                
                return new_1D_decoder, new_2D_decoder

            active_second_order_spikes_df = _prepare_spikes_df_for_recurrsive_decoding(active_one_step_decoder, spikes_df)
            active_second_order_pos_df = _prepare_pos_df_for_recurrsive_decoding(active_one_step_decoder, pos_df)
            active_second_order_pf_1D, active_second_order_pf_2D = perform_compute_placefields(active_second_order_spikes_df, Position(active_second_order_pos_df),
                                                                                            next_order_computation_config.pf_params, None, None, included_epochs=None, should_force_recompute_placefields=True)
            # build the second_order decoders:
            active_second_order_1D_decoder, active_second_order_2D_decoder = _next_order_decode(active_second_order_pf_1D, active_second_order_pf_2D, next_order_computation_config.pf_params) 
            
            return active_second_order_pf_1D, active_second_order_pf_2D, active_second_order_1D_decoder, active_second_order_2D_decoder

        pos_linearization_method='isomap'
        prev_one_step_bayesian_decoder = computation_result.computed_data['pf2D_Decoder']

        ## Builds a duplicate of the current computation config but sets the speed_thresh to 0.0:
        next_order_computation_config = deepcopy(computation_result.computation_config) # make a deepcopy of the active computation config
        next_order_computation_config.pf_params.speed_thresh = 0.0 # no speed thresholding because the speeds aren't real for the second-order fields

        enable_pf1D = True # Enabled on 2022-12-15 afte fixing bug in implementaion of ffill
        enable_pf2D = True

        # Start with empty lists, which will accumulate the different levels of recurrsive depth:
        computation_result.computed_data['pf1D_RecursiveLatent'] = []
        computation_result.computed_data['pf2D_RecursiveLatent'] = []

        # 1st Order:
        if enable_pf1D:
            computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':computation_result.computed_data['pf1D'], 'pf1D_Decoder':computation_result.computed_data.get('pf1D_Decoder', {})}))
        if enable_pf2D:
            computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':computation_result.computed_data['pf2D'], 'pf2D_Decoder':computation_result.computed_data['pf2D_Decoder']}))

        # 2nd Order:
        active_second_order_pf_1D, active_second_order_pf_2D, active_second_order_1D_decoder, active_second_order_2D_decoder = _subfn_build_recurrsive_placefields(prev_one_step_bayesian_decoder, next_order_computation_config=next_order_computation_config, spikes_df=prev_one_step_bayesian_decoder.spikes_df, pos_df=prev_one_step_bayesian_decoder.pf.filtered_pos_df, pos_linearization_method=pos_linearization_method)
        if enable_pf1D:
            computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':active_second_order_pf_1D, 'pf1D_Decoder':active_second_order_1D_decoder}))
        if enable_pf2D:
            computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':active_second_order_pf_2D, 'pf2D_Decoder':active_second_order_2D_decoder}))

        # # 3rd Order:
        active_third_order_pf_1D, active_third_order_pf_2D, active_third_order_1D_decoder, active_third_order_2D_decoder = _subfn_build_recurrsive_placefields(active_second_order_2D_decoder, next_order_computation_config=next_order_computation_config, spikes_df=active_second_order_2D_decoder.spikes_df, pos_df=active_second_order_2D_decoder.pf.filtered_pos_df, pos_linearization_method=pos_linearization_method)
        if enable_pf1D:
            computation_result.computed_data['pf1D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf1D':active_third_order_pf_1D, 'pf1D_Decoder':active_third_order_1D_decoder}))
        if enable_pf2D:
            computation_result.computed_data['pf2D_RecursiveLatent'].append(DynamicContainer.init_from_dict({'pf2D':active_third_order_pf_2D, 'pf2D_Decoder':active_third_order_2D_decoder}))

        return computation_result

    def _perform_specific_epochs_decoding(computation_result: ComputationResult, active_config, decoder_ndim:int=2, filter_epochs='ripple', decoding_time_bin_size=0.02, **kwargs):
        """ TODO: meant to be used by `_display_plot_decoded_epoch_slices` but needs a smarter way to cache the computations and etc. 
        Eventually to replace `pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError._compute_specific_decoded_epochs`

        Usage:
            ## Test _perform_specific_epochs_decoding
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import DefaultComputationFunctions
            computation_result = curr_active_pipeline.computation_results['maze1_PYR']
            computation_result = DefaultComputationFunctions._perform_specific_epochs_decoding(computation_result, curr_active_pipeline.active_configs['maze1_PYR'], filter_epochs='ripple', decoding_time_bin_size=0.02)
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = computation_result.computed_data['specific_epochs_decoding'][('Ripples', 0.02)]

        """
        ## BEGIN_FUNCTION_BODY _perform_specific_epochs_decoding:
        ## Check for previous computations:
        needs_compute = True # default to needing to recompute.
        computation_tuple_key = (filter_epochs, decoding_time_bin_size, decoder_ndim) # used to be (default_figure_name, decoding_time_bin_size) only

        curr_result = computation_result.computed_data.get('specific_epochs_decoding', {})
        found_result = curr_result.get(computation_tuple_key, None)
        if found_result is not None:
            # Unwrap and reuse the result:
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = found_result # computation_result.computed_data['specific_epochs_decoding'][('Laps', decoding_time_bin_size)]
            needs_compute = False # we don't need to recompute

        if needs_compute:
            ## Do the computation:
            filter_epochs_decoder_result, active_filter_epochs, default_figure_name = _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, decoder_ndim=decoder_ndim)

            ## Cache the computation result via the tuple key: (default_figure_name, decoding_time_bin_size) e.g. ('Laps', 0.02) or ('Ripples', 0.02)
            curr_result[computation_tuple_key] = (filter_epochs_decoder_result, active_filter_epochs, default_figure_name)

        computation_result.computed_data['specific_epochs_decoding'] = curr_result
        return computation_result




# ==================================================================================================================== #
# Private Methods                                                                                                      #
# ==================================================================================================================== #

from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum

class KnownFilterEpochs(ExtendedEnum):
    """Describes the type of file progress actions that can be performed to get the right verbage.
    Used by `_subfn_compute_decoded_epochs(...)`
   
    """
    LAP = "lap"
    PBE = "pbe"
    RIPPLE = "ripple"
    REPLAY = "replay"
    GENERIC = "GENERIC"


    # BEGIN PROBLEMATIC ENUM CODE ________________________________________________________________________________________ #
    @property
    def default_figure_name(self):
        return KnownFilterEpochs.default_figure_nameList()[self]

    # Static properties
    @classmethod
    def default_figure_nameList(cls):
        return cls.build_member_value_dict([f'Laps',f'PBEs',f'Ripples',f'Replays',f'Generic'])



    @classmethod
    def _perform_get_filter_epochs_df(cls, sess, filter_epochs, min_epoch_included_duration=None, debug_print=False):
        """DOES NOT WORK due to messed-up `.epochs.get_non_overlapping_df`

        Args:
            sess (_type_): computation_result.sess
            filter_epochs (_type_): _description_
            min_epoch_included_duration: only applies to Replay for some reason?

        Raises:
            NotImplementedError: _description_
        """
        # post_process_epochs_fn = lambda filter_epochs_df: filter_epochs_df.epochs.get_non_overlapping_df(debug_print=debug_print) # post_process_epochs_fn should accept an epochs dataframe and return a clean copy of the epochs dataframe
        post_process_epochs_fn = lambda filter_epochs_df: filter_epochs_df.epochs.get_valid_df() # post_process_epochs_fn should accept an epochs dataframe and return a clean copy of the epochs dataframe

        if debug_print:
            print(f'')
        if isinstance(filter_epochs, str):
            try:
                filter_epochs = cls.init(value=filter_epochs) # init an enum object from the string
            except Exception as e:
                print(f'filter_epochs "{filter_epochs}" could not be parsed into KnownFilterEpochs but is string.')
                raise e

        if isinstance(filter_epochs, cls):
            if filter_epochs.name == cls.LAP.name:
                ## Lap-Epochs Decoding:
                laps_copy = deepcopy(sess.laps)
                active_filter_epochs = laps_copy.as_epoch_obj() # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)

            elif filter_epochs.name == cls.PBE.name:
                ## PBEs-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.pbe) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)
            
            elif filter_epochs.name == cls.RIPPLE.name:
                ## Ripple-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.ripple) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                # note we need to make sure we have a valid label to start because `.epochs.get_non_overlapping_df()` requires one.
                active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)
                active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing
                
            elif filter_epochs.name == cls.REPLAY.name:
                active_filter_epochs = deepcopy(sess.replay) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)
                if min_epoch_included_duration is not None:
                    active_filter_epochs = active_filter_epochs[active_filter_epochs.duration >= min_epoch_included_duration] # only include those epochs which are greater than or equal to two decoding time bins
            else:
                print(f'filter_epochs "{filter_epochs.name}" could not be parsed into KnownFilterEpochs but is string.')
                active_filter_epochs = None
                raise NotImplementedError

        else:
            # Use it filter_epochs raw, hope it's right. It should be some type of Epoch or pd.DataFrame object.
            active_filter_epochs = filter_epochs
            ## TODO: why even allow passing in a raw Epoch object? It's not clear what the use case is.
            raise NotImplementedError


        # Finally, convert back to Epoch object:
        assert isinstance(active_filter_epochs, pd.DataFrame)
        # active_filter_epochs = Epoch(active_filter_epochs)
        if debug_print:
            print(f'active_filter_epochs: {active_filter_epochs}')
        return active_filter_epochs


    @classmethod
    def perform_get_filter_epochs_df(cls, sess, filter_epochs, min_epoch_included_duration=None, **kwargs):
        """Temporary wrapper for `process_functionList` to replace `_perform_get_filter_epochs_df`

        Args:
            sess (_type_): computation_result.sess
            filter_epochs (_type_): _description_
            min_epoch_included_duration: only applies to Replay for some reason?

        """
        # `process_functionList` version:
        return cls.process_functionList(sess=sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration, default_figure_name='', **kwargs)[0] # [0] gets the returned active_filter_epochs
        # proper `_perform_get_filter_epochs_df` version:
        # return cls._perform_get_filter_epochs_df(sess=computation_result.sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration)

    @classmethod
    def process_functionList(cls, sess, filter_epochs, min_epoch_included_duration, default_figure_name='stacked_epoch_slices_matplotlib_subplots'):
        # min_epoch_included_duration = decoding_time_bin_size * float(2) # 0.06666
        # min_epoch_included_duration = 0.06666

        if isinstance(filter_epochs, str):
            print(f'filter_epochs string: "{filter_epochs}"')
            filter_epochs = cls.init(value=filter_epochs) # init an enum object from the string
            # filter_epochs = cls.init(filter_epochs, fallback_value=cls.GENERIC) # init an enum object from the string
            default_figure_name = filter_epochs.default_figure_name

            if filter_epochs.name == KnownFilterEpochs.LAP.name:
                ## Lap-Epochs Decoding:
                laps_copy = deepcopy(sess.laps)
                active_filter_epochs = laps_copy.as_epoch_obj() # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                # pre_exclude_n_epochs = active_filter_epochs.n_epochs
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                # post_exclude_n_epochs = active_filter_epochs.n_epochs                    
                # num_excluded_epochs = post_exclude_n_epochs - pre_exclude_n_epochs
                # if num_excluded_epochs > 0:
                #     print(f'num_excluded_epochs: {num_excluded_epochs} due to overlap.')
                # ## Build Epochs:
                epoch_description_list = [f'lap[{epoch_tuple.lap_id}]' for epoch_tuple in active_filter_epochs[['lap_id']].itertuples()] # Short

            elif filter_epochs.name == KnownFilterEpochs.PBE.name:
                ## PBEs-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.pbe) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
            
            elif filter_epochs.name == KnownFilterEpochs.RIPPLE.name:
                ## Ripple-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.ripple) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing
                epoch_description_list = [f'ripple[{epoch_tuple.label}]' for epoch_tuple in active_filter_epochs[['label']].itertuples()] # SHORT
                
            elif filter_epochs.name == KnownFilterEpochs.REPLAY.name:
                active_filter_epochs = deepcopy(sess.replay) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                if min_epoch_included_duration is not None:
                    active_filter_epochs = active_filter_epochs[active_filter_epochs.duration >= min_epoch_included_duration] # only include those epochs which are greater than or equal to two decoding time bins
                epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs[['label']].itertuples()]
                
            else:
                print(f'filter_epochs "{filter_epochs.name}" could not be parsed into KnownFilterEpochs but is string.')
                raise NotImplementedError

            # Finally, convert back to Epoch object:
            assert isinstance(active_filter_epochs, pd.DataFrame)
            # active_filter_epochs = Epoch(active_filter_epochs)

        else:
            # Use it raw, hope it's right
            active_filter_epochs = filter_epochs
            default_figure_name = f'{default_figure_name}_CUSTOM'
            epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs.to_dataframe()[['label']].itertuples()]

        return active_filter_epochs, default_figure_name, epoch_description_list

def _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs='ripple', decoding_time_bin_size=0.02, decoder_ndim:int=2):
    """ compuites a plot with the 1D Marginals either (x and y position axes): the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. 
    
    It determines which epochs are being referred to (enabling specifying them by a simple string identifier, like 'ripple', 'pbe', or 'laps') and then gets the coresponding data that's needed to recompute the decoded data for them.
    This decoding is done by calling:
        active_decoder.decode_specific_epochs(...) which returns a result that can then be plotted.

    Used by: _perform_specific_epochs_decoding

    # TODO: 2022-12-20: Need to convert to work with 1D
    
    """    
    default_figure_name = 'stacked_epoch_slices_matplotlib_subplots'
    min_epoch_included_duration = decoding_time_bin_size * float(2) # 0.06666 # all epochs shorter than min_epoch_included_duration will be excluded from analysis
    active_filter_epochs, default_figure_name, epoch_description_list = KnownFilterEpochs.process_functionList(sess=computation_result, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration, default_figure_name=default_figure_name)

    ## BEGIN_FUNCTION_BODY _subfn_compute_decoded_epochs:
    if decoder_ndim is None:
        decoder_ndim = 2 # add the 2D version if no alterantive is passed in.
    if decoder_ndim == 1:
        one_step_decoder_key = 'pf1D_Decoder'
        # two_step_decoder_key = 'pf1D_TwoStepDecoder'

    elif decoder_ndim == 2:
        one_step_decoder_key = 'pf2D_Decoder'
        # two_step_decoder_key = 'pf2D_TwoStepDecoder'
    else:
        raise NotImplementedError # dimensionality must be 1 or 2
   
    active_decoder = computation_result.computed_data[one_step_decoder_key]
    filter_epochs_decoder_result = active_decoder.decode_specific_epochs(computation_result.sess.spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
    filter_epochs_decoder_result.epoch_description_list = epoch_description_list
    return filter_epochs_decoder_result, active_filter_epochs, default_figure_name


# ==================================================================================================================== #
# 2023-03-15 Surprise/Leave-One-Out Analyses                                                                           #
# ==================================================================================================================== #

# def _subfn_reshape_for_each_epoch_to_for_each_cell(data, epoch_IDXs, neuron_IDs):
#     """ UNUSED: Reshape to -for-each-epoch instead of -for-each-cell
#         from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _subfn_reshape_for_each_epoch_to_for_each_cell
#         flat_all_epochs_cell_data, all_epochs_cell_data = _subfn_reshape_for_each_epoch_to_for_each_cell(data, epoch_IDXs=np.arange(active_filter_epochs.n_epochs), neuron_IDs=original_1D_decoder.neuron_IDs)
#     """
#     all_epochs_cell_data = []
#     for decoded_epoch_idx in epoch_IDXs:
#         all_epochs_cell_data.append(np.array([data[aclu][decoded_epoch_idx] for aclu in neuron_IDs]))
#     flat_all_epochs_cell_data = np.hstack(all_epochs_cell_data) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
#     return flat_all_epochs_cell_data, all_epochs_cell_data

def _analyze_leave_one_out_decoding_results(active_pos_df, active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict):
    """ 2023-03-15 - Kamran's Leave-One-Out-Surprise - Main leave-one-out surprise computation:

        [9:28 AM] Diba, Kamran
        then for bins when cell i fires, you calculate the surprise between cell i’s place field and the posterior probability (which is calculated by excluding cell i)

        #### Minimize binning artifacts by smoothing replay spikes:
        [10:26 AM] Diba, Kamran
        I think van der Meer (Hippocampus) did a 25ms Gaussian convolution

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _subfn_compute_leave_one_out_analysis
        original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict = perform_leave_one_aclu_out_decoding_analysis(pyramidal_only_spikes_df, active_pos_df, active_filter_epochs)
        flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus = _subfn_compute_leave_one_out_analysis(active_pos_df, active_filter_epochs, original_1D_decoder, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict)


    """
    all_cells_decoded_epoch_time_bins = {}
    all_cells_computed_epoch_surprises = {}
    all_cells_computed_epoch_one_left_out_to_global_surprises = {}


    # Secondary computations
    all_cells_decoded_expected_firing_rates = {}
    ## Compute the impact leaving each aclu out had on the average encoding performance:
    one_left_out_omitted_aclu_distance = {}

    num_cells = original_1D_decoder.num_neurons

    ## for each cell:
    for i, left_out_aclu in enumerate(original_1D_decoder.neuron_IDs):
        # aclu = original_1D_decoder.neuron_IDs[i]
        left_out_neuron_IDX = original_1D_decoder.neuron_IDXs[i] # should just be i, but just to be safe
        ## TODO: only look at bins where the cell fires (is_cell_firing_time_bin[i])
        curr_cell_tuning_curve = original_1D_decoder.pf.ratemap.tuning_curves[left_out_neuron_IDX]
        # curr_cell_spike_curve = original_1D_decoder.pf.ratemap.spikes_maps[unit_IDX] ## not occupancy weighted... is this the right one to use for computing the expected spike rate? NO... doesn't seem like it

        left_out_decoder_result = one_left_out_filter_epochs_decoder_result_dict[left_out_aclu]
        ## single cell outputs:
        curr_cell_decoded_epoch_time_bins = [] # will be a list of the time bins in each epoch that correspond to each surprise in the corresponding list in curr_cell_computed_epoch_surprises 
        curr_cell_computed_epoch_surprises = [] # will be a list of np.arrays, with each array representing the surprise of each time bin in each epoch
        all_cells_decoded_expected_firing_rates[left_out_aclu] = [] 
        all_cells_computed_epoch_one_left_out_to_global_surprises[left_out_aclu] = []


        # have one list of posteriors p_x_given_n for each decoded epoch (active_filter_epochs.n_epochs):
        assert len(left_out_decoder_result.p_x_given_n_list) == active_filter_epochs.n_epochs == left_out_decoder_result.num_filter_epochs

        ## Compute the impact leaving each aclu out had on the average encoding performance:
        ### 1. The distance between the actual measured position and the decoded position at each timepoint for each decoder. A larger magnitude difference implies a stronger, more positive effect on the decoding quality.
        one_left_out_omitted_aclu_distance[left_out_aclu] = [] # list to hold the distance results from the epochs
        ## Iterate through each of the epochs for the given left_out_aclu (and its decoder), each of which has its own result

        for decoded_epoch_idx in np.arange(left_out_decoder_result.num_filter_epochs):
            curr_epoch_time_bin_container = left_out_decoder_result.time_bin_containers[decoded_epoch_idx]
            curr_time_bins = curr_epoch_time_bin_container.centers
            curr_epoch_p_x_given_n = left_out_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_cell_tuning_curve.shape[0]
            
            ## Get the all-included values too for this decoded_epoch_idx:
            curr_epoch_all_included_p_x_given_n = all_included_filter_epochs_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_epoch_all_included_p_x_given_n.shape[0]

            ## Need to exclude estimates from bins that didn't have any spikes in them (in general these glitch around):
            curr_total_spike_counts_per_window = np.sum(left_out_decoder_result.spkcount[decoded_epoch_idx], axis=0) # left_out_decoder_result.spkcount[i].shape # (69, 222) - (nCells, nTimeWindowCenters)
            curr_is_time_bin_non_firing = (curr_total_spike_counts_per_window == 0) # this would mean that no cells fired in this time bin
            # curr_non_firing_time_bin_indicies = np.where(curr_is_time_bin_non_firing)[0] # TODO: could also filter on a minimum number of spikes larger than zero (e.g. at least 2 spikes are required).
            curr_posterior_container = left_out_decoder_result.marginal_x_list[decoded_epoch_idx]
            curr_posterior = curr_posterior_container.p_x_given_n # TODO: check the posteriors too!
            curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D

            ## Compute the distance metric for this epoch:

            # Interpolate the measured positions to the window center times:
            window_center_measured_pos_x = np.interp(curr_time_bins, active_pos_df.t, active_pos_df.lin_pos)
            # ## PLOT_ONLY: NaN out the most_likely_positions that don't have spikes.
            # curr_most_likely_valid_positions = deepcopy(curr_most_likely_positions)
            # curr_most_likely_valid_positions[curr_non_firing_time_bin_indicies] = np.nan
            
            ## Computed the distance metric finally:
            # is it fair to only compare the valid (windows containing at least one spike) windows?
            curr_omit_aclu_distance = distance.cdist(np.atleast_2d(window_center_measured_pos_x[~curr_is_time_bin_non_firing]), np.atleast_2d(curr_most_likely_positions[~curr_is_time_bin_non_firing]), 'sqeuclidean') # squared-euclidian distance between the two vectors
            # curr_omit_aclu_distance comes back double-wrapped in np.arrays for some reason (array([[659865.11994352]])), so .item() extracts the scalar value
            curr_omit_aclu_distance = curr_omit_aclu_distance.item()
            one_left_out_omitted_aclu_distance[left_out_aclu].append(curr_omit_aclu_distance)

            # Compute the expected firing rate for this cell during each bin by taking the computed position posterior and taking the sum of the element-wise product with the cell's placefield.
            # curr_epoch_expected_fr = np.array([np.sum(curr_cell_spike_curve * curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) / original_1D_decoder.time_bin_size
            curr_epoch_expected_fr = original_1D_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates[left_out_neuron_IDX] * np.array([np.sum(curr_cell_tuning_curve * curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) # * original_1D_decoder.pf.ratemap.
            all_cells_decoded_expected_firing_rates[left_out_aclu].append(curr_epoch_expected_fr)
            
            # Compute the Jensen-Shannon Distance as a measure of surprise between the placefield and the posteriors
            curr_epoch_surprises = np.array([distance.jensenshannon(curr_cell_tuning_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) # works! Finite! [0.5839003679903784, 0.5839003679903784, 0.6997779781969289, 0.7725622595699131, 0.5992295785891731]
            curr_cell_computed_epoch_surprises.append(curr_epoch_surprises)
            curr_cell_decoded_epoch_time_bins.append(curr_epoch_time_bin_container)
            # Compute the Jensen-Shannon Distance as a measure of surprise between the all-included and the one-left-out posteriors:
            all_cells_computed_epoch_one_left_out_to_global_surprises[left_out_aclu].append(np.array([distance.jensenshannon(curr_all_included_p_x_given_n, curr_p_x_given_n) for curr_all_included_p_x_given_n, curr_p_x_given_n in zip(curr_epoch_all_included_p_x_given_n.T, curr_epoch_p_x_given_n.T)])) 

        ## End loop over decoded epochs
        assert len(curr_cell_decoded_epoch_time_bins) == len(curr_cell_computed_epoch_surprises)
        all_cells_decoded_epoch_time_bins[left_out_aclu] = curr_cell_decoded_epoch_time_bins
        all_cells_computed_epoch_surprises[left_out_aclu] = curr_cell_computed_epoch_surprises


    ## End loop over cells
    # build a dataframe version to hold the distances:
    one_left_out_omitted_aclu_distance_df = pd.DataFrame({'omitted_aclu':np.array(list(one_left_out_omitted_aclu_distance.keys())),
                                                        'distance': list(one_left_out_omitted_aclu_distance.values()),
                                                        'avg_dist': [np.mean(v) for v in one_left_out_omitted_aclu_distance.values()]}
                                                        )
    one_left_out_omitted_aclu_distance_df.sort_values(by='avg_dist', ascending=False, inplace=True) # this sort reveals the aclu values that when omitted had the largest performance decrease on decoding (as indicated by a larger distance)
    most_contributing_aclus = one_left_out_omitted_aclu_distance_df.omitted_aclu.values

    ## Reshape to -for-each-epoch instead of -for-each-cell
    all_epochs_decoded_epoch_time_bins = []
    all_epochs_computed_surprises = []
    all_epochs_computed_expected_cell_firing_rates = []
    all_epochs_computed_one_left_out_to_global_surprises = []
    for decoded_epoch_idx in np.arange(active_filter_epochs.n_epochs):
        all_epochs_decoded_epoch_time_bins.append(np.array([all_cells_decoded_epoch_time_bins[aclu][decoded_epoch_idx].centers for aclu in original_1D_decoder.neuron_IDs])) # these are duplicated (and the same) for each cell
        all_epochs_computed_surprises.append(np.array([all_cells_computed_epoch_surprises[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))
        all_epochs_computed_expected_cell_firing_rates.append(np.array([all_cells_decoded_expected_firing_rates[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))
        all_epochs_computed_one_left_out_to_global_surprises.append(np.array([all_cells_computed_epoch_one_left_out_to_global_surprises[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))

    assert len(all_epochs_computed_surprises) == active_filter_epochs.n_epochs
    assert len(all_epochs_computed_surprises[0]) == original_1D_decoder.num_neurons
    flat_all_epochs_decoded_epoch_time_bins = np.hstack(all_epochs_decoded_epoch_time_bins) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_surprises = np.hstack(all_epochs_computed_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_expected_cell_firing_rates = np.hstack(all_epochs_computed_expected_cell_firing_rates) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_one_left_out_to_global_surprises = np.hstack(all_epochs_computed_one_left_out_to_global_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs


    ## Could also do but would need to loop over all epochs for each of the three variables:
    # flat_all_epochs_computed_expected_cell_firing_rates, all_epochs_computed_expected_cell_firing_rates = _subfn_reshape_for_each_epoch_to_for_each_cell(all_cells_decoded_expected_firing_rates, epoch_IDXs=np.arange(active_filter_epochs.n_epochs), neuron_IDs=original_1D_decoder.neuron_IDs)

    ## Aggregates over all time bins in each epoch:
    all_epochs_decoded_epoch_time_bins_mean = np.vstack([np.mean(curr_epoch_time_bins, axis=1) for curr_epoch_time_bins in all_epochs_decoded_epoch_time_bins]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_cell_surprises_mean = np.vstack([np.mean(curr_epoch_surprises, axis=1) for curr_epoch_surprises in all_epochs_computed_surprises]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_cell_one_left_out_to_global_surprises_mean = np.vstack([np.mean(curr_epoch_surprises, axis=1) for curr_epoch_surprises in all_epochs_computed_one_left_out_to_global_surprises]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)

    ## Aggregates over all cells and all time bins in each epoch:
    all_epochs_all_cells_computed_surprises_mean = np.mean(all_epochs_computed_cell_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)
    all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean = np.mean(all_epochs_computed_cell_one_left_out_to_global_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)

    """ Returns:
        one_left_out_omitted_aclu_distance_df: a dataframe of the distance metric for each of the decoders in one_left_out_decoder_dict. The index is the aclu that was omitted from the decoder.
        most_contributing_aclus: a list of aclu values, sorted by the largest performance decrease on decoding (as indicated by a larger distance)
    """
    ## Output variables: flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean
    return flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus


from attrs import define, field, Factory


@define
class LeaveOneOutDecodingResult(object):
    """Docstring for ClassName."""
    one_left_out_to_global_surprises: type = Factory(dict)
    one_left_out_posterior_to_pf_surprises: type = Factory(dict)
    one_left_out_posterior_to_scrambled_pf_surprises: type = Factory(dict)

    one_left_out_to_global_surprises_mean: type = Factory(dict)
    shuffle_IDXs: np.array = None


def _SHELL_analyze_leave_one_out_decoding_results(active_pos_df, active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict):
    """ 2023-03-23 - Aims to generalize the `_analyze_leave_one_out_decoding_results`

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _subfn_compute_leave_one_out_analysis
        original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict = perform_leave_one_aclu_out_decoding_analysis(pyramidal_only_spikes_df, active_pos_df, active_filter_epochs)
        flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus = _subfn_compute_leave_one_out_analysis(active_pos_df, active_filter_epochs, original_1D_decoder, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict)


    """
    all_cells_decoded_epoch_time_bins = {}
    all_cells_computed_epoch_surprises = {}



    all_cells_computed_epoch_one_left_out_to_global_surprises = {}


    def shuffle_ids(neuron_ids, seed:int=1337):
        import random
        shuffle_IDXs = list(range(len(neuron_ids)))
        random.Random(seed).shuffle(shuffle_IDXs) # shuffle the list of indicies
        shuffle_IDXs = np.array(shuffle_IDXs)
        return neuron_ids[shuffle_IDXs], shuffle_IDXs

    shuffled_aclus, shuffle_IDXs = shuffle_ids(original_1D_decoder.neuron_IDs)

    result = LeaveOneOutDecodingResult(shuffle_IDXs=shuffle_IDXs)
    result.one_left_out_posterior_to_scrambled_pf_surprises


    # Secondary computations
    all_cells_decoded_expected_firing_rates = {}
    ## Compute the impact leaving each aclu out had on the average encoding performance:
    one_left_out_omitted_aclu_distance = {}

    ## for each cell:
    for i, left_out_aclu in enumerate(original_1D_decoder.neuron_IDs):
        # aclu = original_1D_decoder.neuron_IDs[i]
        left_out_neuron_IDX = original_1D_decoder.neuron_IDXs[i] # should just be i, but just to be safe
        ## TODO: only look at bins where the cell fires (is_cell_firing_time_bin[i])
        curr_cell_pf_curve = original_1D_decoder.pf.ratemap.tuning_curves[left_out_neuron_IDX]
        # curr_cell_spike_curve = original_1D_decoder.pf.ratemap.spikes_maps[unit_IDX] ## not occupancy weighted... is this the right one to use for computing the expected spike rate? NO... doesn't seem like it

        shuffled_cell_pf_curve = original_1D_decoder.pf.ratemap.tuning_curves[shuffle_IDXs[i]]

        left_out_decoder_result = one_left_out_filter_epochs_decoder_result_dict[left_out_aclu]
        ## single cell outputs:
        curr_cell_decoded_epoch_time_bins = [] # will be a list of the time bins in each epoch that correspond to each surprise in the corresponding list in curr_cell_computed_epoch_surprises 
        curr_cell_computed_epoch_surprises = [] # will be a list of np.arrays, with each array representing the surprise of each time bin in each epoch
        all_cells_decoded_expected_firing_rates[left_out_aclu] = [] 
        all_cells_computed_epoch_one_left_out_to_global_surprises[left_out_aclu] = []

        # have one list of posteriors p_x_given_n for each decoded epoch (active_filter_epochs.n_epochs):
        assert len(left_out_decoder_result.p_x_given_n_list) == active_filter_epochs.n_epochs == left_out_decoder_result.num_filter_epochs

        ## Compute the impact leaving each aclu out had on the average encoding performance:
        ### 1. The distance between the actual measured position and the decoded position at each timepoint for each decoder. A larger magnitude difference implies a stronger, more positive effect on the decoding quality.
        one_left_out_omitted_aclu_distance[left_out_aclu] = [] # list to hold the distance results from the epochs
        ## Iterate through each of the epochs for the given left_out_aclu (and its decoder), each of which has its own result
        for decoded_epoch_idx in np.arange(left_out_decoder_result.num_filter_epochs):
            curr_epoch_time_bin_container = left_out_decoder_result.time_bin_containers[decoded_epoch_idx]
            curr_time_bins = curr_epoch_time_bin_container.centers
            curr_epoch_p_x_given_n = left_out_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_cell_pf_curve.shape[0]
            
            ## Get the all-included values too for this decoded_epoch_idx:
            curr_epoch_all_included_p_x_given_n = all_included_filter_epochs_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_epoch_all_included_p_x_given_n.shape[0]

            ## Need to exclude estimates from bins that didn't have any spikes in them (in general these glitch around):
            curr_total_spike_counts_per_window = np.sum(left_out_decoder_result.spkcount[decoded_epoch_idx], axis=0) # left_out_decoder_result.spkcount[i].shape # (69, 222) - (nCells, nTimeWindowCenters)
            curr_is_time_bin_non_firing = (curr_total_spike_counts_per_window == 0) # this would mean that no cells fired in this time bin
            # curr_non_firing_time_bin_indicies = np.where(curr_is_time_bin_non_firing)[0] # TODO: could also filter on a minimum number of spikes larger than zero (e.g. at least 2 spikes are required).
            curr_posterior_container = left_out_decoder_result.marginal_x_list[decoded_epoch_idx]
            curr_posterior = curr_posterior_container.p_x_given_n # TODO: check the posteriors too!
            curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D

            ## Compute the distance metric for this epoch:

            # Interpolate the measured positions to the window center times:
            window_center_measured_pos_x = np.interp(curr_time_bins, active_pos_df.t, active_pos_df.lin_pos)
            # ## PLOT_ONLY: NaN out the most_likely_positions that don't have spikes.
            # curr_most_likely_valid_positions = deepcopy(curr_most_likely_positions)
            # curr_most_likely_valid_positions[curr_non_firing_time_bin_indicies] = np.nan
            
            ## Computed the distance metric finally:
            # is it fair to only compare the valid (windows containing at least one spike) windows?
            curr_omit_aclu_distance = distance.cdist(np.atleast_2d(window_center_measured_pos_x[~curr_is_time_bin_non_firing]), np.atleast_2d(curr_most_likely_positions[~curr_is_time_bin_non_firing]), 'sqeuclidean') # squared-euclidian distance between the two vectors
            # curr_omit_aclu_distance comes back double-wrapped in np.arrays for some reason (array([[659865.11994352]])), so .item() extracts the scalar value
            curr_omit_aclu_distance = curr_omit_aclu_distance.item()
            one_left_out_omitted_aclu_distance[left_out_aclu].append(curr_omit_aclu_distance)

            # Compute the expected firing rate for this cell during each bin by taking the computed position posterior and taking the sum of the element-wise product with the cell's placefield.
            # curr_epoch_expected_fr = np.array([np.sum(curr_cell_spike_curve * curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) / original_1D_decoder.time_bin_size
            curr_epoch_expected_fr = original_1D_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates[left_out_neuron_IDX] * np.array([np.sum(curr_cell_pf_curve * curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) # * original_1D_decoder.pf.ratemap.
            all_cells_decoded_expected_firing_rates[left_out_aclu].append(curr_epoch_expected_fr)
            
            # Compute the Jensen-Shannon Distance as a measure of surprise between the placefield and the posteriors
            curr_epoch_surprises = np.array([distance.jensenshannon(curr_cell_pf_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) # works! Finite! [0.5839003679903784, 0.5839003679903784, 0.6997779781969289, 0.7725622595699131, 0.5992295785891731]
            curr_cell_computed_epoch_surprises.append(curr_epoch_surprises)
            curr_cell_decoded_epoch_time_bins.append(curr_epoch_time_bin_container)

            # Compute the Jensen-Shannon Distance as a measure of surprise between the all-included and the one-left-out posteriors:
            all_cells_computed_epoch_one_left_out_to_global_surprises[left_out_aclu].append(np.array([distance.jensenshannon(curr_all_included_p_x_given_n, curr_p_x_given_n) for curr_all_included_p_x_given_n, curr_p_x_given_n in zip(curr_epoch_all_included_p_x_given_n.T, curr_epoch_p_x_given_n.T)])) 

            # The shuffled cell's placefield and the posterior from leaving a cell out:
            result.one_left_out_posterior_to_scrambled_pf_surprises[left_out_aclu].append(np.array([distance.jensenshannon(shuffled_cell_pf_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]))
            result.one_left_out_posterior_to_pf_surprises[left_out_aclu].append(np.array([distance.jensenshannon(curr_cell_pf_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T])) # works! Finite! [0.5839003679903784, 0.5839003679903784, 0.6997779781969289, 0.7725622595699131, 0.5992295785891731]

        ## End loop over decoded epochs
        assert len(curr_cell_decoded_epoch_time_bins) == len(curr_cell_computed_epoch_surprises)
        all_cells_decoded_epoch_time_bins[left_out_aclu] = curr_cell_decoded_epoch_time_bins
        all_cells_computed_epoch_surprises[left_out_aclu] = curr_cell_computed_epoch_surprises


    ## End loop over cells
    # build a dataframe version to hold the distances:
    one_left_out_omitted_aclu_distance_df = pd.DataFrame({'omitted_aclu':np.array(list(one_left_out_omitted_aclu_distance.keys())),
                                                        'distance': list(one_left_out_omitted_aclu_distance.values()),
                                                        'avg_dist': [np.mean(v) for v in one_left_out_omitted_aclu_distance.values()]}
                                                        )
    one_left_out_omitted_aclu_distance_df.sort_values(by='avg_dist', ascending=False, inplace=True) # this sort reveals the aclu values that when omitted had the largest performance decrease on decoding (as indicated by a larger distance)
    most_contributing_aclus = one_left_out_omitted_aclu_distance_df.omitted_aclu.values

    ## Reshape to -for-each-epoch instead of -for-each-cell
    all_epochs_decoded_epoch_time_bins = []
    all_epochs_computed_surprises = []
    all_epochs_computed_expected_cell_firing_rates = []
    all_epochs_computed_one_left_out_to_global_surprises = []
    for decoded_epoch_idx in np.arange(active_filter_epochs.n_epochs):
        all_epochs_decoded_epoch_time_bins.append(np.array([all_cells_decoded_epoch_time_bins[aclu][decoded_epoch_idx].centers for aclu in original_1D_decoder.neuron_IDs])) # these are duplicated (and the same) for each cell
        all_epochs_computed_surprises.append(np.array([all_cells_computed_epoch_surprises[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))
        all_epochs_computed_expected_cell_firing_rates.append(np.array([all_cells_decoded_expected_firing_rates[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))
        all_epochs_computed_one_left_out_to_global_surprises.append(np.array([all_cells_computed_epoch_one_left_out_to_global_surprises[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))

    assert len(all_epochs_computed_surprises) == active_filter_epochs.n_epochs
    assert len(all_epochs_computed_surprises[0]) == original_1D_decoder.num_neurons
    flat_all_epochs_decoded_epoch_time_bins = np.hstack(all_epochs_decoded_epoch_time_bins) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_surprises = np.hstack(all_epochs_computed_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_expected_cell_firing_rates = np.hstack(all_epochs_computed_expected_cell_firing_rates) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_one_left_out_to_global_surprises = np.hstack(all_epochs_computed_one_left_out_to_global_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs


    ## Could also do but would need to loop over all epochs for each of the three variables:
    # flat_all_epochs_computed_expected_cell_firing_rates, all_epochs_computed_expected_cell_firing_rates = _subfn_reshape_for_each_epoch_to_for_each_cell(all_cells_decoded_expected_firing_rates, epoch_IDXs=np.arange(active_filter_epochs.n_epochs), neuron_IDs=original_1D_decoder.neuron_IDs)

    ## Aggregates over all time bins in each epoch:
    all_epochs_decoded_epoch_time_bins_mean = np.vstack([np.mean(curr_epoch_time_bins, axis=1) for curr_epoch_time_bins in all_epochs_decoded_epoch_time_bins]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_cell_surprises_mean = np.vstack([np.mean(curr_epoch_surprises, axis=1) for curr_epoch_surprises in all_epochs_computed_surprises]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_cell_one_left_out_to_global_surprises_mean = np.vstack([np.mean(curr_epoch_surprises, axis=1) for curr_epoch_surprises in all_epochs_computed_one_left_out_to_global_surprises]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)

    ## Aggregates over all cells and all time bins in each epoch:
    all_epochs_all_cells_computed_surprises_mean = np.mean(all_epochs_computed_cell_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)
    all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean = np.mean(all_epochs_computed_cell_one_left_out_to_global_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)

    """ Returns:
        one_left_out_omitted_aclu_distance_df: a dataframe of the distance metric for each of the decoders in one_left_out_decoder_dict. The index is the aclu that was omitted from the decoder.
        most_contributing_aclus: a list of aclu values, sorted by the largest performance decrease on decoding (as indicated by a larger distance)
    """
    ## Output variables: flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean
    return flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus, result


# ==================================================================================================================== #
# Average Speed/Acceleration per Position Bin                                                                          #
# ==================================================================================================================== #
def _subfn_compute_group_stats_for_var(active_position_df, xbin, ybin, variable_name:str = 'speed', debug_print=False):
    """Can compute aggregate statistics (such as the mean) for any column of the position dataframe.

    Args:
        active_position_df (_type_): _description_
        xbin (_type_): _description_
        ybin (_type_): _description_
        variable_name (str, optional): _description_. Defaults to 'speed'.
        debug_print (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if ybin is None:
        # Assume 1D:
        ndim = 1
        binned_col_names = ['binned_x',]
        
    else:
        # otherwise assume 2D:
        ndim = 2
        binned_col_names = ['binned_x','binned_y']
        

    # For each unique binned_x and binned_y value, what is the average velocity_x at that point?
    position_bin_dependent_specific_average_velocities = active_position_df.groupby(binned_col_names)[variable_name].agg([np.nansum, np.nanmean, np.nanmin, np.nanmax]).reset_index() #.apply(lambda g: g.mean(skipna=True)) #.agg((lambda x: x.mean(skipna=False)))
    if debug_print:
        print(f'np.shape(position_bin_dependent_specific_average_velocities): {np.shape(position_bin_dependent_specific_average_velocities)}')
    # position_bin_dependent_specific_average_velocities # 1856 rows
    if debug_print:
        print(f'np.shape(output): {np.shape(output)}')
        print(f"np.shape(position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()): {np.shape(position_bin_dependent_specific_average_velocities['binned_x'])}")
        max_binned_x_index = np.max(position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1) # 63
        print(f'max_binned_x_index: {max_binned_x_index}')
    # (len(xbin), len(ybin)): (59, 21)

    if ndim == 1:
        # 1D Position:
        output = np.zeros((len(xbin),)) # (65,)
        output[position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1] = position_bin_dependent_specific_average_velocities['nanmean'].to_numpy() # ValueError: shape mismatch: value array of shape (1856,) could not be broadcast to indexing result of shape (1856,2,30)

    else:
        # 2D+ Position:
        if debug_print:
            max_binned_y_index = np.max(position_bin_dependent_specific_average_velocities['binned_y'].to_numpy()-1) # 28
            print(f'max_binned_y_index: {max_binned_y_index}')
        output = np.zeros((len(xbin), len(ybin))) # (65, 30)
        output[position_bin_dependent_specific_average_velocities['binned_x'].to_numpy()-1, position_bin_dependent_specific_average_velocities['binned_y'].to_numpy()-1] = position_bin_dependent_specific_average_velocities['nanmean'].to_numpy() # ValueError: shape mismatch: value array of shape (1856,) could not be broadcast to indexing result of shape (1856,2,30)

    return output


def _compute_avg_speed_at_each_position_bin(active_position_df, active_pf_computation_config, xbin, ybin, show_plots=False, debug_print=False):
    """ compute the average speed at each position x. Used only by the two-step decoder above. """
    outputs = dict()
    outputs['speed'] = _subfn_compute_group_stats_for_var(active_position_df, xbin, ybin, 'speed')
    # outputs['velocity_x'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'velocity_x')
    # outputs['acceleration_x'] = _compute_group_stats_for_var(active_position_df, xbin, ybin, 'acceleration_x')
    return outputs['speed']

