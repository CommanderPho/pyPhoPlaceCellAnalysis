
# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #

from copy import deepcopy
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
from neuropy.utils.mixins.binning_helpers import transition_matrix
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder #typehinting only
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability


@metadata_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-14 00:00', related_items=[])
class TransitionMatrixComputations:
    """ 
    from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
    
    # Visualization ______________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
    
    out = TransitionMatrixComputations.plot_transition_matricies(decoders_dict=decoders_dict, binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list_dict)
    out

    """
    ### 1D Transition Matrix:

    def _compute_position_transition_matrix(xbin_labels, binned_x: np.ndarray, n_powers:int=3):
        """  1D Transition Matrix from binned positions (e.g. 'binned_x')

            pf1D.xbin_labels # array([  1,   2,   3,   4,  ...)
            pf1D.filtered_pos_df['binned_x'].to_numpy() # array([116, 115, 115, ...,  93,  93,  93], dtype=int64)
            
        Usage:
        
            # pf1D = deepcopy(curr_active_pipeline.computation_results['maze1'].computed_data['pf1D'])
            pf1D = deepcopy(global_pf1D)
            # pf1D = deepcopy(short_pf1D)
            # pf1D = deepcopy(long_pf1D)
            binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix(pf1D.xbin_labels, pf1D.filtered_pos_df['binned_x'].to_numpy())

        """
        num_position_states = len(xbin_labels)
        # binned_x = pos_1D.to_numpy()
        binned_x_indicies = binned_x - 1
        binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_indicies), markov_order=1, max_state_index=num_position_states)
        # binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, transition_matrix(deepcopy(binned_x_indicies), markov_order=2, max_state_index=num_position_states), transition_matrix(deepcopy(binned_x_indicies), markov_order=3, max_state_index=num_position_states)]

        binned_x_transition_matrix[np.isnan(binned_x_transition_matrix)] = 0.0
        binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix] + [np.linalg.matrix_power(binned_x_transition_matrix, n) for n in np.arange(2, n_powers+1)]
        # , np.linalg.matrix_power(binned_x_transition_matrix, 2), np.linalg.matrix_power(binned_x_transition_matrix, 3)
        # binned_x_transition_matrix.shape # (64, 64)
        return binned_x_transition_matrix_higher_order_list

    def _build_decoded_positions_transition_matrix(active_one_step_decoder):
        """ Compute the transition_matrix from the decoded positions 

        TODO: make sure that separate events (e.g. separate replays) are not truncated creating erronious transitions

        """
        # active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
        # active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
        # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
        active_one_step_decoder.most_likely_position_flat_indicies
        # active_most_likely_positions = active_one_step_decoder.revised_most_likely_positions.T
        # active_most_likely_positions #.shape # (36246,)

        most_likely_position_indicies = np.squeeze(np.array(np.unravel_index(active_one_step_decoder.most_likely_position_flat_indicies, active_one_step_decoder.original_position_data_shape))) # convert back to an array
        most_likely_position_xbins = most_likely_position_indicies + 1 # add 1 to convert back to a bin label from an index
        # most_likely_position_indicies # (1, 36246)

        xbin_labels = np.arange(active_one_step_decoder.original_position_data_shape[0]) + 1

        decoded_binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix(xbin_labels, most_likely_position_indicies)
        return decoded_binned_x_transition_matrix_higher_order_list, xbin_labels

    # ==================================================================================================================== #
    # 2024-08-02 Likelihoods of observed transitions from transition matricies                                             #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _generate_testing_posteriors(decoders_dict, a_decoder_name, n_generated_t_bins: int = 4, test_time_bin_size: float = 0.25):
        """ generates sample posteriors for testing 
        
        test_posterior, (test_tbins, test_pos_bins) = _generate_testing_posteriors(decoders_dict, a_decoder_name)
        n_xbins = len(test_pos_bins)

        """
        # number time bins in generated posterior

        n_xbins = len(decoders_dict[a_decoder_name].xbin)
        test_tbins = np.arange(n_generated_t_bins).astype(float) * test_time_bin_size
        # test_pos_bins = np.arange(n_xbins).astype(float)
        test_pos_bins = deepcopy(decoders_dict[a_decoder_name].xbin)

        print(f'{n_xbins =}, {n_generated_t_bins =}')
        test_posterior = np.zeros((n_xbins, n_generated_t_bins))

        # ## Separated position bins 
        # positions_ratio_space = [0.75, 0.5, 0.4, 0.25]
        # positions_bin_space = [int(round(x * n_xbins)) for x in positions_ratio_space]
        # for i, a_bin in enumerate(positions_bin_space):
        #     test_posterior[a_bin, i] = 1.0
            

        ## Directly adjacent position bins 
        start_idx = 14
        positions_bin_space = start_idx + np.arange(4)
        positions_bin_space
        for i, a_bin in enumerate(positions_bin_space):
            test_posterior[a_bin, i] = 1.0

        return test_posterior, (test_tbins, test_pos_bins)

    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _likelihood_of_observation(observed_posterior, pos_likelihoods) -> float:
        """ likelihood of the observed posterior for a single time bin """
        # Using numpy.dot() function
        assert np.shape(observed_posterior) == np.shape(pos_likelihoods)
        # Squeeze to remove single-dimensional entries
        observed_posterior = np.squeeze(observed_posterior)  # Shape (3,)
        pos_likelihoods = np.squeeze(pos_likelihoods)  # Shape (3,)

        return np.dot(observed_posterior, pos_likelihoods)

    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _predicted_probabilities_likelihood(a_binned_x_transition_matrix_higher_order_list, test_posterior, transition_matrix_order: int=5) -> NDArray:
        """ Computes likelihoods giveen posteriors and transition matricies for a certain `transition_matrix_order`
        
        next_pos_likelihood = _predicted_probabilities_likelihood(a_binned_x_transition_matrix_higher_order_list, test_posterior, transition_matrix_order=5)
        predicted_probabilities_dict[a_decoder_name] = next_pos_likelihood

        """
        ## single order
        next_pos_likelihood = []
        # for a_tbin_idx, a_tbin in enumerate(test_tbins):
        for a_tbin_idx in np.arange(np.shape(test_posterior)[1]): # each time bin in posterior
            ## single time-bin
            a_trans_prob_mat = a_binned_x_transition_matrix_higher_order_list[transition_matrix_order] # (n_x, n_x)
            a_next_t_predicted_pos_probs = a_trans_prob_mat @ np.atleast_2d(test_posterior[:, a_tbin_idx]).T # (n_x, 1)
            next_pos_likelihood.append(a_next_t_predicted_pos_probs) # @ is matrix multiplication
            
            # a_next_t_step_obs_posterior = np.atleast_2d(test_posterior[:, a_tbin_idx+1]).T # (n_x, 1)
            # next_pos_likelihood = _likelihood_of_observation(a_next_t_step_obs_posterior, next_pos_likelihood)

        next_pos_likelihood = np.hstack(next_pos_likelihood) # (n_x, 1)
        return next_pos_likelihood


    @function_attributes(short_name=None, tags=['transition_matrix', 'plot'], input_requires=[], output_provides=[], uses=['BasicBinnedImageRenderingWindow'], used_by=[], creation_date='2024-08-02 09:55', related_items=[])
    def plot_transition_matricies(decoders_dict: Dict[types.DecoderName, BasePositionDecoder], binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray], power_step:int=7, enable_all_titles=True) -> BasicBinnedImageRenderingWindow:
        """ plots each decoder as a separate column
        each order of matrix as a separate row
        
        Works well
        
        Usage:

            out = TransitionMatrixComputations.plot_transition_matricies(decoders_dict=decoders_dict, binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list_dict)
            out

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
        
        out = None
        all_decoders_label_kwargs = dict(name=f'binned_x_transition_matrix for all decoders', title=f"Transition Matrix for binned x (from, to) for all decoders", variable_label='Transition Matrix')
        for a_decoder_idx, (a_decoder_name, a_binned_x_transition_matrix_higher_order_list) in enumerate(binned_x_transition_matrix_higher_order_list_dict.items()):
            a_decoder_label_kwargs = dict(name=f'binned_x_transition_matrix["{a_decoder_name}"]', title=f"Transition Matrix for binned x (from, to) for '{a_decoder_name}'", variable_label='Transition Matrix')

            def _subfn_plot_all_rows(start_idx:int=0):
                for row_idx, transition_power_idx in enumerate(np.arange(start=start_idx, stop=len(a_binned_x_transition_matrix_higher_order_list), step=power_step)):
                    row_idx = row_idx + start_idx
                    a_title = ''
                    if enable_all_titles:
                        a_title = f'{a_decoder_label_kwargs["name"]}[{transition_power_idx}]'
                    else:
                        if row_idx == 0:
                            a_title = f'decoder: "{a_decoder_name}"'
                    
                    out.add_data(row=row_idx, col=a_decoder_idx, matrix=a_binned_x_transition_matrix_higher_order_list[transition_power_idx], xbins=decoders_dict[a_decoder_name].xbin_centers, ybins= decoders_dict[a_decoder_name].xbin_centers,
                                name=f'{a_decoder_label_kwargs["name"]}[{transition_power_idx}]', title=a_title, variable_label=f'{a_decoder_label_kwargs["name"]}[{transition_power_idx}]')  

            if out is None:
                ## only VERy first (0, 0) item
                out = BasicBinnedImageRenderingWindow(a_binned_x_transition_matrix_higher_order_list[0], decoders_dict[a_decoder_name].xbin_centers, decoders_dict[a_decoder_name].xbin_centers,
                                                    **all_decoders_label_kwargs, scrollability_mode=LayoutScrollability.NON_SCROLLABLE,
                                                    grid_opacity=0.4)
                # add remaining rows for this decoder:
                _subfn_plot_all_rows(start_idx=1)
                
            else:
                # add to existing plotter:
                _subfn_plot_all_rows()
                

        return out