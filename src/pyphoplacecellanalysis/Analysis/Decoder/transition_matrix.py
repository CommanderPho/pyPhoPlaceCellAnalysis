
# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #

from copy import deepcopy
import numpy as np
from neuropy.utils.mixins.binning_helpers import transition_matrix
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


@metadata_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-14 00:00', related_items=[])
class TransitionMatrixComputations:
    """ 
    from PendingNotebookCode import TransitionMatrixComputations
    
    # Visualization ______________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
    out = BasicBinnedImageRenderingWindow(binned_x_transition_matrix_higher_order_list[0], pf1D.xbin_labels, pf1D.xbin_labels, name='binned_x_transition_matrix', title="Transition Matrix for binned x (from, to)", variable_label='Transition Matrix', scrollability_mode=LayoutScrollability.NON_SCROLLABLE)
    
    
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


