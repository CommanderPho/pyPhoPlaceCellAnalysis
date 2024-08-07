
# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #

from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
import neuropy.utils.type_aliases as types
from attrs import define, field, Factory
from neuropy.utils.mixins.binning_helpers import transition_matrix
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder #typehinting only
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability

def vertical_gaussian_blur(arr, sigma:float=1, **kwargs):
    """ blurs each column over the rows """
    return np.apply_along_axis(gaussian_filter1d, axis=0, arr=arr, sigma=sigma, **kwargs)


def horizontal_gaussian_blur(arr, sigma:float=1, **kwargs):
    """ blurs each row over the columns """
    return np.apply_along_axis(gaussian_filter1d, axis=1, arr=arr, sigma=sigma, **kwargs)
    




@define(slots=False, eq=False)
@metadata_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-14 00:00', related_items=[])
class TransitionMatrixComputations:
    """ 
    from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
    
    # Visualization ______________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
    
    out = TransitionMatrixComputations.plot_transition_matricies(decoders_dict=decoders_dict, binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list_dict)
    out

    """
    binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, List[NDArray]] = field(factory=dict)
    n_powers: int = field(default=20)
    time_bin_size: float = field(default=None)
    pos_bin_size: float = field(default=None)
    

    ### 1D Transition Matrix:
    @classmethod
    def _compute_position_transition_matrix(cls, xbin_labels, binned_x_indicies: np.ndarray, n_powers:int=3, use_direct_observations_for_order:bool=True, should_validate_normalization:bool=True):
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
        max_state_index: int = num_position_states - 1

        # validate the state sequences are indeed consistent:    
        assert max(binned_x_indicies) <= max_state_index, f"VIOLATED! max(binned_x_indicies): {max(binned_x_indicies)} <= max_state_index: {max_state_index}"
        assert max(binned_x_indicies) < num_position_states, f"VIOLATED! max(binned_x_indicies): {max(binned_x_indicies)} < num_position_states: {num_position_states}"
        # assert 0 in state_sequence, f"does not contain zero! Make sure that it is not a 1-indexed sequence!"
        
        # 0th order:
        binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_indicies), markov_order=1, max_state_index=max_state_index, nan_entries_replace_value=0.0, should_validate_normalization=should_validate_normalization) # #TODO 2024-08-02 21:10: - [ ] max_state_index != num_position_states
        if should_validate_normalization:
            ## test row normalization (only considering non-zero entries):
            _row_normalization_sum = np.sum(binned_x_transition_matrix, axis=1)
            assert np.allclose(_row_normalization_sum[np.nonzero(_row_normalization_sum)], 1), f"0th order not row normalized!\n\t_row_normalization_sum: {_row_normalization_sum}"
            
        if not use_direct_observations_for_order:
            ## use exponentiation version: only works if Markov Property is not violated!
            binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix] + [np.linalg.matrix_power(binned_x_transition_matrix, n) for n in np.arange(2, n_powers+1)]
        else:
            binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix] + [transition_matrix(deepcopy(binned_x_indicies), markov_order=n, max_state_index=max_state_index, nan_entries_replace_value=0.0, should_validate_normalization=should_validate_normalization) for n in np.arange(2, n_powers+1)]
            
        if should_validate_normalization:
            ## test row normalization (only considering non-zero entries):
            _row_normalization_sums = [np.sum(a_mat, axis=1) for a_mat in binned_x_transition_matrix_higher_order_list]
            _is_row_normalization_all_valid = [np.allclose(v[np.nonzero(v)], 1.0) for v in _row_normalization_sums]
            assert np.alltrue(_is_row_normalization_all_valid), f"not row normalized!\n\t_is_row_normalization_all_valid: {_is_row_normalization_all_valid}\n\t_row_normalization_sums: {_row_normalization_sums}"

        # binned_x_transition_matrix.shape # (64, 64)
        return binned_x_transition_matrix_higher_order_list


    @classmethod
    def _compute_time_transition_matrix(cls, df: pd.DataFrame, t_bin_size=0.25): # "0.25S"
        """ attempts to determine a spike transition matrix from continuously sampled events (like spike times) given a specified bin_size 
        
        #TODO 2024-08-06 08:50: - [ ] Unfinished
        
        
        """
        from scipy.sparse import csr_matrix
        
        # df["bin"] = pd.cut(df["t_rel_seconds"], bins=pd.timedelta_range(df["t_rel_seconds"].min(), df["t_rel_seconds"].max(), freq=bin_size))
        # df["bin"] = pd.cut(
        # 	pd.to_timedelta(df["t_rel_seconds"], unit="s"),
        # 	bins=pd.timedelta_range(df["t_rel_seconds"].min(), df["t_rel_seconds"].max(), freq=bin_size),
        # )
        def get_bin(seconds):
            bin_number = int(seconds // t_bin_size)
            return bin_number

        df["bin"] = df["t_rel_seconds"].apply(get_bin) # .astype(np.int64)
        # df = df.astype({'aclu': 'int64', 'bin': 'int64'})
        df = df[['aclu', 'bin']] # .astype({'aclu': 'float', 'bin': 'float'})
        print(f'df.dtypes: {df.dtypes}')
        # , dtype=np.int8
        transition_matrix = csr_matrix((
            counts, (i, j)
        ) for i, (bin, group) in df.groupby(["bin", "aclu"]).agg(counts=("aclu", "count")).iterrows()
            for j, counts in group.iteritems()
        )
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        return transition_matrix


    # def _build_decoded_positions_transition_matrix(active_one_step_decoder):
    #     """ Compute the transition_matrix from the decoded positions 

    #     TODO: make sure that separate events (e.g. separate replays) are not truncated creating erronious transitions

    #     """
    #     # active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
    #     # active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
    #     # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
    #     active_one_step_decoder.most_likely_position_flat_indicies
    #     # active_most_likely_positions = active_one_step_decoder.revised_most_likely_positions.T
    #     # active_most_likely_positions #.shape # (36246,)

    #     most_likely_position_indicies = np.squeeze(np.array(np.unravel_index(active_one_step_decoder.most_likely_position_flat_indicies, active_one_step_decoder.original_position_data_shape))) # convert back to an array
    #     most_likely_position_xbins = most_likely_position_indicies + 1 # add 1 to convert back to a bin label from an index
    #     # most_likely_position_indicies # (1, 36246)

    #     xbin_labels = np.arange(active_one_step_decoder.original_position_data_shape[0]) + 1

    #     decoded_binned_x_transition_matrix_higher_order_list = TransitionMatrixComputations._compute_position_transition_matrix(xbin_labels, most_likely_position_indicies)
    #     return decoded_binned_x_transition_matrix_higher_order_list, xbin_labels

    # ==================================================================================================================== #
    # 2024-08-02 Likelihoods of observed transitions from transition matricies                                             #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _generate_testing_posteriors(cls, decoders_dict, a_decoder_name, n_generated_t_bins: int = 4, test_time_bin_size: float = 0.25, test_posterior_type:str='directly_adjacent_pos_bins', debug_print=True,
                                     blur_vertical_std_dev=None, blur_horizontal_std_dev=None):
        """ generates sample position posteriors for testing 
        
        test_posterior, (test_tbins, test_pos_bins) = _generate_testing_posteriors(decoders_dict, a_decoder_name)
        n_xbins = len(test_pos_bins)

        """
        # number time bins in generated posterior

        n_xbins = (len(decoders_dict[a_decoder_name].xbin) - 1) # the -1 is to get the counts for the centers only
        test_tbins = np.arange(n_generated_t_bins).astype(float) * test_time_bin_size
        # test_pos_bins = np.arange(n_xbins).astype(float)
        test_pos_bins = deepcopy(decoders_dict[a_decoder_name].xbin_centers)

        print(f'{n_xbins =}, {n_generated_t_bins =}')
        test_posterior: NDArray = np.zeros((n_xbins, n_generated_t_bins))

        # ## Separated position bins 
        # positions_ratio_space = [0.75, 0.5, 0.4, 0.25]
        # positions_bin_space = [int(round(x * n_xbins)) for x in positions_ratio_space]
        # for i, a_bin in enumerate(positions_bin_space):
        #     test_posterior[a_bin, i] = 1.0
            
        known_test_posterior_types = ['directly_adjacent_pos_bins', 'step_adjacent_pos_bins']
        ## Directly adjacent position bins 
        if test_posterior_type == 'directly_adjacent_pos_bins':
            start_idx = 14
            positions_bin_space = start_idx + np.arange(n_generated_t_bins)
            for i, a_bin in enumerate(positions_bin_space):
                test_posterior[a_bin, i] = 1.0
        elif test_posterior_type == 'step_adjacent_pos_bins':
            start_idx = 9
            
            ## Step mode:
            step_size = 2
            stop_idx = (n_generated_t_bins * step_size)
            
            # ## Span mode:
            # span_size = 5 # span 5 bins sequentially, then step
            # step_size = 1
            # stop_idx = (n_generated_t_bins * step_size)
            # step_arr = np.repeat(1.0, span_size) + np.repeat(0.0, step_size)
            
            positions_bin_space = start_idx + np.arange(start=0, stop=stop_idx, step=step_size)
            for i, a_bin in enumerate(positions_bin_space):
                test_posterior[a_bin, i] = 1.0
                                
        else:
            raise NotImplementedError(f'test_posterior_type: "{test_posterior_type}" was not recognized! Known types: {known_test_posterior_types}')

        if debug_print:
            print(f"positions_bin_space: {positions_bin_space}")
            
        ## WARN: posteriors must be normalized over all possible positions in each time bin (all columns), the opposite of normalizing transition matricies
        # Normalize posterior by columns (over all positions)
        test_posterior = normalize(test_posterior, axis=0, norm='l1')
        
        ## apply vertical and horizontal blurs:
        if (blur_vertical_std_dev is not None) and (blur_vertical_std_dev > 0.0):
            # blur_vertical_std_dev: Standard deviation for Gaussian kernel
            test_posterior = vertical_gaussian_blur(test_posterior, sigma=blur_vertical_std_dev)
            test_posterior = normalize(test_posterior, axis=0, norm='l1') # ensure re-normalized posterior
            
        if (blur_horizontal_std_dev is not None) and (blur_horizontal_std_dev > 0.0):
            test_posterior = horizontal_gaussian_blur(test_posterior, sigma=blur_horizontal_std_dev)
            test_posterior = normalize(test_posterior, axis=0, norm='l1') # ensure re-normalized posterior
    
        return test_posterior, (test_tbins, test_pos_bins)

    @classmethod
    def _generate_expected_replay_sequences(cls, a_binned_x_transition_matrix_higher_order_list, n_generated_events: int = 10, n_generated_t_bins: int = 4, test_time_bin_size: float = 0.25, transition_matrix_order_start_idx:int=0, debug_print=True, blur_vertical_std_dev=None, blur_horizontal_std_dev=None):
        """ takes a a_binned_x_transition_matrix_higher_order_list and a position posterior
        
        Uses the first time bin to and the `a_binned_x_transition_matrix_higher_order_list` to predict the future bins:

        Usage:        
            predicited_posteriors = TransitionMatrixComputations._perform_forward_prediction(a_binned_x_transition_matrix_higher_order_list, test_posterior)
            predicited_posteriors

            # # Compare real:
            ## Observed posteriors:
            observed_posteriors = deepcopy(test_posterior[:, 1:])

            assert np.shape(observed_posteriors) == np.shape(predicited_posteriors)
            _prediction_observation_diff = observed_posteriors - predicited_posteriors
            _prediction_observation_diff

        """
        print(f'n_generated_events: {n_generated_events}, n_generated_t_bins: {n_generated_t_bins}')
        
        generated_events = []
        
        n_pos_bins: int = np.shape(a_binned_x_transition_matrix_higher_order_list[0])[0]

        for gen_evt_idx in np.arange(n_generated_events):
            a_trans_prob_mat = a_binned_x_transition_matrix_higher_order_list[transition_matrix_order_start_idx] # (n_x, n_x)

            # Generate a random start location:            
            arr = np.zeros((n_pos_bins,))
            arr[np.random.randint(0, len(arr))] = 1.0
            an_observed_posterior = np.atleast_2d(arr).T # (n_x, 1)
            ## Decode the posterior
            n_time_bins: int = n_generated_t_bins
            n_predicted_time_bins = n_time_bins - 1
            # print(f'np.shape(test_posterior): {np.shape(an_observed_posterior)}, n_time_bins: {n_time_bins}, n_predicted_time_bins: {n_predicted_time_bins}')
            predicited_posteriors = []
            predicited_posteriors.append(an_observed_posterior) ## add randomly generated one:
            # for a_tbin_idx in np.arange(start=1, stop=n_time_bins):
            for i in np.arange(n_predicted_time_bins):
                a_tbin_idx = i + 1
                # an_observed_posterior = np.atleast_2d(test_posterior[:, i]).T # (n_x, 1)
                # an_actual_observed_next_step_posterior = np.atleast_2d(test_posterior[:, a_tbin_idx]).T # (n_x, 1)
                # an_observed_posterior

                ## NOTE: transition_matrix_order does seem to do the identity transformation
                transition_matrix_order_idx: int = transition_matrix_order_start_idx + a_tbin_idx
                # transition_matrix_order: int = transition_matrix_order_idx + 1 ## order is index + 1

                ## single time-bin
                a_trans_prob_mat = a_binned_x_transition_matrix_higher_order_list[transition_matrix_order_idx] # (n_x, n_x)
                # a_next_t_predicted_pos_probs = a_trans_prob_mat @ an_observed_posterior # (n_x, 1)
                a_next_t_predicted_pos_probs = a_trans_prob_mat @ an_observed_posterior # (n_x, 1)

                # a_next_t_predicted_pos_probs
                predicited_posteriors.append(a_next_t_predicted_pos_probs)

            predicited_posteriors = np.hstack(predicited_posteriors) # (n_x, n_predicted_time_bins)
            generated_events.append(predicited_posteriors)
            

        return generated_events
    



    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _likelihood_of_observation(cls, observed_posterior, pos_likelihoods) -> float:
        """ likelihood of the observed posterior for a single time bin """
        # Using numpy.dot() function
        assert np.shape(observed_posterior) == np.shape(pos_likelihoods)
        # Squeeze to remove single-dimensional entries
        observed_posterior = np.squeeze(observed_posterior)  # Shape (3,)
        pos_likelihoods = np.squeeze(pos_likelihoods)  # Shape (3,)

        return np.dot(observed_posterior, pos_likelihoods)

    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _predicted_probabilities_likelihood(cls, a_binned_x_transition_matrix_higher_order_list, test_posterior, transition_matrix_order: int=5) -> NDArray:
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

    @classmethod
    def _perform_forward_prediction(cls, a_binned_x_transition_matrix_higher_order_list, test_posterior, transition_matrix_order_start_idx:int=1):
        """ takes a a_binned_x_transition_matrix_higher_order_list and a position posterior
        
        Uses the first time bin to and the `a_binned_x_transition_matrix_higher_order_list` to predict the future bins:

        Usage:        
            predicited_posteriors = TransitionMatrixComputations._perform_forward_prediction(a_binned_x_transition_matrix_higher_order_list, test_posterior)
            predicited_posteriors

            # # Compare real:
            ## Observed posteriors:
            observed_posteriors = deepcopy(test_posterior[:, 1:])

            assert np.shape(observed_posteriors) == np.shape(predicited_posteriors)
            _prediction_observation_diff = observed_posteriors - predicited_posteriors
            _prediction_observation_diff

        """
        n_time_bins: int = np.shape(test_posterior)[1]
        n_predicted_time_bins = n_time_bins - 1
        print(f'np.shape(test_posterior): {np.shape(test_posterior)}, n_time_bins: {n_time_bins}, n_predicted_time_bins: {n_predicted_time_bins}')
        # Only use the first posterior
        an_observed_posterior = np.atleast_2d(test_posterior[:, 0]).T # (n_x, 1)


        predicited_posteriors = []
        # for a_tbin_idx in np.arange(start=1, stop=n_time_bins):
        for i in np.arange(n_predicted_time_bins):
            a_tbin_idx = i + 1
            # an_observed_posterior = np.atleast_2d(test_posterior[:, i]).T # (n_x, 1)
            # an_actual_observed_next_step_posterior = np.atleast_2d(test_posterior[:, a_tbin_idx]).T # (n_x, 1)
            # an_observed_posterior

            ## NOTE: transition_matrix_order does seem to do the identity transformation
            transition_matrix_order_idx: int = transition_matrix_order_start_idx + a_tbin_idx
            # transition_matrix_order: int = transition_matrix_order_idx + 1 ## order is index + 1

            ## single time-bin
            a_trans_prob_mat = a_binned_x_transition_matrix_higher_order_list[transition_matrix_order_idx] # (n_x, n_x)
            # a_next_t_predicted_pos_probs = a_trans_prob_mat @ an_observed_posterior # (n_x, 1)
            a_next_t_predicted_pos_probs = a_trans_prob_mat @ an_observed_posterior # (n_x, 1)

            # a_next_t_predicted_pos_probs
            predicited_posteriors.append(a_next_t_predicted_pos_probs)

        predicited_posteriors = np.hstack(predicited_posteriors) # (n_x, n_predicted_time_bins)
        return predicited_posteriors

    # ==================================================================================================================== #
    # Plot/Display                                                                                                         #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix', 'plot'], input_requires=[], output_provides=[], uses=['BasicBinnedImageRenderingWindow'], used_by=[], creation_date='2024-08-02 09:55', related_items=[])
    def plot_transition_matricies(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, NDArray],
                                   power_step:int=7, grid_opacity=0.4, enable_all_titles=True) -> BasicBinnedImageRenderingWindow:
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
                                                    grid_opacity=grid_opacity)
                # add remaining rows for this decoder:
                _subfn_plot_all_rows(start_idx=1)
                
            else:
                # add to existing plotter:
                _subfn_plot_all_rows()
                

        return out
    

    # ==================================================================================================================== #
    # Save/Load                                                                                                            #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix', 'save'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=[])
    def save_transition_matricies(cls, binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, List[NDArray]], save_path:Path='transition_matrix_data.h5'): # decoders_dict: Dict[types.DecoderName, BasePositionDecoder], 
        """Save the transitiion matrix info to a file
        
        save_path = TransitionMatrixComputations.save_transition_matricies(binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list_dict, save_path='output/transition_matrix_data.h5')
        save_path
        
        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path).resolve()
        
        import h5py
        # Save to .h5 file
        with h5py.File(save_path, 'w') as f:
            for a_name, an_array_list in binned_x_transition_matrix_higher_order_list_dict.items():
                decoder_prefix: str = a_name
                for markov_order, array in enumerate(an_array_list):
                    a_dset = f.create_dataset(f'{decoder_prefix}/array_{markov_order}', data=array)
                    a_dset.attrs['decoder_name'] = a_name
                    a_dset.attrs['markov_order'] = markov_order
                    
                                
                # Add metadata
                f.attrs['decoder_name'] = a_name

            

        return save_path
    

    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix', 'load'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=[])
    def load_transition_matrices(cls, load_path: Path) -> Dict[types.DecoderName, List[np.ndarray]]:
        """
        Load the transition matrix info from a file
        
        load_path = Path('output/transition_matrix_data.h5')
        binned_x_transition_matrix_higher_order_list_dict = TransitionMatrixComputations.load_transition_matrices(load_path)
        binned_x_transition_matrix_higher_order_list_dict
        """
        if not isinstance(load_path, Path):
            load_path = Path(load_path).resolve()

        import h5py
        binned_x_transition_matrix_higher_order_list_dict: Dict[types.DecoderName, List[np.ndarray]] = {}

        with h5py.File(load_path, 'r') as f:
            for decoder_prefix in f.keys():
                arrays_list = []
                group = f[decoder_prefix]
                for dataset_name in group.keys():
                    array = group[dataset_name][()]
                    markov_order = group[dataset_name].attrs['markov_order']
                    arrays_list.append((markov_order, array))
                
                arrays_list.sort(key=lambda x: x[0])  # Sort by markov_order
                binned_x_transition_matrix_higher_order_list_dict[decoder_prefix] = [array for _, array in arrays_list]

        return binned_x_transition_matrix_higher_order_list_dict
