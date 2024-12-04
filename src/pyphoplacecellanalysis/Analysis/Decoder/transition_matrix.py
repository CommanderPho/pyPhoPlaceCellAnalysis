from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
from nptyping import NDArray
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize
import neuropy.utils.type_aliases as types
from attrs import define, field, Factory
from neuropy.utils.mixins.binning_helpers import transition_matrix
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder #typehinting only
    from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult


# Custom Type Definitions ____________________________________________________________________________________________ #
T = TypeVar('T')
DecoderListDict: TypeAlias = Dict[types.DecoderName, List[T]] # Use like `v: DecoderListDict[NDArray]`


# used for `_compute_expected_velocity_out_per_node`
class VelocityType(Enum):
    OUTGOING = 'out'
    INCOMING = 'in'
    

@define(slots=False, eq=False)
class ExpectedVelocityTuple(UnpackableMixin, object):
    """ the specific heuristic measures for a single decoded epoch
    
    from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import ExpectedVelocityTuple
    
    
    """
    # Incoming:
    in_combined: NDArray = field()
    in_fwd: NDArray = field()
    in_bkwd: NDArray = field()

    
    # Outgoing
    out_combined: NDArray = field() # this seems to be combined incorrectly when fwd or bkwd is NaN, it becomes NaN
    out_fwd: NDArray = field()
    out_bkwd: NDArray = field()

    

    @classmethod
    def init_by_computing_from_transition_matrix(cls, A: NDArray) -> "ExpectedVelocityTuple":
        """ computes all incomming/outgoing x fwd/bkwd velocities from a position bin transition matrix"""
        combined_expected_incoming_velocity, (fwd_expected_incoming_velocity, bkwd_expected_incoming_velocity) = TransitionMatrixComputations._compute_expected_velocity_out_per_node(A, should_split_fwd_and_bkwd_velocities=True, velocity_type='in')
        combined_expected_out_velocity, (fwd_expected_out_velocity, bkwd_expected_out_velocity) = TransitionMatrixComputations._compute_expected_velocity_out_per_node(A, should_split_fwd_and_bkwd_velocities=True, velocity_type='out')
        return cls(in_fwd=fwd_expected_incoming_velocity, in_bkwd=bkwd_expected_incoming_velocity, in_combined=combined_expected_incoming_velocity,
                              out_fwd=fwd_expected_out_velocity, out_bkwd=bkwd_expected_out_velocity, out_combined=combined_expected_out_velocity)
        

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
    binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray] = field(factory=dict)
    n_powers: int = field(default=20)
    time_bin_size: float = field(default=None)
    pos_bin_size: float = field(default=None)
    

    ### 1D Transition Matrix:
    @classmethod
    def _compute_position_transition_matrix(cls, xbin_labels, binned_x_index_sequence: np.ndarray, n_powers:int=3, use_direct_observations_for_order:bool=True, should_validate_normalization:bool=True):
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
        assert max(binned_x_index_sequence) <= max_state_index, f"VIOLATED! max(binned_x_index_sequence): {max(binned_x_index_sequence)} <= max_state_index: {max_state_index}"
        assert max(binned_x_index_sequence) < num_position_states, f"VIOLATED! max(binned_x_index_sequence): {max(binned_x_index_sequence)} < num_position_states: {num_position_states}"
        # assert 0 in state_sequence, f"does not contain zero! Make sure that it is not a 1-indexed sequence!"
        
        # 0th order:
        binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_index_sequence), markov_order=1, max_state_index=max_state_index, nan_entries_replace_value=0.0, should_validate_normalization=should_validate_normalization) # #TODO 2024-08-02 21:10: - [ ] max_state_index != num_position_states
        if should_validate_normalization:
            ## test row normalization (only considering non-zero entries):
            _row_normalization_sum = np.sum(binned_x_transition_matrix, axis=1)
            assert np.allclose(_row_normalization_sum[np.nonzero(_row_normalization_sum)], 1), f"0th order not row normalized!\n\t_row_normalization_sum: {_row_normalization_sum}"
            
        if not use_direct_observations_for_order:
            ## use exponentiation version: only works if Markov Property is not violated!
            binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix] + [np.linalg.matrix_power(binned_x_transition_matrix, n) for n in np.arange(2, n_powers+1)]
        else:
            binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix] + [transition_matrix(deepcopy(binned_x_index_sequence), markov_order=n, max_state_index=max_state_index, nan_entries_replace_value=0.0, should_validate_normalization=should_validate_normalization) for n in np.arange(2, n_powers+1)]
            
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

        Usage:        
            from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations

            test_spikes_df = deepcopy(rank_order_results.LR_laps.spikes_df)
            a_decoder = deepcopy(track_templates.get_decoders_dict()['long_LR'])
            filtered_pos_df: pd.DataFrame = deepcopy(a_decoder.pf.filtered_pos_df)
            filtered_pos_df = filtered_pos_df.dropna(axis='index', how='any', subset=['binned_x']).reset_index(drop=True)
            filtered_pos_df

            # test_spikes_df = test_spikes_df.spikes.add_binned_time_column()

            # [['t_rel_seconds', 'aclu', 'flat_spike_idx', 'Probe_Epoch_id']]]


            # test_spikes_df
            print(f'list(test_spikes_df.columns): {list(test_spikes_df.columns)}')

            # spikes_columns_list = ["t_rel_seconds", "aclu", 'shank', 'cluster', 'aclu', 'qclu', 'x', 'y', 'speed', 'traj', 'lap', 'maze_relative_lap', 'maze_id', 'flat_spike_idx', 'x_loaded', 'y_loaded', 'lin_pos', 'fragile_linear_neuron_IDX', 'PBE_id', 'neuron_IDX', 'Probe_Epoch_id']
            spikes_columns_list = ["t_rel_seconds", "aclu"]
            test_spikes_df = test_spikes_df[spikes_columns_list]
            # Change column type to object for column: 'aclu'
            test_spikes_df = test_spikes_df.astype({'aclu': 'int'}).reset_index(drop=True)
            transition_matrix = TransitionMatrixComputations._compute_time_transition_matrix(df=test_spikes_df, bin_size=0.5) #  bin_size="0.25S"
            transition_matrix

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
    def _generate_expected_replay_sequences(cls, a_binned_x_transition_matrix_higher_order_list: List[NDArray], n_generated_events: int = 10, n_generated_t_bins: int = 4, test_time_bin_size: float = 0.25, transition_matrix_order_start_idx:int=0, debug_print=True, blur_vertical_std_dev=None, blur_horizontal_std_dev=None):
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
        
        ## Euclidian Distance
        return np.sqrt(np.sum(np.power((pos_likelihoods - observed_posterior), 2)))
        # return np.dot(observed_posterior, pos_likelihoods)

    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-02 09:53', related_items=[])
    def _predicted_probabilities_likelihood(cls, a_binned_x_transition_matrix_higher_order_list: List[NDArray], test_posterior: NDArray, transition_matrix_order: int=5) -> NDArray:
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
    def _perform_forward_prediction(cls, a_binned_x_transition_matrix_higher_order_list: List[NDArray], test_posterior: NDArray, transition_matrix_order_start_idx:int=1, transition_matrix_order_growth_factor:float=1.0, debug_print=False):
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
        max_required_transition_matrix_order: int = transition_matrix_order_start_idx + int(round(transition_matrix_order_growth_factor * n_predicted_time_bins)) 
        print(f'np.shape(test_posterior): {np.shape(test_posterior)}, n_time_bins: {n_time_bins}, n_predicted_time_bins: {n_predicted_time_bins}, max_required_transition_matrix_order: {max_required_transition_matrix_order}')
        assert len(a_binned_x_transition_matrix_higher_order_list) > max_required_transition_matrix_order, f"Not enough len(a_binned_x_transition_matrix_higher_order_list): {len(a_binned_x_transition_matrix_higher_order_list)} to meet max_required_transition_matrix_order: {max_required_transition_matrix_order}"
        # Only use the first posterior
        an_observed_posterior = np.atleast_2d(test_posterior[:, 0]).T # (n_x, 1)

        predicited_posteriors = []
        # for a_tbin_idx in np.arange(start=1, stop=n_time_bins):
        for i in np.arange(n_predicted_time_bins):
            a_tbin_idx = i + 1
            # an_observed_posterior = np.atleast_2d(test_posterior[:, i]).T # (n_x, 1)
            # an_actual_observed_next_step_posterior = np.atleast_2d(test_posterior[:, a_tbin_idx]).T # (n_x, 1)
            # an_observed_posterior

            if transition_matrix_order_growth_factor is not None:
                transition_matrix_order_idx: int = transition_matrix_order_start_idx + int(round(transition_matrix_order_growth_factor * a_tbin_idx))
            else:
                ## NOTE: transition_matrix_order does seem to do the identity transformation
                transition_matrix_order_idx: int = transition_matrix_order_start_idx + a_tbin_idx
                # transition_matrix_order: int = transition_matrix_order_idx + 1 ## order is index + 1
                
            if debug_print:
                print(f'\t{i = }, {a_tbin_idx = }, {transition_matrix_order_idx = }')

            ## single time-bin
            a_trans_prob_mat = a_binned_x_transition_matrix_higher_order_list[transition_matrix_order_idx] # (n_x, n_x)
            a_next_t_predicted_pos_probs = a_trans_prob_mat @ an_observed_posterior # (n_x, 1)

            # a_next_t_predicted_pos_probs
            predicited_posteriors.append(a_next_t_predicted_pos_probs)

        predicited_posteriors = np.hstack(predicited_posteriors) # (n_x, n_predicted_time_bins)
        return predicited_posteriors

    @classmethod
    def agreement_with_observation(cls, binned_x_transition_matrix_higher_order_list_dict: List[NDArray], active_decoded_results: Dict[str, DecodedFilterEpochsResult], transition_matrix_order_start_idx:int=1, transition_matrix_order_growth_factor:float=1.0, debug_print=False):
        """ Computes the degree of agreement between predicition and observation for a set of transition matricies
        
        
        """
        # INPUTS: active_decoded_results, binned_x_transition_matrix_higher_order_list_dict, makov_assumed_binned_x_transition_matrix_higher_order_list_dict, expected_velocity_list_dict, transition_matrix_t_bin_size

        # SINGLE DECODER: ____________________________________________________________________________________________________ #
        # single decoder for now
        a_decoder_name = 'long_LR'
        # a_decoder_name = 'long_RL'
        a_binned_x_transition_matrix_higher_order_list: List[NDArray] = deepcopy(binned_x_transition_matrix_higher_order_list_dict[a_decoder_name])
        # single-decoder

        an_epoch_idx: int = 1 #2 ## SINGLE EPOCH
        # markov_order_idx: int = 1 ## SINGLE MARKOV ORDER

        # V = expected_velocity_list_dict['long_LR'][0].in_combined
        # V = np.stack([v.in_combined for v in expected_velocity_list_dict['long_LR']])

        predicited_posteriors_list_dict = {}
        predicited_posteriors_agreement_with_observation_list_dict = {}

        # for a_decoder_name, a_binned_x_transition_matrix_higher_order_list in binned_x_transition_matrix_higher_order_list_dict.items():
        # decoder_laps_filter_epochs_decoder_result_dict[a_decoder_name]

        a_decoded_results: DecodedFilterEpochsResult = active_decoded_results[a_decoder_name]
        # a_train_decoded_results.
        # V = np.stack([v.in_combined for v in a_velocity_tuple_list]) # Stack over orders. This doesn't make much sense actually
        # # V
        # display(np.atleast_2d(V[markov_order_idx]).T)

        # a_decoded_results.validate_time_bins()

        # an_epoch_idx: int # ## SINGLE EPOCH
        test_posterior = deepcopy(a_decoded_results.p_x_given_n_list[an_epoch_idx])
        print(f'np.shape(test_posterior): {np.shape(test_posterior)}')

        # ## Traditional
        # predicited_posteriors: NDArray = TransitionMatrixComputations._perform_forward_prediction(a_binned_x_transition_matrix_higher_order_list, test_posterior=test_posterior, transition_matrix_order_growth_factor=3.0) # (n_pos_bins, (n_time_bins - 1))

        # growth_order_factors = np.linspace(start=1.0, stop=3.0, num=5)
        # several_predicited_posteriors: NDArray = [TransitionMatrixComputations._perform_forward_prediction(a_binned_x_transition_matrix_higher_order_list, test_posterior=test_posterior, transition_matrix_order_growth_factor=G) for G in growth_order_factors] # (n_pos_bins, (n_time_bins - 1))

        # agreement_with_observation = [TransitionMatrixComputations._likelihood_of_observation(test_posterior[:, (a_predicted_timestamp_index + 1)], a_predicted_posterior) for a_predicted_timestamp_index, a_predicted_posterior in enumerate(predicited_posteriors.T)]
        # agreement_with_observation = np.array(agreement_with_observation)

        # ==================================================================================================================== #
        # Expanding `_perform_forward_prediction` because it seems to be returning almost entirely the same things:            #
        # ==================================================================================================================== #

        debug_print = False
        transition_matrix_order_start_idx = 1
        transition_matrix_order_growth_factor = 1.0

        ## INPUTS: a_binned_x_transition_matrix_higher_order_list, test_posterior, transition_matrix_order_start_idx, transition_matrix_order_growth_factor
        n_time_bins: int = np.shape(test_posterior)[1]
        n_predicted_time_bins: int = n_time_bins - 1
        if debug_print:
            print(f'np.shape(test_posterior): {np.shape(test_posterior)}, n_time_bins: {n_time_bins}, n_predicted_time_bins: {n_predicted_time_bins}')

        # # Only use the first posterior
        # an_observed_posterior = np.atleast_2d(test_posterior[:, 0]).T # (n_x, 1)
        # n_predicted_time_bins: int = 5 # arrbitrary number of time bins to predict in the future

        specific_observation_t_bin_indicies = np.arange(n_time_bins-1) # all time bins except the last
        all_t_bins_predicited_corresponding_t_bins = []
        all_t_bins_predicited_posteriors = []
        all_t_bins_agreement_with_observation = []

        should_empty_fill_outputs = True # if `should_empty_fill_outputs == True`, all outputs will be the same size with np.nan values filling the "missing" entries.

        for a_specific_t_bin_idx in specific_observation_t_bin_indicies:
            # # Only use a specific posterior
            an_observed_posterior = np.atleast_2d(test_posterior[:, a_specific_t_bin_idx]).T # (n_x, 1)
            n_predicted_time_bins: int = ((n_time_bins-a_specific_t_bin_idx) - 1) # number of time bins to predict computed by how many remain in the current epoch after the specific_t_bin_idx (in the future)

            # """ Takes a single observation time bin and returns its best prediction for all future bins

            # """
            ## INPUTS: n_predicted_time_bins, an_observed_posterior
            an_observed_posterior = np.atleast_2d(an_observed_posterior) # (n_x, 1)
            if np.shape(an_observed_posterior)[-1] != 1:
                # last dimension should be 1 (n_x, 1), transpose it
                an_observed_posterior = an_observed_posterior.T
                
            n_xbins: int = np.shape(an_observed_posterior)[0]
            if debug_print:
                print(f'Single observation at a_specific_t_bin_idx = {a_specific_t_bin_idx}:\n\tnp.shape(an_observed_posterior): {np.shape(an_observed_posterior)}, n_predicted_time_bins: {n_predicted_time_bins}')

            predicited_posteriors = []
            # for a_tbin_idx in np.arange(start=1, stop=n_time_bins):
            for i in np.arange(n_predicted_time_bins):
                a_start_rel_tbin_idx = i + 1 # a_start_rel_tbin_idx: relative to the current observation start t_bin (a_start_rel_tbin_idx == 0 at the observation time)
                
                # an_observed_posterior = np.atleast_2d(test_posterior[:, i]).T # (n_x, 1)
                # an_actual_observed_next_step_posterior = np.atleast_2d(test_posterior[:, a_tbin_idx]).T # (n_x, 1)

                if transition_matrix_order_growth_factor is not None:
                    transition_matrix_order_idx: int = transition_matrix_order_start_idx + int(round(transition_matrix_order_growth_factor * a_start_rel_tbin_idx))
                else:
                    ## NOTE: transition_matrix_order does seem to do the identity transformation
                    transition_matrix_order_idx: int = transition_matrix_order_start_idx + a_start_rel_tbin_idx
                    # transition_matrix_order: int = transition_matrix_order_idx + 1 ## order is index + 1
                    
                if debug_print:
                    print(f'\t{i = }, {a_start_rel_tbin_idx = }, {transition_matrix_order_idx = }')
                    
                ## single time-bin
                a_trans_prob_mat = a_binned_x_transition_matrix_higher_order_list[transition_matrix_order_idx] # (n_x, n_x)
                a_next_t_predicted_pos_probs = a_trans_prob_mat @ an_observed_posterior # (n_x, 1)

                predicited_posteriors.append(a_next_t_predicted_pos_probs)

            predicited_posteriors = np.hstack(predicited_posteriors) # (n_x, n_predicted_time_bins)
            # ## OUTPUTS: predicited_posteriors
            # predicited_posteriors_list_dict[a_decoder_name] = predicited_posteriors

            ## Compute the agreement of this time bin with observation:
            agreement_with_observation = []
            for a_predicted_timestamp_index, a_predicted_posterior in enumerate(predicited_posteriors.T):
                a_predicted_timestamp = a_predicted_timestamp_index + 1
                # a_predicted_posterior = np.atleast_2d(a_predicted_posterior).T
                an_observed_posterior_single_t_bin = test_posterior[:, a_predicted_timestamp]
                if debug_print:
                    print(f'\t {a_predicted_timestamp} - np.shape(an_observed_posterior_single_t_bin): {np.shape(an_observed_posterior_single_t_bin)}, np.shape(a_predicted_posterior): {np.shape(a_predicted_posterior)}')
                agreement_with_observation.append(TransitionMatrixComputations._likelihood_of_observation(an_observed_posterior_single_t_bin, a_predicted_posterior))
            agreement_with_observation = np.array(agreement_with_observation)
            if debug_print:
                print(agreement_with_observation)

            ## Build outputs:
            if should_empty_fill_outputs:
                n_fill_bins: int = a_specific_t_bin_idx+1
                pre_prediction_start_timebins = np.arange(n_fill_bins) # observation is at index `a_specific_t_bin_idx`, so to get the full range before the first prediction (inclusive) we need to do +1
                fill_arr = np.full((n_xbins, n_fill_bins), fill_value=np.nan)
                all_t_bins_predicited_corresponding_t_bins.append(np.arange(n_predicted_time_bins + n_fill_bins))
                predicited_posteriors = np.concatenate((fill_arr, predicited_posteriors), axis=-1)
                agreement_with_observation = np.concatenate((np.full((n_fill_bins,), fill_value=np.nan), agreement_with_observation))
                if debug_print:
                    print(f'\tnp.shape(agreement_with_observation): {np.shape(agreement_with_observation)}')
            else:
                all_t_bins_predicited_corresponding_t_bins.append((np.arange(n_predicted_time_bins)+a_specific_t_bin_idx))
            if debug_print:    
                print(f'\tnp.shape(predicited_posteriors): {np.shape(predicited_posteriors)}')
            all_t_bins_predicited_posteriors.append(predicited_posteriors)
            
            ## OUTPUTS: agreement_with_observation
            all_t_bins_agreement_with_observation.append(agreement_with_observation)
            

            # Aggregate over all predictors we have for the next timestep
            

            # END Expanding `_perform_forward_prediction` ________________________________________________________________________________________________________________ #


            ## only for this epoch:
            # predicited_posteriors_agreement_with_observation_list_dict[a_decoder_name] = agreement_with_observation

            # agreement_with_observation

            # # # Compare real:
            # ## Observed posteriors:
            # observed_posteriors = deepcopy(test_posterior[:, 1:])

            # assert np.shape(observed_posteriors) == np.shape(predicited_posteriors)
            # _prediction_observation_diff = observed_posteriors - predicited_posteriors
            # _prediction_observation_diff


        ## Construct into 3D arrays for visualization in Napari and future operations
        ## INPUTS: all_t_bins_predicited_corresponding_t_bins, all_t_bins_predicited_posteriors, all_t_bins_agreement_with_observation
        all_t_bins_predicited_posteriors: NDArray = np.stack(all_t_bins_predicited_posteriors) # (n_predicted_t_bins, n_x_bins, n_t_bins), single posterior (n_x_bins, n_t_bins)
        all_t_bins_agreement_with_observation: NDArray = np.stack(all_t_bins_agreement_with_observation) # (n_predicted_t_bins, n_t_bins)
        all_t_bins_predicited_corresponding_t_bins: NDArray = np.stack(all_t_bins_predicited_corresponding_t_bins) # (n_predicted_t_bins, n_t_bins)

        ## OUTPUTS: all_t_bins_predicited_posteriors, all_t_bins_agreement_with_observation, all_t_bins_predicited_corresponding_t_bins
        # all_t_bins_predicited_posteriors.shape: (30, 57, 31) (n_t_prediction_bins, n_x_bins, n_t_bins)
        # test_posterior.shape: (57, 31)

        n_past_observations: NDArray = np.sum(np.logical_not(np.isnan(all_t_bins_agreement_with_observation)), axis=0) # (n_t_bins,)  -- sum over each predicted time bin to see how many valid past predictions we have # e.g. [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
        ## aggregate over all possible predictions to determine the observed likelihood for each real time bin
        collided_across_predictions_observation_likelihoods = np.nansum(all_t_bins_agreement_with_observation, axis=0) # (n_t_bins,)
        # collided_across_predictions_observation_likelihoods

        ## Average across all prediction timesteps and renormalize
        # all_t_bins_predicited_posteriors # TypeError: Invalid shape (30, 57, 31) for image data

        _arr: NDArray = np.nansum(deepcopy(all_t_bins_predicited_posteriors), axis=0) # (n_x_bins, n_t_bins)
        ## Divide by the number of observations to prevent over-representation:
        _arr = _arr / n_past_observations

        ## re normalize over first position
        _arr = _arr / np.nansum(_arr, axis=0, keepdims=True)
        # _arr

        # INPUTS: n_past_observations

        # `collided_across_prediction_position_likelihoods` - this isn't "fair" because it includes the predictions from t+1 timesteps which are very close to the actual observed positions. Mostly exploring how to combine across time bin lags.
        # collided_across_prediction_position_likelihoods = np.nansum(all_t_bins_predicited_posteriors, axis=-1) # (n_t_prediction_bins, n_x_bins)
        collided_across_prediction_position_likelihoods = np.nansum(all_t_bins_predicited_posteriors, axis=0) # (n_t_prediction_bins, n_x_bins)
        return (all_t_bins_predicited_posteriors, all_t_bins_agreement_with_observation, all_t_bins_predicited_corresponding_t_bins), collided_across_prediction_position_likelihoods




    # ==================================================================================================================== #
    # Plot/Display                                                                                                         #
    # ==================================================================================================================== #
    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix', 'plot'], input_requires=[], output_provides=[], uses=['BasicBinnedImageRenderingWindow'], used_by=[], creation_date='2024-08-02 09:55', related_items=[])
    def plot_transition_matricies(cls, decoders_dict: Dict[types.DecoderName, BasePositionDecoder], binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray],
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
    @function_attributes(short_name=None, tags=['transition_matrix', 'save', 'export'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=[])
    def save_transition_matricies(cls, binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray], save_path:Path='transition_matrix_data.h5', out_context=None, debug_print=False): # decoders_dict: Dict[types.DecoderName, BasePositionDecoder], 
        """Save the transitiion matrix info to a file
        
        _save_context: IdentifyingContext = curr_active_pipeline.build_display_context_for_session('save_transition_matricies')
        _save_path = TransitionMatrixComputations.save_transition_matricies(binned_x_transition_matrix_higher_order_list_dict=binned_x_transition_matrix_higher_order_list_dict, out_context=_save_context, save_path='output/transition_matrix_data.h5')
        _save_path

        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path).resolve()
        
        import h5py
        # Save to .h5 file
        with h5py.File(save_path, 'w') as f:
            if out_context is not None:
                # add context to the file
                if not isinstance(out_context, dict):
                    flat_context_desc: str = out_context.get_description(separator='|') # 'kdiba|gor01|one|2006-6-08_14-26-15|save_transition_matricies'
                    _out_context_dict = out_context.to_dict() | {'session_context_desc': flat_context_desc}
                else:
                    # it is a raw dict
                    _out_context_dict = deepcopy(out_context)
                    
                for k, v in _out_context_dict.items():
                    ## add the context as file-level metadata
                    f.attrs[k] = v



            for a_name, an_array_list in binned_x_transition_matrix_higher_order_list_dict.items():
                decoder_prefix: str = a_name
                if isinstance(an_array_list, NDArray):
                    ## 3D NDArray version:
                    assert np.ndim(an_array_list) == 3, f"np.ndim(an_array_list): {np.ndim(an_array_list)}, np.shape(an_array_list): {np.shape(an_array_list)}"
                    n_markov_orders, n_xbins, n_xbins2 = np.shape(an_array_list)
                    assert n_xbins == n_xbins2, f"n_xbins: {n_xbins} != n_xbins2: {n_xbins2}" 
                    a_dset = f.create_dataset(f'{decoder_prefix}/binned_x_transition_matrix_higher_order_mat', data=an_array_list)
                    a_dset.attrs['decoder_name'] = a_name
                    a_dset.attrs['max_markov_order'] = n_markov_orders
                    a_dset.attrs['n_xbins'] = n_xbins

                else:
                    # list
                    assert isinstance(an_array_list, (list, tuple)), f"type(an_array_list): {type(an_array_list)}\nan_array_list: {an_array_list}\n"
                    # Determine how much zero padding is needed so that the array entries sort correctly
                    max_markov_order: int = np.max([len(an_array_list) for an_array_list in binned_x_transition_matrix_higher_order_list_dict.values()])
                    if debug_print:
                        print(f'max_markov_order: {max_markov_order}')
                    padding_length: int = len(str(max_markov_order)) + 1 ## how long is the string?
                    if debug_print:
                        print(f'padding_length: {padding_length}')
                        
                    for markov_order, array in enumerate(an_array_list):
                        _markov_order_str: str = f"{markov_order:0{padding_length}d}" # Determine how much zero padding is needed so that the array entries sort correctly
                        a_dset = f.create_dataset(f'{decoder_prefix}/array_{_markov_order_str}', data=array)
                        a_dset.attrs['decoder_name'] = a_name
                        a_dset.attrs['markov_order'] = markov_order
                    
                                
                # Add metadata
                f.attrs['decoder_name'] = a_name

            

        return save_path
    

    @classmethod
    @function_attributes(short_name=None, tags=['transition_matrix', 'load'], input_requires=[], output_provides=[], uses=['h5py'], used_by=[], creation_date='2024-08-05 10:47', related_items=[])
    def load_transition_matrices(cls, load_path: Path) -> DecoderListDict[NDArray]:
        """
        Load the transition matrix info from a file
        
        load_path = Path('output/transition_matrix_data.h5')
        binned_x_transition_matrix_higher_order_list_dict = TransitionMatrixComputations.load_transition_matrices(load_path)
        binned_x_transition_matrix_higher_order_list_dict
        """
        if not isinstance(load_path, Path):
            load_path = Path(load_path).resolve()

        import h5py
        binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray] = {}

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


    # ==================================================================================================================== #
    # Sampling Sequences                                                                                                   #
    # ==================================================================================================================== #
    @classmethod
    def sample_sequences(cls, transition_matrix_mat: NDArray, sequence_length: int, num_sequences: int, initial_P_x: Optional[NDArray]=None, initial_state: Optional[int]=None) -> Tuple[NDArray, NDArray, int]:
        """ Generates sample sequences given a transition_matrix_mat of order O

        USage 1:
        
            T_mat = deepcopy(binned_x_transition_matrix_higher_order_mat_dict['long_LR']) # (200, 57, 57)
        
            initial_state = 0  # Starting from state 0
            sequence_length = 10  # Length of each sequence
            num_sequences = 1000  # Number of sequences to generate        
            sequences, sequence_likelihoods, num_states = TransitionMatrixComputations.sample_sequences(T_mat.copy(), initial_state=10, sequence_length=sequence_length, num_sequences=num_sequences) # (1000, 10)
            sequences
        
        Usage 2:
        
            T_mat = deepcopy(binned_x_transition_matrix_higher_order_mat_dict['long_LR']) # (200, 57, 57)
            probability_normalized_occupancy = deepcopy(decoders_dict['long_LR'].pf.probability_normalized_occupancy) # BasePositionDecoder

            sequence_length = 10  # Length of each sequence
            num_sequences = 1000  # Number of sequences to generate
            sequences, sequence_likelihoods, num_states = TransitionMatrixComputations.sample_sequences(T_mat.copy(), sequence_length=sequence_length, num_sequences=num_sequences, initial_P_x=probability_normalized_occupancy) # (1000, 10)
            sequences


        """
        n_orders, n_x_bins, _n_x_bins2 = np.shape(transition_matrix_mat)
        assert n_x_bins == _n_x_bins2
        assert n_orders > sequence_length
        
        num_states = n_x_bins
        sequences = np.zeros((num_sequences, sequence_length), dtype=int)
        sequence_likelihoods = np.zeros((num_sequences, sequence_length))
        

        for i in range(num_sequences):
            ## Start from the initial constrained sequences:
            if initial_state is None:
                assert initial_P_x is not None
                assert len(initial_P_x) == num_states
                current_state = np.random.choice(num_states, p=initial_P_x)
                current_state_likelihood = initial_P_x[current_state]
                a_sequence = [current_state]
                a_sequence_probability = [current_state_likelihood]
                
            else:
                assert initial_P_x is None, "initial_P_x will not be used if initial_state is provided!"
                current_state = initial_state
                a_sequence = [current_state]
                a_sequence_probability = [1.0] # specified
                
            ## Begin sequence generation:
            len_initial_constrained_sequence: int = len(a_sequence) # 1

            # ## (Initial Position , Increasing Order) Dependent:
            # fixed_initial_state = current_state ## capture the initial state
            # for an_order in range(sequence_length - len_initial_constrained_sequence):
            #     next_state = np.random.choice(num_states, p = transition_matrix_mat[an_order, fixed_initial_state, :]) # always `initial_state` with increasing timesteps
            #     next_state_likelihood = transition_matrix_mat[an_order, fixed_initial_state, next_state] # always `initial_state`
            #     a_sequence.append(next_state)
            #     a_sequence_probability.append(next_state_likelihood)


            # (Fixed Order, Previous Timestep's Position) Dependent:
            for an_order in range(sequence_length - len_initial_constrained_sequence):
                next_state = np.random.choice(num_states, p = transition_matrix_mat[0, current_state, :])
                next_state_likelihood = transition_matrix_mat[0, current_state, next_state]
                a_sequence.append(next_state)
                a_sequence_probability.append(next_state_likelihood)
                current_state = next_state
                

            # # TODO: (All time lags, All Positions) Dependent:
            # for a_max_order in range(sequence_length - len_initial_constrained_sequence):
            #     a_sub_sequence = []
            #     a_sub_sequence_likelihoods = []
                
            #     for prev_t_idx, an_order in np.arange(a_max_order):
            #         next_state = np.random.choice(num_states, p = transition_matrix_mat[an_order, current_state, :])
            #         next_state_likelihood = transition_matrix_mat[an_order, current_state, next_state]
            #         a_sub_sequence.append(next_state)
            #         a_sub_sequence_likelihoods.append(next_state_likelihood)
            #         current_state = next_state
                    
            #     a_sequence.append(a_sub_sequence)
            #     a_sequence_probability.append(a_sub_sequence_likelihoods)

            # for an_order in range(sequence_length - len_initial_constrained_sequence):
            #     next_state = np.random.choice(num_states, p = transition_matrix_mat[an_order, current_state, :])
            #     next_state_likelihood = transition_matrix_mat[an_order, current_state, next_state]
            #     a_sequence.append(next_state)
            #     a_sequence_probability.append(next_state_likelihood)
            #     current_state = next_state
                
            sequences[i] = a_sequence # append to the output array
            sequence_likelihoods[i] = a_sequence_probability
            
        return sequences, sequence_likelihoods, num_states
        

    @classmethod
    def sample_sequences_stationary(cls, transition_matrix: NDArray, initial_state: int, sequence_length: int, num_sequences: int) -> NDArray: 
        """
        
        # Example usage:
        # transition_matrix = np.array([[0.1, 0.9], [0.5, 0.5]])  # Example transition matrix
        transition_matrix = T_mat[0, :, :].copy()

        initial_state = 0  # Starting from state 0
        sequence_length = 10  # Length of each sequence
        num_sequences = 1000  # Number of sequences to generate

        sequences = TransitionMatrixComputations.sample_sequences_stationary(transition_matrix, initial_state=initial_state, sequence_length=sequence_length, num_sequences=num_sequences) # (1000, 10)
        sequences

        """
        num_states = transition_matrix.shape[0]
        sequences = np.zeros((num_sequences, sequence_length), dtype=int)

        for i in range(num_sequences):
            current_state = initial_state
            sequence = [current_state]
            for _ in range(sequence_length - 1):
                next_state = np.random.choice(num_states, p = transition_matrix[current_state])
                sequence.append(next_state)
                current_state = next_state
            sequences[i] = sequence

        return sequences


    @classmethod
    def _sequences_index_list_to_matrix_occupancy(cls, sequences: NDArray, num_states:int) -> NDArray:
        """Turn `sequences`, a list of x_bin indicies, into a flattened matrix occupancy representation for visualization of the path frequency
        
        sequence_frames_occupancy = TransitionMatrixComputations._sequences_index_list_to_matrix_occupancy(sequences=sequences, num_states=num_states)
        sequence_frames_occupancy

        """
        num_sequences, sequence_length = np.shape(sequences)

        sequences_mat = np.zeros((num_states, sequence_length))
        for sample_idx in np.arange(num_sequences):
            # sequences_mat[sequences[i, :]] += 1
            for t in np.arange(sequence_length):
                # sequences_mat[sequences[sample_idx, t], t] += 1
                # for x_idx in np.arange(num_states):0
                sequences_mat[sequences[sample_idx, t], t] += 1
            
        return sequences_mat



    
    @classmethod
    def _sequences_index_list_to_sparse_matrix_stack(cls, sequences: NDArray, num_states:int) -> List[csr_matrix]:
        """Turn `sequences`, a list of x_bin indicies, into a flattened matrix occupancy representation for visualization of the path frequency
        
        Usage:
            sequence_frames_sparse = TransitionMatrixComputations._sequences_index_list_to_sparse_matrix_stack(sequences=sequences, num_states=num_states)
            sequence_frames_sparse

        """
        num_sequences, sequence_length = np.shape(sequences)

        sequence_frames_list = []
        for sample_idx in np.arange(num_sequences):
            a_sequence_mat = np.zeros((num_states, sequence_length))
            for t in np.arange(sequence_length):
                a_sequence_mat[sequences[sample_idx, t], t] = 1
            # end sequence
            sequence_frames_list.append(csr_matrix(a_sequence_mat))
            
        # seq_indicies = np.arange(num_sequences)
        # seq_t_indicies = np.arange(sequence_length)
        # col_indices = np.arange(num_states)

        # coo = coo_matrix((sequences, (seq_indicies, seq_t_indicies, col_indices)), shape=(num_sequences, sequence_length, num_states))
        # return coo.tocsr()
        return sequence_frames_list



    @classmethod
    def compute_arbitrary_sequence_likelihood(cls, transition_matrix_mat: NDArray, initial_P_x: NDArray, test_posterior_most_likely_index_sequence: NDArray):
        """ Generates sample sequences given a transition_matrix_mat of order O

        USage 1:
        
            T_mat = deepcopy(binned_x_transition_matrix_higher_order_mat_dict['long_LR']) # (200, 57, 57)
            probability_normalized_occupancy = deepcopy(decoders_dict['long_LR'].pf.probability_normalized_occupancy) # BasePositionDecoder
            sequence_likelihood, num_states = TransitionMatrixComputations.compute_arbitrary_sequence_likelihood(T_mat.copy(), initial_P_x=probability_normalized_occupancy, test_posterior_most_likely_index_sequence=test_posterior_most_likely_index_sequence)
            sequence_likelihood
        
        Usage 2:
        
            T_mat = deepcopy(binned_x_transition_matrix_higher_order_mat_dict['long_LR']) # (200, 57, 57)
            probability_normalized_occupancy = deepcopy(decoders_dict['long_LR'].pf.probability_normalized_occupancy) # BasePositionDecoder

            sequence_length = 10  # Length of each sequence
            num_sequences = 1000  # Number of sequences to generate
            sequences, sequence_likelihoods, num_states = TransitionMatrixComputations.sample_sequences(T_mat.copy(), sequence_length=sequence_length, num_sequences=num_sequences, initial_P_x=probability_normalized_occupancy) # (1000, 10)
            sequences


        """
        sequence_length: int = len(test_posterior_most_likely_index_sequence)
        n_orders, n_x_bins, _n_x_bins2 = np.shape(transition_matrix_mat)
        assert n_x_bins == _n_x_bins2
        assert n_orders > sequence_length
        
        num_states = n_x_bins
        sequence_likelihood = np.zeros((sequence_length, ))
        sequence_bin_maximal_likelihood = np.zeros((sequence_length, ))
        
        ## Start from the initial constrained sequences:
        assert initial_P_x is not None
        assert len(initial_P_x) == num_states
        
        for sequence_i in np.arange(sequence_length):
            current_state = test_posterior_most_likely_index_sequence[sequence_i]
            
            if (sequence_i == 0):            
                # sequence_i: int = 0    
                current_state_likelihood = initial_P_x[current_state]
                # a_sequence = [current_state]
                # a_sequence_probability = [current_state_likelihood] # list-style
                sequence_likelihood[0] = current_state_likelihood
            else:
                ## Test sequence
                an_order: int = (sequence_i-1) # (sequence_i-1): markov order
                # len_initial_constrained_sequence: int = 1 # len(a_sequence) # 1

                # ## (Initial Position , Increasing Order) Dependent:
                # sequence_likelihood[sequence_i] = transition_matrix_mat[an_order, test_posterior_most_likely_index_sequence[0], current_state] 
                
                # (Fixed Order, Previous Timestep's Position) Dependent:
                sequence_likelihood[sequence_i] = transition_matrix_mat[0, test_posterior_most_likely_index_sequence[sequence_i-1], current_state] # (sequence_i-1): previous timestep's state
                ## get the most likely likelihood:
                maximally_likely_likelihood = np.nanmax(transition_matrix_mat[0, test_posterior_most_likely_index_sequence[sequence_i-1], :], axis=-1) # most likely state
                sequence_bin_maximal_likelihood[sequence_i] = maximally_likely_likelihood

                # # TODO: (All time lags, All Positions) Dependent:
                # for a_max_order in range(sequence_length - len_initial_constrained_sequence):
                #     a_sub_sequence = []
                #     a_sub_sequence_likelihoods = []
                    
                #     for prev_t_idx, an_order in np.arange(a_max_order):
                #         next_state = np.random.choice(num_states, p = transition_matrix_mat[an_order, current_state, :])
                #         next_state_likelihood = transition_matrix_mat[an_order, current_state, next_state]
                #         a_sub_sequence.append(next_state)
                #         a_sub_sequence_likelihoods.append(next_state_likelihood)
                #         current_state = next_state
                        
                #     a_sequence.append(a_sub_sequence)
                #     a_sequence_probability.append(a_sub_sequence_likelihoods)

                # for an_order in range(sequence_length - len_initial_constrained_sequence):
                #     next_state = np.random.choice(num_states, p = transition_matrix_mat[an_order, current_state, :])
                #     next_state_likelihood = transition_matrix_mat[an_order, current_state, next_state]
                #     a_sequence.append(next_state)
                #     a_sequence_probability.append(next_state_likelihood)
                #     current_state = next_state
                
        return sequence_likelihood, sequence_bin_maximal_likelihood, num_states, sequence_length
       


    # ==================================================================================================================== #
    # Expected Position/Velocity                                                                                           #
    # ==================================================================================================================== #
    @classmethod
    def _compute_expected_velocity_list_dict(cls, binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray]) -> Dict[types.DecoderName, List[ExpectedVelocityTuple]]:
        """ working expected velocity for each transition matrix.
        
        """
        raise NotImplementedError
    


    @classmethod
    def compute_expected_positions(cls, binned_x_transition_matrix_higher_order_mat_dict, decoders_dict):
        """ working expected positions <x> in real position coordinates
         
        """
                
        _expected_x_dict = {}
        for a_decoder_name, arr in binned_x_transition_matrix_higher_order_mat_dict.items():
            
            arr = deepcopy(arr)
            # decoders_dict['long_LR'].xbin
            xbin_centers = deepcopy(decoders_dict[a_decoder_name].xbin_centers) # (57, )
            n_x_bins = len(xbin_centers)
            # xbin_centers
            # n_x_bins

            _expected_x = []
            
            for row_i in np.arange(n_x_bins):
                # arr = binned_x_transition_matrix_higher_order_mat_dict['long_LR']
                _expected_val = arr[:,row_i,:] @ xbin_centers # (200, 57) @ (57, ) = (200, )
                _expected_x.append(_expected_val)
                

            _expected_x = np.stack(_expected_x).T # (57, 200)
            _expected_x_dict[a_decoder_name] = _expected_x
            # _expected_x    
            # binned_x_transition_matrix_higher_order_mat_dict['long_LR'] * xbin_centers
            
        ## OUTPUT: _expected_x_dict
        return _expected_x_dict



    @classmethod
    def _compute_expected_velocity_out_per_node(cls, A: np.ndarray, should_split_fwd_and_bkwd_velocities: bool = False, velocity_type: Union[VelocityType, str] = VelocityType.OUTGOING) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """ 
        Compute the expected velocity in/out per node (position bin) of the transition matrix.

        Parameters:
        A (np.ndarray): A square transition matrix where A[i][j] represents the transition rate from node i to node j.
        should_split_fwd_and_bkwd_velocities (bool): Flag to determine whether to return separate forward and backward velocities.

        Returns:
        Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
            - If should_split_fwd_and_bkwd_velocities is False: returns the combined expected velocity as a numpy array.
            - If should_split_fwd_and_bkwd_velocities is True: returns a tuple containing:
                - combined_expected_velocity (np.ndarray): Combined expected velocity as a numpy array.
                - (fwd_expected_velocity, bkwd_expected_velocity): A tuple of numpy arrays containing forward and backward expected velocities respectively.
                

        Usage:    
            
            combined_expected_velocity = TransitionMatrixComputations._compute_expected_velocity_out_per_node(A)
            combined_expected_velocity
        Example 2:
            combined_expected_incoming_velocity, (fwd_expected_incoming_velocity, bkwd_expected_incoming_velocity) = TransitionMatrixComputations._compute_expected_velocity_out_per_node(A, should_split_fwd_and_bkwd_velocities=True, velocity_type='in')
            fwd_expected_incoming_velocity
            bkwd_expected_incoming_velocity

            combined_expected_out_velocity, (fwd_expected_out_velocity, bkwd_expected_out_velocity) = TransitionMatrixComputations._compute_expected_velocity_out_per_node(A, should_split_fwd_and_bkwd_velocities=True, velocity_type='out')
            fwd_expected_out_velocity
            bkwd_expected_out_velocity

        """
        num_states = np.shape(A)[0]
        assert np.shape(A)[0] == np.shape(A)[1], "must be a square matrix"
        
        if isinstance(velocity_type, str):
            velocity_type = VelocityType(velocity_type)
            
        if velocity_type.name == VelocityType.INCOMING.name:
            # compute incoming instead by transposing the transition matrix
            A = deepcopy(A).T
            ## NOTE: fwd_expected_velocity and bkwd_expected_velocity will be swapped and will need to be exchanged after computation
            
        fwd_expected_velocity = []
        bkwd_expected_velocity = []
        combined_expected_velocity = []

        for i in np.arange(num_states):
            _curr_node_fwd_vel = []
            _curr_node_bkwd_vel = []
            _curr_node_combined_vel = []
            for j in np.arange(num_states):
                rate = A[i][j]
                distance_n_xbins = j - i # distance from current node     
                if distance_n_xbins > 0:
                    _curr_node_fwd_vel.append(rate * distance_n_xbins)
                elif distance_n_xbins < 0:
                    _curr_node_bkwd_vel.append(rate * abs(distance_n_xbins))
                _curr_node_combined_vel.append(rate * distance_n_xbins)
            
            _curr_node_fwd_vel = np.nansum(np.array(_curr_node_fwd_vel)) # sum over all forward terms
            _curr_node_bkwd_vel = np.nansum(np.array(_curr_node_bkwd_vel)) # sum over all backward terms
            _curr_node_combined_vel = np.nansum(np.array(_curr_node_combined_vel)) # sum over all terms

            fwd_expected_velocity.append(_curr_node_fwd_vel)
            bkwd_expected_velocity.append(_curr_node_bkwd_vel)
            combined_expected_velocity.append(_curr_node_combined_vel)
        
        fwd_expected_velocity = np.array(fwd_expected_velocity)
        bkwd_expected_velocity = np.array(bkwd_expected_velocity)
        combined_expected_velocity = np.array(combined_expected_velocity)
        
        if velocity_type.name == VelocityType.INCOMING.name:
            # swap "fwd"/"bkwd" velocity so they're correct relative to the track for incoming velocities as well
            _tmp_expected_velocity = deepcopy(bkwd_expected_velocity)
            bkwd_expected_velocity = deepcopy(fwd_expected_velocity)
            fwd_expected_velocity = _tmp_expected_velocity

        if should_split_fwd_and_bkwd_velocities:
            return combined_expected_velocity, (fwd_expected_velocity, bkwd_expected_velocity)
        else:
            return combined_expected_velocity
        

    @classmethod
    def _compute_expected_velocity_list_dict(cls, binned_x_transition_matrix_higher_order_list_dict: DecoderListDict[NDArray]) -> Dict[types.DecoderName, List[ExpectedVelocityTuple]]:
        """ working expected velocity for each transition matrix.
        
        """
        expected_velocity_list_dict: Dict[types.DecoderName, List[ExpectedVelocityTuple]] = {}

        for a_decoder_name, a_transition_mat_list in binned_x_transition_matrix_higher_order_list_dict.items():
            expected_velocity_list_dict[a_decoder_name] = []
            for markov_order_idx, A in enumerate(a_transition_mat_list):
                markov_order = markov_order_idx + 1
                combined_expected_incoming_velocity, (fwd_expected_incoming_velocity, bkwd_expected_incoming_velocity) = cls._compute_expected_velocity_out_per_node(A, should_split_fwd_and_bkwd_velocities=True, velocity_type='in')
                # expected_velocity_list_dict[a_decoder_name].append(fwd_expected_incoming_velocity,
                combined_expected_out_velocity, (fwd_expected_out_velocity, bkwd_expected_out_velocity) = cls._compute_expected_velocity_out_per_node(A, should_split_fwd_and_bkwd_velocities=True, velocity_type='out')

                # the order is O number of timesteps away
                if markov_order > 1:
                    ## #TODO 2024-08-07 11:12: - [ ] Not confidant about whether to divide by the number of distant timestamps or not
                    combined_expected_incoming_velocity /= markov_order
                    fwd_expected_incoming_velocity /= markov_order
                    bkwd_expected_incoming_velocity /= markov_order
                    combined_expected_out_velocity /= markov_order
                    fwd_expected_out_velocity /= markov_order
                    bkwd_expected_out_velocity /= markov_order
                    
                expected_velocity_list_dict[a_decoder_name].append(
                    ExpectedVelocityTuple(in_fwd=fwd_expected_incoming_velocity, in_bkwd=bkwd_expected_incoming_velocity, in_combined=combined_expected_incoming_velocity,
                                    out_fwd=fwd_expected_out_velocity, out_bkwd=bkwd_expected_out_velocity, out_combined=combined_expected_out_velocity)
                )
        return expected_velocity_list_dict
