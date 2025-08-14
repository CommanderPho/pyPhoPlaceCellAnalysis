from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
# ==================================================================================================================== #
# 2023-11-14 - Transition Matrix                                                                                       #
# ==================================================================================================================== #
from copy import deepcopy
from pathlib import Path
from enum import Enum
from neuropy.analyses.decoders import BinningContainer
from neuropy.utils.result_context import IdentifyingContext
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias  # "from typing_extensions" in Python 3.9 and earlier
import nptyping as ND
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
    from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult


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




    @function_attributes(short_name=None, tags=['testing', 'transition-matrix', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-04-22 01:08', related_items=[])
    @classmethod
    def estimate_transition_matrix_weighted_avg(cls, state_probs, epsilon=1e-12) -> NDArray:
        """
        Estimates the transition matrix using weighted averaging based on state probabilities.

        Args:
            state_probs (np.ndarray): Array of shape (N, num_states) where N is the
                                    number of time bins and num_states is the number
                                    of states (e.g., 2). state_probs[t, i] is the
                                    probability of being in state i at time t.
                                    Rows should sum to 1.
            epsilon (float): Small value to add to denominator to avoid division by zero.

        Returns:
            np.ndarray: Estimated transition matrix of shape (num_states, num_states).
                        T[i, j] is the probability of transitioning from state i to state j.
                        
        Usage:
        
            transition_matrix: NDArray = TransitionMatrixComputations.estimate_transition_matrix_weighted_avg(state_probs=a_p_x_given_n)
            
        """
        if not np.allclose(np.sum(state_probs, axis=1), 1.0):
            raise ValueError("Input probabilities for each time bin must sum to 1.")

        num_time_bins, num_states = state_probs.shape

        if num_time_bins < 2:
            raise ValueError("Need at least 2 time bins to estimate transitions.")

        # Initialize transition matrix
        transition_matrix = np.zeros((num_states, num_states))
        
        # For each time step t, compute the outer product of probabilities at t and t+1
        for t in range(num_time_bins - 1):
            p_t = state_probs[t, :]
            p_t_plus_1 = state_probs[t + 1, :]
            
            # Outer product gives transition probabilities weighted by current state probabilities
            transition_matrix += np.outer(p_t, p_t_plus_1)
            
        # # Probabilities at time t (from states)
        # probs_t = state_probs[:-1, :] # Shape (N-1, num_states)
        # # Probabilities at time t+1 (to states)
        # probs_t_plus_1 = state_probs[1:, :] # Shape (N-1, num_states)

        # # Calculate expected counts (numerators and denominators)
        # # Numerator: Sum over t of P(state=i at t) * P(state=j at t+1)
        # # Denominator: Sum over t of P(state=i at t)

        # # Expected transition counts (Numerator for T_ij)
        # # Element-wise multiplication and sum:
        # # We want Sum_t [ p_t(i) * p_{t+1}(j) ] for each (i, j)
        # # This can be computed via matrix multiplication: probs_t.T @ probs_t_plus_1
        # # (num_states, N-1) @ (N-1, num_states) -> (num_states, num_states)
        # expected_transition_counts = probs_t.T @ probs_t_plus_1

        # # Expected occurrences of 'from' states (Denominator for T_ij)
        # # Sum over t of p_t(i) for each i
        # expected_state_counts = np.sum(probs_t, axis=0) # Shape (num_states,)

        # # Estimate Transition Matrix
        # transition_matrix = np.zeros((num_states, num_states))

        # # Add epsilon to avoid division by zero if a state has near-zero probability mass
        # denominator = expected_state_counts + epsilon

        # # Calculate T[i, j] = expected_transition_counts[i, j] / expected_state_counts[i]
        # # We need to divide each row i of expected_transition_counts by denominator[i]
        # transition_matrix = expected_transition_counts / denominator[:, np.newaxis]

        # Normalize rows to ensure they sum to 1 (handles epsilon effect and floating point)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        
        # Avoid division by zero for rows that are all zero
        non_zero_mask = (row_sums > epsilon)
        
        # Safe division - only divide non-zero rows
        safe_row_sums = np.where(non_zero_mask, row_sums, 1.0)
        transition_matrix = transition_matrix / safe_row_sums
        
        # For rows that were all zeros, set to uniform distribution
        zero_rows = ~non_zero_mask.squeeze()
        if np.any(zero_rows):
            transition_matrix[zero_rows, :] = 1.0 / num_states

        return transition_matrix


    @classmethod
    def normalize_probabilities_along_axis(cls, data: NDArray, axis=-1, equal_probs_for_zeros=True, debug_print=False) -> NDArray:
        """
        Normalizes probabilities along the specified axis, handling edge cases like zeros and NaNs.
        
        Args:
            data: NDArray - The array to normalize
            axis: int - The axis along which to normalize (default: -1, the last axis)
            equal_probs_for_zeros: bool - Whether to set equal probabilities for rows that sum to zero
            debug_print: bool - Whether to print debug information
            
        Returns:
            NDArray - The normalized array where values along the specified axis sum to 1.0
        """
        # Make a copy to avoid modifying the original data
        normalized_data = data.copy()
        
        # Check for zeros or NaNs before normalization
        sum_before_norm = np.nansum(normalized_data, axis=axis, keepdims=True)
        if debug_print:
            print(f"Before normalization - min sum: {np.min(sum_before_norm)}, has NaNs: {np.isnan(sum_before_norm).any()}")

        # Safe normalization with handling for zeros
        # Replace zeros with ones in the denominator to avoid division by zero
        safe_sums = np.where(sum_before_norm == 0, 1.0, sum_before_norm)
        normalized_data = normalized_data / safe_sums

        # For rows that summed to zero, set all values to equal probabilities (e.g., 1/n)
        if equal_probs_for_zeros:
            zero_sum_rows = (sum_before_norm == 0).squeeze()
            if np.any(zero_sum_rows):
                n_values = normalized_data.shape[axis]
                equal_probs = np.ones(n_values) / n_values
                # Create indexing for the array
                idx = [slice(None)] * normalized_data.ndim
                
                # Iterate through the array and replace zero-sum rows with equal probabilities
                # This is a simpler approach that avoids complex reshaping
                it = np.nditer(zero_sum_rows, flags=['multi_index'])
                for x in it:
                    if x:
                        # Get the multi_index but replace the axis dimension with a full slice
                        idx_list = list(it.multi_index)
                        # Remove the last dimension which was kept by keepdims=True
                        if axis == -1:
                            idx_list = idx_list[:-1]
                        else:
                            idx_list.pop(axis)
                        
                        # Create the full indexing tuple
                        full_idx = tuple(idx_list)
                        
                        # Set the values along the axis to equal probabilities
                        if axis == -1:
                            normalized_data[full_idx] = equal_probs
                        else:
                            # For other axes, we need to use advanced indexing
                            idx = [slice(None)] * normalized_data.ndim
                            for i, val in enumerate(full_idx):
                                dim = i if i < axis else i + 1
                                idx[dim] = val
                            idx[axis] = slice(None)
                            normalized_data[tuple(idx)] = equal_probs

                # # Apply equal probabilities to rows with zero sums
                # if zero_sum_rows.ndim > 0:  # If it's not a scalar
                #     # Create indexing tuple to properly assign values
                #     idx = [slice(None)] * normalized_data.ndim
                #     idx[axis] = slice(None)
                #     idx_tuple = tuple(idx)
                    
                #     # Create a boolean mask for indexing
                #     mask_idx = [slice(None)] * normalized_data.ndim
                #     mask_idx[axis] = np.newaxis
                #     mask = zero_sum_rows.reshape([zero_sum_rows.shape[0] if i == j else 1 
                #                                 for i, j in enumerate(range(normalized_data.ndim)) 
                #                                 if i != axis])
                    
                #     # Broadcast the mask to the correct shape
                #     broadcast_shape = list(normalized_data.shape)
                #     broadcast_shape[axis] = 1
                #     mask = np.broadcast_to(mask.reshape(tuple(broadcast_shape)), normalized_data.shape)
                    
                #     # Apply equal probabilities where the mask is True
                #     normalized_data = np.where(mask, equal_probs, normalized_data)
                # else:
                #     # Handle the case where there's only one row
                #     normalized_data[:] = equal_probs

        # Verify normalization
        verification = np.sum(normalized_data, axis=axis)
        if debug_print:
            print(f"Normalized sums: min={verification.min()}, max={verification.max()}, has NaNs: {np.isnan(verification).any()}")
        
        return normalized_data




    @function_attributes(short_name=None, tags=['transition_matrix', 'position', 'decoder_id', '2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 10:05', related_items=[])
    @classmethod
    def build_position_by_decoder_transition_matrix(cls, p_x_given_n: NDArray, debug_print=False):
        """
        given a decoder that gives a probability that the generating process is one of two possibilities, what methods are available to estimate the probability for a contiguous epoch made of many time bins?
        Note: there is most certainly temporal dependence, how should I go about dealing with this?

        Usage:

            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_position_by_decoder_transition_matrix, plot_blocked_transition_matrix

            ## INPUTS: p_x_given_n
            n_position_bins, n_decoding_models, n_time_bins = p_x_given_n.shape
            A_position, A_model, A_combined = TransitionMatrixComputations.build_position_by_decoder_transition_matrix(a_p_x_given_n)
            
            ## Plotting:
            import matplotlib.pyplot as plt; import seaborn as sns

            # plt.figure(figsize=(8,6)); sns.heatmap(A_big, cmap='viridis'); plt.title("Transition Matrix A_big"); plt.show()
            plt.figure(figsize=(8,6)); sns.heatmap(A_position, cmap='viridis'); plt.title("Transition Matrix A_position"); plt.show()
            plt.figure(figsize=(8,6)); sns.heatmap(A_model, cmap='viridis'); plt.title("Transition Matrix A_model"); plt.show()

            plot_blocked_transition_matrix(A_big, n_position_bins, n_decoding_models)


        """
        # Assume p_x_given_n is already loaded with shape (57, 4, 29951).
        # We'll demonstrate by generating random data:
        # p_x_given_n = np.random.rand(57, 4, 29951)

        n_position_bins, n_decoding_models, n_time_bins = p_x_given_n.shape
        if debug_print:
            print(f'\tn_position_bins, n_decoding_models, n_time_bins = p_x_given_n.shape')
            print(f'\t\tn_position_bins: {n_position_bins}, n_decoding_models: {n_decoding_models}, n_time_bins: {n_time_bins}')
            

        # 1. Determine the most likely model for each time bin
        sum_over_positions = p_x_given_n.sum(axis=0)  # (n_decoding_models, n_time_bins)
        best_model_each_bin = sum_over_positions.argmax(axis=0)  # (n_time_bins,)
        if debug_print:
            print(f'\tsum_over_positions.shape: {np.shape(sum_over_positions)}')
            print(f'\tbest_model_each_bin.shape: {np.shape(best_model_each_bin)}')
            

        sum_over_context_states = np.squeeze(p_x_given_n.sum(axis=1))  # (n_position_bins, n_time_bins)
        if debug_print:
            print(f'\tsum_over_context_states.shape: {np.shape(sum_over_context_states)}')

        # 2. Determine the most likely position for each time bin (conditional on chosen model)
        best_position_each_bin = np.array([
            p_x_given_n[:, best_model_each_bin[t], t].argmax()
            for t in range(n_time_bins)
        ])
        if debug_print:
            print(f'\tbest_position_each_bin.shape: {np.shape(best_position_each_bin)}')
            

        # Context Marginal Computation _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        marginal_p_x_given_n_over_positions = deepcopy(sum_over_positions).T
        if debug_print:
            print(f'\tmarginal_p_x_given_n_over_positions: {np.shape(marginal_p_x_given_n_over_positions)}')
        marginal_p_x_given_n_over_positions = cls.normalize_probabilities_along_axis(marginal_p_x_given_n_over_positions, axis=-1)
        if debug_print:
            print(f'\tmarginal_p_x_given_n_over_positions: {np.shape(marginal_p_x_given_n_over_positions)}')

        A_model_transition_matrix: NDArray = cls.estimate_transition_matrix_weighted_avg(state_probs=marginal_p_x_given_n_over_positions)
        A_model = A_model_transition_matrix
        A_model = np.nan_to_num(A_model)
        if debug_print:
            print(f'\tnp.shape(A_model): {np.shape(A_model)}')



        # Position Marginal Computation _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        marginal_p_x_given_n_over_context_states = deepcopy(sum_over_context_states).T
        if debug_print:
            print(f'\tmarginal_p_x_given_n_over_context_states: {np.shape(marginal_p_x_given_n_over_context_states)}')
        marginal_p_x_given_n_over_context_states = cls.normalize_probabilities_along_axis(marginal_p_x_given_n_over_context_states, axis=-1)
        if debug_print:
            print(f'\tmarginal_p_x_given_n_over_context_states: {np.shape(marginal_p_x_given_n_over_context_states)}')

        A_position_transition_matrix: NDArray = cls.estimate_transition_matrix_weighted_avg(state_probs=marginal_p_x_given_n_over_context_states)
        A_position = A_position_transition_matrix
        A_position = np.nan_to_num(A_position)
        if debug_print:
            print(f'\tnp.shape(A_position): {np.shape(A_position)}')


        # # 3. Build position transition matrix
        # A_position_counts = np.zeros((n_position_bins, n_position_bins))
        # for t in range(n_time_bins - 1):
        #     A_position_counts[best_position_each_bin[t], best_position_each_bin[t+1]] += 1
        # A_position = A_position_counts / A_position_counts.sum(axis=1, keepdims=True)
        # A_position = np.nan_to_num(A_position)  # handle rows with zero counts

        # # 4. Build model transition matrix
        # A_model_counts = np.zeros((n_decoding_models, n_decoding_models))
        # for t in range(n_time_bins - 1):
        #     A_model_counts[best_model_each_bin[t], best_model_each_bin[t+1]] += 1
        # A_model = A_model_counts / A_model_counts.sum(axis=1, keepdims=True)
        # A_model = np.nan_to_num(A_model)

        # 5. Construct combined transition matrix (Kronecker product)
        A_combined = np.kron(A_position, A_model)
        # if debug_print:
        #     print("\tA_position:", A_position)
        #     print("\tA_model:", A_model)
        #     print("\tA_big shape:", A_combined.shape)
        return A_position, A_model, A_combined




# ==================================================================================================================================================================================================================================================================================== #
# Unsorted - Extracted from PendingNotebookCode.py on 2025-05-14 14:05                                                                                                                                                                                                                 #
# ==================================================================================================================================================================================================================================================================================== #


@function_attributes(short_name=None, tags=['pre_post_delta', 'transition-matrix', 'TO_REFACTOR'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-04-22 18:09', related_items=[])
def split_transition_matricies_results_pre_post_delta_category(an_out_decoded_marginal_posterior_df: pd.DataFrame, a_context_state_transition_matrix_list: List[NDArray]) -> Dict[str, NDArray]:
    """ 
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import split_transition_matricies_results_pre_post_delta_category
        ## INPUTS: laps_context_state_transition_matrix_context_dict
        out_context_state_transition_matrix_context_dict = deepcopy(laps_context_state_transition_matrix_context_dict)
        out_matched_result_tuple_context_dict = deepcopy(laps_matched_result_tuple_context_dict)

        an_out_best_matching_context, an_out_result, an_out_decoder, an_out_decoded_marginal_posterior_df = list(out_matched_result_tuple_context_dict.values())[0] # [-1]
        a_context_state_transition_matrix_list: List[NDArray] = list(out_context_state_transition_matrix_context_dict.values())[0]

        a_mean_context_state_transition_matrix_dict = split_transition_matricies_results_pre_post_delta_category(an_out_decoded_marginal_posterior_df=an_out_decoded_marginal_posterior_df, a_context_state_transition_matrix_list=a_context_state_transition_matrix_list)
        
    """
    ## INPUTS: an_out_decoded_marginal_posterior_df, a_context_state_transition_matrix_list

    assert 'pre_post_delta_category' in an_out_decoded_marginal_posterior_df
    is_pre_delta = (an_out_decoded_marginal_posterior_df['pre_post_delta_category'] == 'pre-delta')

    a_context_state_transition_matrix: NDArray = np.stack(a_context_state_transition_matrix_list) # np.stack(out_context_state_transition_matrix_context_dict[a_ctxt]).shape

    ## split on first index:
    a_context_state_transition_matrix_dict = {'pre-delta': a_context_state_transition_matrix[is_pre_delta], 'post-delta': a_context_state_transition_matrix[np.logical_not(is_pre_delta)]}
    a_mean_context_state_transition_matrix_dict = {k:np.nanmean(v, axis=0) for k, v in a_context_state_transition_matrix_dict.items()}

    # np.shape(a_context_state_transition_matrix) # (84, 4, 4) - (n_epochs, n_states, n_states)
    # a_mean_context_state_transition_matrix: NDArray = np.nanmean(a_context_state_transition_matrix, axis=0) #.shape (4, 4)
    # a_mean_context_state_transition_matrix
    return a_mean_context_state_transition_matrix_dict


@function_attributes(short_name=None, tags=['transition-matrix', 'TO_REFACTOR'], input_requires=[], output_provides=[], uses=['TransitionMatrixComputations.build_position_by_decoder_transition_matrix'], used_by=['complete_all_transition_matricies'], creation_date='2025-04-22 13:52', related_items=[])
def build_transition_matricies(a_result: DecodedFilterEpochsResult, debug_print: bool = False):
    """ 
    
    
    (out_time_bin_container_list, out_position_transition_matrix_list, out_context_state_transition_matrix_list, out_combined_transition_matrix_list), (a_mean_context_state_transition_matrix, a_mean_position_transition_matrix) = build_transition_matricies(a_result=a_result)
    out_mean_context_state_transition_matrix_context_dict[a_ctxt] = deepcopy(a_mean_context_state_transition_matrix)
    
    """
    p_x_given_n_list = deepcopy(a_result.p_x_given_n_list)
    
    ## initialize result output
    out_time_bin_container_list = []
    out_position_transition_matrix_list = []
    out_context_state_transition_matrix_list = []
    out_combined_transition_matrix_list = []
    ## Iterate through the epochs:
    for a_time_bin_container, a_p_x_given_n in zip(a_result.time_bin_containers, p_x_given_n_list):
        if debug_print:
            print(f'np.shape(a_p_x_given_n): {np.shape(a_p_x_given_n)}')
        try:
            A_position, A_model, A_combined = TransitionMatrixComputations.build_position_by_decoder_transition_matrix(a_p_x_given_n, debug_print=debug_print)    
            out_position_transition_matrix_list.append(A_position)
            out_context_state_transition_matrix_list.append(A_model)
            out_combined_transition_matrix_list.append(A_combined)
            out_time_bin_container_list.append(deepcopy(a_time_bin_container))

        except ValueError as err:
            print(f'err: {err}')
        except Exception as err:
            raise ## unhandled exception
        
    ## END for a_...
    
    a_context_state_transition_matrix: NDArray = np.stack(out_context_state_transition_matrix_list) # np.stack(out_context_state_transition_matrix_context_dict[a_ctxt]).shape
    a_mean_context_state_transition_matrix: NDArray = np.nanmean(a_context_state_transition_matrix, axis=0) #.shape (4, 4)
    
    a_position_transition_matrix: NDArray = np.stack(out_position_transition_matrix_list) # np.stack(out_context_state_transition_matrix_context_dict[a_ctxt]).shape
    a_mean_position_transition_matrix: NDArray = np.nanmean(a_position_transition_matrix, axis=0) #.shape (4, 4)
    
    
    return (out_time_bin_container_list, out_position_transition_matrix_list, out_context_state_transition_matrix_list, out_combined_transition_matrix_list), (a_mean_context_state_transition_matrix, a_mean_position_transition_matrix)


@function_attributes(short_name=None, tags=['MAIN', 'transition-matrix', 'TO_REFACTOR'], input_requires=[], output_provides=[], uses=['build_transition_matricies'], used_by=[], creation_date='2025-04-22 14:49', related_items=[])
def complete_all_transition_matricies(a_new_fully_generic_result: "GenericDecoderDictDecodedEpochsDictResult", a_target_context: IdentifyingContext, debug_print: bool = False):
    """ Computes all transition matrix outputs for all found results for the provided target_context
    
    
    complete_all_transition_matricies(a_new_fully_generic_result=a_new_fully_generic_result, a_target_context=a_target_context)
    
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import complete_all_transition_matricies, build_transition_matricies
        ## Laps context:
        a_laps_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore', data_grain='per_epoch') ## Laps
        laps_target_context_results = complete_all_transition_matricies(a_new_fully_generic_result=a_new_fully_generic_result, a_target_context=a_laps_target_context)
        out_matched_result_tuple_context_dict, (out_time_bin_container_context_dict, out_position_transition_matrix_context_dict, out_context_state_transition_matrix_context_dict, out_combined_transition_matrix_context_dict), (out_mean_context_state_transition_matrix_context_dict, out_mean_position_transition_matrix_context_dict) = laps_target_context_results
        # a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = out_matched_result_tuple_context_dict[a_ctxt]



    """
    any_matching_contexts_list, result_context_dict, decoder_context_dict, decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=a_target_context)

    
    
    out_matched_result_tuple_context_dict = {}
    
    out_time_bin_container_context_dict: Dict[IdentifyingContext, List[BinningContainer]] = {}

    out_position_transition_matrix_context_dict: Dict[IdentifyingContext, List[NDArray]] = {}
    out_context_state_transition_matrix_context_dict: Dict[IdentifyingContext, List[NDArray]] = {}
    out_combined_transition_matrix_context_dict: Dict[IdentifyingContext, List[NDArray]] = {}

    out_mean_context_state_transition_matrix_context_dict: Dict[IdentifyingContext, NDArray] = {}
    out_mean_position_transition_matrix_context_dict: Dict[IdentifyingContext, NDArray] = {}

    # for a_ctxt, a_posterior_df in decoded_marginal_posterior_df_context_dict.items():
    for a_ctxt, a_result in result_context_dict.items():
        a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_ctxt)
        a_result: DecodedFilterEpochsResult = a_result
        if debug_print:
            print(f'a_ctxt: {a_ctxt}')
        
        # Drop Epochs that are too short from all results: ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        replay_epochs_df = deepcopy(a_result.filter_epochs)
        if not isinstance(replay_epochs_df, pd.DataFrame):
            replay_epochs_df = replay_epochs_df.to_dataframe()
        # min_possible_ripple_time_bin_size: float = find_minimum_time_bin_duration(replay_epochs_df['duration'].to_numpy())
        # min_bounded_ripple_decoding_time_bin_size: float = min(desired_ripple_decoding_time_bin_size, min_possible_ripple_time_bin_size) # 10ms # 0.002
        # if desired_ripple_decoding_time_bin_size < min_bounded_ripple_decoding_time_bin_size:
        #     print(f'WARN: desired_ripple_decoding_time_bin_size: {desired_ripple_decoding_time_bin_size} < min_bounded_ripple_decoding_time_bin_size: {min_bounded_ripple_decoding_time_bin_size}... hopefully it works.')
        decoding_time_bin_size: float = a_best_matching_context.get('time_bin_size',  a_ctxt.get('time_bin_size', None)) 
        assert decoding_time_bin_size is not None
        minimum_event_duration: float = 2.0 * decoding_time_bin_size

        ## Drop those less than the time bin duration
        print(f'DropShorterMode:')
        pre_drop_n_epochs = len(replay_epochs_df)
        assert minimum_event_duration is not None
        replay_epochs_df = replay_epochs_df[replay_epochs_df['duration'] > minimum_event_duration]
        post_drop_n_epochs = len(replay_epochs_df)
        n_dropped_epochs = post_drop_n_epochs - pre_drop_n_epochs
        print(f'\tminimum_event_duration present (minimum_event_duration={minimum_event_duration}).\n\tdropping {n_dropped_epochs} that are shorter than our minimum_event_duration of {minimum_event_duration}.', end='\t')
        print(f'{post_drop_n_epochs} remain.')

        epoch_data_indicies = a_result.find_data_indicies_from_epoch_times(replay_epochs_df['start'])
        ## filter the output
        a_result = a_result.filtered_by_epoch_times(replay_epochs_df['start'])
        # a_decoded_marginal_posterior_df = a_decoded_marginal_posterior_df.epochs.filtered_by_epoch_times(replay_epochs_df['start'])
        a_decoded_marginal_posterior_df = a_decoded_marginal_posterior_df.loc[epoch_data_indicies]

        out_matched_result_tuple_context_dict[a_ctxt] = (a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df)

        (out_time_bin_container_list, out_position_transition_matrix_list, out_context_state_transition_matrix_list, out_combined_transition_matrix_list), (a_mean_context_state_transition_matrix, a_mean_position_transition_matrix) = build_transition_matricies(a_result=a_result)
        out_time_bin_container_context_dict[a_ctxt] = out_time_bin_container_list
        out_position_transition_matrix_context_dict[a_ctxt] = out_position_transition_matrix_list
        out_context_state_transition_matrix_context_dict[a_ctxt] = out_context_state_transition_matrix_list
        out_combined_transition_matrix_context_dict[a_ctxt] = out_combined_transition_matrix_list

        out_mean_context_state_transition_matrix_context_dict[a_ctxt] = deepcopy(a_mean_context_state_transition_matrix)
        out_mean_position_transition_matrix_context_dict[a_ctxt] = deepcopy(a_mean_position_transition_matrix)

        # ## Plotting:
        # # # plt.figure(figsize=(8,6)); sns.heatmap(A_big, cmap='viridis'); plt.title("Transition Matrix A_big"); plt.show()
        # # plt.figure(figsize=(8,6)); sns.heatmap(A_position, cmap='viridis'); plt.title("Transition Matrix A_position"); plt.show()
        # # plt.figure(figsize=(8,6)); sns.heatmap(A_model, cmap='viridis'); plt.title("Transition Matrix A_model"); plt.show()

        # # plot_blocked_transition_matrix(A_combined, n_position_bins, n_decoding_models)
    # END for a_ctxt, a_result in result_con....

    return out_matched_result_tuple_context_dict, (out_time_bin_container_context_dict, out_position_transition_matrix_context_dict, out_context_state_transition_matrix_context_dict, out_combined_transition_matrix_context_dict), (out_mean_context_state_transition_matrix_context_dict, out_mean_position_transition_matrix_context_dict)



# ==================================================================================================================== #
# 2025-04-22 - Transition Matrix                                                                                       #
# ==================================================================================================================== #
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
import matplotlib.pyplot as plt
import seaborn as sns


def _perform_plot_P_Context_State_Transition_Matrix(context_state_transition_matrix: NDArray, num='laps', **kwargs):
    """ 
    
    Usage:    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_plot_P_Context_State_Transition_Matrix, _perform_plot_position_Transition_Matrix
        _perform_plot_P_Context_State_Transition_Matrix(context_state_transition_matrix=laps_mean_context_state_transition_matrix_context_dict[a_laps_best_matching_context], num='laps')
        _perform_plot_P_Context_State_Transition_Matrix(context_state_transition_matrix=pbes_mean_context_state_transition_matrix_context_dict[a_pbes_best_matching_context], num='PBEs')

    """
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8,6), num=num, **kwargs); sns.heatmap(context_state_transition_matrix, cmap='viridis'); plt.title("Transition Matrix P_Context State"); 
    plt.xlabel('P[t+1]')
    plt.ylabel('P[t]')
    state_labels = ['Long_LR', 'Long_RL', 'Short_LR', 'Short_RL']
    plt.xticks(ticks=(np.arange(len(state_labels))+0.5), labels=state_labels)
    plt.yticks(ticks=(np.arange(len(state_labels))+0.5), labels=state_labels)
    plt.show()

def _perform_plot_position_Transition_Matrix(a_position_transition_matrix: NDArray, num='laps', clear=True, **kwargs):
    """ 
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_plot_P_Context_State_Transition_Matrix, _perform_plot_position_Transition_Matrix
        _perform_plot_position_Transition_Matrix(a_position_transition_matrix=laps_mean_position_transition_matrix_context_dict[a_laps_best_matching_context], num='laps')
        _perform_plot_position_Transition_Matrix(a_position_transition_matrix=pbes_mean_position_transition_matrix_context_dict[a_pbes_best_matching_context], num='PBEs')

    """
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8,6), num=num, **kwargs); sns.heatmap(a_position_transition_matrix, cmap='viridis'); plt.title("Transition Matrix Position"); 
    plt.xlabel('P(Pos[t+1])')
    plt.ylabel('P(Pos[t]]')
    # state_labels = ['Long_LR', 'Long_RL', 'Short_LR', 'Short_RL']
    # plt.xticks(ticks=(np.arange(len(state_labels))+0.5), labels=state_labels)
    # plt.yticks(ticks=(np.arange(len(state_labels))+0.5), labels=state_labels)
    plt.show()


@function_attributes(short_name=None, tags=['figure', 'heatmap', 'matplotlib', 'transition-matrix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-04-22 15:52', related_items=[])
def plot_blocked_transition_matrix(A_big: NDArray, n_position_bins: int, n_decoding_models: int, tick_labels=('long_LR', 'long_RL', 'short_LR', 'short_RL'), should_show_marginals:bool=True, extra_title_suffix:str=''):
    """

    Usage:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.gridspec as gridspec
        from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import plot_blocked_transition_matrix
        
        # plt.figure(figsize=(8,6)); sns.heatmap(A_big, cmap='viridis'); plt.title("Transition Matrix A_big"); plt.show()
        plt.figure(figsize=(8,6)); sns.heatmap(A_position, cmap='viridis'); perform_update_title_subtitle(title_string=f"Transition Matrix A_position - t_bin: {a_time_bin_size}"); plt.show(); 
        plt.figure(figsize=(8,6)); sns.heatmap(A_model, cmap='viridis'); perform_update_title_subtitle(title_string=f"Transition Matrix A_model - t_bin: {a_time_bin_size}"); plt.show()

        _out = plot_blocked_transition_matrix(A_big, n_position_bins, n_decoding_models, extra_title_suffix=f' - t_bin: {a_time_bin_size}')

    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle

    if should_show_marginals:
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(2, 2, width_ratios=[10, 1], height_ratios=[1, 10])

        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_row_sums = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
        ax_col_sums = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)

        # Hide tick labels on margin plots
        plt.setp(ax_row_sums.get_yticklabels(), visible=False)
        plt.setp(ax_col_sums.get_xticklabels(), visible=False)

        # Main heatmap
        sns.heatmap(A_big, cmap='viridis', ax=ax_heatmap, cbar=False)

        # Draw lines separating decoder blocks
        for i in range(1, n_decoding_models):
            ax_heatmap.axhline(i * n_position_bins, color='white')
            ax_heatmap.axvline(i * n_position_bins, color='white')

        # Row sums (marginal over columns)
        row_sums = A_big.sum(axis=1)
        ax_row_sums.barh(np.arange(len(row_sums)), row_sums, color='gray')
        ax_row_sums.invert_xaxis()

        # Column sums (marginal over rows)
        col_sums = A_big.sum(axis=0)
        ax_col_sums.bar(np.arange(len(col_sums)), col_sums, color='gray')

        # Tick positions (centered in each block)
        tick_locs = [i * n_position_bins + n_position_bins / 2 for i in range(n_decoding_models)]
        if tick_labels is not None:
            assert len(tick_labels) == n_decoding_models, f"n_decoding_models: {n_decoding_models}, len(tick_labels): {len(tick_labels)}"
            tick_labels = list(tick_labels)
        else:
            tick_labels = [f'Decoder {i}' for i in range(n_decoding_models)]

        # Apply block-centered labels
        ax_heatmap.set_xticks(tick_locs)
        ax_heatmap.set_xticklabels(tick_labels, rotation=90)
        ax_heatmap.set_yticks(tick_locs)
        ax_heatmap.set_yticklabels(tick_labels)

        plt.tight_layout()
        title_text =  "Transition Matrix Blocks by Decoder w/ Marginals"

    else:
        fig = plt.figure(figsize=(8,8))
        ax_heatmap = sns.heatmap(A_big, cmap='viridis')

        for i in range(1, n_decoding_models):
            plt.axhline(i * n_position_bins, color='white')
            plt.axvline(i * n_position_bins, color='white')

        tick_locs = [i * n_position_bins + n_position_bins / 2 for i in range(n_decoding_models)]
        if tick_labels is not None:
            assert len(tick_labels) == n_decoding_models, f"n_decoding_models: {n_decoding_models}, len(tick_labels): {len(tick_labels)}"
            tick_labels = list(tick_labels)
        else:
            tick_labels = [f'Decoder {i}' for i in range(n_decoding_models)]

        plt.xticks(tick_locs, tick_labels, rotation=90)
        plt.yticks(tick_locs, tick_labels, rotation=0)
        title_text = "Transition Matrix Blocks by Decoder"


    perform_update_title_subtitle(fig=fig, ax=None, title_string=f"plot_blocked_transition_matrix()- {title_text}{extra_title_suffix}", subtitle_string=None)
    plt.show()
    return fig, ax_heatmap

