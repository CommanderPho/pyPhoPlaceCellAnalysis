# HeuristicReplayScoring

# ==================================================================================================================== #
# 2024-02-29 - Pho Replay Heuristic Metric                                                                             #
# ==================================================================================================================== #
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
import attrs
from attrs import asdict, astuple, define, field, Factory

import numpy as np
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult # used in compute_pho_heuristic_replay_scores
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult, TrackTemplates
from pyphoplacecellanalysis.Analysis.position_derivatives import _compute_pos_derivs

from scipy.ndimage import convolve # used in `expand_peaks_mask`


# HeuristicScoresTuple = attrs.make_class("HeuristicScoresTuple", {k:field() for k in ("longest_sequence_length", "longest_sequence_length_ratio", "direction_change_bin_ratio", "congruent_dir_bins_ratio", "total_congruent_direction_change", 
#                                                                                      "total_variation", "integral_second_derivative", "stddev_of_diff",
#                                                                                      "position_derivatives_df")}, bases=(UnpackableMixin, object,))
# longest_sequence_length, longest_sequence_length_ratio, direction_change_bin_ratio, congruent_dir_bins_ratio, total_congruent_direction_change, total_variation, integral_second_derivative, stddev_of_diff, position_derivatives_df = a_tuple

@define(slots=False)
class HeuristicScoresTuple(UnpackableMixin, object):
    """ the specific heuristic measures for a single decoded epoch
    """
    longest_sequence_length: int = field(default=1),
    longest_sequence_length_ratio: float = field(default=None),
    direction_change_bin_ratio: float = field(default=None),
    congruent_dir_bins_ratio: float = field(default=None),
    total_congruent_direction_change: float = field(default=None),
    total_variation: float = field(default=None),
    integral_second_derivative: float = field(default=None),
    stddev_of_diff: float = field(default=None),
    position_derivatives_df: pd.DataFrame = field(default=None)
    

def is_valid_sequence_index(sequence, test_index: int) -> bool:
    """ checks if the passed index is a valid index without wrapping.        
    Usage:

    """
    min_sequence_index: int = 0
    max_sequence_index: int = len(sequence)-1
    return ((test_index >= min_sequence_index) and (test_index <= max_sequence_index))


def _compute_sequences_spanning_ignored_intrusions(split_first_order_diff_arrays, continuous_sequence_lengths, longest_sequence_start_idx: int, max_ignore_bins: int = 2):
    """ an "intrusion" refers to one or more time bins that interrupt a longer sequence that would be monotonic if the intrusions were removed.

    The quintessential example would be a straight ramp that is interrupted by a single point discontinuity intrusion.  

    iNPUTS:
        split_first_order_diff_arrays: a list of split arrays of 1st order differences.
        continuous_sequence_lengths: a list of integer sequence lengths
        longest_sequence_start_idx: int - the sequence start index to start the scan in each direction
        max_ignore_bins: int = 2 - this means to disolve sequences shorter than 2 time bins if the sequences on each side are in the same direction

        
    Usage:

    
    TODO: convert to use `is_valid_sequence_index(...)` and `max_ignore_bins`

    """
    ## INPUTS: split_first_order_diff_arrays, continuous_sequence_lengths, max_ignore_bins

    left_congruent_flanking_index = None
    left_congruent_flanking_sequence = None

    right_congruent_flanking_index = None
    right_congruent_flanking_sequence = None        

    ## from the location of the longest sequence, check for flanking sequences <= `max_ignore_bins` in each direction.
    # Scan left:
    if (longest_sequence_start_idx-1) >= 0:
        left_flanking_index = (longest_sequence_start_idx-1)
        left_flanking_seq_length = continuous_sequence_lengths[left_flanking_index]
        if (left_flanking_seq_length <= max_ignore_bins):
            ## left flanking sequence can be considered
            ### Need to look even FURTHER to the left to see the prev sequence:
            if (longest_sequence_start_idx-2) >= 0:
                left_congruent_flanking_index = (longest_sequence_start_idx-2)
                ## Have a sequence to concatenate with
                left_congruent_flanking_sequence = split_first_order_diff_arrays[left_congruent_flanking_index]

            
    # Scan right:
    if ((longest_sequence_start_idx+1) < len(continuous_sequence_lengths)):
        right_flanking_index = (longest_sequence_start_idx+1)
        right_flanking_seq_length = continuous_sequence_lengths[right_flanking_index]
        if (right_flanking_seq_length <= max_ignore_bins):
            ### Need to look even FURTHER to the left to see the prev sequence:
            if (longest_sequence_start_idx+2) >= 0:
                right_congruent_flanking_index = (longest_sequence_start_idx+2)
                ## Have a sequence to concatenate with
                right_congruent_flanking_sequence = split_first_order_diff_arrays[right_congruent_flanking_index]


    return (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index)


def _compute_diffusion_value(a_p_x_given_n: NDArray) -> float:
    """ The amount of the track that is represented by the decoding. More is better (indicating a longer replay).

    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import _compute_diffusion_value

    
    """
    ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
    n_pos_bins, n_time_bins = np.shape(a_p_x_given_n) # np.shape(a_p_x_given_n): (62, 9)
    # Determine baseline (uniform) value for equally distributed bins
    uniform_diffusion_prob = (1.0 / float(n_pos_bins)) # equally diffuse everywhere on the track
    # uniform_diffusion_cumprob_all_bins = float(uniform_diffusion_prob) * float(n_time_bins) # can often be like 0.3 or so. Seems a little high.

    # is_higher_than_diffusion = (cum_pos_bin_probs > uniform_diffusion_cumprob_all_bins)

    # num_bins_higher_than_diffusion_across_time: int = np.nansum(is_higher_than_diffusion, axis=0)
    # ratio_bins_higher_than_diffusion_across_time: float = (float(num_bins_higher_than_diffusion_across_time) / float(n_pos_bins))

    return uniform_diffusion_prob

def _compute_total_variation(arr) -> float:
    """ very simple description of how much each datapoint varies
     from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import _compute_total_variation
      
    """
    return np.nansum(np.abs(np.diff(arr)))


def _compute_integral_second_derivative(arr, dx=1) -> float:
    """ very simple description of how much each datapoint varies
     from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import _compute_integral_second_derivative
      
    """
    return (np.nansum((np.diff(arr, n=2) ** 2.0))/dx) 



def _compute_stddev_of_diff(arr) -> float:
    """ very simple description of how much each datapoint varies
     from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import _compute_stddev_of_diff
      
    """
    return np.std(np.diff(arr, n=1))



@define(slots=False)
class SubsequencesPartitioningResult:
    """ returned by `partition_subsequences_ignoring_repeated_similar_positions` 

    diff_index_subsequence_indicies = np.split(np.arange(n_diff_bins), list_split_indicies)
    no_low_magnitude_diff_index_subsequence_indicies = [v[np.isin(v, low_magnitude_change_indicies, invert=True)] for v in diff_index_subsequence_indicies] # get the list of indicies for each subsequence without the low-magnitude ones
    num_subsequence_bins: List[int] = [len(v) for v in diff_index_subsequence_indicies]
    num_subsequence_bins_no_repeats: List[int] = [len(v) for v in no_low_magnitude_diff_index_subsequence_indicies]

    total_num_subsequence_bins = np.sum(num_subsequence_bins)
    total_num_subsequence_bins_no_repeats = np.sum(num_subsequence_bins_no_repeats)

    """
    first_order_diff_lst: List = field() # the original list
    list_parts: List = field() # factory=list
    diff_split_indicies: NDArray = field() # for the 1st-order diff array
    split_indicies: NDArray = field() # for the original array
    low_magnitude_change_indicies: NDArray = field() # specified in diff indicies

    @property
    def n_diff_bins(self) -> int:
        return len(self.first_order_diff_lst)


    @property
    def subsequence_index_lists_omitting_repeats(self):
        """The subsequence_index_lists_omitting_repeats property."""
        return np.array_split(len(self.split_indicies), self.split_indicies)


@function_attributes(short_name=None, tags=['partition'], input_requires=[], output_provides=[], uses=['SubsequencesPartitioningResult'], used_by=[], creation_date='2024-05-09 02:47', related_items=[])
def partition_subsequences_ignoring_repeated_similar_positions(first_order_diff_lst: Union[List, NDArray], same_thresh: float = 4.0, debug_print=False) -> "SubsequencesPartitioningResult":
    """ function partitions the list according to an iterative rule and the direction changes, ignoring changes less than or equal to `same_thresh`.
     
    NOTE: This helps "ignore" false-positive direction changes for bins with spatially-nearby (stationary) positions that happen to be offset in the wrong direction.
        It does not "bridge" discontiguous subsequences like `_compute_sequences_spanning_ignored_intrusions` was intended to do, that's a different problem.

        
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import partition_subsequences_ignoring_repeated_similar_positions

        # lst1 = [0, 3.80542, -3.80542, -19.0271, 0, -19.0271]
        # list_parts1, list_split_indicies1 = partition_subsequences_ignoring_repeated_similar_positions(lst=lst1) # [[0, 3.80542, -3.80542, 0, -19.0271]]

        # lst2 = [0, 3.80542, 5.0, -3.80542, -19.0271, 0, -19.0271]
        # list_parts2, list_split_indicies2 = partition_subsequences_ignoring_repeated_similar_positions(lst=lst2) # [[0, 3.80542, -3.80542], [-19.0271, 0, -19.0271]]

        
        ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
        partition_result1: SubsequencesPartitioningResult = partition_subsequences_ignoring_repeated_similar_positions(lst1, same_thresh=4)

    """
    n_diff_bins: int = len(first_order_diff_lst)
    n_original_bins: int = n_diff_bins + 1


    list_parts = []
    list_split_indicies = []
    low_magnitude_change_indicies = []

    prev_i = None
    prev_v = None
    prev_accum_dir = None # sentinal value
    prev_accum = []

    for i, v in enumerate(first_order_diff_lst):
        curr_dir = np.sign(v)
        did_accum_dir_change: bool = (prev_accum_dir != curr_dir)# and (prev_accum_dir is not None) and (prev_accum_dir != 0)
        if debug_print:
            print(f'i: {i}, v: {v}, curr_dir: {curr_dir}, prev_accum_dir: {prev_accum_dir}')

        if (np.abs(v) > same_thresh) and did_accum_dir_change:
            ## sign changed, split here.
            should_split: bool = True # (prev_accum_dir is None)
            if (curr_dir != 0):
                # only for non-zero directions should we set the prev_accum_dir, otherwise leave it what it was (or blank)
                if prev_accum_dir is None:
                    should_split = False # don't split for the first direction change (since it's a change from None/0.0
                if debug_print:
                    print(f'\t accum_dir changing! {prev_accum_dir} -> {curr_dir}')
                prev_accum_dir = curr_dir
                
            if should_split:
                if debug_print:
                    print(f'\t splitting.')
                list_parts.append(prev_accum)
                list_split_indicies.append(i)
                prev_accum = [] # zero the accumulator part
                # prev_accum_dir = None # reset the accum_dir

            ## accumulate the new entry, and potentially the change direction
            prev_accum.append(v)


        else:
            ## continue accumulating
            prev_accum.append(v)
            low_magnitude_change_indicies.append(i)

            # if curr_dir != 0:
            #     # only for non-zero directions should we set the prev_accum_dir, otherwise leave it what it was (or blank)
            #     prev_accum_dir = curr_dir

        ## update prev variables
        prev_i = i
        prev_v = v
        
    if len(prev_accum) > 0:
        # finish up with any points remaining
        list_parts.append(prev_accum)


    diff_split_indicies = np.array(list_split_indicies)
    list_split_indicies = diff_split_indicies + 1 # the +1 is because we pass a diff list/array which has one less index than the original array.
    low_magnitude_change_indicies = np.array(low_magnitude_change_indicies)

    list_parts = [np.array(l) for l in list_parts]

    # split_diff_index_subsequence_index_arrays = np.split(np.arange(n_diff_bins), diff_split_indicies) # subtract 1 again to get the diff_split_indicies instead
    # no_low_magnitude_diff_index_subsequence_indicies = [v[np.isin(v, low_magnitude_change_indicies, invert=True)] for v in split_diff_index_subsequence_index_arrays] # get the list of indicies for each subsequence without the low-magnitude ones
    # num_subsequence_bins = np.array([len(v) for v in split_diff_index_subsequence_index_arrays])
    # num_subsequence_bins_no_repeats = np.array([len(v) for v in no_low_magnitude_diff_index_subsequence_indicies])

    # total_num_subsequence_bins = np.sum(num_subsequence_bins)
    # total_num_subsequence_bins_no_repeats = np.sum(num_subsequence_bins_no_repeats)
    
    # continuous_sequence_lengths = np.array([len(a_split_first_order_diff_array) for a_split_first_order_diff_array in split_diff_index_subsequence_index_arrays])
    # longest_sequence_length: int = np.nanmax(continuous_sequence_lengths) # Now find the length of the longest non-changing sequence
    # longest_sequence_start_idx: int = np.nanargmax(continuous_sequence_lengths)

    # longest_sequence_length_no_repeats: int = np.nanmax(num_subsequence_bins_no_repeats) # Now find the length of the longest non-changing sequence
    # longest_sequence_no_repeats_start_idx: int = np.nanargmax(num_subsequence_bins_no_repeats)

    # return (list_parts, list_split_indicies, )
    _result = SubsequencesPartitioningResult(first_order_diff_lst=first_order_diff_lst, list_parts=list_parts, diff_split_indicies=diff_split_indicies, split_indicies=list_split_indicies, low_magnitude_change_indicies=low_magnitude_change_indicies)
    return _result



@metadata_attributes(short_name=None, tags=['heuristic', 'replay', 'ripple', 'scoring', 'pho'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 06:00', related_items=[])
class HeuristicReplayScoring:
    """ Measures of replay quality ("scores") that are better aligned with my (human-rated) intuition. Mostly based on the decoded posteriors.
    
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, HeuristicScoresTuple

    """

    @classmethod
    @function_attributes(short_name='jump', tags=['bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 17:50', related_items=[])
    def bin_wise_jump_distance(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ provides a metric that punishes long jumps in sequential maximal prob. position bins
        """
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
        time_window_centers = a_result.time_window_centers[an_epoch_idx]

        n_track_position_bins: int = np.shape(a_p_x_given_n)[0]
        
        max_indicies = np.argmax(a_p_x_given_n, axis=0)
        a_first_order_diff = np.diff(max_indicies, n=1, prepend=[max_indicies[0]]) # max index change
        max_jump_index_distance = np.nanmax(np.abs(a_first_order_diff)) # find the maximum jump size (in number of indicies) during this period
        # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
        max_jump_index_distance_ratio = (float(max_jump_index_distance) / float(n_track_position_bins-1))
        max_jump_index_distance_score = max_jump_index_distance_ratio / a_decoder_track_length
        ## RETURNS: total_first_order_change_score
        return max_jump_index_distance_score
    
    @classmethod
    @function_attributes(short_name='travel', tags=['bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 17:50', related_items=[])
    def bin_wise_position_difference(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ Bin-wise most-likely position difference. Contiguous trajectories have small deltas between adjacent time bins, while non-contiguous ones can jump wildly (up to the length of the track)
        ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
        """
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        if n_time_bins <= 1:
            ## only a single bin, return 0.0
            return 0.0
        else:
            # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
            time_window_centers = a_result.time_window_centers[an_epoch_idx]

            # compute the 1st-order diff of all positions
            a_first_order_diff = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]])
            a_first_order_diff
            # add up the differences over all time bins
            total_first_order_change: float = np.nansum(np.abs(a_first_order_diff[1:])) # use .abs() to sum the total distance traveled in either direction
            total_first_order_change
            ## convert to a score

            # normalize by the number of bins to allow comparions between different Epochs (so epochs with more bins don't intrinsically have a larger score.
            total_first_order_change_score: float = float(total_first_order_change) / float(n_time_bins - 1)
            total_first_order_change_score
            # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
            total_first_order_change_score = total_first_order_change_score / a_decoder_track_length
            ## RETURNS: total_first_order_change_score
            return total_first_order_change_score

    @classmethod
    @function_attributes(short_name='coverage', tags=['bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-12 01:05', related_items=[])
    def bin_wise_track_coverage_score_fn(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ The amount of the track that is represented by the decoding. More is better (indicating a longer replay).

        """
        ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        n_pos_bins: int = np.shape(a_p_x_given_n)[0]

        # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
        time_window_centers = a_result.time_window_centers[an_epoch_idx]

        cum_pos_bin_probs = np.nansum(a_p_x_given_n, axis=1) # sum over the time bins, leaving the accumulated probability per pos bin.
        
        # 2024-04-30 - New Idea - give the animal a "cursor" with which it can sweep the track. This prevents the small jumps in decoder position from messing up the bins.

        
        

        # Determine baseline (uniform) value for equally distributed bins
        uniform_diffusion_prob = (1.0 / float(n_pos_bins)) # equally diffuse everywhere on the track
        uniform_diffusion_cumprob_all_bins = float(uniform_diffusion_prob) * float(n_time_bins) # can often be like 0.3 or so. Seems a little high.

        is_higher_than_diffusion = (cum_pos_bin_probs > uniform_diffusion_cumprob_all_bins)

        num_bins_higher_than_diffusion_across_time: int = np.nansum(is_higher_than_diffusion, axis=0)
        ratio_bins_higher_than_diffusion_across_time: float = (float(num_bins_higher_than_diffusion_across_time) / float(n_pos_bins))

        ## convert to a score
        # track_portion_covered: float = (ratio_bins_higher_than_diffusion_across_time * float(a_decoder_track_length))
        # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
        # total_first_order_change_score = total_first_order_change_score / a_decoder_track_length
        ## RETURNS: ratio_bins_higher_than_diffusion_across_time
        return ratio_bins_higher_than_diffusion_across_time

    @classmethod
    @function_attributes(short_name='continuous_seq_sort', tags=['bin-size', 'score', 'replay', 'UNFINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-12 01:05', related_items=[])
    def bin_wise_continuous_sequence_sort_score_fn(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float, same_thresh: float=4) -> float:
        """ The amount of the track that is represented by the decoding. More is better (indicating a longer replay).

        - Finds the longest continuous sequence (perhaps with some intrusions allowed?)
                Intrusions!
        - Do something with the start/end periods
        - 

        TODO 2024-03-27 - Not yet working:  

        # 2024-03-27 - Finish `bin_wise_continuous_sequence_sort_score_fn`
            from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, is_valid_sequence_index, _compute_sequences_spanning_ignored_intrusions

            # track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict

            a_result: DecodedFilterEpochsResult = deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict['short_RL'])
            an_epoch_idx: int = 8
            a_decoder_track_length = 144.0 # {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}
            HeuristicReplayScoring.bin_wise_continuous_sequence_sort_score_fn(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
            start_t: float = 105.4005

            active_filter_epochs: pd.DataFrame = deepcopy(a_result.active_filter_epochs)
            active_filter_epochs
            active_filter_epochs.epochs.find_data_indicies_from_epoch_times(epoch_times=np.array([start_t])) # array([8], dtype=int64)
            active_filter_epochs

            an_epoch_idx: int = 8

            a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
            a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
            n_time_bins: int = a_result.nbins[an_epoch_idx]
            n_pos_bins: int = np.shape(a_p_x_given_n)[0]
            time_window_centers = a_result.time_window_centers[an_epoch_idx]

            a_most_likely_positions_list

            # n_time_bins

        """
        ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        n_pos_bins: int = np.shape(a_p_x_given_n)[0]
        time_window_centers = a_result.time_window_centers[an_epoch_idx]

        ## Begin computations:

        # Get confidence of each position
        each_t_bin_max_confidence = np.nanmax(a_p_x_given_n, axis=0)

        # the confidence in the differences depends on the confidence in the two bins that the differences are taken from
        each_t_bin_max_confidence
        # each_t_bin_max_confidence


        a_first_order_diff = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]])
        assert len(a_first_order_diff) == len(a_most_likely_positions_list), f"the prepend above should ensure that the sequence and its first-order diff are the same length."
        # Calculate the signs of the differences
        # a_first_order_diff_sign = np.sign(a_first_order_diff)
        # Calculate where the sign changes occur (non-zero after taking diff of signs)
        # sign_change_indices = np.where(np.diff(a_first_order_diff_sign) != 0)[0] + 1  # Add 1 because np.diff reduces the index by 1

        ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
        partition_result: SubsequencesPartitioningResult = partition_subsequences_ignoring_repeated_similar_positions(a_first_order_diff, same_thresh=same_thresh)  # Add 1 because np.diff reduces the index by 1
        # sign_change_indices = np.where(np.abs(np.diff(a_first_order_diff_sign.astype(float))) > 0.0)[0] + 1  # Add 1 because np.diff reduces the index by 1 -- Oh no, is this right when I preserve length?

        total_first_order_change: float = np.nansum(a_first_order_diff[1:]) # this is very susceptable to misplaced bins
        epoch_change_direction: float = np.sign(total_first_order_change) # -1.0 or 1.0

        # position = deepcopy(a_most_likely_positions_list)
        # velocity = a_first_order_diff / float(a_result.decoding_time_bin_size) # velocity with real world units of cm/sec
        # acceleration = np.diff(velocity, n=1, prepend=[velocity[0]])

        # position_derivatives_df: pd.DataFrame = pd.DataFrame({'t': time_window_centers, 'x': position, 'vel_x': velocity, 'accel_x': acceleration})

        # position_derivative_column_names = ['x', 'vel_x', 'accel_x']
        # position_derivative_means = position_derivatives_df.mean(axis='index')[position_derivative_column_names].to_numpy()
        # position_derivative_medians = position_derivatives_df.median(axis='index')[position_derivative_column_names].to_numpy()
        # # position_derivative_medians = position_derivatives_df(axis='index')[position_derivative_column_names].to_numpy()

        position_derivatives_df: pd.DataFrame = _compute_pos_derivs(time_window_centers=time_window_centers, position=a_most_likely_positions_list, debug_print=False)

        # Now split the array at each point where a direction change occurs
        
        
        # num_direction_changes: int = len(sign_change_indices)
        # direction_change_bin_ratio: float = float(num_direction_changes) / (float(n_time_bins)-1) ## OUT: direction_change_bin_ratio

        # Split the array at each index where a sign change occurs
        relative_indicies_arr = np.arange(n_pos_bins)
        split_relative_indicies = np.split(relative_indicies_arr, partition_result.split_indicies)
        split_most_likely_positions_arrays = np.split(a_most_likely_positions_list, partition_result.split_indicies)
        # split_first_order_diff_arrays = np.split(a_first_order_diff, partition_result.split_indicies)
        split_first_order_diff_arrays = np.split(a_first_order_diff, partition_result.diff_split_indicies)

        
        # continuous_sequence_lengths = np.array([len(a_split_first_order_diff_array) for a_split_first_order_diff_array in split_first_order_diff_arrays])
        # longest_sequence_length: int = np.nanmax(continuous_sequence_lengths) # Now find the length of the longest non-changing sequence
        # longest_sequence_start_idx: int = np.nanargmax(continuous_sequence_lengths)

        # longest_sequence_indicies = split_relative_indicies[longest_sequence_start_idx]
        # longest_sequence = split_most_likely_positions_arrays[longest_sequence_start_idx]
        # longest_sequence_diff = split_first_order_diff_arrays[longest_sequence_start_idx]

        # longest_sequence
        split_diff_index_subsequence_index_arrays = np.split(np.arange(partition_result.n_diff_bins), partition_result.diff_split_indicies) # subtract 1 again to get the diff_split_indicies instead
        no_low_magnitude_diff_index_subsequence_indicies = [v[np.isin(v, partition_result.low_magnitude_change_indicies, invert=True)] for v in split_diff_index_subsequence_index_arrays] # get the list of indicies for each subsequence without the low-magnitude ones
        num_subsequence_bins = np.array([len(v) for v in split_diff_index_subsequence_index_arrays])
        num_subsequence_bins_no_repeats = np.array([len(v) for v in no_low_magnitude_diff_index_subsequence_indicies])

        total_num_subsequence_bins = np.sum(num_subsequence_bins)
        total_num_subsequence_bins_no_repeats = np.sum(num_subsequence_bins_no_repeats)
        
        # continuous_sequence_lengths = np.array([len(a_split_first_order_diff_array) for a_split_first_order_diff_array in split_diff_index_subsequence_index_arrays])
        # longest_sequence_length: int = np.nanmax(continuous_sequence_lengths) # Now find the length of the longest non-changing sequence
        # longest_sequence_start_idx: int = np.nanargmax(continuous_sequence_lengths)

        longest_sequence_length_no_repeats: int = np.nanmax(num_subsequence_bins_no_repeats) # Now find the length of the longest non-changing sequence
        longest_sequence_no_repeats_start_idx: int = np.nanargmax(num_subsequence_bins_no_repeats)
        
        ## 
        max_ignore_bins: int = 2
        # (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index) = _compute_sequences_spanning_ignored_intrusions(split_first_order_diff_arrays, continuous_sequence_lengths, longest_sequence_start_idx=longest_sequence_start_idx, max_ignore_bins=max_ignore_bins)
        (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index) = _compute_sequences_spanning_ignored_intrusions(split_first_order_diff_arrays, num_subsequence_bins_no_repeats, longest_sequence_start_idx=longest_sequence_no_repeats_start_idx, max_ignore_bins=max_ignore_bins)

        
        ## alternatively can determine the direction of momentum by building an array of direction vectors between each bin.
        # change in position over change in time. Applied at point t.



        # longest_sequence_length_ratio: float = float(longest_sequence_length) /  float(n_time_bins) # longest_sequence_length_ratio: the ratio of the bins that form the longest contiguous sequence to the total num bins

        ## Compensate for repeating bins, not counting them towards the score but also not against.
        if total_num_subsequence_bins_no_repeats > 0:
            longest_sequence_length_ratio: float = float(longest_sequence_length_no_repeats) /  float(total_num_subsequence_bins_no_repeats) # longest_sequence_length_ratio: the ratio of the bins that form the longest contiguous sequence to the total num bins
        else:
            longest_sequence_length_ratio: float = 0.0 # zero it out if they are all repeats

        

        # cum_pos_bin_probs = np.nansum(a_p_x_given_n, axis=1) # sum over the time bins, leaving the accumulated probability per time bin.
        
        # # Determine baseline (uniform) value for equally distributed bins
        # uniform_diffusion_prob = (1.0 / float(n_pos_bins)) # equally diffuse everywhere on the track
        # uniform_diffusion_cumprob_all_bins = float(uniform_diffusion_prob) * float(n_time_bins)

        # is_higher_than_diffusion = (cum_pos_bin_probs > uniform_diffusion_cumprob_all_bins)

        # num_bins_higher_than_diffusion_across_time: int = np.nansum(is_higher_than_diffusion, axis=0)
        # ratio_bins_higher_than_diffusion_across_time: float = (float(num_bins_higher_than_diffusion_across_time) / float(n_pos_bins))

        ## convert to a score
        # track_portion_covered: float = (ratio_bins_higher_than_diffusion_across_time * float(a_decoder_track_length))
        # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
        # total_first_order_change_score = total_first_order_change_score / a_decoder_track_length
        ## RETURNS: ratio_bins_higher_than_diffusion_across_time
        # return ratio_bins_higher_than_diffusion_across_time
        return longest_sequence_length_ratio
    


    @classmethod
    @function_attributes(short_name=None, tags=['heuristic', 'replay', 'score', 'OLDER'], input_requires=[], output_provides=[],
                         uses=['_compute_pos_derivs', 'partition_subsequences_ignoring_repeated_similar_positions', '_compute_total_variation', '_compute_integral_second_derivative', '_compute_stddev_of_diff', 'HeuristicScoresTuple'],
                         used_by=['compute_all_heuristic_scores'], creation_date='2024-02-29 00:00', related_items=[])
    def compute_pho_heuristic_replay_scores(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, debug_print=False, use_bin_units_instead_of_realworld:bool=False, **kwargs) -> HeuristicScoresTuple:
        """ 2024-02-29 - New smart replay heuristic scoring

        
        use_bin_units_instead_of_realworld: bool = True # if False, uses the real-world units (cm/seconds). If True, uses nbin units (n_posbins/n_timebins)
        
        
        For a single_decoder, single_epoch

        a_result: DecodedFilterEpochsResult = a_decoded_filter_epochs_decoder_result_dict['long_LR']

        Want to maximize: longest_nonchanging_sequence, total_congruent_direction_change
        Want to minimize: num_direction_changes

        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring
            _out_new_scores = {}
            an_epoch_idx: int = 4 # 7
            for a_name, a_result in a_decoded_filter_epochs_decoder_result_dict.items():
                print(f'\na_name: {a_name}')
                _out_new_scores[a_name] = HeuristicReplayScoring.compute_pho_heuristic_replay_scores(a_result=a_result, an_epoch_idx=an_epoch_idx)

            _out_new_scores

            
            
        #TODO 2024-08-01 07:44: - [ ] `np.sign(...)` does not do what I thought it did. It returns *three* possible values for each element: {-1, 0, +1}, not just two {-1, +1} like I thought
        
        """
        same_thresh: float = kwargs.pop('same_thresh', 4.0)

        # use_bin_units_instead_of_realworld: bool = True # if False, uses the real-world units (cm/seconds). If True, uses nbin units (n_posbins/n_timebins)
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]

        if use_bin_units_instead_of_realworld:
            time_window_centers = np.arange(n_time_bins) + 0.5 # time-bin units, plot range would then be from (0.0, (float(n_time_bins) + 0.5))
            a_most_likely_positions_list = a_result.most_likely_position_indicies_list[an_epoch_idx] # pos-bins
            if np.ndim(a_most_likely_positions_list) == 2:
                a_most_likely_positions_list = a_most_likely_positions_list.flatten()
                
            
        else:
            # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
            time_window_centers = a_result.time_window_centers[an_epoch_idx]
            a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
            

        # a_p_x_given_n
        # a_result.p_x_given_n_list
        # a_result.marginal_x_list
        # a_result.marginal_y_list

        # Usage example:
        # # Set the number of adjacent bins you want to include on either side of the peak
        # n_adjacent_position_bins = 0  # or any other integer value you need
        # # a_p_x_given_n should already be defined as a (62, 9) shape array
        # peak_probabilities_with_adjacent, peak_positions = compute_local_peak_probabilities(a_p_x_given_n, n_adjacent_position_bins)
        # print("Local peak probabilities including adjacent position bins for each time bin:", peak_probabilities_with_adjacent)
        # print("Position indices corresponding to the local peaks for each time bin:", peak_positions)
        # Local peak probabilities including adjacent position bins for each time bin: [0.31841 0.321028 0.374347 0.367907 0.2261 0.172176 0.140867 0.0715084 0.172176]
        # Position indices corresponding to the local peaks for each time bin: [55 54 55 58 58 59 57  0 59]
        # Local peak probabilities including adjacent position bins for each time bin: [0.784589 0.785263 0.851714 0.840573 0.607828 0.478891 0.40594 0.185163 0.478891]
        # Position indices corresponding to the local peaks for each time bin: [55 54 55 58 58 59 57  1 59]
        if debug_print:
            print(f'np.shape(a_p_x_given_n): {np.shape(a_p_x_given_n)}')


        track_coverage = np.nansum(a_p_x_given_n, axis=-1) # sum over all time bins
        if debug_print:
            print(f'track_coverage: {track_coverage}')

        # assert n_time_bins > 1, f"n_time_bins must be greater than 1 for the first_order_diff to make any sense, but it it isn't. n_time_bins: {n_time_bins}"

        if n_time_bins <= 1:
            if debug_print:
                print(f"WARN: n_time_bins must be greater than 1 for the first_order_diff to make any sense, but it it isn't. n_time_bins: {n_time_bins}")
            return HeuristicScoresTuple(1, None, None, None, None, None, None, None, None)
        else:

            # The idea here was to look at the most-likely positions and their changes (derivatives) to see if these were predictive of good vs. bad ripples. For example, bad ripples might have extreme accelerations while good ones fall within a narrow window of physiologically consistent accelerations
            position_derivatives_df: pd.DataFrame = _compute_pos_derivs(time_window_centers=time_window_centers, position=a_most_likely_positions_list, debug_print=debug_print)

            ## TODO: reuse the values computed (properly) in `position_derivatives_df` instead of doing it from scratch again here:
            # a_first_order_diff = np.diff(a_most_likely_positions_list, n=1, prepend=[0.0])
            a_first_order_diff = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]]) ## #TODO ⚠️ 2024-08-01 07:27: - [ ] :this diff represents the velocity doesn't it, so prepending the position in the first bin makes zero sense. UPDATE: prepend means its prepended to the input before the diff, meaning the first value should always be zero
            
            
            # a_first_order_diff
            total_first_order_change: float = np.nansum(a_first_order_diff[1:]) # the *NET* position change over all epoch bins
            # total_first_order_change
            epoch_change_direction: float = np.sign(total_first_order_change) # -1.0 or 1.0, the general direction trend for the entire epoch
            # epoch_change_direction

            # a_result

            # a_track_length: float = 170.0
            # effectively_same_location_size = 0.1 * a_track_length # 10% of the track length
            # effectively_same_location_num_bins: int = np.rint(effectively_same_location_size)
            # effectively_same_location_num_bins: int = 4

            # non_same_indicies = (np.abs(a_first_order_diff) > float(effectively_same_location_num_bins))
            # effectively_same_indicies = np.logical_not(non_same_indicies)

            # an_effective_change_first_order_diff = deepcopy(a_first_order_diff)
            # an_effective_change_first_order_diff[effectively_same_location_num_bins] = 0.0 # treat as non-changing


            # Now split the array at each point where a direction change occurs
            # Calculate the signs of the differences
            a_first_order_diff_sign = np.sign(a_first_order_diff)

            # an_effective_first_order_diff_sign = deepcopy(a_first_order_diff_sign)
            # an_effective_first_order_diff_sign[effectively_same_indicies] = 0.0

            # Calculate where the sign changes occur (non-zero after taking diff of signs)
            # sign_change_indices = np.where(np.diff(a_first_order_diff_sign) != 0)[0] + 1  # Add 1 because np.diff reduces the index by 1

            ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
            partition_result: SubsequencesPartitioningResult = partition_subsequences_ignoring_repeated_similar_positions(a_first_order_diff, same_thresh=same_thresh)
            num_direction_changes: int = len(partition_result.split_indicies)
            direction_change_bin_ratio: float = float(num_direction_changes) / (float(n_time_bins)-1) ## OUT: direction_change_bin_ratio

            if debug_print:
                print(f'num_direction_changes: {num_direction_changes}')
                print(f'direction_change_bin_ratio: {direction_change_bin_ratio}')

            # Split the array at each index where a sign change occurs
            split_most_likely_positions_arrays = np.split(a_most_likely_positions_list, partition_result.split_indicies)
            # split_first_order_diff_arrays = np.split(a_first_order_diff, partition_result.split_indicies)
            split_first_order_diff_arrays = np.split(a_first_order_diff, partition_result.diff_split_indicies)

            # Pre-2024-05-09 Sequence Determination ______________________________________________________________________________ #
            continuous_sequence_lengths = [len(a_split_first_order_diff_array) for a_split_first_order_diff_array in split_first_order_diff_arrays]
            if debug_print:
                print(f'continuous_sequence_lengths: {continuous_sequence_lengths}')
            longest_sequence_length: int = np.nanmax(continuous_sequence_lengths) # Now find the length of the longest non-changing sequence
            if debug_print:
                print("Longest sequence of time bins without a direction change:", longest_sequence_length)
            longest_sequence_start_idx: int = np.nanargmax(continuous_sequence_lengths)
            longest_sequence = split_first_order_diff_arrays[longest_sequence_start_idx]
            
            longest_sequence_length_ratio: float = float(longest_sequence_length) /  float(n_time_bins) # longest_sequence_length_ratio: the ratio of the bins that form the longest contiguous sequence to the total num bins


            # 2024-05-09 Sequence Determination with ignored repeats (not yet working) ___________________________________________ #
            # split_diff_index_subsequence_index_arrays = np.split(np.arange(partition_result.n_diff_bins), partition_result.diff_split_indicies) # subtract 1 again to get the diff_split_indicies instead
            # no_low_magnitude_diff_index_subsequence_indicies = [v[np.isin(v, partition_result.low_magnitude_change_indicies, invert=True)] for v in split_diff_index_subsequence_index_arrays] # get the list of indicies for each subsequence without the low-magnitude ones
            # num_subsequence_bins = np.array([len(v) for v in split_diff_index_subsequence_index_arrays])
            # num_subsequence_bins_no_repeats = np.array([len(v) for v in no_low_magnitude_diff_index_subsequence_indicies])

            # total_num_subsequence_bins = np.sum(num_subsequence_bins)
            # total_num_subsequence_bins_no_repeats = np.sum(num_subsequence_bins_no_repeats)
            
            # longest_sequence_length_no_repeats: int = np.nanmax(num_subsequence_bins_no_repeats) # Now find the length of the longest non-changing sequence
            # # longest_sequence_no_repeats_start_idx: int = np.nanargmax(num_subsequence_bins_no_repeats)
            
            # longest_sequence_length = longest_sequence_length_no_repeats

            # ## Compensate for repeating bins, not counting them towards the score but also not against.
            # if total_num_subsequence_bins_no_repeats > 0:
            #     longest_sequence_length_ratio: float = float(longest_sequence_length_no_repeats) /  float(total_num_subsequence_bins_no_repeats) # longest_sequence_length_ratio: the ratio of the bins that form the longest contiguous sequence to the total num bins
            # else:
            #     longest_sequence_length_ratio: float = 0.0 # zero it out if they are all repeats


            contiguous_total_change_quantity = [np.nansum(a_split_first_order_diff_array) for a_split_first_order_diff_array in split_first_order_diff_arrays]
            if debug_print:
                print(f'contiguous_total_change_quantity: {contiguous_total_change_quantity}')
            max_total_change_quantity = np.nanmax(np.abs(contiguous_total_change_quantity))
            if debug_print:
                print(f'max_total_change_quantity: {max_total_change_quantity}')

            # for i, (a_split_most_likely_positions_array, a_split_first_order_diff_array) in enumerate(zip(split_most_likely_positions_arrays, split_first_order_diff_arrays)):
            #     print(f"Sequence {i}: {a_split_most_likely_positions_array}, {a_split_first_order_diff_array}")
            #     a_split_first_order_diff_array
            #     np.nansum(a_split_first_order_diff_array)

            ## a bin's direction is said to be "congruent" if it's consistent with the general trend in direction across the entire epoch duration:
            is_non_congruent_direction_bin = (a_first_order_diff_sign != epoch_change_direction)
            is_congruent_direction_bins = np.logical_not(is_non_congruent_direction_bin)

            congruent_bin_diffs = a_first_order_diff[is_congruent_direction_bins]
            incongruent_bin_diffs = a_first_order_diff[is_non_congruent_direction_bin]

            congruent_dir_bins_ratio: float = float(len(congruent_bin_diffs)) / float(n_time_bins - 1) # the ratio of the number of tbins where the velocity is moving in the general direction of the trajectory across the whole event. Not a great metric because minor variations around a constant position can dramatically drop score despite not looking bad.
            if debug_print:
                print(f'num_congruent_direction_bins_score: {congruent_dir_bins_ratio}')
            total_congruent_direction_change: float = np.nansum(np.abs(congruent_bin_diffs)) # the total quantity of change in the congruent direction
            total_incongruent_direction_change: float = np.nansum(np.abs(incongruent_bin_diffs))
            if debug_print:
                print(f'total_congruent_direction_change: {total_congruent_direction_change}, total_incongruent_direction_change: {total_incongruent_direction_change}')

            ## 2024-05-09 - New - "integral_second_derivative", "stddev_of_diff"
            total_variation = _compute_total_variation(a_most_likely_positions_list)
            integral_second_derivative = _compute_integral_second_derivative(a_most_likely_positions_list)
            stddev_of_diff = _compute_stddev_of_diff(a_most_likely_positions_list)

            return HeuristicScoresTuple(longest_sequence_length, longest_sequence_length_ratio, direction_change_bin_ratio, congruent_dir_bins_ratio, total_congruent_direction_change,
                                        total_variation=total_variation, integral_second_derivative=integral_second_derivative, stddev_of_diff=stddev_of_diff,
                                        position_derivatives_df=position_derivatives_df)

    @classmethod
    @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=['compute_all_heuristic_scores'], creation_date='2024-03-07 19:54', related_items=[])
    def _run_all_score_computations(cls, track_templates: TrackTemplates, a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], all_score_computations_fn_dict: Dict):
        """ 
        Performs the score computations specified in `all_score_computations_fn_dict` 
        Ideas is to have a general format for the functions that can be ran, and this function loops through all of them passing them what they need to run (all decoders, all epochs) and then collects their outputs to get simple DataFrames of scores for each epoch.

        Currently only have one implemented.
        #TODO 2024-03-07 20:05: - [ ] generalize the older `compute_pho_heuristic_replay_scores` to be called from here too!

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import bin_wise_position_difference ## functions to run
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _run_all_score_computations

            all_score_computations_fn_dict = {'bin_wise_position_difference': bin_wise_position_difference}
            all_epochs_scores_df = _run_score_computations(track_templates, a_decoded_filter_epochs_decoder_result_dict, all_score_computations_fn_dict=all_score_computations_fn_dict)
            all_epochs_scores_df

        """
        from neuropy.utils.misc import adding_additional_df_columns
        from neuropy.utils.indexing_helpers import NumpyHelpers
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import get_track_length_dict

        ## INPUTS: track_templates, a_decoded_filter_epochs_decoder_result_dict
        decoder_grid_bin_bounds_dict = {a_name:a_decoder.pf.config.grid_bin_bounds for a_name, a_decoder in track_templates.get_decoders_dict().items()}
        assert NumpyHelpers.all_allclose(list(decoder_grid_bin_bounds_dict.values())), f"all decoders should have the same grid_bin_bounds (independent of whether they are built on long/short, etc but they do not! This violates following assumptions."
        grid_bin_bounds = list(decoder_grid_bin_bounds_dict.values())[0] # tuple
        actual_track_length_dict, idealized_track_length_dict = get_track_length_dict(grid_bin_bounds, grid_bin_bounds)
        # idealized_track_length_dict # {'long': 214.0, 'short': 144.0}
        decoder_track_length_dict = {a_name:idealized_track_length_dict[a_name.split('_', maxsplit=1)[0]] for a_name, a_result in a_decoded_filter_epochs_decoder_result_dict.items()} # 
        decoder_track_length_dict # {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}
        ## OUTPUTS: decoder_track_length_dict

        ## INPUTS: a_decoded_filter_epochs_decoder_result_dict, decoder_track_length_dict
        all_epochs_scores_dict = {} # holds a single flat dataframe with scores from across all decoders
        separate_decoder_new_scores_df = {} # holds one df for each decoder name in a_decoded_filter_epochs_decoder_result_dict

        for a_name, a_result in a_decoded_filter_epochs_decoder_result_dict.items():
            ## all four decoders are guaranteed to be independent
            a_decoder_track_length: float = decoder_track_length_dict[a_name]

            _a_separate_decoder_new_scores_dict = {}

            ## compute all scores for this decoder:
            for score_computation_name, computation_fn in all_score_computations_fn_dict.items():
                score_name: str = score_computation_name # bin_wise_position_difference.short_name or bin_wise_position_difference.__name__
                single_decoder_column_name = f"{score_name}"
                unique_full_decoder_score_column_name: str = f"{score_name}_{a_name}"
                
                # all_epochs_scores_dict[column_name] = [bin_wise_position_difference(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length) for an_epoch_idx in np.arange(a_result.num_filter_epochs)]
                all_epochs_scores_dict[unique_full_decoder_score_column_name] = [computation_fn(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length) for an_epoch_idx in np.arange(a_result.num_filter_epochs)]
                _a_separate_decoder_new_scores_dict[single_decoder_column_name] = deepcopy(all_epochs_scores_dict[unique_full_decoder_score_column_name]) # a single column, all epochs
            # END for all_score_computations_fn_dict

            ## once done with all scores for this decoder, have `_a_separate_decoder_new_scores_dict`:
            separate_decoder_new_scores_df[a_name] =  pd.DataFrame(_a_separate_decoder_new_scores_dict)
            assert np.shape(separate_decoder_new_scores_df[a_name])[0] == np.shape(a_result.filter_epochs)[0], f"np.shape(separate_decoder_new_scores_df[a_name])[0]: {np.shape(separate_decoder_new_scores_df[a_name])[0]} != np.shape(a_result.filter_epochs)[0]: {np.shape(a_result.filter_epochs)[0]}"
            a_result.filter_epochs = adding_additional_df_columns(original_df=a_result.filter_epochs, additional_cols_df=separate_decoder_new_scores_df[a_name]) # update the filter_epochs with the new columns

        # END for `a_decoded_filter_epochs_decoder_result_dict`
        ## OUTPUTS: all_epochs_scores_dict, all_epochs_scores_df
        all_epochs_scores_df = pd.DataFrame(all_epochs_scores_dict)
        return a_decoded_filter_epochs_decoder_result_dict, all_epochs_scores_df

    @classmethod
    @function_attributes(short_name=None, tags=['heuristic', 'main', 'computation'], input_requires=[], output_provides=[], uses=['_run_all_score_computations', 'cls.compute_pho_heuristic_replay_scores'], used_by=[], creation_date='2024-03-12 00:59', related_items=[])
    def compute_all_heuristic_scores(cls, track_templates: TrackTemplates, a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult]):
        """ Computes all heuristic scoring metrics (for each epoch) and adds them to the DecodedFilterEpochsResult's .filter_epochs as columns
        
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_heuristic_scores

        a_decoded_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(a_decoded_filter_epochs_decoder_result_dict=a_decoded_filter_epochs_decoder_result_dict)


        """
        from neuropy.utils.misc import adding_additional_df_columns

        # positions __________________________________________________________________________________________________________ #
        # def directionality_ratio(positions):
        #     """
        #     Computes the directionality ratio (DR), which measures the degree to which the trajectory follows
        #     a consistent direction (increasing or decreasing) in position over time. It is calculated as the
        #     ratio of the net displacement to the total distance traveled.

        #     Args:
        #         positions (np.ndarray): 1D array of position bin indices.

        #     Returns:
        #         float: The directionality ratio, ranging from 0 to 1.
        #     """
        #     net_displacement = np.abs(positions[-1] - positions[0])
        #     total_distance = np.sum(np.abs(np.diff(positions)))
        #     return net_displacement / total_distance if total_distance != 0 else 0

        def sweep_score(positions, num_pos_bins: int):
            """
            Computes the sweep score (SS), which measures how well the trajectory sweeps across the available
            position bins over time. It is calculated as the number of unique position bins visited during the event,
            divided by the total number of position bins.

            Args:
                positions (np.ndarray): 1D array of position bin indices.

            Returns:
                float: The sweep score, ranging from 0 to 1.
            """
            unique_positions = np.unique(positions)
            return float(len(unique_positions)) / float(num_pos_bins)


        # def transition_entropy(positions):
        #     """
        #     Computes the transition entropy (TE), which quantifies the uncertainty or randomness in the transitions
        #     between position bins over time. It is calculated as the entropy of the transition probability matrix.

        #     Args:
        #         positions (np.ndarray): 1D array of position bin indices.

        #     Returns:
        #         float: The transition entropy score.
        #     """
        #     from scipy.stats import entropy
        #     transitions = np.diff(positions)
        #     transition_counts = np.bincount(transitions)
        #     transition_probs = transition_counts / np.sum(transition_counts)
        #     return entropy(transition_probs, base=2)

        # _positions_fns = [sweep_score] # directionality_ratio, transition_entropy 
        _positions_fns = []

        # positions, times ___________________________________________________________________________________________________ #
        def sequential_correlation(positions, times):
            """
            Computes the sequential correlation (SC) score, which quantifies the degree of sequential order
            in the trajectory by calculating the correlation between the position bin indices and the time bins.

            Args:
                positions (np.ndarray): 1D array of position bin indices.
                times (np.ndarray): 1D array of time bin indices.

            Returns:
                float: The sequential correlation score, ranging from -1 to 1.
            """
            return np.corrcoef(positions, times)[0, 1]

        def monotonicity_score(positions, times):
            """
            Computes the monotonicity score (MS), which measures how well the trajectory follows a monotonic
            (increasing or decreasing) pattern in position over time. It is calculated as the absolute value
            of the correlation between the position bin indices and the time bins.

            Args:
                positions (np.ndarray): 1D array of position bin indices.
                times (np.ndarray): 1D array of time bin indices.

            Returns:
                float: The monotonicity score, ranging from 0 to 1.
            """
            return np.abs(np.corrcoef(positions, times)[0, 1])

        def laplacian_smoothness(positions, times):
            """
            Computes the Laplacian smoothness (LS) score, which quantifies how smooth or continuous the trajectory
            is in terms of position changes over time. It is calculated as the sum of the squared differences
            between adjacent position bin values, weighted by the time bin differences.

            Args:
                positions (np.ndarray): 1D array of position bin indices.
                times (np.ndarray): 1D array of time bin indices.

            Returns:
                float: The Laplacian smoothness score.
            """
            position_diffs = np.diff(positions)
            time_diffs = np.diff(times)
            weighted_diffs = position_diffs ** 2 / time_diffs
            return np.sum(weighted_diffs)

        _positions_times_fns = [sequential_correlation, monotonicity_score, laplacian_smoothness]

        # positions, measured_positions ______________________________________________________________________________________ #
        # def replay_fidelity(positions, original_trajectory):
        #     """
        #     Computes the replay fidelity (RF) score, which measures the similarity between the decoded trajectory
        #     and the original trajectory or environment that is being replayed. It is calculated as the correlation
        #     between the two trajectories.

        #     Args:
        #         positions (np.ndarray): 1D array of position bin indices.
        #         original_trajectory (np.ndarray): 1D array representing the original trajectory.

        #     Returns:
        #         float: The replay fidelity score, ranging from -1 to 1.
        #     """
        #     return np.corrcoef(positions, original_trajectory)[0, 1]

        def bin_wise_wrapper_score_fn(a_fn, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float, needs_times=False) -> float:
            """ """
            ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float

            final_args = []

            a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
            # a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
            # positions = deepcopy(np.argmax(a_p_x_given_n, axis=1)) # peak indicies
            positions = deepcopy(a_most_likely_positions_list) # actual x positions
            final_args.append(positions)

            if needs_times:
                # n_time_bins: int = a_result.nbins[an_epoch_idx]
                # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
                time_window_centers = a_result.time_window_centers[an_epoch_idx]    
                times = deepcopy(time_window_centers)
                final_args.append(times)

            try:
                return a_fn(*final_args)
            except ValueError as e:
                # ValueError: 
                return np.nan
            except Exception as e:
                raise e
            # ValueError
            # return a_fn(*final_args)
            


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        ## Wrap them:
        positions_fns_dict = {fn.__name__:(lambda *args, **kwargs: bin_wise_wrapper_score_fn(fn, *args, **kwargs, needs_times=False)) for fn in _positions_fns}
        positions_times_fns_dict = {fn.__name__:(lambda *args, **kwargs: bin_wise_wrapper_score_fn(fn, *args, **kwargs, needs_times=True)) for fn in _positions_times_fns}
            
        all_score_computations_fn_dict = {'travel': cls.bin_wise_position_difference, 'coverage': cls.bin_wise_track_coverage_score_fn, 'jump': cls.bin_wise_jump_distance, **positions_fns_dict, **positions_times_fns_dict} # a_result, an_epoch_idx, a_decoder_track_length 
        a_decoded_filter_epochs_decoder_result_dict, all_epochs_scores_df = cls._run_all_score_computations(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=a_decoded_filter_epochs_decoder_result_dict, all_score_computations_fn_dict=all_score_computations_fn_dict)

        _out_new_scores: Dict[str, pd.DataFrame] = {}

        for a_name, a_result in a_decoded_filter_epochs_decoder_result_dict.items():
            _out_new_scores[a_name] =  pd.DataFrame([asdict(cls.compute_pho_heuristic_replay_scores(a_result=a_result, an_epoch_idx=an_epoch_idx), filter=lambda a, v: a.name not in ['position_derivatives_df']) for an_epoch_idx in np.arange(a_result.num_filter_epochs)])
            assert np.shape(_out_new_scores[a_name])[0] == np.shape(a_result.filter_epochs)[0], f"np.shape(_out_new_scores[a_name])[0]: {np.shape(_out_new_scores[a_name])[0]} != np.shape(a_result.filter_epochs)[0]: {np.shape(a_result.filter_epochs)[0]}"
            a_result.filter_epochs = adding_additional_df_columns(original_df=a_result.filter_epochs, additional_cols_df=_out_new_scores[a_name]) # update the filter_epochs with the new columns


        return a_decoded_filter_epochs_decoder_result_dict, _out_new_scores


    # ==================================================================================================================== #
    # OLD/UNINTEGRATED                                                                                                     #
    # ==================================================================================================================== #

    # # Single-time bin metrics: `sb_metric_*` ____________________________________________________________________________________________ #
    # # these metrics act on a single decoded time bin
    # def sb_metric_position_spread(self):
    #     """ provides a metric that punishes diffuse decoded posterior positions. For example, a posterior bin with two peaks far apart from one another. """
    #     pass


    # # Across time-bin metrics ____________________________________________________________________________________________ #
    # # These metrics operate on a series of decoded posteriors (multiple time bins)
    # def metric_position_covered_distance(self, a_p_x_given_n: NDArray, time_window_centers):
    #     """ provides a metric that punishes posteriors focused only on a small fraction of the environment, favoring those that sweep the track """
    #     # max_indicies = a_p_x_given_n.idxmax(axis=0)
    #     # a_first_order_diff = np.diff(max_indicies, n=1, prepend=[max_indicies[0]])
    #     pass

    @classmethod
    @function_attributes(short_name=None, tags=['DEP', 'OLD'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 19:54', related_items=[])
    def _DEP_run_all_compute_pho_heuristic_replay_scores(cls, filter_epochs: pd.DataFrame, a_decoded_filter_epochs_decoder_result_dict, t_start, t_delta, t_end, labels_column_name='lap_id'):
        """ 
        
        version from earlier in the day that only computes `compute_pho_heuristic_replay_scores`. The `_run_score_computations` was generalized from this one.

        Usage:    
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _run_all_compute_pho_heuristic_replay_scores

            ## INPUTS: all_directional_laps_filter_epochs_decoder_result_value, labels_column_name
            # Creates Columns: 'maze_id', 'truth_decoder_name':
            labels_column_name='label'
            # labels_column_name='lap_id'

            # all_directional_laps_filter_epochs_decoder_result_value.filter_epochs # has 'lap_id',  'lap_dir', -- needs 'maze_id'
            filter_epochs = all_directional_laps_filter_epochs_decoder_result_value.filter_epochs.to_dataframe()
            filter_epochs, all_epochs_position_derivatives_df = _run_all_compute_pho_heuristic_replay_scores(filter_epochs, labels_column_name=labels_column_name)
            filter_epochs

        """
        from neuropy.core.epoch import Epoch, ensure_dataframe
        ## INPUTS: filter_epochs: pd.DataFrame
        filter_epochs = ensure_dataframe(filter_epochs).epochs.adding_maze_id_if_needed(t_start, t_delta, t_end, replace_existing=True, labels_column_name=labels_column_name)
        # # Creates Columns: 'truth_decoder_name':
        # filter_epochs = _add_lap_extended_info_columns(filter_epochs, t_start, t_delta, t_end, labels_column_name=labels_column_name)
        # # # Update result's .filter_epochs
        # # all_directional_laps_filter_epochs_decoder_result_value.filter_epochs = filter_epochs.epochs.to_Epoch()

        # ## INPUT: a_decoded_filter_epochs_decoder_result_dict, all_directional_laps_filter_epochs_decoder_result_value
        # # num_filter_epochs: int = all_directional_laps_filter_epochs_decoder_result_value

        _out_true_decoder_new_scores = {}

        for i, row in enumerate(filter_epochs.itertuples()): # np.arange(num_filter_epochs)
            ## For each epoch, it gets the known-true decoder so that it can be compared to all the others.
            # print(row.truth_decoder_name)
            curr_decoder_name: str = row.truth_decoder_name
            a_result = a_decoded_filter_epochs_decoder_result_dict[curr_decoder_name]
            _out_true_decoder_new_scores[i] = cls.compute_pho_heuristic_replay_scores(a_result=a_result, an_epoch_idx=i)

            # all_directional_laps_filter_epochs_decoder_result_value.it

        all_epochs_position_derivatives_df = pd.concat([a_scores.position_derivatives_df for a_scores in _out_true_decoder_new_scores.values()], ignore_index=True)
        return filter_epochs, _out_true_decoder_new_scores, all_epochs_position_derivatives_df


    @staticmethod
    def _add_lap_extended_info_columns(filter_epochs: pd.DataFrame, t_start, t_delta, t_end, labels_column_name='lap_id'):
        """ Ensures the laps df passed has the required track and directional information ('maze_id', 'lap_dir'], and from this info builds a new 'truth_decoder_name' column containing the name of the decoder built from the corresponding lap
        
        Usage:    
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_lap_extended_info_columns

            ## INPUTS: all_directional_laps_filter_epochs_decoder_result_value, labels_column_name
            # Creates Columns: 'maze_id', 'truth_decoder_name':
            labels_column_name='label'
            # labels_column_name='lap_id'
            filter_epochs = all_directional_laps_filter_epochs_decoder_result_value.filter_epochs.to_dataframe()
            filter_epochs, all_epochs_position_derivatives_df = _add_lap_extended_info_columns(filter_epochs, labels_column_name=labels_column_name)
            filter_epochs

        """
        from neuropy.core.epoch import Epoch, ensure_dataframe
        ## INPUTS: filter_epochs: pd.DataFrame
        filter_epochs = ensure_dataframe(filter_epochs).epochs.adding_maze_id_if_needed(t_start, t_delta, t_end, replace_existing=True, labels_column_name=labels_column_name)
        # Creates Columns: 'truth_decoder_name':
        lap_dir_keys = ['LR', 'RL']
        maze_id_keys = ['long', 'short']
        filter_epochs['truth_decoder_name'] = filter_epochs['maze_id'].map(dict(zip(np.arange(len(maze_id_keys)), maze_id_keys))) + '_' + filter_epochs['lap_dir'].map(dict(zip(np.arange(len(lap_dir_keys)), lap_dir_keys)))
        return filter_epochs



@function_attributes(short_name=None, tags=['UNUSED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-13 15:57', related_items=[])
def compute_local_peak_probabilities(probs, n_adjacent: int):
    n_positions, n_time_bins = probs.shape
    local_peak_probabilities = np.zeros(n_time_bins)
    peak_position_indices = np.zeros(n_time_bins, dtype=int)

    for t in range(n_time_bins):
        time_slice = probs[:, t]
        for pos in range(n_positions):
            # The lower and upper bounds ensuring we don't go beyond the array bounds
            lower_bound = max(pos - n_adjacent, 0)
            upper_bound = min(pos + n_adjacent + 1, n_positions)  # The upper index is exclusive

            # Summing the local probabilities, including the adjacent bins
            local_sum = np.nansum(time_slice[lower_bound:upper_bound])

            # If this local sum is higher than a previous local peak, we record it
            if local_sum > local_peak_probabilities[t]:
                local_peak_probabilities[t] = local_sum
                peak_position_indices[t] = pos  # Save the position index

    return local_peak_probabilities, peak_position_indices


# For a decoded posterior probability matrix t `a_p_x_given_n` where `np.shape(a_p_x_given_n): (62, 9) = (n_pos_bins, n_time_bins)`, how do I get a boolean matrix with zeros everywhere except at the location of the position bin with the peak value for each time bin? How would I expand this peaks_mask_matrix to include the elements adjacent to the peaks?


def get_peaks_mask(a_p_x_given_n):
    """
    Returns a boolean mask matrix with True values at the position bins
    with the peak value for each time bin.
    """
    peaks_mask = np.zeros_like(a_p_x_given_n, dtype=bool)
    for t in range(a_p_x_given_n.shape[1]):  # iterate over time bins
        peaks_mask[np.argmax(a_p_x_given_n[:, t]), t] = True
    return peaks_mask

def expand_peaks_mask(peaks_mask, kernel=np.ones((3, 3))):
    """
    Expands the peaks_mask to include adjacent elements using a convolution kernel.

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import get_peaks_mask, expand_peaks_mask
        
        a_p_x_given_n = np.random.rand(62, 9)  # dummy data
        peaks_mask = get_peaks_mask(a_p_x_given_n)
        expanded_mask = expand_peaks_mask(peaks_mask) # expand in both position and time
        expanded_mask = expand_peaks_mask(peaks_mask, kernel=np.ones((3, 1))) # expand only in position

        expanded_mask

    """
    expanded_mask = convolve(peaks_mask.astype(int), kernel, mode='constant') >= 1
    return expanded_mask




# ==================================================================================================================== #
# 2023-12-21 - Inversion Count Concept                                                                                 #
# ==================================================================================================================== #

@metadata_attributes(short_name=None, tags=['inversion-count', 'heuristic', 'concept', 'untested', 'UNUSED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-21 00:00', related_items=[])
class InversionCount:
    """ 2023-12-21 - "Inversion Count" Quantification of Order (as an alternative to Spearman?

        computes the number of swap operations required to sort the list `arr` 



    # Example usage

        from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import InversionCount
        # list1 = [3, 1, 5, 2, 4]
        list1 = [1, 2, 4, 3, 5] # 1
        list1 = [1, 3, 4, 5, 2] # 3
        num_swaps = count_swaps_to_sort(list1)
        print("Number of swaps required:", num_swaps)

        >>> Number of swaps required: 3



    """
    @classmethod
    def merge_sort_and_count(cls, arr):
        """ Inversion Count - computes the number of swap operations required to sort the list `arr` 
        """
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, count_left = cls.merge_sort_and_count(arr[:mid])
        right, count_right = cls.merge_sort_and_count(arr[mid:])
        merged, count_split = cls.merge_and_count(left, right)

        return merged, (count_left + count_right + count_split)

    @classmethod
    def merge_and_count(cls, left, right):
        """ Inversion Count """
        merged = []
        count = 0
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                count += len(left) - i
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, count

    @classmethod
    def count_swaps_to_sort(cls, arr):
        _, swaps = cls.merge_sort_and_count(arr)
        return swaps


