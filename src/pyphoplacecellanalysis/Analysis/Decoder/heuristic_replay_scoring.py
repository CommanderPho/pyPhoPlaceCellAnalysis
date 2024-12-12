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
from pyphocorehelpers.assertion_helpers import Assert

from neuropy.utils.mixins.indexing_helpers import UnpackableMixin, get_dict_subset
from neuropy.utils.indexing_helpers import ListHelpers
from neuropy.utils.indexing_helpers import flatten

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


# def merge_subsequences(curr_subsequence, remaining_subsequence_list, max_ignore_bins: int = 2):
#     """ 
    
#     curr_subsequence, remaining_subsequence_list = merge_subsequences(
#     """
#     if len(remaining_subsequence_list) == 0:
#         return curr_subsequence, remaining_subsequence_list
#     else:
#         # iteratively combine
#         # next_attempted_subsequence = remaining_subsequence_list.pop(0)
#         next_attempted_subsequence = deepcopy(remaining_subsequence_list[0])
#         if len(next_attempted_subsequence) > max_ignore_bins:
#             # cannot merge, give up and return
#             return curr_subsequence, remaining_subsequence_list
#         else:
#             # merge and then continue trying trying to merge
#             return merge_subsequences([*curr_subsequence, *next_attempted_subsequence], remaining_subsequence_list[1:])
        



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
    flat_positions: NDArray = field(metadata={'desc': "the list of most-likely positions (in [cm]) for each time bin in a decoded posterior"})

    first_order_diff_lst: List = field() # the original list

    n_pos_bins: int = field(metadata={'desc': "the total number of unique position bins along the track, unrelated to the number of *positions* in `flat_positions` "})
    max_ignore_bins: int = field(default=2, metadata={'desc': "the maximum number of sequential time bins that can be merged over to form a larger subsequence."})
    same_thresh: float = field(default=4, metadata={'desc': "if the difference (in [cm]) between the positions of two sequential time bins is less than this value, it will be treated as a non-changing direction and effectively treated as a single bin."})
    max_jump_distance_cm: Optional[float] = field(default=None, metadata={'desc': 'The maximum allowed distance between adjacent bins'})
    
    flat_time_window_centers: Optional[NDArray] = field(default=None)
    flat_time_window_edges: Optional[NDArray] = field(default=None)
    
    # computed ___________________________________________________________________________________________________________ #
    
    list_parts: List = field(default=None, repr=False) # factory=list
    diff_split_indicies: NDArray = field(default=None, repr=False, metadata={'desc': "the indicies into the for the 1st-order diff array (`first_order_diff_lst`) where a split into subsequences should occur. "}) # for the 1st-order diff array
    split_indicies: NDArray = field(default=None, repr=False, metadata={'desc': "the indicies into `flat_positions` where a split into subsequences should occur. "}) # for the original array
    low_magnitude_change_indicies: NDArray = field(default=None, repr=False, metadata={'desc': "indicies where a change in direction occurs but it's below the threshold indicated by `same_thresh`"}) # specified in diff indicies
    bridged_intrusion_bin_indicies: NDArray = field(default=None, repr=False, metadata={'desc': "indicies where an intrusion previously existed that was bridged. "}) # specified in diff indicies
    

    # main subsequence splits ____________________________________________________________________________________________ #
    split_positions_arrays: List[NDArray] = field(default=None, repr=False, metadata={'desc': "the positions in `flat_positions` but partitioned into subsequences determined by changes in direction exceeding `self.same_thresh`"})
    split_position_flatindicies_arrays: List[NDArray] = field(default=None, repr=False, metadata={'WARN': 'UNIMPLEMENTED', 'desc': "the positions in `flat_positions` but partitioned into subsequences determined by changes in direction exceeding `self.same_thresh`"})


    merged_split_positions_arrays: List[NDArray] = field(default=None, metadata={'desc': "the subsequences from `split_positions_arrays` but merged into larger subsequences by briding-over (ignoring) sequences of intrusive tbins (with the max ignored length specified by `self.max_ignore_bins`"})
    merged_split_position_flatindicies_arrays: List[NDArray] = field(default=None, metadata={'WARN': 'UNIMPLEMENTED', 'desc': "the subsequences from `split_positions_arrays` but merged into larger subsequences by briding-over (ignoring) sequences of intrusive tbins (with the max ignored length specified by `self.max_ignore_bins`"})
    


    sequence_info_df: pd.DataFrame = field(default=None, repr=False)
    

    def __attrs_post_init__(self):
        if isinstance(self.flat_positions, list):
            self.flat_positions = np.array(self.flat_positions)        

        if self.sequence_info_df is None:
            ## initialize new
            self.sequence_info_df = self.rebuild_sequence_info_df()

    # Computed Properties ________________________________________________________________________________________________ #
    @property
    def n_diff_bins(self) -> int:
        return len(self.first_order_diff_lst)
    
    @property
    def flat_position_indicies(self) -> NDArray:
        """ the list of corresponding indicies for `self.flat_positions`."""
        if self.flat_positions is None:
            return None
        return np.arange(len(self.flat_positions))


    @property
    def subsequence_index_lists_omitting_repeats(self):
        """The subsequence_index_lists_omitting_repeats property. BROKEN """
        return np.array_split(np.arange(len(self.split_indicies)), self.split_indicies)

    @property
    def total_num_subsequence_bins_no_repeats(self) -> int:
        """ Calculates the total number of subsequence bins without repeats.
        """
        _, all_value_equiv_group_idxs_list = SubsequencesPartitioningResult.find_value_equiv_groups(self.flat_positions, same_thresh_cm=self.same_thresh)
        total_num_equiv_values: int = len(all_value_equiv_group_idxs_list) # the number of equivalence value sets in the longest subsequence
        return int(total_num_equiv_values)


    # Longest Sequence ___________________________________________________________________________________________________ #
    @property
    def longest_sequence_length_no_repeats(self) -> int:
        """ Finds the length of the longest non-repeating subsequence. """
        # longest_length = int(np.nanmax(self.num_merged_subsequence_bins))
        _, value_equiv_group_idxs_list = SubsequencesPartitioningResult.find_value_equiv_groups(self.longest_sequence_subsequence, same_thresh_cm=self.same_thresh)
        # num_items_per_equiv_list: List[int] = [len(v) for v in value_equiv_group_idxs_list] ## number of items in each equiv-list
        num_equiv_values: int = len(value_equiv_group_idxs_list) # the number of equivalence value sets in the longest subsequence
        return num_equiv_values

    @property
    def longest_no_repeats_sequence_length_ratio(self) -> float:
        """  Compensate for repeating bins, not counting them towards the score but also not against. """
        ## Compensate for repeating bins, not counting them towards the score but also not against.
        return self.get_longest_sequence_length(should_use_no_repeat_values=True)
            
    @property
    def longest_sequence_length_ratio(self) -> float:
        """  Compensate for repeating bins, not counting them towards the score but also not against. """
        return self.get_longest_sequence_length(should_use_no_repeat_values=False)
            

    # 2024-12-04 09:21 Replacement Properties ____________________________________________________________________________ #

    @property
    def num_subsequence_bins(self) -> NDArray:
        """Number of time bins in each split sequence."""
        if self.split_positions_arrays is None:
            return None
        return np.array([len(v) for v in self.split_positions_arrays])

    @property
    def num_merged_subsequence_bins(self) -> NDArray:
        """Number of time bins in each merged subsequence."""
        if self.merged_split_positions_arrays is None:
            return None
        return np.array([len(v) for v in self.merged_split_positions_arrays])

    @property
    def total_num_subsequence_bins(self) -> int:
        """Calculates the total number of subsequence bins."""
        if self.split_positions_arrays is None:
            return 0
        total = np.sum(self.num_subsequence_bins)
        return int(total)

    @property
    def total_num_merged_subsequence_bins(self) -> int:
        """Calculates the total number of merged subsequence bins."""
        if self.merged_split_positions_arrays is None:
            return 0
        total = np.sum(self.num_merged_subsequence_bins)
        return int(total)

    @property
    def longest_subsequence_length(self) -> int:
        """Finds the length of the longest merged subsequence."""
        if self.merged_split_positions_arrays is None:
            return 0
        longest_length = int(np.nanmax(self.num_merged_subsequence_bins))
        return longest_length

    @property
    def longest_sequence_subsequence_idx(self) -> int:
        """Finds the start index of the longest merged subsequence."""
        if self.merged_split_positions_arrays is None:
            return 0
        start_idx = int(np.nanargmax(self.num_merged_subsequence_bins))
        return start_idx

    @property
    def longest_sequence_subsequence(self) -> NDArray:
        """Returns the positions of the longest merged subsequence."""
        if self.merged_split_positions_arrays is None:
            return np.array([])
        return self.merged_split_positions_arrays[self.longest_sequence_subsequence_idx]
    

    @property
    def longest_sequence_flatindicies(self) -> NDArray:
        """Returns the flatindicies for the elements in the longest merged subsequence."""
        if self.merged_split_positions_arrays is None:
            return np.array([])
        if self.sequence_info_df is None:
            self.rebuild_sequence_info_df()
        is_longest_sequence_bin = (self.sequence_info_df['subsequence_idx'] == self.longest_sequence_subsequence_idx) #['flat_idx'].to_numpy()
        return self.sequence_info_df[is_longest_sequence_bin]['flat_idx'].to_numpy()


    @property
    def longest_sequence_subsequence_excluding_intrusions(self) -> NDArray:
        """ the longest merged subsequence excluding intrusions."""
        if self.sequence_info_df is None:
            self.rebuild_sequence_info_df()
        intrusion_flat_indicies = self.sequence_info_df[self.sequence_info_df['is_intrusion']]['flat_idx'].to_numpy()
        longest_sequence_non_intrusion_flatindicies = np.setdiff1d(self.longest_sequence_flatindicies, intrusion_flat_indicies)
        return self.flat_positions[longest_sequence_non_intrusion_flatindicies]
    

    @property
    def longest_subsequence_non_intrusion_nbins(self) -> int:
        """Finds the length of the longest merged subsequence excluding intrusions."""
        if self.merged_split_positions_arrays is None:
            return 0
        return len(self.longest_sequence_subsequence_excluding_intrusions)
    

    def get_longest_sequence_length(self, return_ratio:bool=True, should_use_no_repeat_values: bool = False, should_ignore_intrusion_bins: bool=True) -> float:
        """  Compensate for repeating bins, not counting them towards the score but also not against. """
        ## Compensate for repeating bins, not counting them towards the score but also not against.
        if not should_ignore_intrusion_bins:
            flat_positions = self.flat_positions
            longest_subsequence = self.longest_sequence_subsequence
        else:
            ## Ignoring intrusion bins
            non_intrusion_flat_indicies = self.sequence_info_df[np.logical_not(self.sequence_info_df['is_intrusion'])]['flat_idx'].to_numpy()
            flat_positions = self.flat_positions[non_intrusion_flat_indicies]
            longest_subsequence = self.longest_sequence_subsequence_excluding_intrusions
            

        if should_use_no_repeat_values:
            _, all_value_equiv_group_idxs_list = SubsequencesPartitioningResult.find_value_equiv_groups(flat_positions, same_thresh_cm=self.same_thresh)
            total_num_all_good_values: int = len(all_value_equiv_group_idxs_list) # the number of equivalence value sets in the longest subsequence
            
            _, value_equiv_group_idxs_list = SubsequencesPartitioningResult.find_value_equiv_groups(longest_subsequence, same_thresh_cm=self.same_thresh)
            # num_items_per_equiv_list: List[int] = [len(v) for v in value_equiv_group_idxs_list] ## number of items in each equiv-list
            num_longest_subsequence_good_values: int = len(value_equiv_group_idxs_list) # the number of equivalence value sets in the longest subsequence
            
        else:
            ## version that doesn't ignore repeats:
            total_num_all_good_values = self.total_num_subsequence_bins
            num_longest_subsequence_good_values = len(longest_subsequence)
            

        if not return_ratio:
            return int(num_longest_subsequence_good_values)
        else:
            if total_num_all_good_values > 0:
                return (float(num_longest_subsequence_good_values) / float(total_num_all_good_values)) # longest_sequence_length_ratio: the ratio of the bins that form the longest contiguous sequence to the total num bins
            else:
                return 0.0 # zero it out if they are all repeats
            

    # ==================================================================================================================== #
    # Update/Recompute Functions                                                                                           #
    # ==================================================================================================================== #

    @function_attributes(short_name=None, tags=['sequence'], input_requires=[], output_provides=[], uses=['partition_subsequences_ignoring_repeated_similar_positions', 'partition_subsequences', 'merge_intrusions', 'rebuild_sequence_info_df'], used_by=['bin_wise_continuous_sequence_sort_score_fn'], creation_date='2024-11-27 11:12', related_items=[])
    @classmethod
    def init_from_positions_list(cls, a_most_likely_positions_list: NDArray, n_pos_bins: int, max_ignore_bins: int = 2, same_thresh: float = 4, max_jump_distance_cm: Optional[float]=None, flat_time_window_centers=None, flat_time_window_edges=None, debug_print:bool=False) -> "SubsequencesPartitioningResult":
        """ main initializer """
        
        if isinstance(a_most_likely_positions_list, list):
            a_most_likely_positions_list = np.array(a_most_likely_positions_list)
        
        first_order_diff_lst = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]])
        assert len(first_order_diff_lst) == len(a_most_likely_positions_list), f"the prepend above should ensure that the sequence and its first-order diff are the same length."
        partition_result = cls(flat_positions=deepcopy(a_most_likely_positions_list), n_pos_bins=n_pos_bins, max_ignore_bins=max_ignore_bins, same_thresh=same_thresh, max_jump_distance_cm=max_jump_distance_cm, flat_time_window_centers=flat_time_window_centers, flat_time_window_edges=flat_time_window_edges,
                               first_order_diff_lst=None, list_parts=None, diff_split_indicies=None, split_indicies=None, low_magnitude_change_indicies=None,          
                            )
        
        ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
        partition_result.compute(debug_print=debug_print)

        return partition_result
    

    @function_attributes(short_name=None, tags=['NOT_COMPLETE', 'NOT_NEEDED', 'PhoOriginal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-03 04:17', related_items=[])
    @classmethod
    def detect_repeated_similar_positions(cls, a_most_likely_positions_list: Union[List, NDArray], same_thresh: float = 4.0, debug_print=False, **kwargs) -> "SubsequencesPartitioningResult":
        """ function partitions the list according to an iterative rule and the direction changes, ignoring changes less than or equal to `same_thresh`.
        
        NOTE: This helps "ignore" false-positive direction changes for bins with spatially-nearby (stationary) positions that happen to be offset in the wrong direction.
            It does !NOT! "bridge" discontiguous subsequences like `_compute_sequences_spanning_ignored_intrusions` was intended to do, that's a different problem.

            
        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import partition_subsequences_ignoring_repeated_similar_positions

            # lst1 = [0, 3.80542, -3.80542, -19.0271, 0, -19.0271]
            # list_parts1, list_split_indicies1 = partition_subsequences_ignoring_repeated_similar_positions(lst=lst1) # [[0, 3.80542, -3.80542, 0, -19.0271]]

            # lst2 = [0, 3.80542, 5.0, -3.80542, -19.0271, 0, -19.0271]
            # list_parts2, list_split_indicies2 = partition_subsequences_ignoring_repeated_similar_positions(lst=lst2) # [[0, 3.80542, -3.80542], [-19.0271, 0, -19.0271]]

            
            ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
            partition_result1: SubsequencesPartitioningResult = partition_subsequences_ignoring_repeated_similar_positions(lst1, same_thresh=4)

        """
        if isinstance(a_most_likely_positions_list, list):
            a_most_likely_positions_list = np.array(a_most_likely_positions_list)
        
    
        first_order_diff_lst = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]])
        assert len(first_order_diff_lst) == len(a_most_likely_positions_list), f"the prepend above should ensure that the sequence and its first-order diff are the same length."

        n_diff_bins: int = len(first_order_diff_lst)
        n_original_bins: int = n_diff_bins + 1
        Assert.len_equals(a_most_likely_positions_list, required_length=n_original_bins)
        
        list_parts = []
        list_split_indicies = []
        sub_change_threshold_change_diff_indicies = [] # indicies where a change in direction occurs but it's below the threshold indicated by `same_thresh`

        sub_change_equivalency_groups = []
        
        prev_accum_dir = None # sentinal value
        prev_accum_idxs = []
        prev_accum = []

        for i, v in enumerate(first_order_diff_lst):
            curr_dir = np.sign(v)
            did_accum_dir_change: bool = (prev_accum_dir != curr_dir)# and (prev_accum_dir is not None) and (prev_accum_dir != 0)
            if debug_print:
                print(f'i: {i}, v: {v}, curr_dir: {curr_dir}, prev_accum_dir: {prev_accum_dir}')


            if did_accum_dir_change: 
                if (np.abs(v) > same_thresh):
                    ## Exceeds the `same_thresh` indicating we want to use the change
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
                        sub_change_equivalency_groups.append(prev_accum_idxs)
                        
                        prev_accum = [] # zero the accumulator part
                        prev_accum_idxs = [] 
                        
                        # prev_accum_dir = None # reset the accum_dir
                else:
                    # direction changed but it didn't exceed the threshold, a so-called "low-magnitude" change
                    sub_change_threshold_change_diff_indicies.append(i)
                    ## continue the current `sub_change_equivalency_group`
                    

                # END if (np.abs(v) > same_thresh)
                
                ## accumulate the new entry, and potentially the change direction
                prev_accum.append(v)
                prev_accum_idxs.append(i)
            else:
                ## either below the threshold or the direction didn't change, continue accumulating
                prev_accum.append(v)
                prev_accum_idxs.append(i)

                # if curr_dir != 0:
                #     # only for non-zero directions should we set the prev_accum_dir, otherwise leave it what it was (or blank)
                #     prev_accum_dir = curr_dir

            
        if len(prev_accum) > 0:
            # finish up with any points remaining
            list_parts.append(prev_accum)


        diff_split_indicies = np.array(list_split_indicies)
        list_split_indicies = diff_split_indicies + 1 # the +1 is because we pass a diff list/array which has one less index than the original array.
        # sub_change_threshold_change_indicies = np.array(sub_change_threshold_change_indicies)
        sub_change_threshold_change_diff_indicies = np.array(sub_change_threshold_change_diff_indicies)
        sub_change_threshold_change_indicies = sub_change_threshold_change_diff_indicies + 1
        
        ## convert to non-diff idxs?
        # sub_change_equivalency_groups
        sub_change_equivalency_groups = [np.array(v) for i, v in enumerate(sub_change_equivalency_groups)]
        sub_change_equivalency_group_values = [a_most_likely_positions_list[v] for i, v in enumerate(sub_change_equivalency_groups)]
        

        list_parts = [np.array(l) for l in list_parts]
        
        return (sub_change_equivalency_groups, sub_change_equivalency_group_values), (list_parts, list_split_indicies, sub_change_threshold_change_indicies)


    @function_attributes(short_name=None, tags=['near_equal', 'PhoOriginal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-03 11:15', related_items=[])
    @classmethod
    def find_value_equiv_groups(cls, longest_sequence_subsequence: Union[List, NDArray], same_thresh_cm: float) -> Tuple[List, List]:
        """ returns the positions grouped by whether they are considered the same based on `same_thresh_cm`
        
        Usage:
        
        value_equiv_group_list, value_equiv_group_idxs_list = SubsequencesPartitioningResult.find_value_equiv_groups(longest_sequence_subsequence, same_thresh_cm=same_thresh_cm)
        
        """
        ## INPUTS: longest_sequence_subsequence
        value_equiv_group_list = []
        curr_accum_value_equiv_group = []

        value_equiv_group_idxs_list = []
        curr_accum_value_equiv_group_idxs = []

        initial_v = None
        for i, v in enumerate(longest_sequence_subsequence):
            if initial_v is None:
                initial_v = v
                curr_accum_value_equiv_group.append(v) ## add to equiv group
                curr_accum_value_equiv_group_idxs.append(i)
            else:
                if np.abs((v - initial_v)) > same_thresh_cm:
                    # end this group
                    value_equiv_group_list.append(curr_accum_value_equiv_group) ## add this group to the groups list
                    value_equiv_group_idxs_list.append(curr_accum_value_equiv_group_idxs)
                    
                    curr_accum_value_equiv_group = [v, ] # reset accum
                    curr_accum_value_equiv_group_idxs = [i, ]
                    initial_v = v
                else:
                    # continue this group
                    curr_accum_value_equiv_group.append(v) ## add to equiv group
                    curr_accum_value_equiv_group_idxs.append(i)
                    initial_v = v

        ## end for i, v ...
        ## OUTPUTS: value_equiv_group_list, value_equiv_group_idxs_list
        return value_equiv_group_list, value_equiv_group_idxs_list


    @function_attributes(short_name=None, tags=['partition', 'PhoOriginal'], input_requires=[], output_provides=[], uses=['SubsequencesPartitioningResult', 'rebuild_sequence_info_df'], used_by=['init_from_positions_list'], creation_date='2024-05-09 02:47', related_items=[])
    @classmethod
    def partition_subsequences_ignoring_repeated_similar_positions(cls, a_most_likely_positions_list: Union[List, NDArray], n_pos_bins: int, same_thresh: float = 4.0, debug_print=False, **kwargs) -> dict:
        """ function partitions the list according to an iterative rule and the direction changes, ignoring changes less than or equal to `same_thresh`.
        
        NOTE: This helps "ignore" false-positive direction changes for bins with spatially-nearby (stationary) positions that happen to be offset in the wrong direction.
            It does !NOT! "bridge" discontiguous subsequences like `_compute_sequences_spanning_ignored_intrusions` was intended to do, that's a different problem.

            
        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import partition_subsequences_ignoring_repeated_similar_positions

            # lst1 = [0, 3.80542, -3.80542, -19.0271, 0, -19.0271]
            # list_parts1, list_split_indicies1 = partition_subsequences_ignoring_repeated_similar_positions(lst=lst1) # [[0, 3.80542, -3.80542, 0, -19.0271]]

            # lst2 = [0, 3.80542, 5.0, -3.80542, -19.0271, 0, -19.0271]
            # list_parts2, list_split_indicies2 = partition_subsequences_ignoring_repeated_similar_positions(lst=lst2) # [[0, 3.80542, -3.80542], [-19.0271, 0, -19.0271]]

            
            ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
            partition_result1: SubsequencesPartitioningResult = partition_subsequences_ignoring_repeated_similar_positions(lst1, same_thresh=4)

        """
        if isinstance(a_most_likely_positions_list, list):
            a_most_likely_positions_list = np.array(a_most_likely_positions_list)
        
        first_order_diff_lst = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]])
        assert len(first_order_diff_lst) == len(a_most_likely_positions_list), f"the prepend above should ensure that the sequence and its first-order diff are the same length."

        n_diff_bins: int = len(first_order_diff_lst)
        n_original_bins: int = n_diff_bins + 1

        list_parts = []
        first_order_diff_list_split_indicies = []
        sub_change_threshold_change_indicies = [] # indicies where a change in direction occurs but it's below the threshold indicated by `same_thresh`

        prev_accum_dir = None # sentinal value
        prev_accum = []

        for i, v in enumerate(first_order_diff_lst):
            curr_dir = np.sign(v) # either [-1, 0, +1]
            did_accum_dir_change: bool = (prev_accum_dir != curr_dir) and (curr_dir != 0) # curr_dir == 0 means that we DEFER the split
            if debug_print:
                print(f'i: {i}, v: {v}, curr_dir: {curr_dir}, prev_accum_dir: {prev_accum_dir}')

            if did_accum_dir_change: 
                if (np.abs(v) > same_thresh):
                    ## Exceeds the `same_thresh` indicating we want to use the change
                    ## sign changed, split here.
                    should_split: bool = True # (prev_accum_dir is None)
                    if (curr_dir != 0):
                        # only for non-zero directions should we set the prev_accum_dir, otherwise leave it what it was (or blank)
                        if prev_accum_dir is None:
                            should_split = False # don't split for the first direction change (since it's a change from None/0.0
                        if debug_print:
                            print(f'\t accum_dir changing! {prev_accum_dir} -> {curr_dir}')
                        prev_accum_dir = curr_dir
                        
                    # END if (curr_dir != 0):
                    if should_split:
                        if debug_print:
                            print(f'\t splitting.')
                        list_parts.append(prev_accum)
                        first_order_diff_list_split_indicies.append(i)
                        prev_accum = [] # zero the accumulator part
                        # prev_accum_dir = None # reset the accum_dir
                else:
                    # direction changed but it didn't exceed the threshold, a so-called "low-magnitude" change
                    sub_change_threshold_change_indicies.append(i)
                    prev_accum_dir = None ## HACK fixes index after low-magnitude change
                # END if (np.abs(v) > same_thresh)
                
                ## accumulate the new entry, and potentially the change direction
                prev_accum.append(v)

            else:
                ## either below the threshold or the direction didn't change, continue accumulating
                prev_accum.append(v)
                

                # if curr_dir != 0:
                #     # only for non-zero directions should we set the prev_accum_dir, otherwise leave it what it was (or blank)
                #     prev_accum_dir = curr_dir

        # END for i, v in enu... 
        if len(prev_accum) > 0:
            # finish up with any points remaining
            list_parts.append(prev_accum)


        diff_split_indicies = np.array(first_order_diff_list_split_indicies)
        split_indicies = np.array(deepcopy(first_order_diff_list_split_indicies))# - 1
        # first_order_diff_list_split_indicies = diff_split_indicies + 1 # the +1 is because we pass a diff list/array which has one less index than the original array.
        # first_order_diff_list_split_indicies = deepcopy(diff_split_indicies) - 1 # try -1 
        sub_change_threshold_change_indicies = np.array(sub_change_threshold_change_indicies)
        
        list_parts = [np.array(l) for l in list_parts]
        
        # _result = SubsequencesPartitioningResult(flat_positions=deepcopy(a_most_likely_positions_list), first_order_diff_lst=first_order_diff_lst, list_parts=list_parts, diff_split_indicies=diff_split_indicies, split_indicies=split_indicies, low_magnitude_change_indicies=sub_change_threshold_change_indicies,
        #                                           n_pos_bins=n_pos_bins, same_thresh=same_thresh, **kwargs)
        # return _result
        return dict(flat_positions=deepcopy(a_most_likely_positions_list), first_order_diff_lst=first_order_diff_lst, list_parts=list_parts, diff_split_indicies=diff_split_indicies, split_indicies=split_indicies, low_magnitude_change_indicies=sub_change_threshold_change_indicies, n_pos_bins=n_pos_bins, same_thresh=same_thresh, **kwargs)
     

    @function_attributes(short_name=None, tags=['merge', '_compute_sequences_spanning_ignored_intrusions', 'PhoOriginal'], input_requires=['self.split_positions_arrays'], output_provides=['self.merged_split_positions_arrays'], uses=['_compute_sequences_spanning_ignored_intrusions'], used_by=[], creation_date='2024-11-27 08:17', related_items=['_compute_sequences_spanning_ignored_intrusions'])
    def merge_over_ignored_intrusions(self, max_ignore_bins: int = 2, should_skip_epoch_with_only_short_subsequences: bool = False, debug_print=False):
        """ an "intrusion" refers to one or more time bins that interrupt a longer sequence that would be monotonic if the intrusions were removed.

        The quintessential example would be a straight ramp that is interrupted by a single point discontinuity intrusion.  

        iNPUTS:
            split_first_order_diff_arrays: a list of split arrays of 1st order differences.
            continuous_sequence_lengths: a list of integer sequence lengths
            longest_sequence_start_idx: int - the sequence start index to start the scan in each direction
            max_ignore_bins: int = 2 - this means to disolve sequences shorter than 2 time bins if the sequences on each side are in the same direction

            
        REQUIRES: self.split_positions_arrays
        UPDATES: self.merged_split_positions_arrays
            
        Usage:

        HISTORY: 2024-11-27 08:18 based off of `_compute_sequences_spanning_ignored_intrusions`
        
        """
        def _subfn_resolve_overlapping_replacement_indicies(original_split_positions_arrays, replace_dict, debug_print=False):
            """ De-duplicates the replacement indicies. Prevents an issue that occurs when the replace keys overlap in indicies in `subsequence_replace_dict` that results in a net increase in number of bins in the merged output.
            captures: None
            

                        
            Example Problem Cases resolved by this function:
            #TODO 2024-11-28 09:38: - [X] Issue occurs when the replace keys overlap in indicies in `subsequence_replace_dict` that results in a net increase in number of bins in the merged output
            
            # Example 1))
            subsequence_replace_dict: {(0, 1, 2): (2, array([166.293, 100.721, 170.15, 170.15, 58.2919])), (1, 2, 3): (3, array([170.15, 170.15, 58.2919, 181.722]))}
            
            # Example 2))
            subsequence_replace_dict: {(2, 3, 4): (4, array([58.2919, 58.2919, 58.2919, 58.2919, 77.5778, 100.721, 58.2919, 58.2919])), (3, 4, 5): (5, array([100.721, 58.2919, 58.2919, 231.865])), (6, 7, 8): (8, array([58.2919, 58.2919, 58.2919, 58.2919, 185.579, 58.2919]))}
        
            AssertionError: final_out_subsequences_n_total_tbins (24) should equal the original n_total_tbins (21) but it does not!
                : final_out_subsequences: [array([189.436, 58.2919, 58.2919, 58.2919]), array([189.436, 185.579]), array([58.2919, 58.2919, 58.2919, 58.2919, 77.5778, 100.721, 58.2919, 58.2919]), array([100.721, 58.2919, 58.2919, 231.865]), array([58.2919, 58.2919, 58.2919, 58.2919, 185.579, 58.2919])]
                original_split_positions_arrays: [array([189.436, 58.2919, 58.2919, 58.2919]), array([189.436, 185.579]), array([58.2919, 58.2919, 58.2919, 58.2919, 77.5778]), array([100.721]), array([58.2919, 58.2919]), array([231.865]), array([58.2919, 58.2919, 58.2919, 58.2919]), array([185.579]), array([58.2919])]
                
            
            Usage
                deduplicated_replace_dict = resolve_overlapping_replacement_indicies(original_split_positions_arrays, subsequence_replace_dict)
                deduplicated_replace_dict
            """
            deduplicated_replace_dict = {}
            processed_indices = set()
            for idx_replace_list, (target_idx, replacement_subsequence) in replace_dict.items():
                non_duplicated_replacement_subsequence = []
                for a_replace_idx in idx_replace_list:
                    if a_replace_idx not in processed_indices:
                        non_duplicated_replacement_subsequence.extend(original_split_positions_arrays[a_replace_idx]) # get the original value of the subsequnce from the list and append it
                        processed_indices.add(a_replace_idx) ## add the idx to the set
                    else:
                        ## already found in set, indicates a duplicate is occuring that will need to be removed from this replacement
                        # replacement_subsequence
                        if debug_print:
                            print(f'WARN: skipping duplicated replacement index {a_replace_idx} from idx_replace_list: {idx_replace_list}')                        
                # END for a_replace_idx ...
                deduplicated_replace_dict[idx_replace_list] = (target_idx, np.array(non_duplicated_replacement_subsequence)) ## uses same `idx_replace_list` because we do want to replace these, but it updates the replacement list to be unique
                
            return deduplicated_replace_dict
        

        def _subfn_perform_merge_iteration(original_split_positions_arrays, max_ignore_bins: int = 2, should_skip_epoch_with_only_short_subsequences: bool = False, debug_print=False):
            """ only uses `self` in the call `self._compute_sequences_spanning_ignored_intrusions(...)` which is actually a classmethod, so this whole func could be a classmethod
            
            Captures: self, _subfn_resolve_overlapping_replacement_indicies
            
            original_split_positions_arrays, final_out_subsequences, (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove, final_intrusion_idxs) = _subfn_perform_merge_iteration(original_split_positions_arrays,
                                                                                                                                                                                max_ignore_bins=max_ignore_bins, should_skip_epoch_with_only_short_subsequences=should_skip_epoch_with_only_short_subsequences, debug_print=debug_print)
            """
            # _tmp_merge_split_positions_arrays = deepcopy(original_split_positions_arrays)
            n_tbins_list = np.array([len(v) for v in original_split_positions_arrays])
            n_total_tbins: int = np.sum(n_tbins_list)
            is_subsequence_potential_intrusion = (n_tbins_list <= max_ignore_bins) ## any subsequence shorter than the max ignore distance
            ignored_subsequence_idxs = np.where(is_subsequence_potential_intrusion)[0]

            if debug_print:
                print(f'original_split_positions_arrays: {original_split_positions_arrays}')
            
            subsequences_to_add = [] # not used
            subsequences_to_remove = [] # not used
            subsequence_replace_dict = {}
            
            final_intrusion_idxs = []
            
            # (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove)
            if (should_skip_epoch_with_only_short_subsequences and (np.all(is_subsequence_potential_intrusion))):
                if debug_print:
                    #TODO 2024-11-28 10:23: - [ ] This was previously thought to be the cause of an assertion, but it was later found that merging wasn't working properly in general and this was fixed by `_subfn_resolve_overlapping_replacement_indicies`
                    print(f'WARN: all subsequences are smaller than or equal in length to `max_ignore_bins`: {max_ignore_bins}, n_tbins_list: {n_tbins_list}. Not attempting to merge since we cannot tell which are intrusions.')
                final_out_subsequences = deepcopy(original_split_positions_arrays)

            else:
                # try to merge over intrusions
                for a_subsequence_idx, a_subsequence, a_n_tbins, is_potentially_ignored_intrusion in zip(np.arange(len(n_tbins_list)), original_split_positions_arrays, n_tbins_list, is_subsequence_potential_intrusion):
                    ## loop through ignored subsequences to find out how to merge them
                    #TODO 2024-11-27 08:36: - [ ] 
                    if (not is_potentially_ignored_intrusion):
                        # for an_ignored_subsequence_idx in ignored_subsequence_idxs:
                        active_subsequence_pos = deepcopy(original_split_positions_arrays[a_subsequence_idx])
                        # active_ignored_subsequence_indicies = deepcopy(self.split_indicies[a_subsequence_idx])
                        len_active_subsequence: int = len(active_subsequence_pos)
                        final_is_subsequence_intrusion: bool = False
                        
                        if debug_print:
                            print(f'an_ignored_subsequence_idx: {a_subsequence_idx}, active_ignored_subsequence: {active_subsequence_pos}')
                        (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index) = self._compute_sequences_spanning_ignored_intrusions(original_split_positions_arrays,
                                                                                                                                                n_tbins_list, target_subsequence_idx=a_subsequence_idx, max_ignore_bins=max_ignore_bins)
                        if left_congruent_flanking_sequence is None:
                            left_congruent_flanking_sequence = []
                        if right_congruent_flanking_sequence is None:
                            right_congruent_flanking_sequence = []
                        len_left_congruent_flanking_sequence: int = len(left_congruent_flanking_sequence)
                        len_right_congruent_flanking_sequence: int = len(right_congruent_flanking_sequence)

                        ## impose reasonable size rule
                        # bridging over intrusions only be done when at least one of the adjacent portions to be bridged exceeds a minimum size? This prevents piecing together bad sequences.
                        required_minimum_flanking_sequence_length: int = int((len_active_subsequence) * 2) # one must be at least twice as large as the bridged gap
                        # required_minimum_flanking_sequence_length: int = int((len_active_ignored_subsequence) + 1) # must be at least one larger than the bridged gap                    
                        are_both_flanking_sequences_too_short: bool = ((len_left_congruent_flanking_sequence < required_minimum_flanking_sequence_length) and (len_right_congruent_flanking_sequence < required_minimum_flanking_sequence_length))
                        if are_both_flanking_sequences_too_short:
                            if debug_print:
                                print(f'flanking sequences are too short required_minimum_flanking_sequence_length: {required_minimum_flanking_sequence_length}!\n\t (len_left_congruent_flanking_sequence: {len_left_congruent_flanking_sequence}, len_active_ignored_subsequence: {len_active_subsequence}, len_right_congruent_flanking_sequence: {len_right_congruent_flanking_sequence})')

                        if (are_both_flanking_sequences_too_short or ((len_left_congruent_flanking_sequence == 0) and (len_right_congruent_flanking_sequence == 0))):
                            ## do nothing
                            if debug_print:
                                print(f'\tWARN: merge NULL, both flanking sequences are empty or too short. Skipping')
                        else:
                            # decide on subsequence to merge                    
                            if len_left_congruent_flanking_sequence > len_right_congruent_flanking_sequence:
                                # merge LEFT sequence ________________________________________________________________________________________________ #
                                if debug_print:
                                    print(f'\tmerge LEFT: left_congruent_flanking_sequence: {left_congruent_flanking_sequence}, left_congruent_flanking_index: {left_congruent_flanking_index}')
                                new_subsequence = np.array([*left_congruent_flanking_sequence, *active_subsequence_pos])
                                ## remove old
                                for an_idx_to_remove in np.arange(left_congruent_flanking_index, (a_subsequence_idx+1)):                    
                                    subsequences_to_remove.append(original_split_positions_arrays[an_idx_to_remove])
                                # _tmp_merge_split_positions_arrays[left_congruent_flanking_index:(a_subsequence_idx+1)] = [] # new_subsequence ## or [] to clear it?
                                subsequences_to_add.append(new_subsequence)
                                # subsequence_replace_dict[tuple([original_split_positions_arrays[an_idx_to_remove] for an_idx_to_remove in np.arange(left_congruent_flanking_index, (a_subsequence_idx+1))])] = new_subsequence
                                subsequence_replace_dict[tuple(np.arange(left_congruent_flanking_index, (a_subsequence_idx+1)).tolist())] = (a_subsequence_idx, new_subsequence)
                                final_is_subsequence_intrusion = True
                            else:
                                # merge RIGHT, be biased towards merging right (due to lack of >= above) _____________________________________________ #
                                if debug_print:
                                    print(f'\tmerge RIGHT: right_congruent_flanking_sequence: {right_congruent_flanking_sequence}, right_congruent_flanking_index: {right_congruent_flanking_index}')
                                new_subsequence = np.array([*active_subsequence_pos, *right_congruent_flanking_sequence])
                                ## remove old
                                for an_idx_to_remove in np.arange(a_subsequence_idx, (right_congruent_flanking_index+1)):                    
                                    subsequences_to_remove.append(original_split_positions_arrays[an_idx_to_remove])
                                # _tmp_merge_split_positions_arrays[a_subsequence_idx:(right_congruent_flanking_index+1)] = [] # new_subsequence ## or [] to clear it?
                                subsequences_to_add.append(new_subsequence)
                                # subsequence_replace_dict[tuple([original_split_positions_arrays[an_idx_to_remove] for an_idx_to_remove in np.arange(a_subsequence_idx, (right_congruent_flanking_index+1))])] = new_subsequence
                                subsequence_replace_dict[tuple(np.arange(a_subsequence_idx, (right_congruent_flanking_index+1)).tolist())] = (a_subsequence_idx, new_subsequence)
                                final_is_subsequence_intrusion = True
                            ## END if len_left_congruent_flanking_sequ...
                                
                            #TODO 2024-12-04 04:55: - [ ] Why can't we merge both left AND then right?

                            if final_is_subsequence_intrusion:
                                # final_intrusion_idxs.append(deepcopy(active_ignored_subsequence_indicies))
                                final_intrusion_idxs.append(active_subsequence_pos)
                            
                            if debug_print:
                                print(f'\tnew_subsequence: {new_subsequence}')
                    else:
                        ## not ignored, include?
                        pass
                    
                # END for a_subsequence_idx, a_subsequence,...
                ## detect and prevent overlaps in the replacements, could also fix by finding and replacing iteratively?    
                if debug_print:
                    print(f'subsequence_replace_dict: {subsequence_replace_dict}')
                deduplicated_replace_dict = _subfn_resolve_overlapping_replacement_indicies(original_split_positions_arrays, subsequence_replace_dict, debug_print=debug_print)
                
                if debug_print:
                    print(f'deduplicated_replace_dict: {deduplicated_replace_dict}')
                    
                ## replace with the deduplicated replace dict
                subsequence_replace_dict = deduplicated_replace_dict
                

                final_out_subsequences = []
                replace_idxs = [v[0] for k, v in subsequence_replace_dict.items()]
                replace_idx_value_dict = {v[0]:v[1] for k, v in subsequence_replace_dict.items()} # will there be possible duplicates?
                
                any_remove_idxs = flatten([k for k, v in subsequence_replace_dict.items()])
                
                for a_subsequence_idx, a_subsequence, a_n_tbins, is_potentially_ignored_intrusion in zip(np.arange(len(n_tbins_list)), original_split_positions_arrays, n_tbins_list, is_subsequence_potential_intrusion):
                        is_curr_subsequence_idx_in_replace_list = False
                        if a_subsequence_idx in replace_idxs:
                            # the index to be replaced
                            replacement_subsequence = replace_idx_value_dict[a_subsequence_idx]
                            if debug_print:
                                print(f'final: replacing subsequence[{a_subsequence_idx}] ({a_subsequence}) with {replacement_subsequence}.')
                            final_out_subsequences.append(replacement_subsequence)
                            
                        elif a_subsequence_idx in any_remove_idxs:
                            ## to be excluded only
                            pass
                        else:
                            # included and unmodified
                            final_out_subsequences.append(a_subsequence)
                # END for a_subsequence_idx, a_subsequence, ...
                
            ## check validity of merge result
            final_out_subsequences_n_tbins_list = np.array([len(v) for v in final_out_subsequences])
            final_out_subsequences_n_total_tbins: int = np.sum(final_out_subsequences_n_tbins_list)
            assert final_out_subsequences_n_total_tbins == n_total_tbins, f"final_out_subsequences_n_total_tbins ({final_out_subsequences_n_total_tbins}) should equal the original n_total_tbins ({n_total_tbins}) but it does not!\n\t: final_out_subsequences: {final_out_subsequences}\n\toriginal_split_positions_arrays: {original_split_positions_arrays}"

            return original_split_positions_arrays, final_out_subsequences, (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove, final_intrusion_idxs)
        


        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #

        ## INPUTS: split_first_order_diff_arrays, continuous_sequence_lengths, max_ignore_bins
        assert self.split_positions_arrays is not None
        original_split_positions_arrays = deepcopy(self.split_positions_arrays)

        if debug_print:
            print(f'original_split_positions_arrays: {original_split_positions_arrays}')
        

        ## Begin by finding only the longest sequence
        n_tbins_list = np.array([len(v) for v in original_split_positions_arrays])
        longest_subsequence_idx: int = np.argmax(n_tbins_list)
        
        n_total_tbins: int = np.sum(n_tbins_list)
        is_subsequence_potential_intrusion = (n_tbins_list <= max_ignore_bins) ## any subsequence shorter than the max ignore distance
        ignored_subsequence_idxs = np.where(is_subsequence_potential_intrusion)[0]
        

        # call `_subfn_perform_merge_iteration` ______________________________________________________________________________ #
        original_split_positions_arrays, final_out_subsequences, (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove, final_intrusion_idxs) = _subfn_perform_merge_iteration(original_split_positions_arrays,
                                                                                                                                                                            max_ignore_bins=max_ignore_bins, should_skip_epoch_with_only_short_subsequences=should_skip_epoch_with_only_short_subsequences, debug_print=debug_print)

        ## OUTPUTS: final_out_subsequences
        

        # Update Self ________________________________________________________________________________________________________ #
        ## update self properties
        self.merged_split_positions_arrays = deepcopy(final_out_subsequences)
        
        # self.bridged_intrusion_bin_indicies = final_intrusion_idxs
        
        if len(final_intrusion_idxs) > 0:
            if debug_print:
                print(f'final_intrusion_idxs: {final_intrusion_idxs}')
        
        ## update dataframe
        self.rebuild_sequence_info_df()
        
        # return (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index)
        return original_split_positions_arrays, final_out_subsequences, (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove, final_intrusion_idxs)


    @function_attributes(short_name=None, tags=['merge', '_compute_sequences_spanning_ignored_intrusions', 'PhoOriginal'], input_requires=['self.split_positions_arrays'], output_provides=['self.merged_split_positions_arrays'], uses=['_compute_sequences_spanning_ignored_intrusions'], used_by=[], creation_date='2024-11-27 08:17', related_items=['_compute_sequences_spanning_ignored_intrusions'])
    def enforce_max_jump_distance(self, max_jump_distance_cm: float = 30.0, debug_print=False):
        """ an "intrusion" refers to one or more time bins that interrupt a longer sequence that would be monotonic if the intrusions were removed.

        The quintessential example would be a straight ramp that is interrupted by a single point discontinuity intrusion.  

        iNPUTS:
            split_first_order_diff_arrays: a list of split arrays of 1st order differences.
            continuous_sequence_lengths: a list of integer sequence lengths
            longest_sequence_start_idx: int - the sequence start index to start the scan in each direction
            max_ignore_bins: int = 2 - this means to disolve sequences shorter than 2 time bins if the sequences on each side are in the same direction

            
        REQUIRES: self.merged_split_positions_arrays, self.split_positions_arrays
        UPDATES: self.merged_split_positions_arrays
            
        Usage:

        HISTORY: 2024-11-27 08:18 based off of `_compute_sequences_spanning_ignored_intrusions`
        
        """
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #

        ## INPUTS: split_first_order_diff_arrays, continuous_sequence_lengths, max_ignore_bins
        assert self.split_positions_arrays is not None
        original_split_positions_arrays = deepcopy(self.split_positions_arrays)
        original_merged_split_positions_arrays = deepcopy(self.merged_split_positions_arrays)
        if debug_print:
            print(f'original_split_positions_arrays: {original_split_positions_arrays}, original_merged_split_positions_arrays: {original_merged_split_positions_arrays}')
        

        ## Begin by finding only the longest sequence
        n_tbins_list = np.array([len(v) for v in original_split_positions_arrays])
        
        # new_merged_split_positions_arrays = deepcopy(original_merged_split_positions_arrays)
        new_merged_split_positions_arrays = []
        for a_subsequence_idx, a_subsequence in enumerate(original_merged_split_positions_arrays):
            ## iterate through subsequences
            a_subsequence_first_order_diff = np.diff(a_subsequence, n=1)
            does_jump_exceed_max = (a_subsequence_first_order_diff > max_jump_distance_cm)
            num_jumps_exceeding_max: int = np.count_nonzero(does_jump_exceed_max)
            jump_exceeds_max_idxs = np.where(does_jump_exceed_max)[0]
            if num_jumps_exceeding_max > 0:
                print(f'does_jump_exceed_max: {does_jump_exceed_max}')
                a_split_subsequence = np.split(a_subsequence, jump_exceeds_max_idxs) ## splits into sub bins
                ## TODO: keep track of splits?
                # new_merged_split_positions_arrays[a_subsequence_idx] = a_split_subsequence
                for a_sub_sub_sequence in a_split_subsequence:
                    new_merged_split_positions_arrays.append(a_sub_sub_sequence) ## append the split subsequence
                    
            else:
                ## no split needed
                new_merged_split_positions_arrays.append(a_subsequence)

        ## OUTPUTS: final_out_subsequences
        

        # Update Self ________________________________________________________________________________________________________ #
        ## update self properties
        # self.merged_split_positions_arrays = deepcopy(final_out_subsequences)
        
        # self.bridged_intrusion_bin_indicies = final_intrusion_idxs
        
        # if len(final_intrusion_idxs) > 0:
        #     if debug_print:
        #         print(f'final_intrusion_idxs: {final_intrusion_idxs}')
        
        ## update dataframe
        # self.rebuild_sequence_info_df()
        
        self.merged_split_positions_arrays = new_merged_split_positions_arrays
        
        # return (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index)
        # return original_split_positions_arrays, final_out_subsequences, (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove, final_intrusion_idxs)
        return new_merged_split_positions_arrays
    


    @function_attributes(short_name=None, tags=['merge', 'intrusion', 'subsequences', 'ISSUE'], input_requires=[], output_provides=[], uses=[], used_by=['self.merge_over_ignored_intrusions'], creation_date='2024-12-04 04:43', related_items=[])
    @classmethod
    def _compute_sequences_spanning_ignored_intrusions(cls, subsequence_list, continuous_sequence_lengths, target_subsequence_idx: int, max_ignore_bins: int = 2):
        """ an "intrusion" refers to one or more time bins that interrupt a longer sequence that would be monotonic if the intrusions were removed.

        We aren't just trying to arbitrarily merge short subsequences, we're only trying to bridge over short "intrusions"
        
        The quintessential example would be a straight ramp that is interrupted by a single point discontinuity intrusion.  

        iNPUTS:
            split_first_order_diff_arrays: a list of split arrays of 1st order differences.
            continuous_sequence_lengths: a list of integer sequence lengths
            longest_sequence_start_idx: int - the sequence start index to start the scan in each direction
            max_ignore_bins: int = 2 - this means to disolve sequences shorter than 2 time bins if the sequences on each side are in the same direction

            
        Usage:
        
        TODO: convert to use `is_valid_sequence_index(...)`?

        """
        ## INPUTS: split_first_order_diff_arrays, continuous_sequence_lengths, max_ignore_bins
        Assert.same_length(subsequence_list, continuous_sequence_lengths)
        
        left_congruent_flanking_index = None
        left_congruent_flanking_sequence = None

        right_congruent_flanking_index = None
        right_congruent_flanking_sequence = None        

        ## from the location of the longest sequence, check for flanking sequences <= `max_ignore_bins` in each direction.
        # Scan left:
        if is_valid_sequence_index(subsequence_list, (target_subsequence_idx-1)):
            left_flanking_index = (target_subsequence_idx-1)
            left_flanking_seq_length = continuous_sequence_lengths[left_flanking_index]
            if (left_flanking_seq_length <= max_ignore_bins):
                ## only if shorter than the max_ignore_bins can the left flanking sequence can be considered -- #TODO 2024-12-04 07:52: - [ ] Logic Error introduced here. I was imaginging a long sequence being potentially joined to shorter "intrusion" sequences, but this is the opposite of how I use the code. 
                # left_congruent_flanking_sequence = [*subsequence_list[left_flanking_index]]
                ### Need to look even FURTHER to the left to see the prev sequence:
                if is_valid_sequence_index(subsequence_list, (target_subsequence_idx-2)):
                    left_congruent_flanking_index = (target_subsequence_idx-2)
                    ## Have a sequence to concatenate with
                    left_congruent_flanking_sequence = [*subsequence_list[left_congruent_flanking_index], *subsequence_list[left_flanking_index]]
                    # left_congruent_flanking_sequence.extend(subsequence_list[left_congruent_flanking_index])
                    # left_congruent_flanking_sequence = subsequence_list[left_congruent_flanking_index]
                # else:
                #     left_congruent_flanking_index = left_flanking_index
                
        # Scan right:
        if is_valid_sequence_index(subsequence_list, (target_subsequence_idx+1)):
            right_flanking_index = (target_subsequence_idx+1)
            right_flanking_seq_length = continuous_sequence_lengths[right_flanking_index]
            if (right_flanking_seq_length <= max_ignore_bins):
                # right_congruent_flanking_sequence = [*subsequence_list[right_flanking_index]]
                
                ### Need to look even FURTHER to the left to see the prev sequence:
                if is_valid_sequence_index(subsequence_list, (target_subsequence_idx+2)):
                    right_congruent_flanking_index = (target_subsequence_idx+2)
                    ## Have a sequence to concatenate with
                    right_congruent_flanking_sequence = [*subsequence_list[right_flanking_index], *subsequence_list[right_congruent_flanking_index]]
                    # right_congruent_flanking_sequence.extend(subsequence_list[right_congruent_flanking_index])
                    # right_congruent_flanking_sequence = subsequence_list[right_congruent_flanking_index]
                # else:
                #     right_congruent_flanking_index = right_flanking_index

        return (left_congruent_flanking_sequence, left_congruent_flanking_index), (right_congruent_flanking_sequence, right_congruent_flanking_index)


    @classmethod
    def determine_directionality(cls, subseq: NDArray, return_normalized: bool=True) -> float:
        """ determines the directionality of a subsequence between -1 and +1
        
        """
        if isinstance(subseq, list):
            subseq = np.array(subseq)

        first_order_diff_lst = np.diff(subseq, n=1) # , prepend=[subseq[0]]

        n_diff_bins: int = len(first_order_diff_lst)
        n_original_bins: int = n_diff_bins + 1
        
        n_nonzero_diff_bins: int = np.sum(np.nonzero(first_order_diff_lst)) # len(first_order_diff_lst)
    
        sign_diff = np.sign(first_order_diff_lst) # -1, 0, +1 for each change    
        sign_total = np.sum(sign_diff)
        n_nonzero_sign_bins: int = np.sum(np.nonzero(sign_diff)) # len(first_order_diff_lst)
    
        if not return_normalized:
            return sign_total
        else:
            norm_sign_total = sign_total / float(n_nonzero_sign_bins) # float(len(sign_diff))
            return norm_sign_total
        
        # np.sum(first_order_diff_lst) / float(n_diff_bins))

        # diffs = [subseq[i+1] - subseq[i] for i in range(len(subseq)-1)]
        # if all(diff > 0 for diff in diffs):
        #     return 'increasing'
        # elif all(diff < 0 for diff in diffs):
        #     return 'decreasing'
        # else:
        #     return 'none'
        


    def compute(self, debug_print=False):
        """ recomputes all """
        
        partition_result_dict = self.partition_subsequences_ignoring_repeated_similar_positions(a_most_likely_positions_list=self.flat_positions, n_pos_bins=self.n_pos_bins, flat_time_window_centers=self.flat_time_window_centers, flat_time_window_edges=self.flat_time_window_edges, 
                                                                                                same_thresh=self.same_thresh, max_ignore_bins=self.max_ignore_bins, debug_print=debug_print)  # Add 1 because np.diff reduces the index by 1

        self.__dict__.update(partition_result_dict) ## update self from the result - first_order_diff_lst, list_parts, diff_split_indicies, split_indicies, low_magnitude_change_indicies
        
        # # ChatGPT Method 2024-11-04 - 2pm ____________________________________________________________________________________ #
        # partition_result.partition_subsequences() ## 2024-12-04 09:15 New
        # partition_result.merge_intrusions() ## 2024-12-04 09:15 New


        # Pho Method 2024-11-04 - Pre 2pm ____________________________________________________________________________________ #
        # # Set `partition_result.split_positions_arrays` ______________________________________________________________________ #
        # active_split_indicies = deepcopy(partition_result.split_indicies) ## this is what it should be, but all the splits are +1 later than they should be
        # active_split_indicies = deepcopy(partition_result.diff_split_indicies) - 1 ## this is what it should be, but all the splits are +1 later than they should be
        active_split_indicies = deepcopy(self.split_indicies) ## this is what it should be, but all the splits are +1 later than they should be
        split_most_likely_positions_arrays = np.split(self.flat_positions, active_split_indicies)
        self.split_positions_arrays = split_most_likely_positions_arrays
        
        # Set `merged_split_positions_arrays` ________________________________________________________________________________ #
        _tmp_merge_split_positions_arrays, final_out_subsequences, (subsequence_replace_dict, subsequences_to_add, subsequences_to_remove, final_intrusion_values_list) = self.merge_over_ignored_intrusions(max_ignore_bins=self.max_ignore_bins, debug_print=debug_print)
        # flat_positions_list = deepcopy(partition_result.flat_positions.tolist())
        self.bridged_intrusion_bin_indicies = deepcopy(final_intrusion_values_list) # np.array([flat_positions_list.index(v) for v in final_intrusion_idxs])
        
        if self.max_jump_distance_cm is not None:
            self.enforce_max_jump_distance(max_jump_distance_cm=self.max_jump_distance_cm, debug_print=debug_print)

        ## Common Post-hoc
        self.rebuild_sequence_info_df()

    # 2024-12-04 09:16 New ChatGPT Simplified Functions __________________________________________________________________ #


    def rebuild_sequence_info_df(self) -> pd.DataFrame:
        """Rebuilds the sequence_info_df using the new subsequences."""
        # Initialize the dataframe
        self.sequence_info_df = pd.DataFrame({'pos': self.flat_positions})

        if self.flat_time_window_centers is not None:
            assert len(self.flat_time_window_centers) == len(self.flat_positions)
            self.sequence_info_df['t_center'] = self.flat_time_window_centers

        # Assign original subsequence indices
        if self.split_positions_arrays is not None:
            self.sequence_info_df['orig_subsequence_idx'] = flatten(
                [[i] * len(v) for i, v in enumerate(self.split_positions_arrays)])
            # Mark intrusions based on max_ignore_bins
            is_intrusion: List[bool] = [len(v) <= self.max_ignore_bins for v in self.split_positions_arrays]
            intrusion_dict = {idx: val for idx, val in enumerate(is_intrusion)}
            self.sequence_info_df['is_intrusion'] = self.sequence_info_df['orig_subsequence_idx'].map(intrusion_dict)

        # Assign merged subsequence indices
        if self.merged_split_positions_arrays is not None:
            self.sequence_info_df['subsequence_idx'] = flatten(
                [[i] * len(v) for i, v in enumerate(self.merged_split_positions_arrays)])


        self.sequence_info_df['flat_idx'] = self.sequence_info_df.index.astype(int, copy=True)
        return self.sequence_info_df

    # Existing properties and methods remain unchanged or are adjusted as necessary
    # Update properties that depend on split_positions_arrays and merged_split_positions_arrays

    # @function_attributes(short_name=None, tags=['private', 'ChatGPT', '2024-12-04_09:16_Simplified'], input_requires=[], output_provides=[], uses=[], used_by=['init_from_positions_list'], creation_date='2024-12-04 10:07', related_items=[])
    # def partition_subsequences(self):
    #     """
    #     Partitions the positions into subsequences based on significant direction changes.
    #     Small changes below the threshold (jitter around a stationary position) are ignored.
    #     Requires: self.flat_positions, self.same_thresh
    #     Updates `self.split_positions_arrays`.
    #     """
    #     positions = self.flat_positions
    #     same_thresh = self.same_thresh

    #     # Compute first-order differences and directions
    #     diffs = np.diff(positions, prepend=positions[0])
    #     directions = np.sign(diffs)

    #     subsequences = []
    #     current_subseq = [positions[0]]
    #     prev_dir = directions[0]

    #     for i in range(1, len(positions)):
    #         curr_dir = directions[i]
    #         diff = diffs[i]

    #         # Determine if the direction has changed significantly
    #         if curr_dir != prev_dir and np.abs(diff) > same_thresh:
    #             subsequences.append(np.array(current_subseq))
    #             current_subseq = [positions[i]]
    #             prev_dir = curr_dir
    #         else:
    #             current_subseq.append(positions[i])
    #             # Update the direction if the change is significant
    #             if np.abs(diff) > same_thresh:
    #                 prev_dir = curr_dir

    #     if current_subseq:
    #         subsequences.append(np.array(current_subseq))

    #     self.split_positions_arrays = subsequences

    # @function_attributes(short_name=None, tags=['private', 'ChatGPT', '2024-12-04_09:16_Simplified'], input_requires=[], output_provides=[], uses=[], used_by=['init_from_positions_list'], creation_date='2024-12-04 10:07', related_items=[])
    # def merge_intrusions(self):
    #     """
    #     Merges short subsequences (intrusions) into adjacent longer sequences.
    #     Requires: self.split_positions_arrays, self.max_ignore_bins
    #     Updates: self.merged_split_positions_arrays
    #     """
    #     subsequences = self.split_positions_arrays
    #     max_ignore_bins = self.max_ignore_bins

    #     merged_subsequences = []
    #     i = 0
    #     while i < len(subsequences):
    #         current_seq = subsequences[i]
    #         current_seq_len = len(current_seq)
    #         if current_seq_len <= max_ignore_bins:
    #             # Potential intrusion
    #             left_seq = merged_subsequences[-1] if merged_subsequences else None
    #             right_seq = subsequences[i + 1] if i + 1 < len(subsequences) else None

    #             # Decide which side to merge with
    #             merged = False
    #             if left_seq is not None:
    #                 left_seq_len = len(left_seq)
    #                 if left_seq_len >= (current_seq_len * 2):
    #                     # Merge with left sequence
    #                     merged_seq = np.concatenate([left_seq, current_seq])
    #                     merged_subsequences[-1] = merged_seq
    #                     merged = True

    #             if not merged and right_seq is not None:
    #                 right_seq_len = len(right_seq)
    #                 if right_seq_len >= (current_seq_len * 2):
    #                     # Merge with right sequence
    #                     merged_seq = np.concatenate([current_seq, right_seq])
    #                     i += 1  # Skip the next sequence as it's merged
    #                     merged_subsequences.append(merged_seq)
    #                     merged = True

    #             if not merged:
    #                 # Cannot merge, keep as is
    #                 merged_subsequences.append(current_seq)
    #         else:
    #             # Not an intrusion
    #             merged_subsequences.append(current_seq)
    #         i += 1

    #     self.merged_split_positions_arrays = merged_subsequences
        

    # Visualization/Graphical Debugging __________________________________________________________________________________ #
    @function_attributes(short_name=None, tags=['plot', 'matplotlib', 'figure', 'debug'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-27 06:36', related_items=['SubsequencesPartitioningResult'])
    @classmethod
    def _debug_plot_time_bins_multiple(cls, positions_list, num='debug_plot_time_binned_positions', ax=None, enable_position_difference_indicators=True, defer_show: bool = False, flat_time_window_centers=None, flat_time_window_edges=None,
                                        enable_axes_formatting: bool = False,  arrow_alpha: float = 0.4, subsequence_line_color_alpha: float = 0.55,
                                        is_intrusion: Optional[NDArray] = None, direction_changes: Optional[NDArray] = None, **kwargs):
            """
            Plots positions over fixed-width time bins with vertical lines separating each bin.
            Each sublist in positions_list is plotted in a different color.

            Parameters:
            - positions_list (list of lists/arrays): List of position arrays/lists.
            - is_intrusion (array of bools, optional): Array indicating whether each time bin is an intrusion.
            - direction_changes (array of bools, optional): Array indicating where direction changes occur.
            - num (str or int): Figure number or name.
            - ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.

            Returns:
            - out: MatplotlibRenderPlots object containing the figure, axes, and plot elements.
            """
            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
            import matplotlib.pyplot as plt
            from neuropy.utils.matplotlib_helpers import modify_colormap_alpha
            import matplotlib.transforms as mtransforms


            # Plot Customization _________________________________________________________________________________________________ #
            split_vlines_kwargs = dict(color='black', linestyle='-', linewidth=1) | kwargs.pop('split_vlines_kwargs', {})
            time_bin_edges_vlines_kwargs = dict(color='grey', linestyle='--', linewidth=0.5) | kwargs.pop('time_bin_edges_vlines_kwargs', {})
            direction_change_lines_kwargs = dict(color='yellow', linestyle=':', linewidth=2, zorder=1) | kwargs.pop('direction_change_lines_kwargs', {})

            intrusion_time_bin_shading_kwargs = dict(facecolor='red', alpha=0.15, zorder=0) | kwargs.pop('intrusion_time_bin_shading_kwargs', {})
            sequence_position_hlines_kwargs = dict(linewidth=4, zorder=-1) | kwargs.pop('sequence_position_hlines_kwargs', {})


            
            main_sequence_position_dots_kwargs = dict(linewidths=2, marker ="^", edgecolor="#141414F9", s = 200, zorder=1) | kwargs.pop('main_sequence_position_dots_kwargs', {}) # "#141414F9" -- near black
            
            # Example override dict ______________________________________________________________________________________________ #
            # dict(
            #     split_vlines_kwargs = dict(color='black', linestyle='-', linewidth=1, should_skip=False),
            #     time_bin_edges_vlines_kwargs = dict(color='grey', linestyle='--', linewidth=0.5, should_skip=False) 
            #     direction_change_lines_kwargs = dict(color='yellow', linestyle=':', linewidth=2, zorder=1, should_skip=False),
            #     intrusion_time_bin_shading_kwargs = dict(facecolor='red', alpha=0.15, zorder=0, should_skip=False),
            #     sequence_position_hlines_kwargs = dict(linewidth=4, zorder=-1, should_skip=False),
            #     main_sequence_position_dots_kwargs = dict(linewidths=2, marker ="^", edgecolor ="red", s = 200, zorder=1),
            # )
            

            should_skip_split_vlines: bool = split_vlines_kwargs.pop('should_skip', False)
            should_skip_time_bin_edges_vlines: bool = time_bin_edges_vlines_kwargs.pop('should_skip', False)
            should_skip_direction_change_lines: bool = direction_change_lines_kwargs.pop('should_skip', False)
            should_skip_intrusion_time_bin_shading: bool = intrusion_time_bin_shading_kwargs.pop('should_skip', False)
            should_skip_sequence_position_hlines: bool = sequence_position_hlines_kwargs.pop('should_skip', False)
            should_skip_main_sequence_position_dots: bool = main_sequence_position_dots_kwargs.pop('should_skip', False)
            

            # Begin Function Body ________________________________________________________________________________________________ #
            out_dict = {'time_bin_edges_vlines': None, 'split_vlines': None, 'subsequence_positions_hlines_dict': None,
                        'subsequence_arrows_dict': None, 'subsequence_arrow_labels_dict': None, 'main_sequence_tbins_axhlines': None,
                        'intrusion_shading': None, 'direction_change_lines': None, 'main_sequence_position_dots': None,
                        }

            if ax is None:
                fig, ax = plt.subplots(num=num, clear=True)
            else:
                fig = ax.get_figure()

            if not isinstance(positions_list, (list, tuple,)):
                ## Wrap in a list
                positions_list = [positions_list, ]

            # Flatten the positions_list to get all positions for setting y-limits
            all_positions = np.concatenate(positions_list)
            N = len(all_positions)

            # Prepare x-values for time bins
            if flat_time_window_edges is not None:
                ## Prefer edges over centers
                assert (len(flat_time_window_edges)-1) == N, f"(len(flat_time_window_edges)-1): {(len(flat_time_window_edges)-1)} and len(all_positions): {len(all_positions)}"
                x_bins = deepcopy(flat_time_window_edges[:-1])  ## Left edges of bins
                bin_width: float = np.median(np.diff(x_bins))
                x_starts = x_bins
                x_ends = x_bins + bin_width
            elif flat_time_window_centers is not None:
                assert len(flat_time_window_centers) == N, f"flat_time_window_centers must have the same length as positions, but len(flat_time_window_centers): {len(flat_time_window_centers)} and len(all_positions): {len(all_positions)}"
                x_bins = deepcopy(flat_time_window_centers)
                bin_width: float = np.median(np.diff(x_bins))
                x_starts = x_bins - bin_width / 2
                x_ends = x_bins + bin_width / 2
            else:
                ## Use indices as the x_bins
                x_bins = np.arange(N + 1)  # Time bin edges
                bin_width: float = 1.0
                x_starts = x_bins[:-1]
                x_ends = x_bins[1:]

            # Calculate y-limits with padding
            ymin = min(all_positions) - 10
            ymax = max(all_positions) + 10

            # Calculate group lengths and group end indices
            group_lengths = (float(bin_width) * np.array([len(positions) for positions in positions_list]))
            group_end_indices = np.cumsum(group_lengths)[:-1]  # Exclude the last index

            ## Find the longest subsequence
            longest_subsequence_idx: int = np.argmax(group_lengths)
            longest_subsequence_start_x = None
            longest_subsequence_end_x = None

            # Plot vertical lines at regular time bins excluding group splits
            regular_x_bins = np.setdiff1d(x_bins, group_end_indices)
            if not should_skip_time_bin_edges_vlines:
                out_dict['time_bin_edges_vlines'] = ax.vlines(regular_x_bins, ymin, ymax, **time_bin_edges_vlines_kwargs)

            if not should_skip_split_vlines:
                # Highlight separator lines where splits occur
                out_dict['split_vlines'] = ax.vlines(group_end_indices, ymin, ymax, **split_vlines_kwargs)


            # Define a colormap
            cmap = plt.get_cmap('tab10')
            cmap = modify_colormap_alpha(cmap=cmap, alpha=subsequence_line_color_alpha)
            num_colors: int = cmap.N

            # Initialize intrusion shading
            out_dict['intrusion_shading'] = []

            # Shade intrusion time bins
            if not should_skip_intrusion_time_bin_shading:
                if is_intrusion is not None:
                    for i in range(N):
                        if is_intrusion[i]:
                            ax.axvspan(x_starts[i], x_ends[i], **intrusion_time_bin_shading_kwargs)
                            out_dict['intrusion_shading'].append((x_starts[i], x_ends[i]))

            
            if not should_skip_direction_change_lines:
                # Draw direction change lines
                out_dict['direction_change_lines'] = []
                if direction_changes is not None:
                    for i in range(N):
                        if direction_changes[i]:
                            x_line = x_starts[i]
                            line = ax.axvline(x_line, **direction_change_lines_kwargs)
                            out_dict['direction_change_lines'].append(line)

            # Plot horizontal lines with customizable color
            out_dict['subsequence_positions_hlines_dict'] = {}
            out_dict['subsequence_arrows_dict'] = {}
            out_dict['subsequence_arrow_labels_dict'] = {}
            out_dict['main_sequence_position_dots'] = {}
            
            # Keep track of the current x position
            x_start = x_starts[0]
            position_index = 0  # Global position index across all subsequences

            ## sort the subsequences by length so that the colors are always assigned in a consistent order (longest == color0, 2nd-longests == color1, ....)
            subsequence_lengths = np.array([len(x) for x in positions_list])
            main_subsequence_length: int = np.max(subsequence_lengths)
            subsequence_len_sort_indicies = np.argsort(subsequence_lengths)[::-1] ## reverse sorted for length
            
            for subsequence_idx, subsequence_positions in enumerate(positions_list):
                num_positions: int = len(subsequence_positions)
                curr_subsequence_end_position: float = float(num_positions) * bin_width
                # color = cmap(subsequence_idx % num_colors)
                curr_subsequence_size_sorted_idx: int = subsequence_len_sort_indicies[subsequence_idx]
                color = cmap(curr_subsequence_size_sorted_idx % num_colors) ## curr color set here
                
                is_main_sequence: bool = (num_positions == main_subsequence_length) # (curr_subsequence_size_sorted_idx == 0) # longest subsequence
                x_rel_indices = np.arange(num_positions)
                x_indices = x_start + (x_rel_indices * bin_width)
                x_starts_subseq = x_indices
                x_centers_subseq = x_indices + (float(bin_width) / 2.0)
                x_ends_subseq = x_indices + bin_width

                if (subsequence_idx == longest_subsequence_idx):
                    longest_subsequence_start_x = x_starts_subseq[0]
                    longest_subsequence_end_x = x_ends_subseq[-1]

                if is_main_sequence and (not should_skip_main_sequence_position_dots):
                    ## plot the dots indidicating that this is the main sequence
                    print(f'main_sequence_position_dots -- color: {color}\n\tsubsequence_idx: {subsequence_idx}, subsequence_positions: {subsequence_positions}')
                    out_dict['main_sequence_position_dots'][subsequence_idx] = ax.scatter(x_centers_subseq, subsequence_positions, color=color, **main_sequence_position_dots_kwargs)

                # Plot horizontal lines for position values within each time bin
                # Adjust colors for intrusion time bins
                # colors = [color if is_intrusion is None or not is_intrusion[position_index + i] else 'red' for i in range(num_positions)]
                colors = [color for i in range(num_positions)]
                if not should_skip_sequence_position_hlines:
                    out_dict['subsequence_positions_hlines_dict'][subsequence_idx] = ax.hlines(subsequence_positions, xmin=x_starts_subseq, xmax=x_ends_subseq, colors=colors, **sequence_position_hlines_kwargs)

                if enable_position_difference_indicators:
                    ## Draw "change" arrows between each adjacent bin showing the amount of y-pos change
                    arrow_color = (0, 0, 0, arrow_alpha,)
                    arrow_text_outline_color = (1.0, 1.0, 1.0, arrow_alpha)

                    out_dict['subsequence_arrows_dict'][subsequence_idx] = []
                    out_dict['subsequence_arrow_labels_dict'][subsequence_idx] = []

                    # Now, for each pair of adjacent positions within the group, draw arrows and labels
                    for i in range(num_positions - 1):
                        delta_pos = subsequence_positions[i+1] - subsequence_positions[i]
                        x0 = x_starts_subseq[i] + (bin_width / 2.0)
                        x1 = x_starts_subseq[i+1] + (bin_width / 2.0)
                        y0 = subsequence_positions[i]
                        y1 = subsequence_positions[i+1]

                        # Draw an arrow from (x0, y0) to (x1, y1)
                        arrow = ax.annotate(
                            '',
                            xy=(x1, y1),
                            xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='->', color=arrow_color, shrinkA=0, shrinkB=0, linewidth=1),
                        )
                        out_dict['subsequence_arrows_dict'][subsequence_idx].append(arrow)

                        # Place the label near the midpoint of the arrow
                        xm = (x0 + x1) / 2
                        ym = (y0 + y1) / 2
                        txt = ax.text(
                            xm, ym,
                            f'{delta_pos:+.2f}',  # Format with sign and two decimal places
                            fontsize=6,
                            ha='center',
                            va='bottom',
                            color=arrow_color,
                            bbox=dict(facecolor=arrow_text_outline_color, edgecolor='none', alpha=arrow_alpha, pad=0.5)  # Add background for readability
                        )
                        out_dict['subsequence_arrow_labels_dict'][subsequence_idx].append(txt)

                # Update x_start for next group
                x_start += curr_subsequence_end_position
                position_index += num_positions

            if enable_axes_formatting:
                # Set axis labels and limits
                ax.set_xlabel('Time Bins')
                ax.set_ylabel('Position')
                # ax.set_xlim(0, N)
                # ax.set_xticks(x_bins)
                # ax.set_ylim(ymin, ymax)

            out = MatplotlibRenderPlots(name='test', figures=[fig, ], axes=ax, plots=out_dict, **kwargs)
            if not defer_show:
                plt.show()

            return out  # Return the MatplotlibRenderPlots object



    def plot_time_bins_multiple(self, num='debug_plot_time_binned_positions', ax=None, enable_position_difference_indicators=True, enable_axes_formatting=False, override_positions_list=None, flat_time_window_edges=None,
                                arrow_alpha: float = 0.4, subsequence_line_color_alpha: float = 0.55, **kwargs):
        if override_positions_list is None:
            override_positions_list = self.merged_split_positions_arrays
        if flat_time_window_edges is None:
            flat_time_window_edges = self.flat_time_window_edges

        # Extract 'is_intrusion' array from sequence_info_df
        is_intrusion = None
        direction_changes = None
        if self.sequence_info_df is not None:
            if 'is_intrusion' in self.sequence_info_df.columns:
                is_intrusion = self.sequence_info_df['is_intrusion'].values
            if 'direction_change' in self.sequence_info_df.columns:
                direction_changes = self.sequence_info_df['direction_change'].values

        return self._debug_plot_time_bins_multiple(
            positions_list=override_positions_list,
            num=num,
            ax=ax,
            enable_position_difference_indicators=enable_position_difference_indicators,
            flat_time_window_centers=kwargs.pop('flat_time_window_centers', self.flat_time_window_centers),
            flat_time_window_edges=flat_time_window_edges,
            enable_axes_formatting=enable_axes_formatting,
            arrow_alpha=arrow_alpha,
            subsequence_line_color_alpha=subsequence_line_color_alpha,
            is_intrusion=is_intrusion,  # Pass the intrusion information
            direction_changes=direction_changes,  # Pass the direction change information
            **kwargs
        )



    def _plot_step_by_step_subsequence_partition_process(self):
        """ diagnostic for debugging the step-by-step sequence partitioning heuristics 
        
        out: MatplotlibRenderPlots = partition_result._plot_step_by_step_subsequence_partition_process()
        out

        """
        from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
        import matplotlib.pyplot as plt
        

        common_plot_time_bins_multiple_kwargs = dict(subsequence_line_color_alpha=0.95, arrow_alpha=0.4, enable_axes_formatting=True, )
        

        merged_plots_out_dict = {}
        
        fig = plt.figure(layout="constrained", clear=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["ax_ungrouped_seq"],
                ["ax_grouped_seq"], # "ax_SHORT_activity_v_time", "ax_SHORT_pf_tuning_curve"],
                ["ax_merged_grouped_seq"],
            ],
            # set the height ratios between the rows
            # height_ratios=[8, 1],
            # height_ratios=[1, 1],
            # set the width ratios between the columns
            # width_ratios=[1, 8, 8, 1],
            sharex=True, sharey=True,
            gridspec_kw=dict(wspace=0, hspace=0.15) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
        )


        flat_time_window_edges = np.arange(self.total_num_subsequence_bins+1)
        
        split_most_likely_positions_arrays = deepcopy(self.flat_positions) ## unsplit positions
        # out: MatplotlibRenderPlots = SubsequencesPartitioningResult._debug_plot_time_bins_multiple(positions_list=split_most_likely_positions_arrays, ax=ax_dict["ax_ungrouped_seq"])
        out: MatplotlibRenderPlots = self.plot_time_bins_multiple(num='debug_plot_merged_time_binned_positions', ax=ax_dict["ax_ungrouped_seq"], enable_position_difference_indicators=True,
            flat_time_window_edges=flat_time_window_edges, override_positions_list=split_most_likely_positions_arrays, **common_plot_time_bins_multiple_kwargs,
        )
        merged_plots_out_dict["ax_ungrouped_seq"] = out.plots
        ## Add initially-sequenced result:
        split_positions_arrays = deepcopy(self.split_positions_arrays)
        out2: MatplotlibRenderPlots = self.plot_time_bins_multiple(num='debug_plot_merged_time_binned_positions', ax=ax_dict["ax_grouped_seq"], enable_position_difference_indicators=True,
            flat_time_window_edges=flat_time_window_edges, override_positions_list=split_positions_arrays, **common_plot_time_bins_multiple_kwargs,
            sequence_position_hlines_kwargs=dict(linewidth=3, linestyle='-', zorder=11, alpha=1.0),
        )
        merged_plots_out_dict["ax_grouped_seq"] = out2.plots
        
        #(offset, (on_off_seq)). For example, (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space) with no offset, while (5, (10, 3)), means (10pt line, 3pt space), but skip the first 5pt line.
        # linestyle = (0, (5, 1))
        # linestyle = (0, (1, 1)) # dots with 1pt dot, 0.5pt space
        linestyle = '-'
        # Plot only the positions themselves, as dotted overlaying lines
        pre_merged_debug_sequences_kwargs = dict(sequence_position_hlines_kwargs=dict(linewidth=2, linestyle=linestyle, zorder=10, alpha=1.0), # high-zorder to place it on-top, linestyle is "densely-dashed"
            split_vlines_kwargs = dict(should_skip=False),
            time_bin_edges_vlines_kwargs = dict(should_skip=False),
            direction_change_lines_kwargs = dict(should_skip=True),
            intrusion_time_bin_shading_kwargs = dict(should_skip=False),
        )
                    

        ## Add re-sequenced (merged) result:
        merged_split_positions_arrays = deepcopy(self.merged_split_positions_arrays)
        out3: MatplotlibRenderPlots = self.plot_time_bins_multiple(num='debug_plot_merged_time_binned_positions', ax=ax_dict["ax_merged_grouped_seq"], enable_position_difference_indicators=True,
            flat_time_window_edges=flat_time_window_edges, override_positions_list=merged_split_positions_arrays, **common_plot_time_bins_multiple_kwargs, **pre_merged_debug_sequences_kwargs,
        )        
        merged_plots_out_dict["ax_merged_grouped_seq"] = out3.plots
        
        # out.plots = merged_plots_out_dict ## set main plots to the dict of plots

        # out3: MatplotlibRenderPlots = self.plot_time_bins_multiple(num='debug_plot_merged_time_binned_positions', ax=ax_dict["ax_grouped_seq"], enable_position_difference_indicators=True,
        #     flat_time_window_edges=flat_time_window_edges, subsequence_line_color_alpha=0.95, arrow_alpha=0.9, 
        # )
        # ax = out.ax
        
        merged_out = MatplotlibRenderPlots(name='merged', figures=[fig, ], ax_dict=ax_dict, plots=merged_plots_out_dict)

        # out2: MatplotlibRenderPlots = SubsequencesPartitioningResult._debug_plot_time_bins_multiple(positions_list=final_out_subsequences, num='debug_plot_merged_time_binned_positions')
        return merged_out



@metadata_attributes(short_name=None, tags=['heuristic', 'replay', 'ripple', 'scoring', 'pho'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 06:00', related_items=[])
class HeuristicReplayScoring:
    """ Measures of replay quality ("scores") that are better aligned with my (human-rated) intuition. Mostly based on the decoded posteriors.
    
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, HeuristicScoresTuple

    """

    @classmethod
    @function_attributes(short_name='jump', tags=['bin-by-bin', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 17:50', related_items=[])
    def bin_by_bin_jump_distance(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> Tuple[NDArray, NDArray]:
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
        assert len(time_window_centers) == len(a_first_order_diff)
        ## RETURNS: total_first_order_change_score
        return time_window_centers, a_first_order_diff
    
    @classmethod
    @function_attributes(short_name='jump', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=['cls.bin_by_bin_jump_distance'], used_by=[], creation_date='2024-03-07 17:50', related_items=[])
    def bin_wise_jump_distance_score(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ provides a metric that punishes long jumps in sequential maximal prob. position bins
        """
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        n_track_position_bins: int = np.shape(a_p_x_given_n)[0]        
        time_window_centers, a_first_order_diff = cls.bin_by_bin_jump_distance(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
        max_jump_index_distance = np.nanmax(np.abs(a_first_order_diff)) # find the maximum jump size (in number of indicies) during this period
        # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
        max_jump_index_distance_ratio = (float(max_jump_index_distance) / float(n_track_position_bins-1))
        max_jump_index_distance_score = max_jump_index_distance_ratio / a_decoder_track_length
        ## RETURNS: total_first_order_change_score
        return max_jump_index_distance_score
    
    @classmethod
    @function_attributes(short_name='jump_cm', tags=['bin-by-bin', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 11:39', related_items=[])
    def bin_by_bin_position_jump_distance(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> Tuple[NDArray, NDArray]:
        """ Bin-wise most-likely position difference. Contiguous trajectories have small deltas between adjacent time bins, while non-contiguous ones can jump wildly (up to the length of the track)
        """
        time_window_centers = a_result.time_window_centers[an_epoch_idx]
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        
        ## find diffuse/uncertain bins
        time_bin_max_certainty = np.nanmax(a_p_x_given_n, axis=0) # over all time-bins
        
        ## throw out bins below certainty requirements
        # peak_certainty_min_value: float = 0.2 
        # does_bin_meet_certainty_req = (time_bin_max_certainty >= peak_certainty_min_value)
        # n_valid_time_bins: int = np.sum(does_bin_meet_certainty_req)        
        # included_most_likely_positions_list = deepcopy(a_most_likely_positions_list)[does_bin_meet_certainty_req] ## only the bins meeting the certainty requirements
        # included_time_window_centers = deepcopy(time_window_centers)[does_bin_meet_certainty_req]
        
        included_most_likely_positions_list = deepcopy(a_most_likely_positions_list) ## no certainty requirements
        included_time_window_centers = deepcopy(time_window_centers)
        # compute the 1st-order diff of all positions
        bin_by_bin_jump_distance = np.diff(included_most_likely_positions_list, n=1, prepend=[included_most_likely_positions_list[0]])

        ## convert to cm/sec (jump velocity) by dividing by time_bin_size
        # bin_by_bin_jump_distance = bin_by_bin_jump_distance / time_window_centers
        
        return included_time_window_centers, bin_by_bin_jump_distance

    @classmethod
    @function_attributes(short_name='max_jump_cm', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=['cls.bin_by_bin_position_jump_distance'], used_by=[], creation_date='2024-11-25 11:39', related_items=[])
    def bin_wise_max_position_jump_distance(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ Bin-wise most-likely position difference. Contiguous trajectories have small deltas between adjacent time bins, while non-contiguous ones can jump wildly (up to the length of the track)
        """
        # a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        # a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        if n_time_bins <= 1:
            ## only a single bin, return 0.0 (perfect, no jumps)
            return 0.0
        else:
            time_window_centers, a_first_order_diff = cls.bin_by_bin_position_jump_distance(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
            n_valid_time_bins = len(time_window_centers)

            if n_valid_time_bins < 2:
                return np.inf ## return infinity, meaning it never resolves position appropriately
            
            # normalize by the number of bins to allow comparions between different Epochs (so epochs with more bins don't intrinsically have a larger score.
            max_position_jump_cm: float = float(np.nanmax(np.abs(a_first_order_diff[1:]))) # must skip the first bin because that "jumps" to its initial position on the track            
            # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
            # max_position_jump_cm = max_position_jump_cm / a_decoder_track_length
            
            ## RETURNS: total_first_order_change_score
            return max_position_jump_cm
        

    @classmethod
    @function_attributes(short_name='jump_cm_per_sec', tags=['bin-by-bin', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 11:39', related_items=[])
    def bin_by_bin_position_jump_velocity(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> Tuple[NDArray, NDArray]:
        """
        """
        time_window_centers, bin_by_bin_jump_distance = cls.bin_by_bin_position_jump_distance(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
        assert len(time_window_centers) == len(bin_by_bin_jump_distance)
        ## convert to cm/sec (jump velocity) by dividing by time_bin_size
        bin_by_bin_jump_velocity = deepcopy(bin_by_bin_jump_distance) / time_window_centers
        
        return time_window_centers, bin_by_bin_jump_velocity

    @classmethod
    @function_attributes(short_name='max_jump_cm_per_sec', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=['cls.bin_by_bin_position_jump_velocity'], used_by=[], creation_date='2024-11-25 11:39', related_items=[])
    def bin_wise_max_position_jump_velocity(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ 
        """
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        if n_time_bins <= 1:
            ## only a single bin, return 0.0 (perfect, no jumps)
            return 0.0
        else:
            time_window_centers, a_first_order_diff = cls.bin_by_bin_position_jump_velocity(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
            n_valid_time_bins = len(time_window_centers)
            if n_valid_time_bins < 2:
                return np.inf ## return infinity, meaning it never resolves position appropriately
            
            # normalize by the number of bins to allow comparions between different Epochs (so epochs with more bins don't intrinsically have a larger score.
            max_position_jump_cm: float = float(np.nanmax(np.abs(a_first_order_diff[1:]))) # must skip the first bin because that "jumps" to its initial position on the track            
            # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
            # max_position_jump_cm = max_position_jump_cm / a_decoder_track_length
            
            ## RETURNS: total_first_order_change_score
            return max_position_jump_cm
        
    @classmethod
    @function_attributes(short_name='large_jump_excluding', tags=['bin-by-bin', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 11:39', related_items=[])
    def bin_by_bin_large_jump_filtering_fn(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> Tuple[NDArray, NDArray]:
        """
        """
        from numpy import ma
        
        max_position_jump_distance_cm: float = 50.0
        # max_position_jump_velocity_cm_per_sec: float = 0.3
        an_epoch_n_tbins: int = a_result.nbins[an_epoch_idx]
        time_window_centers, bin_by_bin_jump_distance = HeuristicReplayScoring.bin_by_bin_position_jump_distance(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
        assert len(time_window_centers) == len(bin_by_bin_jump_distance)
        
        # ## average jump distance 
        # avg_bin_by_bin_jump_distance: float = np.nanmean(bin_by_bin_jump_distance) # avg jump distance


        ## convert to cm/sec (jump velocity) by dividing by time_bin_size
        # bin_by_bin_jump_velocity = deepcopy(bin_by_bin_jump_distance) / time_window_centers
        is_included_idx = (np.abs(bin_by_bin_jump_distance) <= max_position_jump_distance_cm)
        does_transition_exceeds_max_jump_distance = np.logical_not(is_included_idx)
        # is_included_idx = np.logical_not(does_transition_exceeds_max_jump_distance)

        # max_excluded_bin_ratio

        # ## masking
        # arr_masked = ma.masked_array(bin_by_bin_jump_distance, mask=does_transition_exceeds_max_jump_distance, fill_value=np.nan)
        # time_window_centers_arr_masked = ma.masked_array(time_window_centers, mask=np.logical_not(is_included_idx), fill_value=np.nan)

        # return time_window_centers_arr_masked, arr_masked

        ## masking
        arr_masked = ma.masked_array(bin_by_bin_jump_distance, mask=does_transition_exceeds_max_jump_distance, fill_value=np.nan)
        time_window_centers_arr_masked = ma.masked_array(time_window_centers, mask=does_transition_exceeds_max_jump_distance, fill_value=np.nan)
        return time_window_centers_arr_masked, arr_masked

    @classmethod
    @function_attributes(short_name='ratio_jump_valid_bins', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=['cls.bin_by_bin_large_jump_filtering_fn'], used_by=[], creation_date='2024-11-25 11:39', related_items=[])
    def bin_wise_large_jump_ratio(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ 
        """
        an_epoch_n_tbins: int = a_result.nbins[an_epoch_idx]
        time_window_centers_arr_masked, arr_masked = cls.bin_by_bin_large_jump_filtering_fn(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length)
        # n_valid_time_bins = len(time_window_centers_arr_masked)
        # if n_valid_time_bins < 2:
        #     return np.inf ## return infinity, meaning it never resolves position appropriately
        ## extract the mask
        does_transition_exceeds_max_jump_distance = deepcopy(time_window_centers_arr_masked.mask)
        is_included_idx = np.logical_not(does_transition_exceeds_max_jump_distance)
        
        total_n_valid_tbin_transitions: int = np.sum(is_included_idx)
        total_n_bad_tbins: int = np.sum(does_transition_exceeds_max_jump_distance)
        
        assert an_epoch_n_tbins == (total_n_valid_tbin_transitions + total_n_bad_tbins), f"an_epoch_n_tbins: {an_epoch_n_tbins} should equal = (total_n_valid_tbin_transitions: {total_n_valid_tbin_transitions} + total_n_bad_tbins: {total_n_bad_tbins})"

        ratio_valid_tbins: float = float(total_n_valid_tbin_transitions) / float(an_epoch_n_tbins) # a valid between 0.0-1.0
        return ratio_valid_tbins





    @classmethod
    @function_attributes(short_name='travel', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 17:50', related_items=[])
    def bin_wise_position_difference_score(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ Bin-wise most-likely position difference. Contiguous trajectories have small deltas between adjacent time bins, while non-contiguous ones can jump wildly (up to the length of the track)
        ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
        """
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        if n_time_bins <= 1:
            ## only a single bin, return 0.0 (perfect, no jumps)
            return 0.0
        else:
            # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
            time_window_centers = a_result.time_window_centers[an_epoch_idx]

            # compute the 1st-order diff of all positions
            a_first_order_diff = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]])
            
            # add up the differences over all time bins
            total_first_order_change: float = np.nansum(np.abs(a_first_order_diff[1:])) # use .abs() to sum the total distance traveled in either direction
            ## convert to a score

            # normalize by the number of bins to allow comparions between different Epochs (so epochs with more bins don't intrinsically have a larger score.
            total_first_order_change_score: float = float(total_first_order_change) / float(n_time_bins - 1)
            # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
            total_first_order_change_score = total_first_order_change_score / a_decoder_track_length
            ## RETURNS: total_first_order_change_score
            return total_first_order_change_score
        
    @classmethod
    @function_attributes(short_name='coverage', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-12 01:05', related_items=[])
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
    @function_attributes(short_name='avg_jump_cm', tags=['bin-wise', 'bin-size', 'score', 'replay'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 17:50', related_items=[])
    def bin_wise_avg_jump_distance_score(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float) -> float:
        """ Bin-wise most-likely position difference. Contiguous trajectories have small deltas between adjacent time bins, while non-contiguous ones can jump wildly (up to the length of the track)
        ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
        """
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        if n_time_bins <= 1:
            ## only a single bin, return 0.0 (perfect, no jumps)
            return 0.0
        else:
            # time_window_centers = a_result.time_bin_containers[an_epoch_idx].centers
            time_window_centers = a_result.time_window_centers[an_epoch_idx]

            # compute the 1st-order diff of all positions
            a_first_order_diff = np.diff(a_most_likely_positions_list, n=1, prepend=[a_most_likely_positions_list[0]]) # bin_by_bin_jump_distance
            ## average jump distance 
            avg_bin_by_bin_jump_distance: float = np.nanmean(np.abs(a_first_order_diff)) # avg jump distance

            #TODO 2024-11-26 19:56: - [ ] return the non-normalized (non-score/absolute) version
            return avg_bin_by_bin_jump_distance
        
            # ## convert to a score
            # # normalize by the number of bins to allow comparions between different Epochs (so epochs with more bins don't intrinsically have a larger score.
            # total_first_order_change_score: float = float(avg_bin_by_bin_jump_distance) / float(n_time_bins - 1)
            # # normalize by the track length (long v. short) to allow fair comparison of the two (so the long track decoders don't intrinsically have a larger score).
            # total_first_order_change_score = total_first_order_change_score / a_decoder_track_length
            # ## RETURNS: total_first_order_change_score
            # return total_first_order_change_score
        


    @classmethod
    @function_attributes(short_name='continuous_seq_sort', tags=['bin-wise', 'bin-size', 'score', 'replay', 'sequence_length'], input_requires=[], output_provides=[],
                          uses=['SubsequencesPartitioningResult', '_compute_sequences_spanning_ignored_intrusions'], used_by=[], creation_date='2024-03-12 01:05', related_items=['SubsequencesPartitioningResult'])
    def bin_wise_continuous_sequence_sort_score_fn(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float, max_ignore_bins:int=2, same_thresh_cm: Optional[float]=6.0, same_thresh_fraction_of_track: Optional[float] = None) -> float:
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
        if (same_thresh_cm is None):
            assert same_thresh_fraction_of_track is not None
            same_thresh_cm: float = float(same_thresh_fraction_of_track * a_decoder_track_length)
        else:
            assert same_thresh_fraction_of_track is None, f"only same_thresh_fraction_of_track or a_decoder_track_length can be provided, NOT BOTH!" 
            ## use it directly

        ## Begin computations:

        # Get confidence of each position
        # each_t_bin_max_confidence = np.nanmax(a_p_x_given_n, axis=0)

        # the confidence in the differences depends on the confidence in the two bins that the differences are taken from
        # each_t_bin_max_confidence
        # each_t_bin_max_confidence


        # INPUTS: a_most_likely_positions_list, n_pos_bins

        ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
        partition_result: SubsequencesPartitioningResult = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list, n_pos_bins=n_pos_bins, max_ignore_bins=max_ignore_bins, same_thresh=same_thresh_cm, flat_time_window_centers=deepcopy(time_window_centers))
        longest_sequence_length_ratio: float = partition_result.get_longest_sequence_length(return_ratio=True, should_ignore_intrusion_bins=True, should_use_no_repeat_values=False) # partition_result.longest_sequence_length_ratio 
        
        assert longest_sequence_length_ratio <= 1.0, f"longest_sequence_length_ratio should not be greater than 1.0, but it is {longest_sequence_length_ratio}!!"
        
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
    @function_attributes(short_name='continuous_seq_len_ratio_no_repeats', tags=['bin-wise', 'bin-size', 'score', 'replay', 'sequence_length'], input_requires=[], output_provides=[],
                          uses=['SubsequencesPartitioningResult', '_compute_sequences_spanning_ignored_intrusions'], used_by=[], creation_date='2024-03-12 01:05', related_items=['SubsequencesPartitioningResult'])
    def bin_wise_continuous_sequence_sort_excluding_near_repeats_score_fn(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float, max_ignore_bins:int=2, same_thresh_cm: Optional[float]=6.0, same_thresh_fraction_of_track: Optional[float] = None) -> float:
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
        if (same_thresh_cm is None):
            assert same_thresh_fraction_of_track is not None
            same_thresh_cm: float = float(same_thresh_fraction_of_track * a_decoder_track_length)
        else:
            assert same_thresh_fraction_of_track is None, f"only same_thresh_fraction_of_track or a_decoder_track_length can be provided, NOT BOTH!" 
            ## use it directly

        ## Begin computations:

        ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
        partition_result: SubsequencesPartitioningResult = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list, n_pos_bins=n_pos_bins, max_ignore_bins=max_ignore_bins, same_thresh=same_thresh_cm, flat_time_window_centers=deepcopy(time_window_centers))
        longest_no_repeats_sequence_length_ratio: float = partition_result.get_longest_sequence_length(return_ratio=True, should_ignore_intrusion_bins=True, should_use_no_repeat_values=True) # partition_result.longest_no_repeats_sequence_length_ratio         
        assert longest_no_repeats_sequence_length_ratio <= 1.0, f"longest_no_repeats_sequence_length_ratio should not be greater than 1.0, but it is {longest_no_repeats_sequence_length_ratio}!!"
        return longest_no_repeats_sequence_length_ratio
    

    @classmethod
    @function_attributes(short_name='main_contiguous_subsequence_len', tags=['bin-wise', 'bin-size', 'score', 'replay', 'sequence_length'], input_requires=[], output_provides=[],
                          uses=['SubsequencesPartitioningResult', '_compute_sequences_spanning_ignored_intrusions'], used_by=[], creation_date='2024-03-12 01:05', related_items=['SubsequencesPartitioningResult'])
    def bin_wise_contiguous_subsequence_num_bins_fn(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int, a_decoder_track_length: float, max_ignore_bins:int=2, same_thresh_cm: Optional[float]=6.0, same_thresh_fraction_of_track: Optional[float] = None) -> int:
        """ The amount of the track that is represented by the decoding. More is better (indicating a longer replay).

        - Finds the longest continuous sequence (perhaps with some intrusions allowed?)
                Intrusions!
        - Do something with the start/end periods
        - 

        """
        ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
        a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        n_pos_bins: int = np.shape(a_p_x_given_n)[0]
        time_window_centers = a_result.time_window_centers[an_epoch_idx]
        if (same_thresh_cm is None):
            assert same_thresh_fraction_of_track is not None
            same_thresh_cm: float = float(same_thresh_fraction_of_track * a_decoder_track_length)
        else:
            assert same_thresh_fraction_of_track is None, f"only same_thresh_fraction_of_track or a_decoder_track_length can be provided, NOT BOTH!" 
            ## use it directly
        
        ## Begin computations:
        # [92.2559,84.6451,84.6451,92.2559,134.116,126.505,126.505,141.726,149.337,38.9801]
        # same_thresh_cm: 7.2
        ## 2024-05-09 Smarter method that can handle relatively constant decoded positions with jitter:
        partition_result: SubsequencesPartitioningResult = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list, n_pos_bins=n_pos_bins, max_ignore_bins=max_ignore_bins, same_thresh=same_thresh_cm,
                                                                                                                    flat_time_window_centers=deepcopy(time_window_centers))
        # longest_sequence_subsequence = deepcopy(partition_result.longest_sequence_subsequence)
        # longest_sequence_subsequence_partition_result: SubsequencesPartitioningResult = SubsequencesPartitioningResult.init_from_positions_list(longest_sequence_subsequence, n_pos_bins=n_pos_bins, max_ignore_bins=max_ignore_bins, same_thresh=same_thresh_cm)
        # longest_sequence_subsequence_partition_result.merged_split_positions_arrays ## makes things worse
        # longest_sequence_subsequence_partition_result.list_parts
        # longest_sequence_subsequence_partition_result.split_positions_arrays
        # longest_no_repeats_sequence_length_ratio: float = partition_result.longest_no_repeats_sequence_length_ratio         
        # assert longest_no_repeats_sequence_length_ratio <= 1.0, f"longest_no_repeats_sequence_length_ratio should not be greater than 1.0, but it is {longest_no_repeats_sequence_length_ratio}!!"
        # return longest_no_repeats_sequence_length_ratio
        # return int(partition_result.longest_subsequence_length)
        # return int(partition_result.longest_subsequence_non_intrusion_nbins) ## excluding intrusions
        # longest_sequence_length: int = int(partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=True, should_use_no_repeat_values=False))
        longest_sequence_length: int = int(partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=False, should_use_no_repeat_values=False))
        
        if np.isclose(a_most_likely_positions_list[0],  92.2559):
            print('HIT IT!')
             
        return longest_sequence_length
    


    # ==================================================================================================================== #
    # All Computation Fns of Type                                                                                          #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['bin_by_bin', 'all_fns'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 22:15', related_items=[])
    @classmethod
    def build_all_bin_by_bin_computation_fn_dict(cls) -> Dict[str, Callable]:
        return {
         'jump': cls.bin_by_bin_jump_distance,
         'jump_cm': cls.bin_by_bin_position_jump_distance, 
         'jump_cm_per_sec': cls.bin_by_bin_position_jump_velocity, 
         'large_jump_excluding': cls.bin_by_bin_large_jump_filtering_fn,
        #  'jump_cm': cls.bin_by_bin_position_jump_distance, 
        }

    @function_attributes(short_name=None, tags=['bin_wise', 'all_fns'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 22:15', related_items=[])
    @classmethod
    def build_all_bin_wise_computation_fn_dict(cls) -> Dict[str, Callable]:
        return {
         'jump': cls.bin_wise_jump_distance_score,
         'avg_jump_cm': cls.bin_wise_avg_jump_distance_score,
         'max_jump_cm': cls.bin_wise_max_position_jump_distance, 
         'max_jump_cm_per_sec': cls.bin_wise_max_position_jump_velocity, 
         'ratio_jump_valid_bins': cls.bin_wise_large_jump_ratio,
         'travel': cls.bin_wise_position_difference_score, 
         'coverage': cls.bin_wise_track_coverage_score_fn, 
         'continuous_seq_sort': cls.bin_wise_continuous_sequence_sort_score_fn,
         'continuous_seq_len_ratio_no_repeats': cls.bin_wise_continuous_sequence_sort_excluding_near_repeats_score_fn, 
         'main_contiguous_subsequence_len': cls.bin_wise_contiguous_subsequence_num_bins_fn,
        }
    
    @function_attributes(short_name=None, tags=['OLDER'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-24 00:00', related_items=[])
    @classmethod
    def build_all_score_computations_fn_dict(cls) -> Dict[str, Callable]:
        """ builds all combined heuristic scoring functions 
        """
        from neuropy.utils.indexing_helpers import PandasHelpers

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
            """ only used for the functions directly above within this function """
            ## INPUTS: a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, a_decoder_track_length: float
            final_args = []
            a_most_likely_positions_list = a_result.most_likely_positions_list[an_epoch_idx]
            positions = deepcopy(a_most_likely_positions_list) # actual x positions
            final_args.append(positions)

            if needs_times:
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
            


        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        ## Wrap them:
        positions_fns_dict = {fn.__name__:(lambda *args, **kwargs: bin_wise_wrapper_score_fn(fn, *args, **kwargs, needs_times=False)) for fn in _positions_fns}
        positions_times_fns_dict = {fn.__name__:(lambda *args, **kwargs: bin_wise_wrapper_score_fn(fn, *args, **kwargs, needs_times=True)) for fn in _positions_times_fns}
        all_bin_wise_computation_fn_dict = get_dict_subset(a_dict=cls.build_all_bin_wise_computation_fn_dict()) # , subset_excludelist=['continuous_seq_sort']
        all_score_computations_fn_dict = {**all_bin_wise_computation_fn_dict, **positions_fns_dict, **positions_times_fns_dict} # a_result, an_epoch_idx, a_decoder_track_length  - 'travel': cls.bin_wise_position_difference, 'coverage': cls.bin_wise_track_coverage_score_fn, 'jump': cls.bin_wise_jump_distance_score, 'max_jump': cls.bin_wise_max_position_jump_distance, 
        return all_score_computations_fn_dict
    

    @classmethod
    def get_all_score_computation_col_names(cls) -> List[str]:
        return list(cls.build_all_score_computations_fn_dict().keys())
     

    # ==================================================================================================================== #
    # End Computation Functions                                                                                            #
    # ==================================================================================================================== #
    

    @classmethod
    @function_attributes(short_name=None, tags=['heuristic', 'replay', 'score', 'OLDER'], input_requires=[], output_provides=[],
                         uses=['_compute_pos_derivs', 'partition_subsequences_ignoring_repeated_similar_positions', '_compute_total_variation', '_compute_integral_second_derivative', '_compute_stddev_of_diff', 'HeuristicScoresTuple'],
                         used_by=['compute_all_heuristic_scores'], creation_date='2024-02-29 00:00', related_items=[])
    def compute_pho_heuristic_replay_scores(cls, a_result: DecodedFilterEpochsResult, an_epoch_idx: int = 1, debug_print=False, use_bin_units_instead_of_realworld:bool=False, max_ignore_bins:float=2, same_thresh_cm: float=6.0, max_jump_distance_cm: float = 30.0, **kwargs) -> HeuristicScoresTuple:
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
        # use_bin_units_instead_of_realworld: bool = True # if False, uses the real-world units (cm/seconds). If True, uses nbin units (n_posbins/n_timebins)
        a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx] # np.shape(a_p_x_given_n): (62, 9)
        n_time_bins: int = a_result.nbins[an_epoch_idx]
        n_pos_bins: int = np.shape(a_p_x_given_n)[0]

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
            partition_result: SubsequencesPartitioningResult = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=a_most_likely_positions_list, n_pos_bins=n_pos_bins, max_ignore_bins=max_ignore_bins, same_thresh=same_thresh_cm, max_jump_distance_cm=max_jump_distance_cm) # (a_first_order_diff, same_thresh=same_thresh)
            num_direction_changes: int = len(partition_result.split_indicies)
            direction_change_bin_ratio: float = float(num_direction_changes) / (float(n_time_bins)-1) ## OUT: direction_change_bin_ratio

            if debug_print:
                print(f'num_direction_changes: {num_direction_changes}')
                print(f'direction_change_bin_ratio: {direction_change_bin_ratio}')

            # Split the array at each index where a sign change occurs
            # split_most_likely_positions_arrays = np.split(a_most_likely_positions_list, partition_result.split_indicies)
            # split_first_order_diff_arrays = np.split(a_first_order_diff, partition_result.split_indicies)
            split_first_order_diff_arrays = np.split(a_first_order_diff, partition_result.diff_split_indicies)

            # Pre-2024-05-09 Sequence Determination ______________________________________________________________________________ #
            # continuous_sequence_lengths = [len(a_split_first_order_diff_array) for a_split_first_order_diff_array in split_first_order_diff_arrays]
            # if debug_print:
            #     print(f'continuous_sequence_lengths: {continuous_sequence_lengths}')
            # longest_sequence_length: int = partition_result.longest_sequence_length_no_repeats
            longest_sequence_length: int = partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=True, should_use_no_repeat_values=False)
            if debug_print:
                print("Longest sequence of time bins without a direction change:", longest_sequence_length)
            # longest_sequence_start_idx: int = np.nanargmax(continuous_sequence_lengths)
            # longest_sequence = split_first_order_diff_arrays[longest_sequence_start_idx]
            
            # longest_sequence_length_ratio: float = partition_result.longest_sequence_length_ratio # float(longest_sequence_length) /  float(n_time_bins) # longest_sequence_length_ratio: the ratio of the bins that form the longest contiguous sequence to the total num bins
            longest_sequence_length_ratio: float = partition_result.get_longest_sequence_length(return_ratio=True, should_ignore_intrusion_bins=True, should_use_no_repeat_values=False)

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
    @function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=[], used_by=['compute_all_heuristic_scores'], creation_date='2024-03-07 19:54', related_items=[])
    def _run_all_score_computations(cls, track_templates: TrackTemplates, a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], all_score_computations_fn_dict: Dict, computation_fn_kwargs_dict: Optional[Dict]=None):
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
        
        from neuropy.utils.indexing_helpers import NumpyHelpers, PandasHelpers
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import get_track_length_dict

        # computation_fn_kwargs_dict: passed to each score function to specify additional required parameters
        if computation_fn_kwargs_dict is None:
            raise NotImplementedError(f'YOU BETTER PASS ONE! 2024-12-05')
            # computation_fn_kwargs_dict = {
            #     'main_contiguous_subsequence_len': dict(same_thresh_cm=same_thresh_cm),
            # }


        ## INPUTS: track_templates, a_decoded_filter_epochs_decoder_result_dict
        # decoder_track_length_dict =  {a_name:idealized_track_length_dict[a_name.split('_', maxsplit=1)[0]] for a_name, a_result in a_decoded_filter_epochs_decoder_result_dict.items()}
        decoder_track_length_dict = track_templates.get_track_length_dict()  # {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}

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

                # 'main_contiguous_subsequence_len_short_LR'
                all_epochs_scores_dict[unique_full_decoder_score_column_name] = [computation_fn(a_result=a_result, an_epoch_idx=an_epoch_idx, a_decoder_track_length=a_decoder_track_length, **computation_fn_kwargs_dict.get(score_computation_name, {})) for an_epoch_idx in np.arange(a_result.num_filter_epochs)]
                _a_separate_decoder_new_scores_dict[single_decoder_column_name] = deepcopy(all_epochs_scores_dict[unique_full_decoder_score_column_name]) # a single column, all epochs
            # END for all_score_computations_fn_dict

            ## once done with all scores for this decoder, have `_a_separate_decoder_new_scores_dict`:
            separate_decoder_new_scores_df[a_name] =  pd.DataFrame(_a_separate_decoder_new_scores_dict)
            assert np.shape(separate_decoder_new_scores_df[a_name])[0] == np.shape(a_result.filter_epochs)[0], f"np.shape(separate_decoder_new_scores_df[a_name])[0]: {np.shape(separate_decoder_new_scores_df[a_name])[0]} != np.shape(a_result.filter_epochs)[0]: {np.shape(a_result.filter_epochs)[0]}"
            a_result.filter_epochs = PandasHelpers.adding_additional_df_columns(original_df=a_result.filter_epochs, additional_cols_df=separate_decoder_new_scores_df[a_name]) # update the filter_epochs with the new columns

        # END for `a_decoded_filter_epochs_decoder_result_dict`
        ## OUTPUTS: all_epochs_scores_dict, all_epochs_scores_df
        all_epochs_scores_df = pd.DataFrame(all_epochs_scores_dict)
        return a_decoded_filter_epochs_decoder_result_dict, all_epochs_scores_df


    @classmethod
    @function_attributes(short_name=None, tags=['heuristic', 'main', 'computation'], input_requires=[], output_provides=[], uses=['_run_all_score_computations', 'cls.compute_pho_heuristic_replay_scores', 'cls.build_all_score_computations_fn_dict'], used_by=['_decoded_epochs_heuristic_scoring'], creation_date='2024-03-12 00:59', related_items=[])
    def compute_all_heuristic_scores(cls, track_templates: TrackTemplates, a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], use_bin_units_instead_of_realworld:bool=False, max_ignore_bins:float=2, same_thresh_cm: float=6.0, max_jump_distance_cm: float = 30.0, **kwargs) -> Tuple[Dict[str, DecodedFilterEpochsResult], Dict[str, pd.DataFrame]]:
        """ Computes all heuristic scoring metrics (for each epoch) and adds them to the DecodedFilterEpochsResult's .filter_epochs as columns
        
        Directly called by the global computation function `_decoded_epochs_heuristic_scoring`
        
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_heuristic_scores

        a_decoded_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(a_decoded_filter_epochs_decoder_result_dict=a_decoded_filter_epochs_decoder_result_dict)

        
        """
        from neuropy.utils.indexing_helpers import PandasHelpers
        
        # computation_fn_kwargs_dict: passed to each score function to specify additional required parameters
        computation_fn_kwargs_dict = {
            'main_contiguous_subsequence_len': dict(same_thresh_cm=same_thresh_cm, max_ignore_bins=max_ignore_bins, same_thresh_fraction_of_track=None),
            'continuous_seq_len_ratio_no_repeats': dict(same_thresh_cm=same_thresh_cm, max_ignore_bins=max_ignore_bins, same_thresh_fraction_of_track=None),
            'continuous_seq_sort': dict(same_thresh_cm=same_thresh_cm, max_ignore_bins=max_ignore_bins, same_thresh_fraction_of_track=None),
        }
        
        all_score_computations_fn_dict = cls.build_all_score_computations_fn_dict()
        a_decoded_filter_epochs_decoder_result_dict, all_epochs_scores_df = cls._run_all_score_computations(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=a_decoded_filter_epochs_decoder_result_dict, all_score_computations_fn_dict=all_score_computations_fn_dict, computation_fn_kwargs_dict=computation_fn_kwargs_dict)

        _out_new_scores: Dict[str, pd.DataFrame] = {}

        for a_name, a_result in a_decoded_filter_epochs_decoder_result_dict.items():
            _out_new_scores[a_name] =  pd.DataFrame([asdict(cls.compute_pho_heuristic_replay_scores(a_result=a_result, an_epoch_idx=an_epoch_idx, use_bin_units_instead_of_realworld=use_bin_units_instead_of_realworld, max_ignore_bins=max_ignore_bins, same_thresh_cm=same_thresh_cm, max_jump_distance_cm=max_jump_distance_cm), filter=lambda a, v: a.name not in ['position_derivatives_df']) for an_epoch_idx in np.arange(a_result.num_filter_epochs)])
            assert np.shape(_out_new_scores[a_name])[0] == np.shape(a_result.filter_epochs)[0], f"np.shape(_out_new_scores[a_name])[0]: {np.shape(_out_new_scores[a_name])[0]} != np.shape(a_result.filter_epochs)[0]: {np.shape(a_result.filter_epochs)[0]}"
            a_result.filter_epochs = PandasHelpers.adding_additional_df_columns(original_df=a_result.filter_epochs, additional_cols_df=_out_new_scores[a_name]) # update the filter_epochs with the new columns


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


