from copy import deepcopy
import unittest
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
import numpy as np
import pandas as pd
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequencesPartitioningResult, SequenceScoringComputations
from neuropy.utils.indexing_helpers import PandasHelpers, NumpyHelpers, flatten
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance

from attr import define, field, Factory

from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequenceDetectionSamples, GroundTruthData

class TestSubsequenceMerging(unittest.TestCase):
    """ #TODO 2024-12-04 12:44: - [ ] These were written by GPT and not tested in any way. Most do not pass
       
    """

    def setUp(self):
        ## Testing Settings:
        self.enable_debug_plotting = False
        self.enable_debug_printing = True
        
        ## Configuration
        self.decoder_track_length_dict = {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}
        self.max_ignore_bins = 2
        self.max_jump_distance_cm = 60.0
        self.same_thresh_fraction_of_track = 0.05 ## up to 5.0% of the track
        self.same_thresh_cm_dict = {k:(v * self.same_thresh_fraction_of_track) for k, v in self.decoder_track_length_dict.items()}
        self.decoder_track_length: float = self.decoder_track_length_dict['long_LR'] # 214.0
        self.a_same_thresh_cm = self.same_thresh_cm_dict['long_LR']
        print(f'a_same_thresh_cm: {self.a_same_thresh_cm}') ## same_thresh_cm: 10.7
        self.pos_bin_edges = np.array([37.0774, 40.8828, 44.6882, 48.4936, 52.2991, 56.1045, 59.9099, 63.7153, 67.5207, 71.3261, 75.1316, 78.937, 82.7424, 86.5478, 90.3532, 94.1586, 97.9641, 101.769, 105.575, 109.38, 113.186, 116.991, 120.797, 124.602, 128.407, 132.213, 136.018, 139.824, 143.629, 147.434, 151.24, 155.045, 158.851, 162.656, 166.462, 170.267, 174.072, 177.878, 181.683, 185.489, 189.294, 193.099, 196.905, 200.71, 204.516, 208.321, 212.127, 215.932, 219.737, 223.543, 227.348, 231.154, 234.959, 238.764, 242.57, 246.375, 250.181, 253.986])
        # self.n_pos_bins = (len(self.pos_bin_edges)-1)  # Adjust as needed

        self.SubsequencesPartitioningResult_common_init_kwargs = dict(same_thresh=self.a_same_thresh_cm,
                                                                        max_ignore_bins=self.max_ignore_bins,
                                                                        max_jump_distance_cm=self.max_jump_distance_cm,
                                                                        pos_bin_edges=deepcopy(self.pos_bin_edges),
                                                                        debug_print=False)

        self.test_dict = SubsequenceDetectionSamples.get_all_example_dict()
        


    def tearDown(self):
        pass
    
    
    @classmethod
    def _add_dynamic_tests(self):
        """Dynamically add test methods based on self.test_dict."""
        for key, example_data in self.test_dict.items():
            test_name = f"test_{key}"  # Name for the test method

            def dynamic_test(self, example_data=example_data, key=key):
                """Dynamic test logic."""
                # Example: Ensure the example_data can be processed
                result = SubsequencesPartitioningResult(**self.SubsequencesPartitioningResult_common_init_kwargs)
                # Add your actual assertions here
                self.assertIsNotNone(result, f"Result should not be None for key: {key}")
                self.assertTrue(isinstance(result, SubsequencesPartitioningResult), f"Result should be a SubsequencesPartitioningResult for key: {key}")

            # Add the method to the class
            setattr(self, test_name, dynamic_test)



    def test_single_sequence(self):
        # Test a single-bin intrusion between two long sequences
        positions_list = np.array([100, 110, 120, 130, 140, 150])
        # Introduce an intrusion
        positions_list[3] = 125  # Intrusion at index 3
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            pos_bin_edges=self.pos_bin_edges,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.a_same_thresh_cm,
            max_jump_distance_cm=self.max_jump_distance_cm
        )
        # Expected merged sequences
        expected_sequences = [np.array([100, 110, 120, 125, 140, 150])]
        # Compare the merged sequences
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)


    def test_many_sequences(self):
        
        all_examples_plot_data_positions_arr_dict = SubsequenceDetectionSamples.get_all_examples()
        all_examples_plot_data_dict = {a_name:SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=a_most_likely_positions_list, **self.SubsequencesPartitioningResult_common_init_kwargs) for a_name, a_most_likely_positions_list in all_examples_plot_data_positions_arr_dict.items()}
        all_examples_plot_data_name_keys = list(all_examples_plot_data_dict.keys())

        for i, (a_name, a_partition_result) in enumerate(all_examples_plot_data_dict.items()):
            # a_most_likely_positions_list = np.array(a_most_likely_positions_list)
            # a_partition_result = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=a_most_likely_positions_list, **SubsequencesPartitioningResult_common_init_kwargs)
            # Access the partitioned subsequences
            subsequences = a_partition_result.split_positions_arrays
            merged_subsequences = a_partition_result.merged_split_positions_arrays
            print("Number of subsequences before merging:", len(subsequences))
            print("Number of subsequences after merging:", len(merged_subsequences))
            
            position_bins_info_df = deepcopy(a_partition_result.position_bins_info_df)
            position_changes_info_df = deepcopy(a_partition_result.position_changes_info_df)
            subsequences_df = deepcopy(a_partition_result.subsequences_df)

            longest_seq_length_dict = {'neither': a_partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=False, should_use_no_repeat_values=False),
                'ignoring_intru': a_partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=True, should_use_no_repeat_values=False),
                '+no_repeat': a_partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=True, should_use_no_repeat_values=True),
            }

            longest_seq_length_multiline_label_str: str = '\n'.join([': '.join([k, str(v)]) for k, v in longest_seq_length_dict.items()])
            print(longest_seq_length_multiline_label_str)


            # print(a_partition_result.get_longest_sequence_length(return_ratio=False, should_ignore_intrusion_bins=True, should_use_no_repeat_values=False))
            # a_fig = figures_dict[a_name] # list(figures_dict.values())[i]
            # an_ax_dict = axs_dict[a_name] # list(axs_dict.values())[i]    
            # _out = a_partition_result._plot_step_by_step_subsequence_partition_process(extant_ax_dict=an_ax_dict)
            

        # Test a single-bin intrusion between two long sequences
        positions_list = np.array([100, 110, 120, 130, 140, 150])
        # Introduce an intrusion
        positions_list[3] = 125  # Intrusion at index 3
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            pos_bin_edges=self.pos_bin_edges,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.a_same_thresh_cm,
            max_jump_distance_cm=self.max_jump_distance_cm
        )
        # Expected merged sequences
        expected_sequences = [np.array([100, 110, 120, 125, 140, 150])]
        # Compare the merged sequences
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)



    def test_single_intrusion_between_long_sequences(self):
        # Test a single-bin intrusion between two long sequences
        positions_list = np.array([100, 110, 120, 130, 140, 150])
        # Introduce an intrusion
        positions_list[3] = 125  # Intrusion at index 3
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Expected merged sequences
        expected_sequences = [np.array([100, 110, 120, 125, 140, 150])]
        # Compare the merged sequences
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    # def test_multiple_intrusions_between_long_sequences(self):
    #     # Test multiple single-bin intrusions between long sequences
    #     positions_list = np.array([100, 110, 120, 130, 140, 150, 160, 170])
    #     # Introduce intrusions at indices 3 and 5
    #     positions_list[3] = 125  # Intrusion
    #     positions_list[5] = 155  # Intrusion
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     # Expected merged sequences
    #     expected_sequences = [np.array([100, 110, 120, 125, 130, 155, 160, 170])]
    #     # Compare the merged sequences
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_intrusion_at_beginning(self):
    #     # Test an intrusion at the beginning of the sequence
    #     positions_list = np.array([95, 100, 110, 120, 130])
    #     # Introduce an intrusion at index 0
    #     positions_list[0] = 105  # Intrusion
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [np.array([105, 100, 110, 120, 130])]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_intrusion_at_end(self):
    #     # Test an intrusion at the end of the sequence
    #     positions_list = np.array([100, 110, 120, 130, 135])
    #     # Introduce an intrusion at the last index
    #     positions_list[-1] = 125  # Intrusion
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [np.array([100, 110, 120, 130, 125])]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_long_intrusion_exceeds_max_ignore_bins(self):
    #     # Test an intrusion longer than max_ignore_bins
    #     positions_list = np.array([100, 110, 120, 50, 55, 60, 130, 140])
    #     # The sequence from indices 3 to 5 is an intrusion longer than max_ignore_bins
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     # Expected sequences: the intrusion should not be merged
    #     expected_sequences = [
    #         np.array([100, 110, 120]),
    #         np.array([50, 55, 60]),
    #         np.array([130, 140])
    #     ]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_adjacent_intrusions(self):
    #     # Test adjacent intrusions
    #     positions_list = np.array([100, 110, 50, 55, 120, 130])
    #     # Introduce two adjacent intrusions at indices 2 and 3
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     # Expected sequences: intrusions should be merged if combined length ≤ max_ignore_bins
    #     expected_sequences = [np.array([100, 110, 50, 55, 120, 130])]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_no_intrusions(self):
    #     # Test sequence with no intrusions
    #     positions_list = np.array([100, 110, 120, 130, 140])
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [positions_list]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
    #     np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    # def test_all_intrusions(self):
    #     # Test sequence where all subsequences are intrusions
    #     positions_list = np.array([100, 105, 110])
    #     # Force all subsequences to be considered intrusions
    #     self.max_ignore_bins = 5  # Set high to make all sequences intrusions
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [positions_list]  # Should not merge since can't determine main sequence
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 3)  # Should not merge
    #     # Each position is a separate subsequence

    # def test_long_intrusion_between_short_sequences(self):
    #     # Test a long intrusion between short sequences
    #     positions_list = np.array([100, 105, 50, 55, 60, 65, 110, 115])
    #     # Intrusion from index 2 to 5 (length 4)
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     # Expected: Intrusion not merged due to length
    #     expected_sequences = [
    #         np.array([100, 105]),
    #         np.array([50, 55, 60, 65]),
    #         np.array([110, 115])
    #     ]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_varying_sequence_lengths(self):
    #     # Test sequences of varying lengths with intrusions
    #     positions_list = np.array([100, 110, 120, 130, 60, 65, 70, 80, 140, 150])
    #     # Intrusion from index 4 to 7 (length 4)
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [
    #         np.array([100, 110, 120, 130]),
    #         np.array([60, 65, 70, 80]),
    #         np.array([140, 150])
    #     ]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_intrusion_with_direction_change(self):
    #     # Test intrusion that includes a direction change
    #     positions_list = np.array([100, 110, 120, 115, 130, 140])
    #     # Intrusion at index 3 (direction change)
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [np.array([100, 110, 120, 115, 130, 140])]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
    #     np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    # def test_large_same_thresh(self):
    #     # Test with a large same_thresh to ignore small changes
    #     positions_list = np.array([100, 100.5, 101, 150, 151, 200])
    #     self.same_thresh = 50  # Large threshold to ignore changes less than 50
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [positions_list]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
    #     np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    # def test_small_same_thresh(self):
    #     # Test with a small same_thresh to detect all changes
    #     positions_list = np.array([100, 100.5, 101, 150, 151, 200])
    #     self.same_thresh = 0.1  # Small threshold to detect all changes
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [
    #         np.array([100]),
    #         np.array([100.5]),
    #         np.array([101]),
    #         np.array([150]),
    #         np.array([151]),
    #         np.array([200])
    #     ]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
    #     for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
    #         np.testing.assert_array_equal(seq, expected_seq)

    # def test_non_integer_positions(self):
    #     # Test positions with decimal values
    #     positions_list = np.array([100.0, 110.5, 120.25, 130.75, 140.5])
    #     # Introduce an intrusion
    #     positions_list[2] = 125.5  # Intrusion at index 2
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [np.array([100.0, 110.5, 125.5, 130.75, 140.5])]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
    #     np.testing.assert_array_almost_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    # def test_large_sequence_with_intrusions(self):
    #     # Test a large sequence with multiple intrusions
    #     positions_list = np.linspace(100, 200, 50)  # Generate 50 positions from 100 to 200
    #     # Introduce intrusions at regular intervals
    #     intrusion_indices = np.arange(5, 50, 10)
    #     positions_list[intrusion_indices] -= 50  # Create intrusions
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     # Expected: Intrusions merged into main sequence
    #     expected_sequences = [positions_list]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
    #     np.testing.assert_array_almost_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    # def test_random_positions_with_intrusions(self):
    #     # Test with random positions and random intrusions
    #     np.random.seed(42)  # For reproducibility
    #     positions_list = np.cumsum(np.random.randn(100)) + 100  # Random walk starting at 100
    #     # Introduce intrusions at random indices
    #     intrusion_indices = np.random.choice(100, size=10, replace=False)
    #     positions_list[intrusion_indices] += np.random.randn(10) * 50  # Larger deviations
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=self.n_pos_bins,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     # Can't specify exact expected sequences, but can check total number of bins
    #     self.assertEqual(partition_result.total_num_subsequence_bins, len(positions_list))

    # def test_empty_positions_list(self):
    #     # Test with an empty positions list
    #     positions_list = np.array([])
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=0,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = []
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 0)

    # def test_single_position(self):
    #     # Test with a single position
    #     positions_list = np.array([100])
    #     partition_result = SubsequencesPartitioningResult.init_from_positions_list(
    #         a_most_likely_positions_list=positions_list,
    #         n_pos_bins=1,
    #         max_ignore_bins=self.max_ignore_bins,
    #         same_thresh=self.same_thresh
    #     )
    #     expected_sequences = [np.array([100])]
    #     self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
    #     np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

class TestNaNTolerantHeuristicSequencePartitioning(unittest.TestCase):
    """Undefined (NaN) most-likely positions must not crash scoring or corrupt direction casts."""


    def setUp(self):
        self.pos_bin_edges = np.array([37.0774, 40.8828, 44.6882, 48.4936, 52.2991, 56.1045, 59.9099, 63.7153, 67.5207, 71.3261, 75.1316, 78.937, 82.7424, 86.5478, 90.3532, 94.1586, 97.9641, 101.769, 105.575, 109.38, 113.186, 116.991, 120.797, 124.602, 128.407, 132.213, 136.018, 139.824, 143.629, 147.434, 151.24, 155.045, 158.851, 162.656, 166.462, 170.267, 174.072, 177.878, 181.683, 185.489, 189.294, 193.099, 196.905, 200.71, 204.516, 208.321, 212.127, 215.932, 219.737, 223.543, 227.348, 231.154, 234.959, 238.764, 242.57, 246.375, 250.181, 253.986])
        self.same_thresh = 10.7
        self.max_ignore_bins = 2
        self.max_jump_distance_cm = 60.0


    def test_total_distance_and_track_coverage_ignore_nans(self):
        positions = np.array([100.0, np.nan, 120.0, 140.0], dtype=float)
        self.assertAlmostEqual(SequenceScoringComputations.total_distance_traveled(positions), 40.0)
        tcov = SequenceScoringComputations.track_coverage_score(positions, pos_bin_edges=self.pos_bin_edges)
        self.assertTrue(np.isfinite(tcov))
        self.assertGreater(tcov, 0.0)


    def test_find_value_equiv_groups_skips_nans(self):
        positions = np.array([100.0, 100.5, np.nan, 130.0], dtype=float)
        groups, idxs = SubsequencesPartitioningResult.find_value_equiv_groups(positions, same_thresh_cm=self.same_thresh)
        flat_idxs = [i for group in idxs for i in group]
        self.assertNotIn(2, flat_idxs)
        self.assertEqual(len(groups), 2)


    def test_compute_shuffle_scalar_scores_handles_embedded_and_all_nan(self):
        positions = np.array([100.0, 110.0, np.nan, 120.0, 130.0], dtype=float)
        scalar_dict, main_positions = SubsequencesPartitioningResult.compute_shuffle_scalar_scores(positions, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        self.assertIn('mseq_dtrav', scalar_dict)
        self.assertTrue(np.isfinite(scalar_dict['mseq_dtrav']) or (scalar_dict['mseq_dtrav'] == 0.0))

        all_nan = np.full(5, np.nan, dtype=float)
        scalar_dict_nan, _ = SubsequencesPartitioningResult.compute_shuffle_scalar_scores(all_nan, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        self.assertEqual(int(scalar_dict_nan['mseq_len']), 5)


    def test_position_changes_info_df_direction_cast_no_intmin(self):
        position_bins_info_df = pd.DataFrame({
            'pos': [100.0, np.nan, 120.0],
            'flat_idx': [0, 1, 2],
            't_bin_end': [0.1, 0.2, 0.3],
            't_bin_center': [0.05, 0.15, 0.25],
        })
        changes_df = SubsequencesPartitioningResult._compute_position_changes_info_df(position_bins_info_df=position_bins_info_df, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        self.assertTrue(np.all(np.isfinite(changes_df['direction'].to_numpy())))
        self.assertFalse(np.any(changes_df['direction'].to_numpy() == np.iinfo(np.int32).min))


class TestHeuristicBugFixes(unittest.TestCase):
    """Focused regressions for the concrete heuristic partitioning/scoring bug fixes."""


    def setUp(self):
        self.pos_bin_edges = np.array([37.0774, 40.8828, 44.6882, 48.4936, 52.2991, 56.1045, 59.9099, 63.7153, 67.5207, 71.3261, 75.1316, 78.937, 82.7424, 86.5478, 90.3532, 94.1586, 97.9641, 101.769, 105.575, 109.38, 113.186, 116.991, 120.797, 124.602, 128.407, 132.213, 136.018, 139.824, 143.629, 147.434, 151.24, 155.045, 158.851, 162.656, 166.462, 170.267, 174.072, 177.878, 181.683, 185.489, 189.294, 193.099, 196.905, 200.71, 204.516, 208.321, 212.127, 215.932, 219.737, 223.543, 227.348, 231.154, 234.959, 238.764, 242.57, 246.375, 250.181, 253.986])
        self.same_thresh = 10.7
        self.max_ignore_bins = 2
        self.max_jump_distance_cm = 60.0
        self.long_track_length = 214.0
        self.short_track_length = 144.0


    def test_subthreshold_reverse_jitter_does_not_split(self):
        ## Forward sweep with a small reverse jitter (< same_thresh) mid-sequence
        positions = np.array([100.0, 110.0, 120.0, 115.0, 125.0, 135.0], dtype=float)  # -5 cm reverse
        result = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=positions, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        self.assertEqual(len(result.split_positions_arrays), 1)


    def test_above_threshold_turnaround_does_split(self):
        ## Real turnaround: reverse by more than same_thresh
        positions = np.array([100.0, 120.0, 140.0, 120.0, 100.0, 80.0], dtype=float)  # -20 cm reverse
        result = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=positions, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        self.assertGreaterEqual(len(result.split_positions_arrays), 2)


    def test_find_value_equiv_groups_anchors_plateau_start(self):
        ## Smooth +8 cm/bin sweep: anchoring initial_v yields multiple groups (old bug collapsed to 1)
        smooth_sweep = np.array([100.0, 108.0, 116.0, 124.0, 132.0, 140.0], dtype=float)
        groups_sweep, _ = SubsequencesPartitioningResult.find_value_equiv_groups(smooth_sweep, same_thresh_cm=self.same_thresh)
        self.assertGreater(len(groups_sweep), 1)
        self.assertGreaterEqual(len(groups_sweep), 3)

        ## Flat plateau within same_thresh collapses to one group
        plateau = np.array([100.0, 102.0, 101.0, 103.0, 100.5], dtype=float)
        groups_plateau, _ = SubsequencesPartitioningResult.find_value_equiv_groups(plateau, same_thresh_cm=self.same_thresh)
        self.assertEqual(len(groups_plateau), 1)


    def test_symmetric_same_direction_intrusion_bridges(self):
        ## 5 + 1 + 5 same-direction flanks with steps > same_thresh so direction is established
        left = np.array([100.0, 120.0, 140.0, 160.0, 180.0])
        intrusion = np.array([40.0])  # reverse into, then forward out → own 1-bin subsequence
        right = np.array([200.0, 220.0, 240.0, 250.0, 251.0])  # continue +; bridge 180→200 = 20 < max_jump
        positions = np.concatenate([left, intrusion, right])
        result = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=positions, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        self.assertGreaterEqual(len(result.split_positions_arrays), 3)
        ## After merge, expect a single bridged subsequence containing the intrusion bin
        self.assertEqual(len(result.merged_split_positions_arrays), 1)
        self.assertEqual(len(result.merged_split_positions_arrays[0]), len(positions))


    def test_opposite_direction_flanks_do_not_merge(self):
        ## Same lengths but opposite travel direction on the two sides
        left = np.array([100.0, 120.0, 140.0, 160.0, 180.0])   # +
        intrusion = np.array([40.0])
        right = np.array([160.0, 140.0, 120.0, 100.0, 80.0])  # -
        positions = np.concatenate([left, intrusion, right])
        result = SubsequencesPartitioningResult.init_from_positions_list(a_most_likely_positions_list=positions, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm)
        ## Should remain unmerged (at least 3 fragments: left, intrusion, right)
        self.assertGreaterEqual(len(result.merged_split_positions_arrays), 3)


    def test_bridged_outlier_does_not_inflate_mseq_tcov_dtrav(self):
        ## Smooth forward sequence with a far outlier intrusion that is bridged
        left = np.array([100.0, 120.0, 140.0, 160.0, 180.0])
        intrusion = np.array([40.0])
        right = np.array([200.0, 220.0, 240.0, 250.0, 251.0])
        positions = np.concatenate([left, intrusion, right])
        scalar_dict, _ = SubsequencesPartitioningResult.compute_shuffle_scalar_scores(positions, pos_bin_edges=self.pos_bin_edges, max_ignore_bins=self.max_ignore_bins, same_thresh=self.same_thresh, max_jump_distance_cm=self.max_jump_distance_cm, decoder_track_length=self.long_track_length)
        non_intrusion = np.concatenate([left, right])
        expected_dtrav = SequenceScoringComputations.total_distance_traveled(non_intrusion)
        expected_tcov = SequenceScoringComputations.track_coverage_score(non_intrusion, pos_bin_edges=self.pos_bin_edges, a_decoder_track_length=self.long_track_length)
        self.assertAlmostEqual(float(scalar_dict['mseq_dtrav']), float(expected_dtrav), places=5)
        self.assertAlmostEqual(float(scalar_dict['mseq_tcov']), float(expected_tcov), places=5)
        ## Full-sequence dtrav is inflated by the detour through the outlier
        inflated_dtrav = SequenceScoringComputations.total_distance_traveled(positions)
        self.assertLess(float(scalar_dict['mseq_dtrav']), float(inflated_dtrav))


    def test_short_vs_long_track_coverage_normalized_by_track_length(self):
        ## Full Long traversal ≈ track length 214; full Short ≈ 144
        long_positions = np.linspace(40.0, 40.0 + self.long_track_length, 12)
        short_positions = np.linspace(40.0, 40.0 + self.short_track_length, 12)
        long_tcov = SequenceScoringComputations.track_coverage_score(long_positions, pos_bin_edges=self.pos_bin_edges, a_decoder_track_length=self.long_track_length)
        short_tcov = SequenceScoringComputations.track_coverage_score(short_positions, pos_bin_edges=self.pos_bin_edges, a_decoder_track_length=self.short_track_length)
        self.assertAlmostEqual(long_tcov, 1.0, places=2)
        self.assertAlmostEqual(short_tcov, 1.0, places=2)


if __name__ == '__main__':
    unittest.main()

