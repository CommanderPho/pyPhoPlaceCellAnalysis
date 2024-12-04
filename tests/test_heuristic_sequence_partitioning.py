import unittest
from typing import List, Dict
import numpy as np
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequencesPartitioningResult






class TestSubsequenceMerging(unittest.TestCase):
    def setUp(self):
        self.max_ignore_bins = 2
        self.same_thresh = 4  # Adjust as needed
        self.n_pos_bins = 200  # Adjust as needed

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

    def test_multiple_intrusions_between_long_sequences(self):
        # Test multiple single-bin intrusions between long sequences
        positions_list = np.array([100, 110, 120, 130, 140, 150, 160, 170])
        # Introduce intrusions at indices 3 and 5
        positions_list[3] = 125  # Intrusion
        positions_list[5] = 155  # Intrusion
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Expected merged sequences
        expected_sequences = [np.array([100, 110, 120, 125, 130, 155, 160, 170])]
        # Compare the merged sequences
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_intrusion_at_beginning(self):
        # Test an intrusion at the beginning of the sequence
        positions_list = np.array([95, 100, 110, 120, 130])
        # Introduce an intrusion at index 0
        positions_list[0] = 105  # Intrusion
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [np.array([105, 100, 110, 120, 130])]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_intrusion_at_end(self):
        # Test an intrusion at the end of the sequence
        positions_list = np.array([100, 110, 120, 130, 135])
        # Introduce an intrusion at the last index
        positions_list[-1] = 125  # Intrusion
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [np.array([100, 110, 120, 130, 125])]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_long_intrusion_exceeds_max_ignore_bins(self):
        # Test an intrusion longer than max_ignore_bins
        positions_list = np.array([100, 110, 120, 50, 55, 60, 130, 140])
        # The sequence from indices 3 to 5 is an intrusion longer than max_ignore_bins
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Expected sequences: the intrusion should not be merged
        expected_sequences = [
            np.array([100, 110, 120]),
            np.array([50, 55, 60]),
            np.array([130, 140])
        ]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_adjacent_intrusions(self):
        # Test adjacent intrusions
        positions_list = np.array([100, 110, 50, 55, 120, 130])
        # Introduce two adjacent intrusions at indices 2 and 3
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Expected sequences: intrusions should be merged if combined length ≤ max_ignore_bins
        expected_sequences = [np.array([100, 110, 50, 55, 120, 130])]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_no_intrusions(self):
        # Test sequence with no intrusions
        positions_list = np.array([100, 110, 120, 130, 140])
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [positions_list]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
        np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    def test_all_intrusions(self):
        # Test sequence where all subsequences are intrusions
        positions_list = np.array([100, 105, 110])
        # Force all subsequences to be considered intrusions
        self.max_ignore_bins = 5  # Set high to make all sequences intrusions
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [positions_list]  # Should not merge since can't determine main sequence
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 3)  # Should not merge
        # Each position is a separate subsequence

    def test_long_intrusion_between_short_sequences(self):
        # Test a long intrusion between short sequences
        positions_list = np.array([100, 105, 50, 55, 60, 65, 110, 115])
        # Intrusion from index 2 to 5 (length 4)
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Expected: Intrusion not merged due to length
        expected_sequences = [
            np.array([100, 105]),
            np.array([50, 55, 60, 65]),
            np.array([110, 115])
        ]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_varying_sequence_lengths(self):
        # Test sequences of varying lengths with intrusions
        positions_list = np.array([100, 110, 120, 130, 60, 65, 70, 80, 140, 150])
        # Intrusion from index 4 to 7 (length 4)
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [
            np.array([100, 110, 120, 130]),
            np.array([60, 65, 70, 80]),
            np.array([140, 150])
        ]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_intrusion_with_direction_change(self):
        # Test intrusion that includes a direction change
        positions_list = np.array([100, 110, 120, 115, 130, 140])
        # Intrusion at index 3 (direction change)
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [np.array([100, 110, 120, 115, 130, 140])]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
        np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    def test_large_same_thresh(self):
        # Test with a large same_thresh to ignore small changes
        positions_list = np.array([100, 100.5, 101, 150, 151, 200])
        self.same_thresh = 50  # Large threshold to ignore changes less than 50
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [positions_list]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
        np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    def test_small_same_thresh(self):
        # Test with a small same_thresh to detect all changes
        positions_list = np.array([100, 100.5, 101, 150, 151, 200])
        self.same_thresh = 0.1  # Small threshold to detect all changes
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [
            np.array([100]),
            np.array([100.5]),
            np.array([101]),
            np.array([150]),
            np.array([151]),
            np.array([200])
        ]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), len(expected_sequences))
        for seq, expected_seq in zip(partition_result.merged_split_positions_arrays, expected_sequences):
            np.testing.assert_array_equal(seq, expected_seq)

    def test_non_integer_positions(self):
        # Test positions with decimal values
        positions_list = np.array([100.0, 110.5, 120.25, 130.75, 140.5])
        # Introduce an intrusion
        positions_list[2] = 125.5  # Intrusion at index 2
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [np.array([100.0, 110.5, 125.5, 130.75, 140.5])]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
        np.testing.assert_array_almost_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    def test_large_sequence_with_intrusions(self):
        # Test a large sequence with multiple intrusions
        positions_list = np.linspace(100, 200, 50)  # Generate 50 positions from 100 to 200
        # Introduce intrusions at regular intervals
        intrusion_indices = np.arange(5, 50, 10)
        positions_list[intrusion_indices] -= 50  # Create intrusions
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Expected: Intrusions merged into main sequence
        expected_sequences = [positions_list]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
        np.testing.assert_array_almost_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

    def test_random_positions_with_intrusions(self):
        # Test with random positions and random intrusions
        np.random.seed(42)  # For reproducibility
        positions_list = np.cumsum(np.random.randn(100)) + 100  # Random walk starting at 100
        # Introduce intrusions at random indices
        intrusion_indices = np.random.choice(100, size=10, replace=False)
        positions_list[intrusion_indices] += np.random.randn(10) * 50  # Larger deviations
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=self.n_pos_bins,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        # Can't specify exact expected sequences, but can check total number of bins
        self.assertEqual(partition_result.total_num_subsequence_bins, len(positions_list))

    def test_empty_positions_list(self):
        # Test with an empty positions list
        positions_list = np.array([])
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=0,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = []
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 0)

    def test_single_position(self):
        # Test with a single position
        positions_list = np.array([100])
        partition_result = SubsequencesPartitioningResult.init_from_positions_list(
            a_most_likely_positions_list=positions_list,
            n_pos_bins=1,
            max_ignore_bins=self.max_ignore_bins,
            same_thresh=self.same_thresh
        )
        expected_sequences = [np.array([100])]
        self.assertEqual(len(partition_result.merged_split_positions_arrays), 1)
        np.testing.assert_array_equal(partition_result.merged_split_positions_arrays[0], expected_sequences[0])

if __name__ == '__main__':
    unittest.main()

