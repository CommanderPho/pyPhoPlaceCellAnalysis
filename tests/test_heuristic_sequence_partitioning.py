import unittest
import numpy as np
# from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequencesPartitioningResult

import numpy as np
from typing import List


## Instead, write `partition_subsequences(self)` and `merge_intrusions(self)` functions for my existing `SubsequencesPartitioningResult` class (so that I don't have to change everywhere SubsequencesPartitioningResult is used in code)
class SubsequencesPartitioningResult:
    def __init__(self, positions: np.ndarray, same_thresh: float = 4.0, max_ignore_bins: int = 2):
        """
        Initializes the SubsequencePartitioner.

        Args:
            positions (np.ndarray): Array of positions over time.
            same_thresh (float): Threshold to ignore small changes in position.
            max_ignore_bins (int): Maximum length of a subsequence to consider it as an intrusion for merging.
        """
        self.positions = positions
        self.same_thresh = same_thresh
        self.max_ignore_bins = max_ignore_bins
        self.subsequences: List[np.ndarray] = []
        self.merged_subsequences: List[np.ndarray] = []
        self.debug_print: bool = False  # Added debug print flag

    @classmethod
    def init_from_positions_list(
        cls,
        a_most_likely_positions_list: np.ndarray,
        n_pos_bins: int = None,
        max_ignore_bins: int = 2,
        same_thresh: float = 4.0,
        flat_time_window_centers=None,
        flat_time_window_edges=None,
        debug_print: bool = False,
    ) -> "SubsequencePartitioner":
        """
        Initializes the SubsequencePartitioner from a positions list, matching the original method signature for compatibility.

        Args:
            a_most_likely_positions_list (np.ndarray): Array of positions over time.
            n_pos_bins (int, optional): Number of position bins (not used in this implementation).
            max_ignore_bins (int): Maximum length of a subsequence to consider it as an intrusion for merging.
            same_thresh (float): Threshold to ignore small changes in position.
            flat_time_window_centers (optional): Not used in this implementation.
            flat_time_window_edges (optional): Not used in this implementation.
            debug_print (bool): If True, enables debug printing.

        Returns:
            SubsequencePartitioner: An instance of SubsequencePartitioner with processed subsequences.
        """
        # Initialize the instance
        instance = cls(
            positions=a_most_likely_positions_list,
            same_thresh=same_thresh,
            max_ignore_bins=max_ignore_bins,
        )
        # Set debug print flag
        instance.debug_print = debug_print
        # Process the partitioning and merging
        instance.process()
        return instance
    

    def partition_subsequences(self):
        """
        Partitions the positions into subsequences based on significant direction changes.
        Small changes below the threshold are ignored.
        """
        positions = self.positions
        same_thresh = self.same_thresh

        # Compute first-order differences and directions
        diffs = np.diff(positions, prepend=positions[0])
        directions = np.sign(diffs)

        subsequences = []
        current_subseq = [positions[0]]
        prev_dir = directions[0]

        for i in range(1, len(positions)):
            curr_dir = directions[i]
            diff = diffs[i]

            # Determine if the direction has changed significantly
            if curr_dir != prev_dir and np.abs(diff) > same_thresh:
                subsequences.append(np.array(current_subseq))
                current_subseq = [positions[i]]
                prev_dir = curr_dir
            else:
                current_subseq.append(positions[i])
                # Update the direction if the change is significant
                if np.abs(diff) > same_thresh:
                    prev_dir = curr_dir

        if current_subseq:
            subsequences.append(np.array(current_subseq))

        self.subsequences = subsequences

    def merge_intrusions(self):
        """
        Merges short subsequences (intrusions) into adjacent longer sequences.
        """
        subsequences = self.subsequences
        max_ignore_bins = self.max_ignore_bins

        merged_subsequences = []
        i = 0
        while i < len(subsequences):
            current_seq = subsequences[i]
            if len(current_seq) <= max_ignore_bins:
                # Potential intrusion
                left_seq = merged_subsequences[-1] if merged_subsequences else None
                right_seq = subsequences[i + 1] if i + 1 < len(subsequences) else None

                # Decide which side to merge with
                if left_seq is not None and right_seq is not None:
                    # Both sides available, merge with the longer one
                    if len(left_seq) >= len(right_seq):
                        merged_seq = np.concatenate([left_seq, current_seq])
                        merged_subsequences[-1] = merged_seq
                    else:
                        merged_seq = np.concatenate([current_seq, right_seq])
                        i += 1  # Skip the next sequence as it's merged
                        merged_subsequences.append(merged_seq)
                elif left_seq is not None:
                    # Only left side available
                    merged_seq = np.concatenate([left_seq, current_seq])
                    merged_subsequences[-1] = merged_seq
                elif right_seq is not None:
                    # Only right side available
                    merged_seq = np.concatenate([current_seq, right_seq])
                    i += 1  # Skip the next sequence as it's merged
                    merged_subsequences.append(merged_seq)
                else:
                    # No sides to merge with
                    merged_subsequences.append(current_seq)
            else:
                # Not an intrusion
                merged_subsequences.append(current_seq)
            i += 1

        self.merged_subsequences = merged_subsequences

    def process(self):
        """
        Executes the partitioning and merging processes.
        """
        self.partition_subsequences()
        self.merge_intrusions()

    def get_subsequences(self) -> List[np.ndarray]:
        """
        Returns the partitioned subsequences before merging.

        Returns:
            List[np.ndarray]: List of subsequences.
        """
        return self.subsequences

    def get_merged_subsequences(self) -> List[np.ndarray]:
        """
        Returns the subsequences after merging intrusions.

        Returns:
            List[np.ndarray]: List of merged subsequences.
        """
        return self.merged_subsequences




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

