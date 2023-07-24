import unittest
import sys, os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent

try:
    import pyphoplacecellanalysis
except ModuleNotFoundError as e:    
    print('root_project_folder: {}'.format(root_project_folder))
    src_folder = root_project_folder.joinpath('src')
    pyphoplacecellanalysis_folder = src_folder.joinpath('pyphoplacecellanalysis')
    print('pyphoplacecellanalysis_folder: {}'.format(pyphoplacecellanalysis_folder))
    sys.path.insert(0, str(src_folder))
finally:
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    

class TestDecodedFilterEpochsResult(unittest.TestCase):

    def test_filter_by_epochs(self):
        # Create a sample DecodedFilterEpochsResult instance for testing
        most_likely_positions_list = [1, 2, 3, 4, 5]
        p_x_given_n_list = [6, 7, 8, 9, 10]
        marginal_x_list = [11, 12, 13, 14, 15]
        marginal_y_list = [16, 17, 18, 19, 20]
        most_likely_position_indicies_list = [21, 22, 23, 24, 25]
        spkcount = [26, 27, 28, 29, 30]
        time_bin_containers = [31, 32, 33, 34, 35]
        epoch_description_list = ['A', 'B', 'C', 'D', 'E']
        num_filter_epochs = 5
        nbins = np.array([36, 37, 38, 39, 40])
        time_bin_edges = [41, 42, 43, 44, 45]

        instance = DecodedFilterEpochsResult(
            most_likely_positions_list=most_likely_positions_list,
            p_x_given_n_list=p_x_given_n_list,
            marginal_x_list=marginal_x_list,
            marginal_y_list=marginal_y_list,
            most_likely_position_indicies_list=most_likely_position_indicies_list,
            spkcount=spkcount,
            time_bin_containers=time_bin_containers,
            epoch_description_list=epoch_description_list,
            num_filter_epochs=num_filter_epochs,
            nbins=nbins,
            time_bin_edges=time_bin_edges,
            decoding_time_bin_size=0.25
        )

        # Perform filtering
        included_epoch_indicies = [1, 3, 4]
        subset = instance.filtered_by_epochs(included_epoch_indicies)

        # Assert the lengths of the filtered fields
        self.assertEqual(len(subset.most_likely_positions_list), len(included_epoch_indicies))
        self.assertEqual(len(subset.p_x_given_n_list), len(included_epoch_indicies))
        self.assertEqual(len(subset.marginal_x_list), len(included_epoch_indicies))
        self.assertEqual(len(subset.marginal_y_list), len(included_epoch_indicies))
        self.assertEqual(len(subset.most_likely_position_indicies_list), len(included_epoch_indicies))
        self.assertEqual(len(subset.spkcount), len(included_epoch_indicies))
        self.assertEqual(len(subset.time_bin_containers), len(included_epoch_indicies))
        self.assertEqual(len(subset.epoch_description_list), len(included_epoch_indicies))

        # Assert the values of the filtered fields
        expected_most_likely_positions = [2, 4, 5]
        expected_p_x_given_n = [7, 9, 10]
        expected_marginal_x = [12, 14, 15]
        expected_marginal_y = [17, 19, 20]
        expected_most_likely_position_indicies = [22, 24, 25]
        expected_spkcount = [27, 29, 30]
        expected_time_bin_containers = [32, 34, 35]
        expected_epoch_description_list = ['B', 'D', 'E']

        self.assertEqual(subset.most_likely_positions_list, expected_most_likely_positions)
        self.assertEqual(subset.p_x_given_n_list, expected_p_x_given_n)
        self.assertEqual(subset.marginal_x_list, expected_marginal_x)
        self.assertEqual(subset.marginal_y_list, expected_marginal_y)
        self.assertEqual(subset.most_likely_position_indicies_list, expected_most_likely_position_indicies)
        self.assertEqual(subset.spkcount, expected_spkcount)
        self.assertEqual(subset.time_bin_containers, expected_time_bin_containers)
        self.assertEqual(subset.epoch_description_list, expected_epoch_description_list)

        # Assert the metadata attributes
        self.assertEqual(subset.num_filter_epochs, len(included_epoch_indicies))
        self.assertEqual(subset.nbins.tolist(), [37, 39, 40])

if __name__ == '__main__':
    unittest.main()
    