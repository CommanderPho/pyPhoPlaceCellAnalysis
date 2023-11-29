import unittest
import sys, os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
print('root_project_folder: {}'.format(root_project_folder))
src_folder = root_project_folder.joinpath('src')
pyphoplacecellanalysis_folder = src_folder.joinpath('pyphoplacecellanalysis')
print('pyphoplacecellanalysis_folder: {}'.format(pyphoplacecellanalysis_folder))

try:
    import pyphoplacecellanalysis
except ModuleNotFoundError as e:    
    print('root_project_folder: {}'.format(root_project_folder))
    src_folder = root_project_folder.joinpath('src')
    pyphoplacecellanalysis_folder = src_folder.joinpath('pyphoplacecellanalysis')
    print('pyphoplacecellanalysis_folder: {}'.format(pyphoplacecellanalysis_folder))
    sys.path.insert(0, str(src_folder))
finally:
    from neuropy.utils.indexing_helpers import paired_incremental_sorting
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_incremental_sort_neurons
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData



class TestPairedIncrementalSorting(unittest.TestCase):

    def setUp(self):
        """ Corresponding load for Neuropy Testing file 'NeuroPy/tests/neuropy_pf_testing.h5': 
            ## Save for NeuroPy testing:
            finalized_testing_file='../NeuroPy/tests/neuropy_pf_testing.h5'
            sess_identifier_key='sess'
            spikes_df.to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
            active_pos.to_dataframe().to_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df', format='table')
        """
        self.enable_debug_plotting = False
        self.enable_debug_printing = True

        # finalized_testing_file = root_project_folder.parent.joinpath('NeuroPy').joinpath('tests').joinpath('neuropy_pf_testing.h5') # The test data is in the Neuropy folder
        tests_folder = Path(os.path.dirname(__file__)).resolve()
        print(f'tests_folder: {tests_folder}')
        # tests_folder = tests_folder.resolve()
        # tests_folder = pyphoplacecellanalysis_folder.joinpath('tests').resolve()
        assert tests_folder.exists()
        finalized_directional_laps_testing_file = tests_folder.joinpath('DirectionalLaps_2Hz.pkl').resolve() # The test data is in the Neuropy folder
        finalized_testing_file = tests_folder.joinpath('2023-11-28_debug_paired_incremental_sort_neurons_data.pkl').resolve() # The test data is in the Neuropy folder

        # Load the data from a file into the pipeline:
        # finalized_directional_laps_testing_file = curr_active_pipeline.get_output_path().joinpath('DirectionalLaps_2Hz.pkl').resolve()
        assert finalized_directional_laps_testing_file.exists()
        self.loaded_directional_laps, self.loaded_rank_order = loadData(finalized_directional_laps_testing_file)
        assert (self.loaded_directional_laps is not None)
        assert (self.loaded_rank_order is not None)
        # load the main file:
        assert finalized_testing_file.exists()
        self.decoders_dict, self.included_any_context_neuron_ids, self.sorted_neuron_IDs_lists, self.sort_helper_neuron_id_to_neuron_colors_dicts, self.sorted_pf_tuning_curves = loadData(finalized_testing_file)



    def tearDown(self):
        pass


    def test_none_included_any_context_neuron_ids(self):
        """ Test with None included_any_context_neuron_ids """
        testNone_included_any_context_neuron_ids = None
        testNone_sorted_neuron_IDs_lists, testNone_sort_helper_neuron_id_to_neuron_colors_dicts, testNone_sorted_pf_tuning_curves = paired_incremental_sort_neurons( decoders_dict=self.decoders_dict, included_any_context_neuron_ids=testNone_included_any_context_neuron_ids )

        test1_included_any_context_neuron_ids = np.array([9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
        test1_sorted_neuron_IDs_lists, test1_sort_helper_neuron_id_to_neuron_colors_dicts, test1_sorted_pf_tuning_curves = paired_incremental_sort_neurons( decoders_dict=self.decoders_dict, included_any_context_neuron_ids=test1_included_any_context_neuron_ids )
        
        self.assertTrue(np.all([np.array_equal(testNone_sorted_pf_tuning_curves[i], test1_sorted_pf_tuning_curves[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
        self.assertTrue(np.all([np.array_equal(testNone_sort_helper_neuron_id_to_neuron_colors_dicts[i], test1_sort_helper_neuron_id_to_neuron_colors_dicts[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
        self.assertTrue(np.all([np.array_equal(testNone_sorted_neuron_IDs_lists[i], test1_sorted_neuron_IDs_lists[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
        # Additional assertions or verifications can be added here

    def test_single_neuron_id(self):
        """ Test with a single neuron ID in included_any_context_neuron_ids """
        test0_included_any_context_neuron_ids = np.array([25])
        test0_sorted_neuron_IDs_lists, test0_sort_helper_neuron_id_to_neuron_colors_dicts, test0_sorted_pf_tuning_curves = paired_incremental_sort_neurons( decoders_dict=self.decoders_dict, included_any_context_neuron_ids=test0_included_any_context_neuron_ids )
        # Additional assertions or verifications can be added here

    def test_multiple_neuron_ids(self):
        """ Test with multiple neuron IDs in included_any_context_neuron_ids and compare with the None case """
        test1_included_any_context_neuron_ids = np.array([9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
        test1_sorted_neuron_IDs_lists, test1_sort_helper_neuron_id_to_neuron_colors_dicts, test1_sorted_pf_tuning_curves = paired_incremental_sort_neurons( decoders_dict=self.decoders_dict, included_any_context_neuron_ids=test1_included_any_context_neuron_ids )
        # Assert comparisons with the None case
        self.assertTrue(np.all([np.array_equal(self.sorted_neuron_IDs_lists[i], test1_sorted_neuron_IDs_lists[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))        
        self.assertTrue(np.all([np.array_equal(self.sorted_pf_tuning_curves[i], test1_sorted_pf_tuning_curves[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
        

    def test_single_omitted_neuron_id(self):
        """ Test with one neuron ID omitted (aclu == 25), the first one in the sorted lists from included_any_context_neuron_ids """
        test2_included_any_context_neuron_ids = np.array([9, 10, 11, 15, 16, 18, 24, 26, 31, 39, 40, 43, 44, 47, 48, 51, 52, 53, 54, 56, 60, 61, 65, 66, 68, 70, 72, 75, 77, 78, 79, 80, 81, 82, 84, 85, 87, 89, 90, 92, 93, 98, 101, 102, 104])
        test2_sorted_neuron_IDs_lists, test2_sort_helper_neuron_id_to_neuron_colors_dicts, test2_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=self.decoders_dict, included_any_context_neuron_ids=test2_included_any_context_neuron_ids)

        testNone_included_any_context_neuron_ids = None
        testNone_sorted_neuron_IDs_lists, testNone_sort_helper_neuron_id_to_neuron_colors_dicts, testNone_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=self.decoders_dict, included_any_context_neuron_ids=testNone_included_any_context_neuron_ids)

        test2_sorted_neuron_IDs_lists

        self.assertTrue(np.all([np.array_equal(self.sorted_neuron_IDs_lists[i][1:], test2_sorted_neuron_IDs_lists[i]) for i in range(len(test2_sorted_neuron_IDs_lists))]))  # sorted lists should just be the same without the omitted element
          
        # self.assertTrue(np.all([np.array_equal(self.sorted_pf_tuning_curves[i], test2_sorted_pf_tuning_curves[i]) for i in range(len(test2_sorted_pf_tuning_curves))]))

        # Additional assertions or verifications can be added here

    def test_many_omitted_neuron_id(self):
        """ Test with one neuron ID omitted (aclu == 25), the first one in the sorted lists from included_any_context_neuron_ids """
        test2_included_any_context_neuron_ids = np.array([9, 10, 11, 15, 16, 18, 24, 26, 31, 39, 40, 43, 44, 47, 48, 51, 52, 53, 54, 56, 60, 61, 65, 66, 68, 70, 72, 75, 77, 78, 79, 80, 81, 82, 84, 85, 87, 89, 90, 92, 93, 98, 101, 102, 104])
        # test3 omits many elements:
        test3_included_any_context_neuron_ids = np.array([54,  56,  70,  72,  75,  77,  78,  80,  81,  82,  84,  85,  87,  92,  93,  98, 101, 102, 104])
        test3_sorted_neuron_IDs_lists, test3_sort_helper_neuron_id_to_neuron_colors_dicts, test3_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=self.decoders_dict, included_any_context_neuron_ids=test3_included_any_context_neuron_ids)

        # _out_test3_directional_template_pfs_debugger = curr_active_pipeline.display(DirectionalPlacefieldGlobalDisplayFunctions._display_directional_template_debugger, included_any_context_neuron_ids=test3_included_any_context_neuron_ids, debug_print=True, figure_name=f'many omitted indicies')

        testNone_included_any_context_neuron_ids = None
        testNone_sorted_neuron_IDs_lists, testNone_sort_helper_neuron_id_to_neuron_colors_dicts, testNone_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=self.decoders_dict, included_any_context_neuron_ids=testNone_included_any_context_neuron_ids)

        self.assertTrue(np.all([np.array_equal(self.sorted_neuron_IDs_lists[i][1:], test3_sorted_neuron_IDs_lists[i]) for i in range(len(test3_sorted_neuron_IDs_lists))]))  # sorted lists should just be the same without the omitted element


        #TODO 2023-11-28 19:30: - [ ] Finish implementing


        # self.assertTrue(np.all([np.array_equal(self.sorted_pf_tuning_curves[i], test2_sorted_pf_tuning_curves[i]) for i in range(len(test2_sorted_pf_tuning_curves))]))

        # Additional assertions or verifications can be added here


    # _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
    # _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
    # _out_data.sorted_pf_tuning_curves = sorted_pf_tuning_curves

    # saveData('output/2023-11-28_debug_paired_incremental_sort_neurons_data.pkl', (decoders_dict, included_any_context_neuron_ids, sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves))

    # testNone_included_any_context_neuron_ids = None
    # testNone_sorted_neuron_IDs_lists, testNone_sort_helper_neuron_id_to_neuron_colors_dicts, testNone_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=testNone_included_any_context_neuron_ids)
    # all_neuron_ids = np.sort(union_of_arrays(*testNone_sorted_neuron_IDs_lists))

    # test0_included_any_context_neuron_ids = np.array([25])
    # test_sorted_neuron_IDs_lists, test_sort_helper_neuron_id_to_neuron_colors_dicts, test_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=test0_included_any_context_neuron_ids)
    
    # test1_included_any_context_neuron_ids = np.array([9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
    # test1_sorted_neuron_IDs_lists, test1_sort_helper_neuron_id_to_neuron_colors_dicts, test1_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=test1_included_any_context_neuron_ids)
    # assert np.all([testNone_sorted_pf_tuning_curves[i] == test1_sorted_pf_tuning_curves[i] for i in np.arange(len(test1_sorted_pf_tuning_curves))])
    # assert np.all([testNone_sorted_neuron_IDs_lists[i] == test1_sorted_neuron_IDs_lists[i] for i in np.arange(len(test1_sorted_pf_tuning_curves))])

    # # test2 omits only one element: aclu == 25
    # test2_included_any_context_neuron_ids = np.array([9, 10, 11,  15,  16,  18,  24,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
    # test2_sorted_neuron_IDs_lists, test2_sort_helper_neuron_id_to_neuron_colors_dicts, test2_sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=test2_included_any_context_neuron_ids)


    # def test_conform_to_position_bins(self):
    #     ## Generate Placefields with varying bin-sizes:
    #     ### Here we use frate_thresh=0.0 which ensures that differently binned ratemaps don't have different numbers of spikes or cells.
    #     smooth_options = [(None, None)]
    #     grid_bin_options = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
    #     all_param_sweep_options, param_sweep_option_n_values = parameter_sweeps(grid_bin=grid_bin_options, smooth=smooth_options, frate_thresh=[0.0])
    #     output_pfs = _compute_parameter_sweep(self.spikes_df, self.active_pos, all_param_sweep_options)
    #     num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(output_pfs)
    #     if self.enable_debug_printing:
    #         print_aligned_columns(['grid_bin x smooth', 'num_good_neurons', 'num_total_spikes'], [all_param_sweep_options, num_good_placefield_neurons_list, num_total_spikes_list], enable_checking_all_values_width=True)
    #     fine_binned_pf = list(output_pfs.values())[0]
    #     coarse_binned_pf = list(output_pfs.values())[-1]

    #     if self.enable_debug_printing:
    #         print(f'{coarse_binned_pf.bin_info = }\n{fine_binned_pf.bin_info = }')
    #     rebinned_fine_binned_pf = deepcopy(fine_binned_pf)
    #     rebinned_fine_binned_pf.conform_to_position_bins(target_pf1D=coarse_binned_pf, force_recompute=True)
    #     self.assertTrue(rebinned_fine_binned_pf.bin_info == coarse_binned_pf.bin_info) # the bins must be equal after conforming

    #     num_good_placefield_neurons_list, num_total_spikes_list, num_spikes_per_spiketrain_list = compare_placefields_info(dict(zip(['coarse', 'original', 'rebinned'],[coarse_binned_pf, fine_binned_pf, rebinned_fine_binned_pf])))
    #     if self.enable_debug_printing:
    #         print_aligned_columns(['pf', 'num_good_neurons', 'num_total_spikes'], [['coarse', 'original', 'rebinned'], num_good_placefield_neurons_list, num_total_spikes_list], enable_checking_all_values_width=True)

    #     self.assertTrue(num_good_placefield_neurons_list[0] == num_good_placefield_neurons_list[-1]) # require the rebinned pf to have the same number of good neurons as the one that it conformed to
    #     self.assertTrue(num_total_spikes_list[0] == num_total_spikes_list[-1]) # require the rebinned pf to have the same number of total spikes as the one that it conformed to
    #     #  self.assertTrue(assert num_spikes_per_spiketrain_list[0] == num_spikes_per_spiketrain_list[-1]) # require the rebinned pf to have the same number of spikes in each spiketrain as the one that it conformed to


    
if __name__ == '__main__':
    unittest.main()