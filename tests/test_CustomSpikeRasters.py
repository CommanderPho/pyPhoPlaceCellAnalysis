import unittest
import sys, os
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots, RenderPlotsData

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

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import RasterPlotParams, Render2DScrollWindowPlotMixin, UnitSortOrderManager, RasterScatterPlotManager, NeuronSpikesConfigTuple, _build_scatter_plotting_managers


class TestScatterPlottingManagers(unittest.TestCase):

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
        assert tests_folder.exists()
        finalized_testing_file = tests_folder.joinpath('test_raster_Functions_spikes_df.pkl').resolve() # The test data is in the Neuropy folder
        # Load the data from a file into the pipeline:
        self.loaded_spikes_df = pd.read_pickle(finalized_testing_file)
        assert (self.loaded_spikes_df is not None)


    def tearDown(self):
        pass


    def test_none_included_any_context_neuron_ids(self):
        """ Test with None included_any_context_neuron_ids """
        active_neuron_ids = np.array([  9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
        active_sort_idxs = np.array([18, 17, 19,  5, 35, 23, 31,  4, 45, 21, 37, 36, 10,  7, 16,  9,  2, 40, 20, 28, 13, 41, 38, 25, 29, 42,  0, 14, 34, 44, 32, 11, 30, 12, 24,  3, 39,  1,  6, 27,  8, 22, 15, 33, 43, 26])
        active_unit_colors_list = None
        active_sorted_neuron_ids = active_neuron_ids[active_sort_idxs] # [ 53  52  54  18  84  65  79  16 104  60  87  85  39  25  51  31  11  92  56  75  44  93  89  68  77  98   9  47  82 102  80  40  78  43  66  15  90  10  24  72  26  61  48  81 101  70]
        print(f"active_sorted_neuron_ids: {active_sorted_neuron_ids}")
        # Get only the spikes for the shared_aclus:
        a_spikes_df = deepcopy(self.loaded_spikes_df).spikes.sliced_by_neuron_id(active_neuron_ids)
        a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

        # make root container for plots
        scatter_app_name = 'test'
        defer_show = True
        active_context = None
        # app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)
        plots = RenderPlots(scatter_app_name)
        plots_data = RenderPlotsData(scatter_app_name)
        if active_context is not None:
            plots_data.active_context = active_context
            

        plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=a_spikes_df, included_neuron_ids=active_neuron_ids, unit_sort_order=active_sort_idxs, unit_colors_list=active_unit_colors_list)

        ## Add the source data (spikes_df) to the plot_data
        plots_data.spikes_df = deepcopy(a_spikes_df)

        # Update the dataframe
        plots_data.spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(plots_data.spikes_df, overwrite_existing=True)

        ## Build the spots for the raster plot:
        plots_data.all_spots, plots_data.all_scatterplot_tooltips_kwargs = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(plots_data.spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

        expected_sorted_y_values = deepcopy(plots_data.unit_sort_manager.series_identity_y_values)[active_sort_idxs]
        
        ## Get actual values:
        # Grouped on columns: 'aclu', 'fragile_linear_neuron_IDX', 'visualization_raster_y_location'
        a_grouped_spikes_df = deepcopy(a_spikes_df).groupby(['aclu', 'fragile_linear_neuron_IDX', 'visualization_raster_y_location']).count().reset_index()[['aclu', 'fragile_linear_neuron_IDX', 'visualization_raster_y_location']]
        # Sort by column: 'visualization_raster_y_location' (ascending)
        a_grouped_spikes_df = a_grouped_spikes_df.sort_values(['visualization_raster_y_location'])
        # Actually recreates the observed sort in the raster plot (that looks "basically random")
        actual_sorted_neuron_IDs = a_grouped_spikes_df.aclu.to_numpy()
        actual_sorted_fragile_linear_neuron_IDX = a_grouped_spikes_df.fragile_linear_neuron_IDX.to_numpy()

        print(f'actual_sorted_neuron_IDs: {actual_sorted_neuron_IDs}') # actual_sorted_neuron_IDs: [ 70  87  51  84  25  15  89  44  92  48  43  79  81  56  72  98  47  10   9  11  53  31  93  18  82  65 104  90  54  66  80  24  78 101  75  16  40  39  61  85  52  60  68 102  77  26]
        print(f'actual_sorted_fragile_linear_neuron_IDX: {actual_sorted_fragile_linear_neuron_IDX}') # actual_sorted_fragile_linear_neuron_IDX: [26 37 16 35  7  3 38 13 40 15 12 31 33 20 27 42 14  1  0  2 18  9 41  5 34 23 45 39 19 24 32  6 30 43 28  4 11 10 22 36 17 21 25 44 29  8]



        self.assertIsNotNone(plots_data)
        
        # testNone_included_any_context_neuron_ids = None
        # testNone_sorted_neuron_IDs_lists, testNone_sort_helper_neuron_id_to_neuron_colors_dicts, testNone_sorted_pf_tuning_curves = paired_incremental_sort_neurons( decoders_dict=self.decoders_dict, included_any_context_neuron_ids=testNone_included_any_context_neuron_ids )

        # test1_included_any_context_neuron_ids = np.array([9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
        # test1_sorted_neuron_IDs_lists, test1_sort_helper_neuron_id_to_neuron_colors_dicts, test1_sorted_pf_tuning_curves = paired_incremental_sort_neurons( decoders_dict=self.decoders_dict, included_any_context_neuron_ids=test1_included_any_context_neuron_ids )
        
        # self.assertTrue(np.all([np.array_equal(testNone_sorted_pf_tuning_curves[i], test1_sorted_pf_tuning_curves[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
        # self.assertTrue(np.all([np.array_equal(testNone_sort_helper_neuron_id_to_neuron_colors_dicts[i], test1_sort_helper_neuron_id_to_neuron_colors_dicts[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
        # self.assertTrue(np.all([np.array_equal(testNone_sorted_neuron_IDs_lists[i], test1_sorted_neuron_IDs_lists[i]) for i in range(len(test1_sorted_pf_tuning_curves))]))
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