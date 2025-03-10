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
    from neuropy.analyses.placefields import PfND
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder


class TestDecodersMethods(unittest.TestCase):

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

        finalized_testing_file = root_project_folder.parent.joinpath('NeuroPy').joinpath('tests').joinpath('neuropy_pf_testing.h5') # The test data is in the Neuropy folder
        # finalized_testing_file = tests_folder.joinpath('neuropy_pf_testing.h5')
        sess_identifier_key='sess'
        # Load the saved .h5 spikes_df and active_pos dataframes for testing:
        self.spikes_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/spikes_df')
        active_pos_df = pd.read_hdf(finalized_testing_file, key=f'{sess_identifier_key}/pos_df')
        self.active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object

    def tearDown(self):
        pass


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


    def test_subset_decoder_by_neuron_id(self):
        # Test excluding certain neurons from the decoder

        ## Build placefield for the decoder to use:
        original_decoder_pf1D = PfND.from_config_values(spikes_df=deepcopy(self.spikes_df).spikes.sliced_by_neuron_type('pyramidal'), position=deepcopy(self.active_pos.linear_pos_obj), frate_thresh=0.0) # all other settings default

        ## Build the new decoder with custom params:
        new_decoder_pf_params = deepcopy(original_decoder_pf1D.config) # should be a PlacefieldComputationParameters
        # override some settings before computation:
        new_decoder_pf_params.time_bin_size = 0.1

        ## 1D Decoder
        new_1D_decoder_spikes_df = original_decoder_pf1D.filtered_spikes_df.copy()

        # Why would it need both the pf1D and the spikes? Doesn't the pf1D include the spikes (and determine the placefields, which are all that are used)???
        original_1D_decoder = BayesianPlacemapPositionDecoder(time_bin_size=new_decoder_pf_params.time_bin_size, pf=original_decoder_pf1D, spikes_df=new_1D_decoder_spikes_df, debug_print=False)
        original_1D_decoder.compute_all()

        original_decoder = original_1D_decoder # strangely this makes original_pf.included_neuron_IDs wrapped in an extra list!
        original_neuron_ids = np.array(original_decoder.pf.ratemap.neuron_ids) # original_pf.included_neuron_IDs
        subset_included_neuron_IDXs = np.arange(10) # only get the first 10 neuron_ids
        subset_included_neuron_ids = original_neuron_ids[subset_included_neuron_IDXs] # only get the first 10 neuron_ids
        if self.enable_debug_printing:
            print(f'{original_neuron_ids = }\n{subset_included_neuron_ids = }')
        neuron_sliced_1D_decoder = original_decoder.get_by_id(subset_included_neuron_ids)
        neuron_sliced_1D_decoder_neuron_ids = np.array(neuron_sliced_1D_decoder.pf.ratemap.neuron_ids)
        if self.enable_debug_printing:
            print(f'{neuron_sliced_1D_decoder_neuron_ids = }')

        self.assertTrue(np.all(neuron_sliced_1D_decoder_neuron_ids == subset_included_neuron_ids)) # ensure that the returned neuron ids actually equal the desired subset
        # self.assertTrue(np.all(np.array(neuron_sliced_pf.ratemap.neuron_ids) == subset_included_neuron_ids)) # ensure that the ratemap neuron ids actually equal the desired subset
        # self.assertTrue(len(neuron_sliced_pf.ratemap.tuning_curves) == len(subset_included_neuron_ids)) # ensure one output tuning curve for each neuron_id
        # self.assertTrue(np.all(np.isclose(neuron_sliced_pf.ratemap.tuning_curves, [original_pf.ratemap.tuning_curves[idx] for idx in subset_included_neuron_IDXs]))) # ensure that the tuning curves built for the neuron_slided_pf are the same as those subset as retrieved from the  original_pf


class TestEpochsSpkcountMethods(unittest.TestCase):

    def setUp(self):
        self.enable_debug_printing = False
        self.test_spikes_df = pd.DataFrame({
            'neuron_id': [1, 1, 2, 2, 3],
            'time': [0.1, 0.2, 0.15, 0.25, 0.3]
        })
        self.test_epochs_df = pd.DataFrame({
            'start': [0.0, 0.5],
            'stop': [0.4, 0.8]
        })

    def test_single_time_bin_per_epoch(self):
        spkcount, nbins, time_bins = epochs_spkcount(
            self.test_spikes_df, 
            self.test_epochs_df,
            bin_size=0.1,
            export_time_bins=True,
            use_single_time_bin_per_epoch=True
        )
        self.assertEqual(len(nbins), 2)
        self.assertTrue(all(nbins == 1))
        self.assertEqual(len(time_bins), 2)

    def test_short_epoch_handling(self):
        short_epochs_df = pd.DataFrame({
            'start': [0.0, 0.5],
            'stop': [0.005, 0.51]
        })
        spkcount, nbins, time_bins = epochs_spkcount(
            self.test_spikes_df,
            short_epochs_df,
            bin_size=0.01,
            export_time_bins=True
        )
        self.assertEqual(len(nbins), 2)
        self.assertTrue(all(nbins == 1))

    def test_included_neuron_ids(self):
        specific_neuron_ids = [1, 2, 3, 4]
        spkcount, nbins, _ = epochs_spkcount(
            self.test_spikes_df,
            self.test_epochs_df,
            included_neuron_ids=specific_neuron_ids,
            bin_size=0.1
        )
        self.assertEqual(len(spkcount[0]), len(specific_neuron_ids))

    def test_variable_slideby(self):
        spkcount, nbins, _ = epochs_spkcount(
            self.test_spikes_df,
            self.test_epochs_df,
            bin_size=0.1,
            slideby=0.05
        )
        self.assertTrue(all(n > 0 for n in nbins))

    def test_empty_epochs(self):
        empty_epochs_df = pd.DataFrame({
            'start': [],
            'stop': []
        })
        spkcount, nbins, time_bins = epochs_spkcount(
            self.test_spikes_df,
            empty_epochs_df,
            bin_size=0.1,
            export_time_bins=True
        )
        self.assertEqual(len(spkcount), 0)
        self.assertEqual(len(nbins), 0)
        self.assertEqual(len(time_bins), 0)
        

if __name__ == '__main__':

    unittest.main()