import unittest
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))

try:
    import pyphoplacecellanalysis
except ModuleNotFoundError as e:    
    root_project_folder = tests_folder.parent
    print('root_project_folder: {}'.format(root_project_folder))
    src_folder = root_project_folder.joinpath('src')
    pyphoplacecellanalysis_folder = src_folder.joinpath('pyphoplacecellanalysis')
    print('pyphoplacecellanalysis_folder: {}'.format(pyphoplacecellanalysis_folder))
    sys.path.insert(0, str(src_folder))
finally:
    from pyphoplacecellanalysis.Analysis.reconstruction import BayesianPlacemapPositionDecoder


class TestPhoCustomPositionDecoderMethods(unittest.TestCase):

    def setUp(self):
        # Hardcoded:
        self.bin_edges = np.array([0, 1, 2, 3, 4, 5])
        # unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)

    def tearDown(self):
        self.bin_edges=None
        

    def test_time_bin_spike_counts_N_i(self, out_digitized_variable_bins, out_binning_info):
        np.shape(out_digitized_variable_bins) # (85842,), array([  22.30206346,   22.32206362,   22.34206378, ..., 1739.09557005, 1739.11557021, 1739.13557036])
        self.assertEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!")
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        
    
    def test_pho_custom_decoder_save_to_disk(self):
        # save the file out to disk:
        curr_posterior_save_path = curr_kdiba_pipeline.active_configs[curr_result_label].plotting_config.active_output_parent_dir.joinpath(curr_kdiba_pipeline.active_configs[curr_result_label].computation_config.str_for_filename(True), 'decoded_posterior.npy') # WindowsPath('output/2006-6-07_11-26-53/maze1/speedThresh_0.00-gridBin_5.00_5.00-smooth_0.00_0.00-frateThresh_0.10/decoded_posterior.npz')
        self.pho_custom_decoder.save(self.curr_posterior_save_path) # output is 417 MB
        
    def test_pho_custom_decoder_load_from_disk(self):
        # try loading again
        test_loaded_custom_decoder = BayesianPlacemapPositionDecoder.from_file(self.curr_posterior_save_path) # output is 417 MB
        test_loaded_custom_decoder


    def build_position_df_resampled_to_time_windows(active_pos_df, time_bin_size=0.02):
        position_time_delta = pd.to_timedelta(active_pos_df[active_pos_df.position.time_variable_name], unit="sec")
        active_pos_df['time_delta_sec'] = position_time_delta
        active_pos_df = active_pos_df.set_index('time_delta_sec')
        return active_pos_df




if __name__ == '__main__':
    unittest.main()
    
    