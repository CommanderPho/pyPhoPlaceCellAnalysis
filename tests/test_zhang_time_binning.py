from copy import deepcopy
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
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation
from neuropy.utils.mixins.binning_helpers import BinningContainer
from pyphocorehelpers.indexing_helpers import build_pairwise_indicies


class TestTimeBinningMethods(unittest.TestCase):

    def setUp(self):
        ## TODO: Convert
        
        ## NEEDS:
        """ 2022-12-15 - TODO: refactor to actually test functionality. Need a pseudo spikes-df to test with.

        In Notebook we use:

            global_epoch_name = curr_active_pipeline.active_completed_computation_result_names[-1] # 'maze'
            global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']
            sess =  curr_active_pipeline.computation_results[global_epoch_name].sess
            active_one_step_decoder = curr_active_pipeline.computation_results[global_epoch_name].computed_data.get('pf2D_Decoder', None)
            active_two_step_decoder = curr_active_pipeline.computation_results[global_epoch_name].computed_data.get('pf2D_TwoStepDecoder', None)
            active_extended_stats = curr_active_pipeline.computation_results[global_epoch_name].computed_data.get('extended_stats', None)
            active_firing_rate_trends = curr_active_pipeline.computation_results[global_epoch_name].computed_data.get('firing_rate_trends', None)
            time_bin_size_seconds, all_session_spikes, pf_included_spikes_only = active_firing_rate_trends['time_bin_size_seconds'], active_firing_rate_trends['all_session_spikes'], active_firing_rate_trends['pf_included_spikes_only']

            active_time_binning_container, active_time_window_edges, active_time_window_edges_binning_info, active_time_binned_unit_specific_binned_spike_rate, active_time_binned_unit_specific_binned_spike_counts = pf_included_spikes_only['time_binning_container'], pf_included_spikes_only['time_window_edges'], pf_included_spikes_only['time_window_edges_binning_info'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_rate'], pf_included_spikes_only['time_binned_unit_specific_binned_spike_counts']
            ZhangReconstructionImplementation._validate_time_binned_spike_rate_df(active_time_binning_container.centers, active_time_binned_unit_specific_binned_spike_counts)


        Requires:
            self.sess.spikes_df


        """

        # Hardcoded:
        self.time_bin_size_seconds = 0.5
        self.bin_edges = np.array([0, 1, 2, 3, 4, 5])
        self.active_session_spikes_df = self.sess.spikes_df.copy()

        # unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)

    def tearDown(self):
        self.bin_edges=None
        self.time_bin_size_seconds = None
        self.active_session_spikes_df = None


    ### Testing `ZhangReconstructionImplementation.time_bin_spike_counts_N_i(...)` and `ZhangReconstructionImplementation.compute_time_binned_spiking_activity(...)`
    def test_time_bin_spike_counts_N_i(self, out_digitized_variable_bins, out_binning_info):
        np.shape(out_digitized_variable_bins) # (85842,), array([  22.30206346,   22.32206362,   22.34206378, ..., 1739.09557005, 1739.11557021, 1739.13557036])
        self.assertEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!")
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")

        ## from `_setup_time_bin_spike_counts_N_i`: using `ZhangReconstructionImplementation.time_bin_spike_counts_N_i(...)` this one now works too, but its output is transposed compared to the `_perform_firing_rate_trends_computation` version:
        unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.active_session_spikes_df.copy(), time_bin_size=self.time_bin_size_seconds, debug_print=False)  # np.shape(unit_specific_spike_counts): (4188, 108)
        time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        ZhangReconstructionImplementation._validate_time_binned_spike_counts(time_binning_container, unit_specific_binned_spike_counts)


    def test_compute_time_binned_spiking_activity(self):
        ## from `_perform_firing_rate_trends_computation`: using `ZhangReconstructionImplementation.compute_time_binned_spiking_activity(...)` this one now all makes sense:
        unit_specific_binned_spike_count_df, sess_time_window_edges, sess_time_window_edges_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(self.active_session_spikes_df.copy(), max_time_bin_size=self.time_bin_size_seconds, debug_print=False) # np.shape(unit_specific_spike_counts): (4188, 108)
        sess_time_binning_container = BinningContainer(edges=sess_time_window_edges, edge_info=sess_time_window_edges_binning_info)
        ZhangReconstructionImplementation._validate_time_binned_spike_rate_df(sess_time_binning_container.centers, unit_specific_binned_spike_count_df)


    def test_compute_time_binned_spiking_activity_from_extant_binning_info(self, extant_time_window_edges, extant_time_window_edges_binning_info):
        """ Test `ZhangReconstructionImplementation.time_bin_spike_counts_N_i(...)` with manual bins -- `_setup_time_bin_spike_counts_N_i`: using `ZhangReconstructionImplementation.time_bin_spike_counts_N_i(...)` this one now works too, but its output is transposed compared to the `_perform_firing_rate_trends_computation` version:
            extant_time_window_edges = deepcopy(time_binning_container.edges)
            extant_time_window_edges_binning_info = deepcopy(time_binning_container.edge_info)
        """
        unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.active_session_spikes_df.copy(), time_bin_size=self.time_bin_size_seconds,
                                                                                                                                                        time_window_edges=extant_time_window_edges, time_window_edges_binning_info=extant_time_window_edges_binning_info, debug_print=False)  # np.shape(unit_specific_spike_counts): (4188, 108)
        time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        ZhangReconstructionImplementation._validate_time_binned_spike_counts(time_binning_container, unit_specific_binned_spike_counts)
         

if __name__ == '__main__':
    unittest.main()
    
    