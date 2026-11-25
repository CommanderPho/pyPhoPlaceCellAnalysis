import unittest
import sys, os
from pathlib import Path
from copy import deepcopy
from typing import Optional, Tuple, Union
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
    from neuropy.analyses.decoders import epochs_spkcount
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, BasePositionDecoder, DecodedFilterEpochsResult, ZhangReconstructionImplementation


DecodingContinuousCacheKey = Tuple[float, float]


def decoding_continuous_cache_key(decoding_time_bin_size: float, slideby: Optional[float]) -> DecodingContinuousCacheKey:
    """Mirror of DirectionalPlacefieldGlobalComputationFunctions.decoding_continuous_cache_key (keep in sync)."""
    w = float(decoding_time_bin_size)
    h = float(slideby) if slideby is not None else w
    return (w, h)


def normalize_continuous_decoding_cache_lookup_key(extant: Union[float, DecodingContinuousCacheKey], slideby: Optional[float] = None) -> DecodingContinuousCacheKey:
    """Mirror of DirectionalPlacefieldGlobalComputationFunctions.normalize_continuous_decoding_cache_lookup_key (keep in sync)."""
    if isinstance(extant, tuple) and len(extant) == 2:
        return (float(extant[0]), float(extant[1]))
    return decoding_continuous_cache_key(float(extant), slideby)


H5_FIXTURE_PATH = root_project_folder.parent.joinpath('NeuroPy').joinpath('tests').joinpath('neuropy_pf_testing.h5')
requires_neuropy_h5 = unittest.skipUnless(H5_FIXTURE_PATH.is_file(), f'missing {H5_FIXTURE_PATH.name}')


@requires_neuropy_h5
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
        self.assertTrue(np.all(neuron_sliced_1D_decoder.neuron_IDs == subset_included_neuron_ids))
        self.assertEqual(neuron_sliced_1D_decoder.F.shape[1], len(subset_included_neuron_ids))
        self.assertIsNotNone(neuron_sliced_1D_decoder.p_x_given_n)
        self.assertIsNotNone(neuron_sliced_1D_decoder.most_likely_positions)

        deferred_sliced_decoder = original_decoder.get_by_id(subset_included_neuron_ids, defer_compute_all=True)
        self.assertTrue(np.all(np.array(deferred_sliced_decoder.pf.ratemap.neuron_ids) == subset_included_neuron_ids))
        self.assertTrue(np.all(deferred_sliced_decoder.neuron_IDs == subset_included_neuron_ids))
        self.assertEqual(deferred_sliced_decoder.F.shape[1], len(subset_included_neuron_ids))
        self.assertIsNone(deferred_sliced_decoder.p_x_given_n)
        self.assertIsNone(deferred_sliced_decoder.most_likely_positions)

        reversed_neuron_ids = subset_included_neuron_ids[::-1]
        reversed_sliced_decoder = original_decoder.get_by_id(reversed_neuron_ids, defer_compute_all=True)
        self.assertTrue(np.all(np.array(reversed_sliced_decoder.pf.ratemap.neuron_ids) == reversed_neuron_ids))
        self.assertTrue(np.all(reversed_sliced_decoder.neuron_IDs == reversed_neuron_ids))
        self.assertEqual(reversed_sliced_decoder.F.shape[1], len(reversed_neuron_ids))
        # self.assertTrue(np.all(np.array(neuron_sliced_pf.ratemap.neuron_ids) == subset_included_neuron_ids)) # ensure that the ratemap neuron ids actually equal the desired subset
        # self.assertTrue(len(neuron_sliced_pf.ratemap.tuning_curves) == len(subset_included_neuron_ids)) # ensure one output tuning curve for each neuron_id
        # self.assertTrue(np.all(np.isclose(neuron_sliced_pf.ratemap.tuning_curves, [original_pf.ratemap.tuning_curves[idx] for idx in subset_included_neuron_IDXs]))) # ensure that the tuning curves built for the neuron_slided_pf are the same as those subset as retrieved from the  original_pf


class TestDecodingContinuousCacheKey(unittest.TestCase):


    def test_decoding_continuous_cache_key_none_slideby_is_width(self):
        self.assertEqual(decoding_continuous_cache_key(0.25, None), (0.25, 0.25))
        self.assertEqual(decoding_continuous_cache_key(0.25, 0.05), (0.25, 0.05))


    def test_normalize_lookup_accepts_float_or_tuple(self):
        self.assertEqual(normalize_continuous_decoding_cache_lookup_key(0.1, None), (0.1, 0.1))
        self.assertEqual(normalize_continuous_decoding_cache_lookup_key(0.1, 0.02), (0.1, 0.02))
        self.assertEqual(normalize_continuous_decoding_cache_lookup_key((0.1, 0.02)), (0.1, 0.02))


class TestDecodeSpecificEpochsSlideby(unittest.TestCase):


    def test_decode_specific_epochs_accepts_and_forwards_slideby(self):
        class SlidebyForwardingDecoder:


            def pre_build_epochs_decoding_result(self, **kwargs):
                self.pre_build_kwargs = kwargs
                return kwargs


            def perform_pre_built_specific_epochs_decoding(self, **kwargs):
                self.perform_kwargs = kwargs
                return kwargs['filter_epochs_decoder_result']

        decoder = SlidebyForwardingDecoder()
        result = BayesianPlacemapPositionDecoder.decode_specific_epochs(decoder, spikes_df=pd.DataFrame(), filter_epochs=pd.DataFrame(), decoding_time_bin_size=0.1, slideby=0.05, debug_print=False)

        self.assertEqual(result['slideby'], 0.05)
        self.assertEqual(decoder.pre_build_kwargs['slideby'], 0.05)


class TestEpochsSpkcountMethods(unittest.TestCase):

    def setUp(self):
        self.enable_debug_printing = False
        self.test_spikes_df = pd.DataFrame({
            'aclu': [1, 1, 2, 2, 3],
            't_rel_seconds': [0.1, 0.2, 0.15, 0.25, 0.3],
            'neuron_type': ['pyramidal'] * 5,
        })
        self.test_epochs_df = pd.DataFrame({
            'start': [0.0, 0.5],
            'stop': [0.4, 0.8]
        })

    def test_single_time_bin_per_epoch(self):
        spkcount, _, nbins, time_bins = epochs_spkcount(
            self.test_spikes_df, 
            self.test_epochs_df,
            bin_size=None,
            export_time_bins=True,
            use_single_time_bin_per_epoch=True
        )
        self.assertEqual(len(nbins), 2)
        self.assertTrue(all(nbins == 1))
        self.assertEqual(len(time_bins), 2)

    def test_short_epoch_handling(self):
        short_epochs_df = pd.DataFrame({
            'start': [0.0, 0.5],
            'stop': [0.005, 0.505]
        })
        spkcount, _, nbins, time_bins = epochs_spkcount(
            self.test_spikes_df,
            short_epochs_df,
            bin_size=0.01,
            export_time_bins=True
        )
        self.assertEqual(len(nbins), 2)
        self.assertTrue(all(nbins == 1))

    def test_included_neuron_ids(self):
        specific_neuron_ids = [1, 2, 3, 4]
        spkcount, _, nbins, _ = epochs_spkcount(
            self.test_spikes_df,
            self.test_epochs_df,
            included_neuron_ids=specific_neuron_ids,
            bin_size=0.1
        )
        self.assertEqual(len(spkcount[0]), len(specific_neuron_ids))

    def test_variable_slideby(self):
        spkcount, _, nbins, _ = epochs_spkcount(
            self.test_spikes_df,
            self.test_epochs_df,
            bin_size=0.1,
            slideby=0.05
        )
        self.assertTrue(all(n > 0 for n in nbins))


    def test_decode_specific_epochs_shell_stores_slideby(self):
        test_spikes_df = self.test_spikes_df.assign(t=self.test_spikes_df['t_rel_seconds'])
        result_shell = BayesianPlacemapPositionDecoder._build_decode_specific_epochs_result_shell(neuron_IDs=np.array([1, 2, 3]), spikes_df=test_spikes_df, filter_epochs=self.test_epochs_df, decoding_time_bin_size=0.1, slideby=0.05, debug_print=False)
        decoded_result = DecodedFilterEpochsResult(**result_shell.to_dict())

        self.assertEqual(decoded_result.slideby, 0.05)


    def test_empty_epochs(self):
        empty_epochs_df = pd.DataFrame({
            'start': [],
            'stop': []
        })
        spkcount, _, nbins, time_bins = epochs_spkcount(
            self.test_spikes_df,
            empty_epochs_df,
            bin_size=0.1,
            export_time_bins=True
        )
        self.assertEqual(len(spkcount), 0)
        self.assertEqual(len(nbins), 0)
        self.assertEqual(len(time_bins), 0)


class TestBayesianZeroSumPosteriorAndMapSentinels(unittest.TestCase):
    """Guards against zero-sum / all-NaN posteriors snapping to xbin_centers[0]."""


    def test_minimum_rate_eps_recovers_peak_despite_out_of_field_zero_ratemap(self):
        """A co-active in-field cell should keep the true peak when another cell fires at an exact-0.0 F bin."""
        n_pos, n_t, n_cells = 5, 1, 2
        tau = 0.025
        P_x = np.ones((n_pos, 1), dtype=float) / float(n_pos)
        F = np.zeros((n_pos, n_cells), dtype=float) # (n_pos, n_cells) — transpose happens inside neuropy_bayesian_prob
        F[3, 0] = 10.0 # cell 0 place field at bin 3
        # cell 1 has exact 0.0 everywhere except a far bin (bin 0 stays 0.0)
        F[1, 1] = 5.0
        n = np.zeros((n_cells, n_t), dtype=float)
        n[0, 0] = 2.0 # in-field spikes for cell 0
        n[1, 0] = 1.0 # out-of-field spike for cell 1 at a bin where F[:,1]==0 at true peak (bin 3)

        posterior_no_floor = ZhangReconstructionImplementation.neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=True, minimum_rate_eps=0.0)
        self.assertTrue(np.all(~np.isfinite(posterior_no_floor[:, 0])) or np.nansum(posterior_no_floor[:, 0]) == 0.0)

        posterior_flat = ZhangReconstructionImplementation.neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=True, minimum_rate_eps=1e-12)
        posterior_full = ZhangReconstructionImplementation.neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=False, minimum_rate_eps=1e-12)
        self.assertTrue(np.all(np.isfinite(posterior_flat[:, 0])))
        self.assertGreater(np.nansum(posterior_flat[:, 0]), 0.0)
        np.testing.assert_allclose(posterior_flat, posterior_full, rtol=1e-9, atol=1e-12)
        self.assertEqual(int(np.argmax(posterior_flat[:, 0])), 3)


    def test_all_nan_and_zero_sum_columns_map_to_neg1_and_nan_positions(self):
        xbin_centers = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        flat_p = np.full((4, 3), np.nan, dtype=float)
        flat_p[:, 1] = np.array([0.1, 0.5, 0.3, 0.1], dtype=float) # valid middle column
        flat_idxs, idxs = BasePositionDecoder.perform_compute_most_likely_positions(flat_p, original_position_data_shape=(4,))
        self.assertEqual(int(flat_idxs[0]), -1)
        self.assertEqual(int(flat_idxs[1]), 1)
        self.assertEqual(int(flat_idxs[2]), -1)
        self.assertTrue(np.all(idxs[:, 0] == -1))
        self.assertEqual(int(idxs[0, 1]), 1)
        self.assertTrue(np.all(idxs[:, 2] == -1))

        positions = BasePositionDecoder.perform_lookup_most_likely_positions(idxs, xbin_centers=xbin_centers, ybin_centers=None, ndim=1)
        self.assertTrue(np.isnan(positions[0]))
        self.assertAlmostEqual(float(positions[1]), 20.0)
        self.assertTrue(np.isnan(positions[2]))
        self.assertFalse(np.isclose(float(np.nan_to_num(positions[0], nan=-999.0)), float(xbin_centers[0])))
        self.assertFalse(np.isclose(float(np.nan_to_num(positions[0], nan=-999.0)), float(xbin_centers[-1])))


    def test_single_time_window_lookup_returns_scalar_nan_without_indexerror(self):
        xbin_centers = np.array([1.0, 2.0, 3.0], dtype=float)
        flat_p = np.full((3, 1), np.nan, dtype=float)
        flat_idxs, idxs = BasePositionDecoder.perform_compute_most_likely_positions(flat_p, original_position_data_shape=(3,))
        self.assertEqual(int(flat_idxs[0]), -1)
        positions = BasePositionDecoder.perform_lookup_most_likely_positions(idxs, xbin_centers=xbin_centers, ybin_centers=None, ndim=1)
        self.assertTrue(np.isscalar(positions) or (np.ndim(positions) == 0))
        self.assertTrue(np.isnan(positions))


    def test_2d_lookup_masks_neg1_without_wrapping_to_last_bin(self):
        xbin_centers = np.array([1.0, 2.0, 3.0], dtype=float)
        ybin_centers = np.array([10.0, 20.0], dtype=float)
        idxs = np.array([[-1, 1], [-1, 0]], dtype=int) # (2, n_t)
        positions = BasePositionDecoder.perform_lookup_most_likely_positions(idxs, xbin_centers=xbin_centers, ybin_centers=ybin_centers, ndim=2)
        self.assertEqual(np.shape(positions), (2, 2))
        self.assertTrue(np.all(np.isnan(positions[0, :])))
        self.assertAlmostEqual(float(positions[1, 0]), 2.0)
        self.assertAlmostEqual(float(positions[1, 1]), 10.0)
        

if __name__ == '__main__':

    unittest.main()