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


class TestOverhangingEpochBinSpikeMaskClipping(unittest.TestCase):
    """Regression for Bug 2A: overhanging final spanning bins past epoch.stop must not stay active from post-epoch spikes.

    Mirrors the clip-then-mask sequence added in ``_do_custom_decode_epochs`` without importing the heavy DirectionalPlacefield module (Qt deps).
    """


    def _build_overhang_decode_result_and_spikes(self):
        """Epoch [0.0, 0.10] + bin_size 0.075 -> edges [0.0, 0.075, 0.15]; spike @0.12 is past epoch.stop but inside overhang bin 1."""
        from neuropy.core.epoch import ensure_dataframe
        from neuropy.utils.mixins.time_slicing import add_epochs_id_identity

        neuron_ids = np.array([1], dtype=int)
        decoding_time_bin_size = 0.075
        epochs_to_decode_df = pd.DataFrame({'start': [0.0], 'stop': [0.10], 'label': [0], 'lap_id': [1]})
        global_spikes_df = pd.DataFrame({
            'aclu': [1, 1],
            't_rel_seconds': [0.03, 0.12], # 0.03 in bin0; 0.12 past epoch.stop=0.10 but inside overhang [0.075, 0.15]
        })
        shell = BasePositionDecoder._build_decode_specific_epochs_result_shell(neuron_IDs=neuron_ids, spikes_df=deepcopy(global_spikes_df), filter_epochs=deepcopy(epochs_to_decode_df), decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
        decoder_result = DecodedFilterEpochsResult(**shell.to_dict())
        for i in range(decoder_result.num_filter_epochs):
            n_bins = int(decoder_result.nbins[i])
            counts = np.asarray(decoder_result.spkcount[i])
            total_per_bin = np.asarray(counts.sum(axis=0) if counts.ndim == 2 else counts, dtype=float).reshape(-1)
            positions = np.where(total_per_bin > 0, 40.0, 200.0).astype(float) # silence-prior-like jump for zero-spike overhang bins
            decoder_result.most_likely_positions_list.append(positions)
            decoder_result.most_likely_position_indicies_list.append(np.zeros((n_bins,), dtype=int))
            decoder_result.p_x_given_n_list.append(np.ones((5, n_bins), dtype=float) / 5.0)
            decoder_result.marginal_x_list.append({'p_x_given_n': np.ones((5, n_bins), dtype=float) / 5.0, 'most_likely_positions_1D': positions})
            decoder_result.marginal_y_list.append(None)
            # Mirror `_perform_decoding_specific_epochs`: populate time_bin_edges from the container (shell leaves this empty).
            _tc = decoder_result.time_bin_containers[i]
            if _tc.window_start_edges is not None:
                decoder_result.time_bin_edges.append(np.column_stack((np.asarray(_tc.window_start_edges), np.asarray(_tc.window_stop_edges))))
            else:
                decoder_result.time_bin_edges.append(np.atleast_1d(_tc.edges))
        ## END for i in range(decoder_result.num_filter_epochs)...

        return decoder_result, global_spikes_df, epochs_to_decode_df, neuron_ids, ensure_dataframe, add_epochs_id_identity


    def test_unclipped_spikes_would_keep_overhang_bin_active(self):
        """Control: without epoch clipping, the post-epoch spike @0.12 keeps overhang bin 1 active (the Bug 2A failure mode)."""
        decoder_result, global_spikes_df, epochs_to_decode_df, neuron_ids, _ensure_dataframe, _add_epochs_id_identity = self._build_overhang_decode_result_and_spikes()
        self.assertEqual(int(decoder_result.nbins[0]), 2)
        edges = np.asarray(decoder_result.time_bin_containers[0].edges, dtype=float)
        self.assertGreater(float(edges[-1]), float(epochs_to_decode_df['stop'].iloc[0]))
        # Decode used clipped spikes, so overhang bin has 0 counts / silence-prior MAP:
        self.assertEqual(float(np.asarray(decoder_result.spkcount[0]).sum(axis=0)[1]), 0.0)
        self.assertAlmostEqual(float(decoder_result.most_likely_positions_list[0][1]), 200.0)
        _masked, mask_index_tuple = decoder_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(global_spikes_df), min_num_spikes_per_bin_to_be_considered_active=1, min_num_unique_active_neurons_per_time_bin=1, masked_bin_fill_mode='nan_filled', included_neuron_ids=neuron_ids)
        is_time_bin_active_list, *_rest = mask_index_tuple
        self.assertTrue(bool(is_time_bin_active_list[0][0]))
        self.assertTrue(bool(is_time_bin_active_list[0][1]), 'unclipped post-epoch spike must keep overhang bin active (pre-fix control)')


    def test_clipped_spikes_mask_overhang_bin_with_only_post_epoch_spikes(self):
        """Same clip-then-mask sequence as ``_do_custom_decode_epochs``: overhang bin 1 becomes inactive and NaN-filled."""
        decoder_result, global_spikes_df, _epochs_to_decode_df, neuron_ids, ensure_dataframe, add_epochs_id_identity = self._build_overhang_decode_result_and_spikes()
        filter_epochs_df: pd.DataFrame = ensure_dataframe(decoder_result.filter_epochs).reset_index(drop=True)
        clipped_global_spikes_df: pd.DataFrame = deepcopy(global_spikes_df)
        if (len(clipped_global_spikes_df) > 0) and (len(filter_epochs_df) > 0):
            clipped_global_spikes_df = add_epochs_id_identity(clipped_global_spikes_df, filter_epochs_df, epoch_id_key_name='_custom_decode_temp_epoch_id', epoch_label_column_name=None, no_interval_fill_value=-1)
            clipped_global_spikes_df = clipped_global_spikes_df[clipped_global_spikes_df['_custom_decode_temp_epoch_id'] != -1].drop(columns=['_custom_decode_temp_epoch_id'])
        elif len(filter_epochs_df) == 0:
            clipped_global_spikes_df = clipped_global_spikes_df.iloc[0:0].copy()
        masked_result, mask_index_tuple = decoder_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=clipped_global_spikes_df, min_num_spikes_per_bin_to_be_considered_active=1, min_num_unique_active_neurons_per_time_bin=1, masked_bin_fill_mode='nan_filled', included_neuron_ids=neuron_ids)
        is_time_bin_active_list, *_rest = mask_index_tuple
        self.assertTrue(bool(is_time_bin_active_list[0][0]))
        self.assertFalse(bool(is_time_bin_active_list[0][1]), 'overhang bin with only post-epoch spikes must be masked inactive after epoch clipping')
        self.assertTrue(np.isnan(float(masked_result.most_likely_positions_list[0][1])))


class TestGetByIdSlicesPfSpikesDfForEpochReplacement(unittest.TestCase):
    """Verify get_by_id keeps pf.spikes_df untouched while replacing_computation_epochs supports preserve_source_neuron_IDs=(False|True)."""

    def _build_synthetic_1d_pf(self):
        from neuropy.core.position import Position
        from neuropy.core.epoch import Epoch
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction_dst import BayesianPlacemapPositionDecoderDST

        dt = 0.025
        t_lap1 = np.arange(0.0, 4.0, dt)
        x_lap1 = np.linspace(20.0, 240.0, len(t_lap1))
        t_lap2 = np.arange(5.0, 9.0, dt)
        x_lap2 = np.linspace(20.0, 240.0, len(t_lap2))
        t = np.concatenate([t_lap1, t_lap2])
        x = np.concatenate([x_lap1, x_lap2])
        pos_df = pd.DataFrame({'t': t, 'x': x, 'speed': np.full_like(t, 55.0)})
        pos_1d = Position(pos_df)

        full_epochs = Epoch(pd.DataFrame({'start': [0.0, 5.0], 'stop': [3.975, 8.975], 'label': ['lap1', 'lap2']}))
        train_epochs = Epoch(pd.DataFrame({'start': [0.5, 5.0], 'stop': [3.975, 8.975], 'label': ['lap1_tr', 'lap2_tr']}))
        test_epochs = pd.DataFrame({'start': [0.0], 'stop': [0.45], 'label': ['lap1_te'], 'lap_id': [1]})

        # aclu 1: curated local field near x ~ 35-50 cm (t ~ 0.2-0.55 and 5.2-5.55)
        t_aclu1 = np.concatenate([np.linspace(0.20, 0.50, 8), np.linspace(5.20, 5.55, 10)])
        # aclu 2: curated mid-track field near x ~ 130 cm (t ~ 2.0 and 7.0)
        t_aclu2 = np.concatenate([np.linspace(1.90, 2.20, 8), np.linspace(6.90, 7.20, 10)])
        # aclu 99: excluded high-rate noisy unit firing heavily across x ~ 20-160 cm (silent only near 235 cm)
        t_aclu99 = np.concatenate([np.linspace(0.05, 2.50, 90), np.linspace(5.05, 7.50, 90)])

        t_all = np.concatenate([t_aclu1, t_aclu2, t_aclu99])
        spikes_df = pd.DataFrame({
            't_rel_seconds': t_all,
            'x': np.interp(t_all, t, x),
            'aclu': np.concatenate([np.full(len(t_aclu1), 1, dtype=int), np.full(len(t_aclu2), 2, dtype=int), np.full(len(t_aclu99), 99, dtype=int)]),
            'qclu': np.concatenate([np.full(len(t_aclu1), 1, dtype=int), np.full(len(t_aclu2), 2, dtype=int), np.full(len(t_aclu99), 5, dtype=int)]),
            'neuron_type': 'PYRAMIDAL',
            'shank': 0,
            'cluster': 0,
        }).sort_values('t_rel_seconds').reset_index(drop=True)
        spikes_df['flat_spike_idx'] = np.arange(len(spikes_df), dtype=int)

        pf1d = PfND.from_config_values(spikes_df=deepcopy(spikes_df), position=pos_1d, epochs=full_epochs, frate_thresh=0.1, speed_thresh=1.0, grid_bin=(10.0, 10.0), grid_bin_bounds=((10.0, 250.0), (0.0, 10.0)), smooth=(1.0, 1.0))
        return pf1d, spikes_df, train_epochs, test_epochs, BayesianPlacemapPositionDecoderDST


    def test_pfnd_replacing_computation_epochs_preserve_vs_refilter_modes(self):
        from neuropy.core.epoch import Epoch

        pf1d, spikes_df, train_epochs, _test_epochs, _DSTCls = self._build_synthetic_1d_pf()
        self.assertIn(99, pf1d.ratemap.neuron_ids)
        sliced_pf = pf1d.get_by_id(np.array([1, 2], dtype=int))
        # pf.spikes_df remains untouched on get_by_id
        self.assertEqual(list(np.unique(sliced_pf.spikes_df['aclu'])), [1, 2, 99])
        self.assertEqual(list(sliced_pf.ratemap.neuron_ids), [1, 2])

        # Default (preserve_source_neuron_IDs=False): refilters from pf.spikes_df using pf.config.frate_thresh on new epochs
        refiltered_pf = sliced_pf.replacing_computation_epochs(train_epochs, preserve_source_neuron_IDs=False)
        self.assertEqual(list(refiltered_pf.ratemap.neuron_ids), [1, 2, 99])

        # preserve_source_neuron_IDs=True: preserves exact source neuron_IDs without mutating pf.spikes_df or pf.config.frate_thresh
        preserved_pf = sliced_pf.replacing_computation_epochs(train_epochs, preserve_source_neuron_IDs=True)
        self.assertEqual(list(np.unique(preserved_pf.spikes_df['aclu'])), [1, 2, 99])
        self.assertAlmostEqual(float(preserved_pf.config.frate_thresh), 0.1)
        self.assertNotIn(99, list(preserved_pf.ratemap.neuron_ids))
        self.assertEqual(list(preserved_pf.ratemap.neuron_ids), [1, 2])

        # Disjoint epoch test: unit 77 fires ONLY in [4.1, 4.9] (outside full_epochs -> excluded initially, recovered when preserve_source_neuron_IDs=False)
        pos_extra_t = np.arange(0.0, 9.0, 0.025)
        pos_extra_x = np.linspace(20.0, 240.0, len(pos_extra_t))
        from neuropy.core.position import Position
        pos_full = Position(pd.DataFrame({'t': pos_extra_t, 'x': pos_extra_x, 'speed': np.full_like(pos_extra_t, 55.0)}))
        t_aclu77 = np.linspace(4.20, 4.70, 25)
        extra_rows = pd.DataFrame({'t_rel_seconds': t_aclu77, 'x': np.interp(t_aclu77, pos_extra_t, pos_extra_x), 'aclu': 77, 'qclu': 1, 'neuron_type': 'PYRAMIDAL', 'shank': 0, 'cluster': 0, 'flat_spike_idx': np.arange(len(spikes_df), len(spikes_df) + len(t_aclu77))})
        aug_spikes = pd.concat([spikes_df, extra_rows], ignore_index=True).sort_values('t_rel_seconds').reset_index(drop=True)
        pf_initial = PfND.from_config_values(spikes_df=aug_spikes, position=pos_full, epochs=pf1d.epochs, frate_thresh=1.0, speed_thresh=1.0, grid_bin=(10.0, 10.0), grid_bin_bounds=((10.0, 250.0), (0.0, 10.0)), smooth=(1.0, 1.0))
        self.assertNotIn(77, list(pf_initial.ratemap.neuron_ids))
        disjoint_epoch = Epoch(pd.DataFrame({'start': [4.05], 'stop': [4.85], 'label': ['inter']}))
        pf_disjoint_refiltered = pf_initial.replacing_computation_epochs(disjoint_epoch, preserve_source_neuron_IDs=False)
        self.assertEqual(list(pf_disjoint_refiltered.ratemap.neuron_ids), [77])
        # Conversely, preserve_source_neuron_IDs=True on disjoint_epoch keeps exact source IDs [1, 2, 99] even with 0 spikes in [4.05, 4.85]
        pf_disjoint_preserved = pf_initial.replacing_computation_epochs(disjoint_epoch, preserve_source_neuron_IDs=True)
        self.assertEqual(list(pf_disjoint_preserved.ratemap.neuron_ids), list(pf_initial.ratemap.neuron_ids))
        self.assertIsInstance(pf_disjoint_preserved.included_neuron_IDs, np.ndarray)
        self.assertEqual(list(pf_disjoint_preserved.included_neuron_IDs), list(pf_initial.ratemap.neuron_ids))
        self.assertTrue(np.allclose(pf_disjoint_preserved.ratemap.tuning_curves, 0.0))


    def test_base_and_stateful_decoders_preserve_exact_order_and_zero_spike_units(self):
        from neuropy.core.epoch import Epoch

        pf1d, spikes_df, train_epochs, test_epochs, BayesianPlacemapPositionDecoderDST = self._build_synthetic_1d_pf()
        curated_ids = np.array([2, 1], dtype=int)  # intentionally non-ascending order

        base_dec = BasePositionDecoder(pf=deepcopy(pf1d), setup_on_init=True)
        sliced_base = base_dec.get_by_id(curated_ids, recompute=False)
        self.assertEqual(list(np.unique(sliced_base.pf.spikes_df['aclu'])), [1, 2, 99])
        self.assertEqual(list(sliced_base.neuron_IDs), [2, 1])

        train_dec = sliced_base.replacing_computation_epochs(train_epochs, preserve_source_neuron_IDs=True)
        self.assertEqual(list(train_dec.neuron_IDs), [2, 1])
        self.assertIsInstance(train_dec.pf.included_neuron_IDs, np.ndarray)
        self.assertEqual(list(train_dec.pf.included_neuron_IDs), [2, 1])
        self.assertEqual(train_dec.F.shape[1], 2)

        # Decode test window [0.0, 0.45] using template-only global_spikes_df (aclu 99 stripped, as in get_proper_global_spikes_df)
        template_only_spikes_df = spikes_df[np.isin(spikes_df['aclu'].to_numpy(), curated_ids)].copy()
        res = train_dec.decode_specific_epochs(spikes_df=template_only_spikes_df, filter_epochs=test_epochs, decoding_time_bin_size=0.15)
        # Bin [0.15, 0.30] contains aclu 1 spikes near true x ~ 35-45 cm -> MAP must stay local (< 80 cm), not jump to ~235 cm
        decoded_pos_bin1 = float(res.most_likely_positions_list[0][1])
        self.assertLess(decoded_pos_bin1, 80.0)

        # Zero-spike subset window [2.8, 3.9] where neither aclu 1 nor aclu 2 fires: preserve_source_neuron_IDs=True still keeps [2, 1]
        silent_sub_epoch = Epoch(pd.DataFrame({'start': [2.80], 'stop': [3.90], 'label': ['silent_sub']}))
        silent_train_dec = sliced_base.replacing_computation_epochs(silent_sub_epoch, preserve_source_neuron_IDs=True)
        self.assertEqual(list(silent_train_dec.neuron_IDs), [2, 1])
        self.assertIsInstance(silent_train_dec.pf.included_neuron_IDs, np.ndarray)
        self.assertEqual(list(silent_train_dec.pf.included_neuron_IDs), [2, 1])
        self.assertEqual(silent_train_dec.F.shape[1], 2)

        stateful_dec = BayesianPlacemapPositionDecoder(time_bin_size=0.075, pf=deepcopy(pf1d), spikes_df=deepcopy(pf1d.filtered_spikes_df), setup_on_init=True)
        sliced_stateful = stateful_dec.get_by_id(curated_ids, recompute=False)
        self.assertEqual(list(np.unique(sliced_stateful.pf.spikes_df['aclu'])), [1, 2, 99])
        self.assertEqual(list(sliced_stateful.pf.replacing_computation_epochs(train_epochs, preserve_source_neuron_IDs=True).ratemap.neuron_ids), [2, 1])

        dst_dec = BayesianPlacemapPositionDecoderDST(time_bin_size=0.075, pf=deepcopy(pf1d), spikes_df=deepcopy(pf1d.filtered_spikes_df), setup_on_init=True)
        sliced_dst = dst_dec.get_by_id(curated_ids, recompute=False)
        self.assertEqual(list(np.unique(sliced_dst.pf.spikes_df['aclu'])), [1, 2, 99])
        self.assertEqual(list(sliced_dst.pf.replacing_computation_epochs(train_epochs, preserve_source_neuron_IDs=True).ratemap.neuron_ids), [2, 1])


    def test_train_test_split_result_safe_slice_and_stale_pickle_detection(self):
        from types import SimpleNamespace
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult, TrainTestLapsSplitting

        pf1d, _spikes_df, train_epochs, test_epochs, _DSTCls = self._build_synthetic_1d_pf()
        leaked_dec = BasePositionDecoder(pf=deepcopy(pf1d), setup_on_init=True)  # neuron_IDs = [1, 2, 99]
        curated_dec = leaked_dec.get_by_id(np.array([1, 2], dtype=int), recompute=False)  # neuron_IDs = [1, 2]

        split_res_curated = TrainTestSplitResult(is_global=True, training_data_portion=0.9, test_data_portion=0.1, laps_decoding_time_bin_size=0.15, should_filter_inactive_bins_before_split=True, min_num_spikes_per_bin_to_be_considered_active=1, min_num_unique_active_neurons_per_time_bin=1, train_test_full_lap=False, test_epochs_dict={'long_LR': test_epochs}, train_epochs_dict={'long_LR': train_epochs.to_dataframe()}, train_lap_specific_pf1D_Decoder_dict={'long_LR': deepcopy(curated_dec)})

        # Safe intersection: requesting [2, 777, 1] where 777 dropped out (0 spikes on 90% split) must not raise AssertionError and must preserve requested order [2, 1]
        sliced_split = split_res_curated.sliced_by_neuron_id(np.array([2, 777, 1], dtype=int))
        self.assertEqual(list(sliced_split.train_lap_specific_pf1D_Decoder_dict['long_LR'].neuron_IDs), [2, 1])

        # Stale-pickle detection: if stored TrainTestSplit decoder contains leaked unit 99 outside TrackTemplates [1, 2], resolve_needs_train_test_split_recompute must return True
        fake_templates = SimpleNamespace(get_decoders_dict=lambda: {'long_LR': curated_dec})
        fake_dir_laps = SimpleNamespace(get_templates=lambda **kwargs: fake_templates)
        fake_rank_order = SimpleNamespace(minimum_inclusion_fr_Hz=2.0, included_qclu_values=[1, 2])
        requested_cfg = {'laps_decoding_time_bin_size': 0.15, 'should_filter_inactive_bins_before_split': True, 'min_num_spikes_per_bin_to_be_considered_active': 1, 'min_num_unique_active_neurons_per_time_bin': 1, 'train_test_full_lap': False}

        clean_pipeline = SimpleNamespace(global_computation_results=SimpleNamespace(computed_data={'TrainTestSplit': split_res_curated, 'DirectionalLaps': fake_dir_laps, 'RankOrder': fake_rank_order}))
        needs_recomp_clean, _ = TrainTestLapsSplitting.resolve_needs_train_test_split_recompute(clean_pipeline, requested_cfg)
        self.assertFalse(needs_recomp_clean)

        stale_split = deepcopy(split_res_curated)
        stale_split.train_lap_specific_pf1D_Decoder_dict['long_LR'] = leaked_dec
        stale_pipeline = SimpleNamespace(global_computation_results=SimpleNamespace(computed_data={'TrainTestSplit': stale_split, 'DirectionalLaps': fake_dir_laps, 'RankOrder': fake_rank_order}))
        needs_recomp_stale, _ = TrainTestLapsSplitting.resolve_needs_train_test_split_recompute(stale_pipeline, requested_cfg)
        self.assertTrue(needs_recomp_stale)


if __name__ == '__main__':

    unittest.main()