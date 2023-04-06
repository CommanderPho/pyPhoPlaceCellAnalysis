from copy import deepcopy
from pathlib import Path
from attrs import define, field, Factory # for BasePositionDecoder
# import pathlib

import numpy as np
import pandas as pd
# from scipy.stats import multivariate_normal
from scipy.special import factorial, logsumexp

# import neuropy
from neuropy.utils.dynamic_container import DynamicContainer # for decode_specific_epochs
from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # for decode_specific_epochs
from neuropy.utils.mixins.unit_slicing import NeuronUnitSlicableObjectProtocol # allows placefields to be sliced by neuron ids
from neuropy.analyses.decoders import epochs_spkcount # for decode_specific_epochs
from neuropy.utils.mixins.binning_helpers import BinningContainer # for epochs_spkcount getting the correct time bins
from neuropy.analyses.placefields import PfND # for BasePositionDecoder


from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.indexing_helpers import BinningInfo, compute_spanning_bins, build_spanning_grid_matrix, np_ffill_1D # for compute_corrected_positions(...)
from pyphocorehelpers.print_helpers import WrappingMessagePrinter, SimplePrintable, safe_get_variable_shape
from pyphocorehelpers.mixins.serialized import SerializedAttributesSpecifyingClass

from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results # for finding common neurons in `prune_to_shared_aclus_only`

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import ZhangReconstructionImplementation # for BasePositionDecoder

# ==================================================================================================================== #
# Stateless Decoders (New 2023-04-06)                                                                                  #
# ==================================================================================================================== #

@define(slots=False)
class BasePositionDecoder(NeuronUnitSlicableObjectProtocol):
    """ 2023-04-06 - A simplified data-only version of the decoder that serves to remove all state related to specific computations to make each run independent 
    Stores only the raw inputs that are used to decode, with the user specifying the specifics for a given decoding (like time_time_sizes, etc.

    """
    pf: PfND

    neuron_IDXs: np.ndarray
    neuron_IDs: np.ndarray
    F: np.ndarray
    P_x: np.ndarray

    setup_on_init:bool = True 
    post_load_on_init:bool = False
    debug_print: bool = False
    
    # # Time Binning:
    # time_bin_size: float
    # time_binning_container: BinningContainer
    # unit_specific_time_binned_spike_counts: np.ndarray
    # total_spike_counts_per_window: np.ndarray

    # # Computed Results:
    # flat_p_x_given_n: np.ndarray
    # p_x_given_n: np.ndarray
    # most_likely_position_flat_indicies: np.ndarray
    # most_likely_position_indicies: type
    # marginal: DynamicContainer
    # most_likely_positions: np.ndarray
    # revised_most_likely_positions: np.ndarray


    # Properties _________________________________________________________________________________________________________ #

    # placefield properties:
    @property
    def ratemap(self):
        return self.pf.ratemap

    @property
    def ndim(self):
        return int(self.pf.ndim)

    @property
    def num_neurons(self):
        """The num_neurons property."""
        return self.ratemap.n_neurons # np.shape(self.neuron_IDs) # or self.ratemap.n_neurons

    # ratemap properties (xbin & ybin)  
    @property
    def xbin(self):
        return self.ratemap.xbin
    @property
    def ybin(self):
        return self.ratemap.ybin
    @property
    def xbin_centers(self):
        return self.ratemap.xbin_centers
    @property
    def ybin_centers(self):
        return self.ratemap.ybin_centers

    @property
    def original_position_data_shape(self):
        """The original_position_data_shape property."""
        return np.shape(self.pf.occupancy)    

    @property
    def flat_position_size(self):
        """The flat_position_size property."""
        return np.shape(self.F)[0] # like 288

    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        # Perform the primary setup to build the placefield
        if self.setup_on_init:
            self.setup()
            if self.post_load_on_init:
                self.post_load()
        else:
            assert (not self.post_load_on_init), f"post_load_on_init can't be true if setup_on_init isn't true!"

    def setup(self):
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        
        self._setup_concatenated_F()

    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        with WrappingMessagePrinter(f'post_load() called.', begin_line_ending='... ', finished_message='all rebuilding completed.', enable_print=self.debug_print):
            self._setup_concatenated_F()
            # self._setup_time_bin_spike_counts_N_i()
            # self._setup_time_window_centers()
            # self.p_x_given_n = self._reshape_output(self.flat_p_x_given_n)
            # self.compute_most_likely_positions()



    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids): # defer_compute_all:bool = False
        """Implementors return a copy of themselves with neuron_ids equal to ids
            Needs to update: neuron_sliced_decoder.pf, ... (much more)

        defer_compute_all: bool - should be set to False if you want to manually decode using custom epochs or something later. Otherwise it will compute for all spikes automatically.
        """
        # call .get_by_id(ids) on the placefield (pf):
        neuron_sliced_pf = self.pf.get_by_id(ids)
        ## apply the neuron_sliced_pf to the decoder:
        neuron_sliced_decoder = BasePositionDecoder(neuron_sliced_pf, setup_on_init=self.setup_on_init, post_load_on_init=self.post_load_on_init, debug_print=self.debug_print)
        return neuron_sliced_decoder

    def conform_to_position_bins(self, target_one_step_decoder, force_recompute=True):
        """ After the underlying placefield (self.pf)'s position bins are changed by calling pf.conform_to_position_bins(...) externally, the computations for the decoder will be messed up (and out of sync).
            Calling this function detects this issue.
            # 2022-12-09 - We want to be able to have both long/short track placefields have the same spatial bins.
            Usage:
                long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D)

            Usage 1D:
                long_pf1D = long_results.pf1D
                short_pf1D = short_results.pf1D
                
                long_one_step_decoder_1D, short_one_step_decoder_1D  = [results_data.get('pf1D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_1D.conform_to_position_bins(long_one_step_decoder_1D)


            Usage 2D:
                long_pf2D = long_results.pf2D
                short_pf2D = short_results.pf2D
                long_one_step_decoder_2D, short_one_step_decoder_2D  = [results_data.get('pf2D_Decoder', None) for results_data in (long_results, short_results)]
                short_one_step_decoder_2D.conform_to_position_bins(long_one_step_decoder_2D)

        """
        did_recompute = False
        # Update the internal placefield's position bins if they're not the same as the target_one_step_decoder's:
        self.pf, did_update_internal_pf_bins = self.pf.conform_to_position_bins(target_one_step_decoder.pf)
        
        # Update the one_step_decoders after the short bins have been updated:
        if (force_recompute or did_update_internal_pf_bins): # or (self.p_x_given_n.shape[0] < target_one_step_decoder.p_x_given_n.shape[0])
            # Compute:
            print(f'self will be re-binned to match target_one_step_decoder...')
            self.setup()
            # self.compute_all()
            did_recompute = True # set the update flag
        else:
            # No changes needed:
            did_recompute = False

        return self, did_recompute


	# ==================================================================================================================== #
    # Non-Modifying Methods:                                                                                               #
    # ==================================================================================================================== #
    @function_attributes(short_name='decode', tags=['decode', 'pure'], input_requires=[], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['BayesianPlacemapPositionDecoder.perform_compute_most_likely_positions', 'ZhangReconstructionImplementation.neuropy_bayesian_prob'],
        used_by=['BayesianPlacemapPositionDecoder.perform_decode_specific_epochs'])
    def decode(self, unit_specific_time_binned_spike_counts, time_bin_size, output_flat_versions=False, debug_print=True):
        """ decodes the neural activity from its internal placefields, returning its posterior and the predicted position 
        Does not alter the internal state of the decoder (doesn't change internal most_likely_positions or posterior, etc)
        
        flat_outputs_container is returned IFF output_flat_versions is True
        
        Inputs:
            unit_specific_time_binned_spike_counts: np.array of shape (num_cells, num_time_bins) - e.g. (69, 20717)

        Requires:
            .P_x
            .F
            .debug_print
            .xbin_centers, .ybin_centers
            .original_position_data_shape
            
        Uses:
            BayesianPlacemapPositionDecoder.perform_compute_most_likely_positions(...)
            ZhangReconstructionImplementation.neuropy_bayesian_prob(...)

        Usages: 
            Used by BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...) to do the actual decoding after building the appropriate spike counts.
        
        """
        num_cells = np.shape(unit_specific_time_binned_spike_counts)[0]    
        num_time_windows = np.shape(unit_specific_time_binned_spike_counts)[1]
        if debug_print:
            print(f'num_cells: {num_cells}, num_time_windows: {num_time_windows}')
        with WrappingMessagePrinter(f'decode(...) called. Computing {num_time_windows} windows for final_p_x_given_n...', begin_line_ending='... ', finished_message='decode completed.', enable_print=(debug_print or self.debug_print)):
            if time_bin_size is None:
                print(f'time_bin_size is None, using internal self.time_bin_size.')
                time_bin_size = self.time_bin_size
            
            # Single sweep decoding:
            curr_flat_p_x_given_n = ZhangReconstructionImplementation.neuropy_bayesian_prob(time_bin_size, self.P_x, self.F, unit_specific_time_binned_spike_counts, debug_print=(debug_print or self.debug_print))
            if debug_print:
                print(f'curr_flat_p_x_given_n.shape: {curr_flat_p_x_given_n.shape}')
            # all computed
            # Reshape the output variables:    
            p_x_given_n = np.reshape(curr_flat_p_x_given_n, (*self.original_position_data_shape, num_time_windows)) # changed for compatibility with 1D decoder
            most_likely_position_flat_indicies, most_likely_position_indicies = self.perform_compute_most_likely_positions(curr_flat_p_x_given_n, self.original_position_data_shape)

            ## Flat properties:
            if output_flat_versions:
                flat_outputs_container = DynamicContainer(flat_p_x_given_n=curr_flat_p_x_given_n, most_likely_position_flat_indicies=most_likely_position_flat_indicies)
            else:
                # No flat versions
                flat_outputs_container = None
                
            if self.ndim > 1:
                most_likely_positions = np.vstack((self.xbin_centers[most_likely_position_indicies[0,:]], self.ybin_centers[most_likely_position_indicies[1,:]])).T # much more efficient than the other implementation. Result is # (85844, 2)
            else:
                # 1D Decoder case:
                most_likely_positions = np.squeeze(self.xbin_centers[most_likely_position_indicies[0,:]])
        
            return most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container
            



    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    @function_attributes(short_name='_setup_concatenated_F', tags=['pr'], input_requires=[], output_provides=[], uses=['ZhangReconstructionImplementation.build_concatenated_F'], used_by=[], creation_date='2023-04-06 13:49')
    def _setup_concatenated_F(self):
        """ Sets up the computation variables F, P_x, neuron_IDs, neuron_IDXs

            maps: (40, 48, 6)
            np.shape(f_i[i]): (48, 6)
            np.shape(F_i[i]): (288, 1)
            np.shape(F): (288, 40)
            np.shape(P_x): (288, 1)
        """
        self.neuron_IDXs, self.neuron_IDs, f_i, F_i, self.F, self.P_x = ZhangReconstructionImplementation.build_concatenated_F(self.pf, debug_print=self.debug_print)

    @function_attributes(short_name='prune_to_shared_aclus_only', tags=['decoder','aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-06 12:19')
    @classmethod
    def prune_to_shared_aclus_only(cls, long_decoder, short_decoder):
        """ determines the neuron_IDs present in both long and short decoders (shared aclus) and returns two copies of long_decoder and short_decoder that only contain the shared_aclus

        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder

            shared_aclus, (long_shared_aclus_only_decoder, short_shared_aclus_only_decoder), long_short_pf_neurons_diff = BayesianPlacemapPositionDecoder.prune_to_shared_aclus_only(long_decoder, short_decoder)
            n_neurons = len(shared_aclus)

        """

        long_neuron_IDs, short_neuron_IDs = long_decoder.neuron_IDs, short_decoder.neuron_IDs
        # long_neuron_IDs, short_neuron_IDs = long_results_obj.original_1D_decoder.neuron_IDs, short_results_obj.original_1D_decoder.neuron_IDs
        long_short_pf_neurons_diff = _compare_computation_results(long_neuron_IDs, short_neuron_IDs)

        ## Get the normalized_tuning_curves only for the shared aclus (that are common across (long/short/global):
        shared_aclus = long_short_pf_neurons_diff.intersection #.shape (56,)
        # n_neurons = len(shared_aclus)
        # long_is_included = np.isin(long_neuron_IDs, shared_aclus)  #.shape # (104, 63)
        # short_is_included = np.isin(short_neuron_IDs, shared_aclus)

        # Restrict the long decoder to only the aclus present on both decoders:
        long_shared_aclus_only_decoder = long_decoder.get_by_id(shared_aclus)
        short_shared_aclus_only_decoder = short_decoder.get_by_id(shared_aclus) # short currently has a subset of long's alcus so this isn't needed, but just for symmetry do it anyway
        return shared_aclus, (long_shared_aclus_only_decoder, short_shared_aclus_only_decoder), long_short_pf_neurons_diff
