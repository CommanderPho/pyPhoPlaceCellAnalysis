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
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder

# ==================================================================================================================== #
# Stateless Decoders (New 2023-04-06)                                                                                  #
# ==================================================================================================================== #

@define(slots=False)
class BasePositionDecoder(NeuronUnitSlicableObjectProtocol):
    """ 2023-04-06 - A simplified data-only version of the decoder that serves to remove all state related to specific computations to make each run independent 
    Stores only the raw inputs that are used to decode, with the user specifying the specifics for a given decoding (like time_time_sizes, etc.


    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_stateless import BasePositionDecoder


    """
    pf: PfND

    neuron_IDXs: np.ndarray = None
    neuron_IDs: np.ndarray = None
    F: np.ndarray = None
    P_x: np.ndarray = None

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

    # ==================================================================================================================== #
    # Initialization                                                                                                       #
    # ==================================================================================================================== #
    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        # Perform the primary setup to build the placefield
        if self.setup_on_init:
            self.setup()
            if self.post_load_on_init:
                self.post_load()
        else:
            assert (not self.post_load_on_init), f"post_load_on_init can't be true if setup_on_init isn't true!"


    @classmethod
    def init_from_stateful_decoder(cls, stateful_decoder: BayesianPlacemapPositionDecoder):
        """ 2023-04-06 - Creates a new instance of this class from a stateful decoder. """
        # Create the new instance:
        new_instance = cls(pf=deepcopy(stateful_decoder.pf), debug_print=stateful_decoder.debug_print)
        # Return the new instance:
        return new_instance



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
    def get_by_id(self, ids, defer_compute_all:bool=False): # defer_compute_all:bool = False
        """Implementors return a copy of themselves with neuron_ids equal to ids
            Needs to update: neuron_sliced_decoder.pf, ... (much more)

        defer_compute_all: bool - should be set to False if you want to manually decode using custom epochs or something later. Otherwise it will compute for all spikes automatically.
            TODO 2023-04-06 - REMOVE this argument. it is unused. It exists just for backwards compatibility with the stateful decoder.
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


    def to_1D_maximum_projection(self, defer_compute_all:bool=True):
        """ returns a copy of the decoder that is 1D 
            defer_compute_all: TODO 2023-04-06 - REMOVE this argument. it is unused. It exists just for backwards compatibility with the stateful decoder.
        """
        # Perform the projection. Can only be ran once.
        new_copy_decoder = deepcopy(self)
        new_copy_decoder.pf = new_copy_decoder.pf.to_1D_maximum_projection() # project the placefields to 1D
        # new_copy_decoder.pf.compute()
        # Test the projection to make sure the dimensionality is correct:
        test_projected_ratemap = new_copy_decoder.pf.ratemap
        assert test_projected_ratemap.ndim == 1, f"projected 1D ratemap must be of dimension 1 but is {test_projected_ratemap.ndim}"
        test_projected_placefields = new_copy_decoder.pf
        assert test_projected_placefields.ndim == 1, f"projected 1D placefields must be of dimension 1 but is {test_projected_placefields.ndim}"
        test_projected_decoder = new_copy_decoder
        assert test_projected_decoder.ndim == 1, f"projected 1D decoder must be of dimension 1 but is {test_projected_decoder.ndim}"
        new_copy_decoder.setup()
        return new_copy_decoder


    # ==================================================================================================================== #
    # Main computation functions:                                                                                          #
    # ==================================================================================================================== #
    @function_attributes(short_name='decode_specific_epochs', tags=['decode'], input_requires=[], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['BayesianPlacemapPositionDecoder.perform_decode_specific_epochs'], used_by=[])
    def decode_specific_epochs(self, spikes_df, filter_epochs, decoding_time_bin_size = 0.05, debug_print=False):
        """ 
        Uses:
            BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...)
        """
        return self.perform_decode_specific_epochs(self, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=debug_print)


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
        if not isinstance(self.neuron_IDs, np.ndarray):
            self.neuron_IDs = np.array(self.neuron_IDs)
        if not isinstance(self.neuron_IDXs, np.ndarray):
            self.neuron_IDXs = np.array(self.neuron_IDXs)


    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #

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

    @function_attributes(short_name='perform_decode_specific_epochs', tags=['decode','specific_epochs','epoch', 'classmethod'], input_requires=[], output_provides=[], uses=['active_decoder.decode', 'add_epochs_id_identity', 'epochs_spkcount', 'cls.perform_build_marginals'], used_by=[''], creation_date='2022-12-04 00:00')
    @classmethod
    def perform_decode_specific_epochs(cls, active_decoder, spikes_df, filter_epochs, decoding_time_bin_size = 0.05, debug_print=False) -> DecodedFilterEpochsResult:
        """Uses the decoder to decode the nerual activity (provided in spikes_df) for each epoch in filter_epochs

        NOTE: Uses active_decoder.decode(...) to actually do the decoding
        
        Args:
            new_2D_decoder (_type_): _description_
            spikes_df (_type_): _description_
            filter_epochs (_type_): _description_
            decoding_time_bin_size (float, optional): _description_. Defaults to 0.05.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            DecodedFilterEpochsResult: _description_
        """
        # build output result object:
        filter_epochs_decoder_result = DynamicContainer(most_likely_positions_list=[], p_x_given_n_list=[], marginal_x_list=[], marginal_y_list=[], most_likely_position_indicies_list=[])
        
        if isinstance(filter_epochs, pd.DataFrame):
            filter_epochs_df = filter_epochs
        else:
            filter_epochs_df = filter_epochs.to_dataframe()
            
        if debug_print:
            print(f'filter_epochs: {filter_epochs.n_epochs}')
        ## Get the spikes during these epochs to attempt to decode from:
        filter_epoch_spikes_df = deepcopy(spikes_df)
        ## Add the epoch ids to each spike so we can easily filter on them:
        filter_epoch_spikes_df = add_epochs_id_identity(filter_epoch_spikes_df, filter_epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name=None, no_interval_fill_value=-1)
        if debug_print:
            print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')
        filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] != -1] # Drop all non-included spikes
        if debug_print:
            print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')

        ## final step is to time_bin (relative to the start of each epoch) the time values of remaining spikes
        spkcount, nbins, time_bin_containers_list = epochs_spkcount(filter_epoch_spikes_df, filter_epochs, decoding_time_bin_size, slideby=decoding_time_bin_size, export_time_bins=True, included_neuron_ids=active_decoder.neuron_IDs, debug_print=debug_print) ## time_bins returned are not correct, they're subsampled at a rate of 1000
        num_filter_epochs = len(nbins) # one for each epoch in filter_epochs

        filter_epochs_decoder_result.spkcount = spkcount
        filter_epochs_decoder_result.nbins = nbins
        filter_epochs_decoder_result.time_bin_containers = time_bin_containers_list
        filter_epochs_decoder_result.decoding_time_bin_size = decoding_time_bin_size
        filter_epochs_decoder_result.num_filter_epochs = num_filter_epochs
        if debug_print:
            print(f'num_filter_epochs: {num_filter_epochs}, nbins: {nbins}') # the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]

        # bins = np.arange(epoch.start, epoch.stop, 0.001)
        filter_epochs_decoder_result.most_likely_positions_list = []
        filter_epochs_decoder_result.p_x_given_n_list = []
        # filter_epochs_decoder_result.marginal_x_p_x_given_n_list = []
        filter_epochs_decoder_result.most_likely_position_indicies_list = []
        # filter_epochs_decoder_result.time_bin_centers = []
        filter_epochs_decoder_result.time_bin_edges = []
        
        filter_epochs_decoder_result.marginal_x_list = []
        filter_epochs_decoder_result.marginal_y_list = []

        # Looks like we're iterating over each epoch in filter_epochs:
        for i, curr_filter_epoch_spkcount, curr_epoch_num_time_bins, curr_filter_epoch_time_bin_container in zip(np.arange(num_filter_epochs), spkcount, nbins, time_bin_containers_list):
            ## New 2022-09-26 method with working time_bin_centers_list returned from epochs_spkcount
            filter_epochs_decoder_result.time_bin_edges.append(curr_filter_epoch_time_bin_container.edges)            
            most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = active_decoder.decode(curr_filter_epoch_spkcount, time_bin_size=decoding_time_bin_size, output_flat_versions=False, debug_print=debug_print)
            filter_epochs_decoder_result.most_likely_positions_list.append(most_likely_positions)
            filter_epochs_decoder_result.p_x_given_n_list.append(p_x_given_n)
            filter_epochs_decoder_result.most_likely_position_indicies_list.append(most_likely_position_indicies)

            # Add the marginal container to the list
            curr_unit_marginal_x, curr_unit_marginal_y = cls.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
            filter_epochs_decoder_result.marginal_x_list.append(curr_unit_marginal_x)
            filter_epochs_decoder_result.marginal_y_list.append(curr_unit_marginal_y)
        
        return DecodedFilterEpochsResult(**filter_epochs_decoder_result.to_dict()) # dump the dynamic dict as kwargs into the class
    
    @classmethod
    def perform_build_marginals(cls, p_x_given_n, most_likely_positions, debug_print=False):
        """ builds the marginal distributions, which for the 1D decoder are the same as the main posterior.


        # For 1D Decoder:
            p_x_given_n.shape # (63, 106)
            most_likely_positions.shape # (106,)

            curr_unit_marginal_x['p_x_given_n'].shape # (63, 106)
            curr_unit_marginal_x['most_likely_positions_1D'].shape # (106,)

            curr_unit_marginal_y: None

        External validations:

            assert np.allclose(curr_epoch_result['marginal_x']['p_x_given_n'], curr_epoch_result['p_x_given_n']), f"1D Decoder should have an x-posterior equal to its own posterior"
            assert np.allclose(curr_epoch_result['marginal_x']['most_likely_positions_1D'], curr_epoch_result['most_likely_positions']), f"1D Decoder should have an x-posterior with most_likely_positions_1D equal to its own most_likely_positions"

        """
        is_1D_decoder = (most_likely_positions.ndim < 2) # check if we're dealing with a 1D decoder, in which case there is no y-marginal (y doesn't exist)
        if debug_print:
            print(f'perform_build_marginals(...): is_1D_decoder: {is_1D_decoder}')
            print(f"\t{p_x_given_n = }\n\t{most_likely_positions = }")
            print(f"\t{np.shape(p_x_given_n) = }\n\t{np.shape(most_likely_positions) = }")

        # p_x_given_n_shape = np.shape(p_x_given_n)

        # Compute Marginal 1D Posterior:
        ## Build a container to hold the marginal distribution and its related values:
        curr_unit_marginal_x = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
        
        if is_1D_decoder:
            # 1D Decoder:
            # p_x_given_n should come in with shape (x_bins, time_bins)
            curr_unit_marginal_x.p_x_given_n = p_x_given_n.copy() # Result should be [x_bins, time_bins]
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n # / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0) # should already be normalized but do it again anyway just in case (so there's a normalized distribution at each timestep)
            # for the 1D decoder case, there are no y-positions
            curr_unit_marginal_y = None

        else:
            # a 2D decoder
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(p_x_given_n, 1)) # sum over all y. Result should be [x_bins x time_bins]
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_x: {curr_unit_marginal_x.p_x_given_n.shape}')

            # y-axis marginal:
            curr_unit_marginal_y = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            curr_unit_marginal_y.p_x_given_n = np.squeeze(np.sum(p_x_given_n, 0)) # sum over all x. Result should be [y_bins x time_bins]
            curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n / np.sum(curr_unit_marginal_y.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_y.p_x_given_n.ndim == 0:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_y.p_x_given_n.ndim == 1:
                curr_unit_marginal_y.p_x_given_n = curr_unit_marginal_y.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior for marginal_y: {curr_unit_marginal_y.p_x_given_n.shape}')
                
        ## Add the most-likely positions to the posterior_x container:
        if is_1D_decoder:
            ## 1D Decoder Case, there is no y marginal (y doesn't exist)
            if debug_print:
                print(f'np.shape(most_likely_positions): {np.shape(most_likely_positions)}')
            curr_unit_marginal_x.most_likely_positions_1D = np.atleast_1d(most_likely_positions).T # already 1D positions, don't need to extract x-component

            # Validate 1D Conditions:
            assert np.allclose(curr_unit_marginal_x['p_x_given_n'], p_x_given_n, equal_nan=True), f"1D Decoder should have an x-posterior equal to its own posterior"
            assert np.allclose(curr_unit_marginal_x['most_likely_positions_1D'], most_likely_positions, equal_nan=True), f"1D Decoder should have an x-posterior with most_likely_positions_1D equal to its own most_likely_positions"


            # # Same as np.amax(x, axis=-1)
            # np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1).squeeze(axis=-1)


        else:
            curr_unit_marginal_x.most_likely_positions_1D = most_likely_positions[:,0].T
            curr_unit_marginal_y.most_likely_positions_1D = most_likely_positions[:,1].T
        
        return curr_unit_marginal_x, curr_unit_marginal_y

    @classmethod
    def perform_compute_most_likely_positions(cls, flat_p_x_given_n, original_position_data_shape):
        """ Computes the most likely positions at each timestep from flat_p_x_given_n and the shape of the original position data """
        most_likely_position_flat_indicies = np.argmax(flat_p_x_given_n, axis=0)        
        most_likely_position_indicies = np.array(np.unravel_index(most_likely_position_flat_indicies, original_position_data_shape)) # convert back to an array
        # np.shape(most_likely_position_flat_indicies) # (85841,)
        # np.shape(most_likely_position_indicies) # (2, 85841)
        return most_likely_position_flat_indicies, most_likely_position_indicies

    @classmethod
    def perform_compute_forward_filled_positions(cls, most_likely_positions: np.ndarray, is_non_firing_bin: np.ndarray) -> np.ndarray:
        """ applies the forward fill to a copy of the positions based on the is_non_firing_bin boolean array
        zero_bin_indicies.shape # (9307,)
        self.most_likely_positions.shape # (11880, 2)
        
        # NaN out the position bins that were determined without any spikes
        # Forward fill the now NaN positions with the last good value (for the both axes)
        
        """
        revised_most_likely_positions = most_likely_positions.copy()
        # NaN out the position bins that were determined without any spikes
        if (most_likely_positions.ndim < 2):
            revised_most_likely_positions[is_non_firing_bin] = np.nan 
        else:
            revised_most_likely_positions[is_non_firing_bin, :] = np.nan 
        # Forward fill the now NaN positions with the last good value (for the both axes):
        revised_most_likely_positions = np_ffill_1D(revised_most_likely_positions.T).T
        return revised_most_likely_positions