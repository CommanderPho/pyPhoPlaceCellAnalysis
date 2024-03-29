from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union
from attrs import define, field, Factory
from neuropy.analyses import Epoch
from neuropy.core.epoch import ensure_dataframe
from nptyping import NDArray # for DecodedFilterEpochsResult
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
from pyphocorehelpers.mixins.serialized import SerializedAttributesAllowBlockSpecifyingClass

from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results # for finding common neurons in `prune_to_shared_aclus_only`
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.peak_location_representing import ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin
    

# cut_bins = np.linspace(59200, 60800, 9)
# pd.cut(df['column_name'], bins=cut_bins)

# # just want counts of number of occurences of each?
# df['column_name'].value_counts(bins=8, sort=False)

# methods of reconstruction/decoding:

""" 
occupancy gives the P(x) in general.
n_i: the number of spikes fired by each cell during the time window of consideration


np.savez_compressed('test_parameters-neuropy_bayesian_prob.npz', **{'tau':time_bin_size, 'P_x':self.P_x, 'F':self.F, 'n':unit_specific_time_binned_spike_counts})


"""

class ZhangReconstructionImplementation:

    # Shared:    
    @staticmethod
    def compute_time_bins(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
        """Given a spikes dataframe, this function temporally bins the spikes, counting the number that fall into each bin.

        # importantly adds 'binned_time' column to spikes_df


        Args:
            spikes_df ([type]): [description]
            time_bin_size ([type]): [description]
            debug_print (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
            
            
        Added Columns to spikes_df:
            'binned_time': the binned time index
        
        Usage:
            time_bin_size=0.02

            curr_result_label = 'maze1'
            sess = curr_kdiba_pipeline.filtered_sessions[curr_result_label]
            pf = curr_kdiba_pipeline.computation_results[curr_result_label].computed_data['pf1D']

            spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = compute_time_binned_spiking_activity(spikes_df, time_bin_size)

        """
        time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'
        time_window_edges, time_window_edges_binning_info = compute_spanning_bins(spikes_df[time_variable_name].to_numpy(), bin_size=max_time_bin_size) # np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]
        spikes_df = spikes_df.spikes.add_binned_time_column(time_window_edges, time_window_edges_binning_info, debug_print=debug_print)
        return time_window_edges, time_window_edges_binning_info, spikes_df
        
    
    @staticmethod
    def compute_unit_specific_bin_specific_spike_counts(spikes_df, time_bin_indicies, debug_print=False):
        """ 
        spikes_df: a dataframe with at least the ['aclu','binned_time'] columns
        time_bin_indicies: np.ndarray of indicies that will be used for the produced output dataframe of the binned spike counts for each unit. (e.g. time_bin_indicies = time_window_edges_binning_info.bin_indicies[1:]).
        
        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'
        assert 'binned_time' in spikes_df.columns
        unit_specific_bin_specific_spike_counts = spikes_df.groupby(['aclu','binned_time'])[time_variable_name].agg('count')
        active_aclu_binned_time_multiindex = unit_specific_bin_specific_spike_counts.index
        active_unique_aclu_values = np.unique(active_aclu_binned_time_multiindex.get_level_values('aclu'))
        unit_specific_binned_spike_counts = np.array([unit_specific_bin_specific_spike_counts[aclu].values for aclu in active_unique_aclu_values]).T # (85841, 40)
        if debug_print:
            print(f'np.shape(unit_specific_spike_counts): {np.shape(unit_specific_binned_spike_counts)}') # np.shape(unit_specific_spike_counts): (40, 85841)
        unit_specific_binned_spike_counts = pd.DataFrame(unit_specific_binned_spike_counts, columns=active_unique_aclu_values, index=time_bin_indicies)
        return unit_specific_binned_spike_counts
    
    @staticmethod
    def compute_time_binned_spiking_activity(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
        """Given a spikes dataframe, this function temporally bins the spikes, counting the number that fall into each bin.

        Args:
            spikes_df ([type]): [description]
            time_bin_size ([type]): [description]
            debug_print (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
            
            
        Added Columns to spikes_df:
            'binned_time': the binned time index
        
        Usage:
            time_bin_size=0.02

            curr_result_label = 'maze1'
            sess = curr_kdiba_pipeline.filtered_sessions[curr_result_label]
            pf = curr_kdiba_pipeline.computation_results[curr_result_label].computed_data['pf1D']

            spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = compute_time_binned_spiking_activity(spikes_df, time_bin_size)

        """
        time_window_edges, time_window_edges_binning_info, spikes_df = ZhangReconstructionImplementation.compute_time_bins(spikes_df, max_time_bin_size=max_time_bin_size, debug_print=debug_print) # importantly adds 'binned_time' column to spikes_df
        unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies[1:], debug_print=debug_print)
        return unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info

    @classmethod
    def _validate_time_binned_spike_rate_df(cls, time_bins, unit_specific_binned_spike_counts_df):
        """ Validates the outputs of `ZhangReconstructionImplementation.compute_time_binned_spiking_activity(...)`
        
        Usage:
            active_session_spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_count_df, sess_time_window_edges, sess_time_window_edges_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(active_session_spikes_df.copy(), max_time_bin_size=time_bin_size_seconds, debug_print=False) # np.shape(unit_specific_spike_counts): (4188, 108)
            sess_time_binning_container = BinningContainer(edges=sess_time_window_edges, edge_info=sess_time_window_edges_binning_info)
            _validate_time_binned_spike_rate_df(sess_time_binning_container.centers, unit_specific_binned_spike_count_df)

        """
        assert isinstance(unit_specific_binned_spike_counts_df, pd.DataFrame)
        assert unit_specific_binned_spike_counts_df.shape[0] == time_bins.shape[0], f"unit_specific_binned_spike_counts_df.shape[0]: {unit_specific_binned_spike_counts_df.shape[0]} and time_bins.shape[0] {time_bins.shape[0]}"
        print(f'unit_specific_binned_spike_counts_df.shape: {unit_specific_binned_spike_counts_df.shape}')
        nCells = unit_specific_binned_spike_counts_df.shape[1]
        nTimeBins = unit_specific_binned_spike_counts_df.shape[0] # many time_bins
        print(f'nCells: {nCells}, nTimeBins: {nTimeBins}')


    @staticmethod
    def build_concatenated_F(pf, debug_print=False):
        """ returns flattened versions of the occupancy (P_x), and the tuning_curves (F) """
        neuron_IDs = pf.ratemap.neuron_ids
        neuron_IDXs = np.arange(len(neuron_IDs))
        ## 2022-09-19 - should this be the non-normalized tuning curves instead of the normalized ones?
        #   2023-04-07 - CONFIRMED: yes, this should be the non-normalized tuning curves instead of the normalized ones
        maps = pf.ratemap.tuning_curves  # (40, 48) for 1D, (40, 48, 10) for 2D
        if debug_print:
            print(f'maps: {np.shape(maps)}') # maps: (40, 48, 10)

        try:
            f_i = [np.squeeze(maps[i,:,:]) for i in neuron_IDXs] # produces a list of (48 x 10) maps
        except IndexError as e:
            # Happens when called on a 1D decoder
            assert np.ndim(maps) == 2, f"Currently only handles special 1D decoder case but np.shape(maps): {np.shape(maps)} and np.ndim(maps): {np.ndim(maps)} != 2"
            f_i = [np.squeeze(maps[i,:]) for i in neuron_IDXs] # produces a list of (48, ) maps
        except Exception as e:
            raise e        
        if debug_print:
            print(f'np.shape(f_i[i]): {np.shape(f_i[0])}') # (48, 6)
        F_i = [np.reshape(f_i[i], (-1, 1)) for i in neuron_IDXs] # Convert each function to a column vector
        if debug_print:
            print(f'np.shape(F_i[i]): {np.shape(F_i[0])}') # (288, 1)
        # Concatenate each individual F_i to produce F
        F = np.hstack(F_i) #@IgnoreException  
        if debug_print:
            print(f'np.shape(F): {np.shape(F)}') # (288, 40)
        P_x = np.reshape(pf.occupancy, (-1, 1)) # occupancy gives the P(x) in general.
        if debug_print:
            print(f'np.shape(P_x): {np.shape(P_x)}') # np.shape(P_x): (48, 6)

        return neuron_IDXs, neuron_IDs, f_i, F_i, F, P_x


    @staticmethod
    def time_bin_spike_counts_N_i(spikes_df, time_bin_size, time_window_edges=None, time_window_edges_binning_info=None, debug_print=False):
        """ Returns the number of spikes that occured for each neuron in each time bin.
        Example:
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
            
        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        if time_window_edges is None or time_window_edges_binning_info is None:
            # build time_window_edges/time_window_edges_binning_info AND adds 'binned_time' column to spikes_df. # NOTE: the added 'binned_time' column is 1-indexed, which may explain why we use `time_window_edges_binning_info.bin_indicies[1:]` down below
            time_window_edges, time_window_edges_binning_info, spikes_df = ZhangReconstructionImplementation.compute_time_bins(spikes_df, max_time_bin_size=time_bin_size, debug_print=debug_print)
        else:
            # already have time bins (time_window_edges/time_window_edges_binning_info) so just add 'binned_time' column to spikes_df if needed:
            if 'binned_time' not in spikes_df.columns:
                # we must have the 'binned_time' column in spikes_df, so add it if needed
                spikes_df = spikes_df.spikes.add_binned_time_column(time_window_edges, time_window_edges_binning_info, debug_print=debug_print)
        
        # either way to compute the unit_specific_binned_spike_counts:
        unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies[1:], debug_print=debug_print) # requires 'binned_time' in spikes_df. TODO 2023-05-16 - ERROR: isn't getting centers. time_window_edges_binning_info.bin_indicies[1:]
        unit_specific_binned_spike_counts = unit_specific_binned_spike_counts.T # Want the outputs to have each time window as a column, with a single time window giving a column vector for each neuron
        if debug_print:
            print(f'unit_specific_binned_spike_counts.to_numpy(): {np.shape(unit_specific_binned_spike_counts.to_numpy())}') # (85841, 40)
        return unit_specific_binned_spike_counts.to_numpy(), time_window_edges, time_window_edges_binning_info

    @classmethod
    def _validate_time_binned_spike_counts(cls, time_binning_container, unit_specific_binned_spike_counts):
        """ Validates the outputs of `ZhangReconstructionImplementation.time_bin_spike_counts_N_i(...)`
        
        Usage:
            active_session_spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(active_session_spikes_df.copy(), time_bin_size=time_bin_size_seconds, debug_print=False)  # np.shape(unit_specific_spike_counts): (4188, 108)
            time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
            _validate_time_binned_spike_counts(time_binning_container, unit_specific_binned_spike_counts)

        """
        assert isinstance(unit_specific_binned_spike_counts, np.ndarray)
        assert unit_specific_binned_spike_counts.shape[1] == time_binning_container.center_info.num_bins, f"unit_specific_binned_spike_counts.shape[1]: {unit_specific_binned_spike_counts.shape[1]} and time_binning_container.center_info.num_bins {time_binning_container.center_info.num_bins}"
        print(f'unit_specific_binned_spike_counts.shape: {unit_specific_binned_spike_counts.shape}')
        nCells = unit_specific_binned_spike_counts.shape[0]
        nTimeBins = unit_specific_binned_spike_counts.shape[1] # many time_bins
        print(f'nCells: {nCells}, nTimeBins: {nTimeBins}')


    # Optimal Functions:
    @staticmethod
    def compute_optimal_functions_G(F):
        G = np.linalg.pinv(F).T # Perform the Moore-Penrose pseudoinverse on F to compute G.
        return G
    
    @staticmethod
    def test_identity_deviation(F):
        """ Tests how close the computed F is to the identity matrix, which indicates independence of the functions. """
        identity_approx = np.matmul(F.T, F)
        print(f'np.shape(identity_approx): {np.shape(identity_approx)}') # np.shape(identity_approx): (40, 40)
        identity_deviation = identity_approx - np.identity(np.shape(identity_approx)[0])
        return identity_deviation

    
    @staticmethod
    def neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=True, debug_intermediates_mode=False, debug_print=False):
        """ 
            n_i: the number of spikes fired by each cell during the time window of consideration
            use_flat_computation_mode: bool - if True, a more memory efficient accumulating computation is performed that avoids `MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64` caused by allocating the full `cell_prob` matrix
            debug_intermediates_mode: bool - if True, the intermediate computations are stored and returned for debugging purposes. MUCH slower.

        NOTES: Flat vs. Full computation modes:
        Originally 
            cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) 
        This was updated throughout the loop, and then after the loop completed np.prod(cell_prob, axis=2) was used to collapse along axis=2 (nCells), leaving the output posterior with dimensions (nFlatPositionBins, nTimeBins)

        To get around this, I introduced a version that accumulates the multilications over the course of the loop.
            cell_prob = np.ones((nFlatPositionBins, nTimeBins))

        Note: This means that the "Flat" implementation may be more susceptible to numerical underflow, as the intermediate products can become very small, whereas the "Full" implementation does not have this issue. However, the "Flat" implementation can be more efficient in terms of memory usage and computation time, as it avoids creating a large number of intermediate arrays.

        2023-04-19 - I'm confused by the lack of use of P_x in the calculation. According to my written formula P_x was used as the occupancy and multiplied in the output equation. After talking to Kourosh and the lab, it seems that the typical modern approach is to use a uniform `P_x` as to not bias the decoded position.
            But when I went to change my code to try a uniform P_x, it seems that P_x isn't used in the computation at all, and instead only its size is used?
            TODO 2023-04-19 - Check agreement with written equations.
        """
        assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'        
        if debug_print:
            print(f'np.shape(P_x): {np.shape(P_x)}, np.shape(F): {np.shape(F)}, np.shape(n): {np.shape(n)}')
        # np.shape(P_x): (1066, 1), np.shape(F): (1066, 66), np.shape(n): (66, 3530)
        
        nCells = n.shape[0]
        nTimeBins = n.shape[1] # many time_bins
        nFlatPositionBins = np.shape(P_x)[0]

        F = F.T # Transpose F so it's of the right form

        if debug_intermediates_mode:
            assert (not use_flat_computation_mode), "debug_intermediates_mode is only supported when use_flat_computation_mode is False"
            print(f'neuropy_bayesian_prob is running in debug_intermediates_mode. This will be much slower than normal.')
            # To save test parameters:
            out_save_path = 'data/test_parameters-neuropy_bayesian_prob_2023-04-07.npz'
            np.savez_compressed(out_save_path, **{'tau':tau, 'P_x':P_x, 'F':F, 'n':n})
            print(f'saved test parameters to {out_save_path}')

        if use_flat_computation_mode:
            ## Single-cell flat version which updates each iteration:
            cell_prob = np.ones((nFlatPositionBins, nTimeBins)) # Must start with ONES (not Zeros) since we're accumulating multiplications

        else:
            # Full Version which leads to MemoryError when nCells is too large:
            cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) ## MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64

        for cell in range(nCells):
            """ Comparing to the Zhang paper: the output posterior is P_n_given_x (Eqn 35)
                cell_ratemap: [f_{i}(x) for i in range(nCells)]
                cell_spkcnt: [n_{i} for i in range(nCells)]            
            """
            cell_spkcnt = n[cell, :][np.newaxis, :] # .shape: (1, nTimeBins)
            cell_ratemap = F[cell, :][:, np.newaxis] # .shape: (nFlatPositionBins, 1)
            coeff = 1.0 / (factorial(cell_spkcnt)) # 1/factorial(n_{i}) term # .shape: (1, nTimeBins)

            if use_flat_computation_mode:
                # Single-cell flat Version:
                cell_prob *= (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap)) # product equal using *=
                # cell_prob.shape (nFlatPositionBins, nTimeBins)
            else:
                # Full Version:
                # broadcasting
                if debug_intermediates_mode:
                    t0 = ((tau * cell_ratemap) ** cell_spkcnt)
                    t1 = coeff
                    t2 = (np.exp(-tau * cell_ratemap))
                    cell_prob[:, :, cell] = (t0 * t1) * t2
                else:
                    cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap))

        if use_flat_computation_mode:
            # Single-cell flat Version:
            posterior = cell_prob # The product has already been accumulating all along
            posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n
        else:
            # Full Version:
            posterior = np.prod(cell_prob, axis=2) # note this product removes axis=2 (nCells)
            posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n

        return posterior



class Zhang_Two_Step:
    """ Two-Step Decoder from Zhang et al. 2018. """

    @classmethod
    def build_all_positions_matrix(cls, x_values, y_values, debug_print=False):
        """ used to build a grid of position points from xbins and ybins.
        Usage:
            all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_all_positions_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
        """
        return build_spanning_grid_matrix(x_values, y_values, debug_print=debug_print)

    @classmethod
    def sigma_t(cls, v_t, K, V, d:float=1.0):
        """ The standard deviation of the Gaussian prior for position. Once computed and normalized, can be used such that it only requires the current position (x_t) to return the correct std_dev at a given timestamp.
        K, V are constants
        d is 0.5 for random walks and 1.0 for linear movements. 
        """
        return K * np.power((v_t / V), d)
    
    @classmethod
    def compute_conditional_probability_x_prev_given_x_t(cls, x_prev, x, sigma_t, C):
        """ Should return a value for all possible current locations x_t. x_prev should be a concrete position, not a matrix of them. """
        # multivariate_normal.pdf()        
        # return C * np.exp(-np.square(np.linalg.norm(x - x_prev, axis=1))/(2.0*np.square(sigma_t)))
        numerator = -np.square(np.linalg.norm(x - x_prev, axis=1)) # (1950,)
        denominator = 2.0*np.square(sigma_t) # (64,29)
        # want output of the shape (64,29)
        return C * np.exp(numerator/denominator)

        # a[..., None] + c[None, None, :]
        # output = multivariate_normal.pdf()
        
    @classmethod
    def compute_scaling_factor_k(cls, flat_p_x_given_n):
        """k can be computed in closed from in a vectorized fashion from the one_step bayesian posterior.
        k is a scaling factor that doesn't depend on x_t. Determined by normalizing P(x_t|n_t, x_prev) over x_t"""        
        out_k = np.append(np.nan, np.nansum(flat_p_x_given_n, axis=0)[:-1]) # we'll have one for each time_window_bin_idx [:, time_window_bin_idx]. Want all except the last element ([:-1])
        return out_k
        
    @classmethod
    def compute_bayesian_two_step_prob_single_timestep(cls, one_step_p_x_given_n, x_prev, all_x, sigma_t, C, k):
        return k * one_step_p_x_given_n * cls.compute_conditional_probability_x_prev_given_x_t(x_prev, all_x, sigma_t, C)
    


@custom_define(slots=False, repr=False)
class DecodedFilterEpochsResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Container for the results of decoding a set of epochs (filter_epochs) using a decoder (active_decoder) 
    
    This class stores results from decoding from multiple non-contiguous time epochs, each containing many time bins (a variable number according to their length)
        The two representational formats are:
            1. 
    
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        
        
    marginal_y:
        DynamicContainer({'p_x_given_n': array([[0, 0, 0.722646, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.722646, 0.722646, 0.722646, 0.722646, 0, 0, 0.722646, 0.722646, 0.722646, 0, 0, 0, 0, 0.722646, 0, 0, 0, 0],
        [1, 1, 0.277354, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.277354, 0.277354, 0.277354, 0.277354, 1, 1, 0.277354, 0.277354, 0.277354, 1, 1, 1, 1, 0.277354, 1, 1, 1, 1]]), 'most_likely_positions_1D': array([1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 1.5, 1.5, 1.5])})
       
    """
    decoding_time_bin_size: float = serialized_attribute_field() # the time bin_size in seconds
    filter_epochs: pd.DataFrame = serialized_field() # the filter epochs themselves
    num_filter_epochs: int = serialized_attribute_field() # depends on the number of epochs (`n_epochs`)
    most_likely_positions_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    p_x_given_n_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    marginal_x_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    marginal_y_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    most_likely_position_indicies_list: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    spkcount: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    nbins: np.ndarray = serialized_field(metadata={'shape': ('n_epochs',)}) # an array of the number of time bins in each epoch
    time_bin_containers: list = non_serialized_field(metadata={'shape': ('n_epochs',)})
    time_bin_edges: list = non_serialized_field(metadata={'shape': ('n_epochs',)}) # depends on the number of epochs, one per epoch
    epoch_description_list: list[str] = non_serialized_field(default=Factory(list), metadata={'shape': ('n_epochs',)}) # depends on the number of epochs, one for each
    

    @property
    def active_filter_epochs(self):
        """ for compatibility """
        return deepcopy(self.filter_epochs)


    @property
    def time_window_centers(self) -> List[NDArray]:
        """ for compatibility """
        return deepcopy([self.time_bin_containers[an_epoch_idx].centers for an_epoch_idx in np.arange(self.num_filter_epochs)])



    @function_attributes(short_name=None, tags=['radon-transform','decoder','line','fit','velocity','speed'], input_requires=[], output_provides=[], uses=['get_radon_transform'], used_by=[], creation_date='2024-02-13 17:25', related_items=[])
    def compute_radon_transforms(self, pos_bin_size:float, nlines:int=8192, margin:int=16, jump_stat=None, n_jobs:int=4, enable_return_neighbors_arr=True) -> pd.DataFrame:
        """ 2023-05-25 - Computes the line of best fit (which gives the velocity) for the 1D Posteriors for each replay epoch using the Radon Transform approch.
        
        # pos_bin_size: the size of the x_bin in [cm]
        if decoder.pf.bin_info is not None:
            pos_bin_size = float(decoder.pf.bin_info['xstep'])
        else:
            ## if the bin_info is for some reason not accessible, just average the distance between the bin centers.
            pos_bin_size = np.diff(decoder.pf.xbin_centers).mean()


        Usage:
        
            epochs_linear_fit_df = compute_radon_transforms(pos_bin_size=pos_bin_size, nlines=8192, margin=16, n_jobs=1)
            
            a_directional_laps_filter_epochs_decoder_result = a_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(curr_active_pipeline.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)
            laps_radon_transform_df = compute_radon_transforms(a_directional_pf1D_Decoder, a_directional_laps_filter_epochs_decoder_result)

            Columns:         ['score', 'velocity', 'intercept', 'speed']
        """
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import get_radon_transform
        # active_time_bins = active_epoch_decoder_result.time_bin_edges[0]
        # active_posterior_container = active_epoch_decoder_result.marginal_x_list[0]
        active_posterior = self.p_x_given_n_list # one for each epoch

        ## compute the Radon transform to get the lines of best fit
        extra_outputs = []
        score, velocity, intercept, *extra_outputs = get_radon_transform(active_posterior, decoding_time_bin_duration=self.decoding_time_bin_size, pos_bin_size=pos_bin_size, posteriors=None, nlines=nlines, margin=margin, jump_stat=jump_stat, n_jobs=n_jobs, enable_return_neighbors_arr=enable_return_neighbors_arr)
        epochs_linear_fit_df = pd.DataFrame({'score': score, 'velocity': velocity, 'intercept': intercept, 'speed': np.abs(velocity)})
        return epochs_linear_fit_df, extra_outputs


    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
            DecodedFilterEpochsResult(decoding_time_bin_size: float,
                filter_epochs: neuropy.core.epoch.Epoch,
                num_filter_epochs: int,
                most_likely_positions_list: list | shape (n_epochs),
                p_x_given_n_list: list | shape (n_epochs),
                marginal_x_list: list | shape (n_epochs),
                marginal_y_list: list | shape (n_epochs),
                most_likely_position_indicies_list: list | shape (n_epochs),
                spkcount: list | shape (n_epochs),
                nbins: numpy.ndarray | shape (n_epochs),
                time_bin_containers: list | shape (n_epochs),
                time_bin_edges: list | shape (n_epochs),
                epoch_description_list: list | shape (n_epochs)
            )
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        # content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        # return f"{type(self).__name__}({content}\n)"
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"


    def flatten(self):
        """ flattens the result over all epochs to produce one per time bin """
        # returns a flattened version of self over all epochs
        n_timebins = np.sum(self.nbins)
        flat_time_bin_containers = np.hstack(self.time_bin_containers)
        # timebins_p_x_given_n = [].extend(self.p_x_given_n_list)
        timebins_p_x_given_n = np.hstack(self.p_x_given_n_list) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)
        # TODO 2023-04-13 -can these squished similar way?: most_likely_positions_list, most_likely_position_indicies_list 
        return n_timebins, flat_time_bin_containers, timebins_p_x_given_n


    def flatten_to_masked_values(self):
        """ appends np.nan values to the beginning and end of each posterior (adding a start and end timebin as well) to allow flat plotting via matplotlib.
        Looks  like it was depricated by `plot_slices_1D_most_likely_position_comparsions` to plot epoch slices (corresponding to certain periods in time) along the continuous session duration.

        """
        # returns a flattened version of self over all epochs
        updated_is_masked_bin = []
        updated_time_bin_containers = []
        updated_timebins_p_x_given_n = []

        decoding_time_bin_size: float = self.decoding_time_bin_size
        desired_n_timebins = self.nbins + 2 # add two to each element for the start/end bin

        total_n_timebins = np.sum(self.nbins)
        desired_total_n_timebins = np.sum(desired_n_timebins)


        for epoch_idx in np.arange(self.num_filter_epochs):
            a_curr_num_bins: int = self.nbins[epoch_idx]
            updated_curr_num_bins = a_curr_num_bins + 2 # add two (start/end) bins
            a_centers = self.time_bin_containers[epoch_idx].centers
            a_posterior = self.p_x_given_n_list[epoch_idx]
            n_pos_bins = np.shape(a_posterior)[0]
            
            updated_posterior = np.full((n_pos_bins, updated_curr_num_bins), np.nan)
            updated_posterior[:,1:-1] = a_posterior

            curr_is_masked_bin = np.full((updated_curr_num_bins,), True)
            curr_is_masked_bin[1:-1] = False

            ## Add the start/end bin
            # a_centers.
            updated_time_bin_containers.append([(a_centers[0]-decoding_time_bin_size), list(a_centers), (a_centers[-1]+decoding_time_bin_size)])
            updated_timebins_p_x_given_n.append(updated_posterior)
            updated_is_masked_bin.append(curr_is_masked_bin)

    
        updated_timebins_p_x_given_n = np.hstack(updated_timebins_p_x_given_n) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)
        updated_time_bin_containers = np.hstack(np.hstack(updated_time_bin_containers))
        updated_is_masked_bin = np.hstack(updated_is_masked_bin)

        assert np.shape(updated_time_bin_containers)[0] == desired_total_n_timebins
        assert np.shape(updated_timebins_p_x_given_n)[1] == desired_total_n_timebins
        assert np.shape(updated_is_masked_bin)[0] == desired_total_n_timebins

        return desired_total_n_timebins, updated_is_masked_bin, updated_time_bin_containers, updated_timebins_p_x_given_n


    def filtered_by_epoch_times(self, included_epoch_start_times) -> "DecodedFilterEpochsResult":
        """ Returns a copy of itself with the fields with the n_epochs related metadata sliced by the included_epoch_indicies found from the rows that match `included_epoch_start_times`.       
        """
        subset = deepcopy(self)
        original_num_filter_epochs = subset.num_filter_epochs
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()
        found_data_indicies = subset.filter_epochs.epochs.find_data_indicies_from_epoch_times(epoch_times=included_epoch_start_times)
        return subset.filtered_by_epochs(found_data_indicies)


    def filtered_by_epochs(self, included_epoch_indicies, debug_print=False) -> "DecodedFilterEpochsResult":
        """Returns a copy of itself with the fields with the n_epochs related metadata sliced by the included_epoch_indicies.
        
        MOSTLY TESTED 2024-03-11 - WORKING

        """
        subset = deepcopy(self)
        original_num_filter_epochs = subset.num_filter_epochs
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()

        ## Convert to the real-deal: pure indicies
        old_fashioned_indicies = np.array([subset.filter_epochs.index.get_loc(a_loc_idx) for a_loc_idx in included_epoch_indicies])
        if debug_print:
            print(f'old_fashioned_indicies: {old_fashioned_indicies}')

        ## Need to have the indicies before applying the filter:
        subset.filter_epochs = subset.filter_epochs.loc[included_epoch_indicies] # the evil `.iloc[...]` creeps in again with `IndexError: positional indexers are out-of-bounds`

        subset.most_likely_positions_list = [subset.most_likely_positions_list[i] for i in old_fashioned_indicies] # that's obviously not going to work because .loc[...] values are used. I need that magic trick that the new AI taught me -- `.index.get_loc(start_index)`
        subset.p_x_given_n_list = [subset.p_x_given_n_list[i] for i in old_fashioned_indicies]
        subset.marginal_x_list = [subset.marginal_x_list[i] for i in old_fashioned_indicies]
        subset.marginal_y_list = [subset.marginal_y_list[i] for i in old_fashioned_indicies]
        subset.most_likely_position_indicies_list = [subset.most_likely_position_indicies_list[i] for i in old_fashioned_indicies]
        subset.spkcount = [subset.spkcount[i] for i in old_fashioned_indicies]
        subset.nbins = subset.nbins[old_fashioned_indicies] # can be subset because it's an ndarray
        subset.time_bin_containers = [subset.time_bin_containers[i] for i in old_fashioned_indicies]
        subset.num_filter_epochs = len(included_epoch_indicies)
        subset.time_bin_edges = [subset.time_bin_edges[i] for i in old_fashioned_indicies]
        if len(subset.epoch_description_list) == original_num_filter_epochs:
            # sometimes epoch_description_list is empty and so it doesn't need to be subsetted.
            subset.epoch_description_list = [subset.epoch_description_list[i] for i in old_fashioned_indicies]
            
        # Only `decoding_time_bin_size` is unchanged
        return subset
    

    @classmethod
    def _test_find_fields_by_shape_metadata(cls):
        """ tries to get all the fields that match the shape criteria. Not completely implemented, but seems to work.
        
        
        # # Get the values at epoch_IDX from a particular instance `active_result`:
        # epoch_IDX: int = 0
        # # values = [getattr(active_result, field)[epoch_IDX] for field in indices_fields_n_epochs]
        # # values = [getattr(active_result, field) for field in indices_fields_n_epochs]
        # values_dict = {field:getattr(active_result, field)[epoch_IDX] for field in indices_fields_n_epochs if field in desired_keys}
        # values_dict

        """
        class_fields = cls.__attrs_attrs__
        # indices_fields_n_epochs = [field.name for field in class_fields if hasattr(field.metadata, 'shape') and field.metadata['shape'][0] == 'n_epochs']
        indices_fields_n_epochs = [field.name for field in class_fields if 'shape' in field.metadata and field.metadata['shape'][0] == 'n_epochs']
        # print(f'indices_fields_n_epochs: {indices_fields_n_epochs}') # ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'most_likely_position_indicies_list', 'spkcount', 'nbins', 'time_bin_containers', 'time_bin_edges', 'epoch_description_list']
        desired_keys = ['most_likely_positions_list', 'p_x_given_n_list', 'marginal_x_list', 'marginal_y_list', 'most_likely_position_indicies_list', 'nbins', 'time_bin_containers', 'time_bin_edges']
        return [a_field for a_field in indices_fields_n_epochs if a_field in desired_keys]


# ==================================================================================================================== #
# Placemap Position Decoders                                                                                           #
# ==================================================================================================================== #


# ==================================================================================================================== #
# Stateless Decoders (New 2023-04-06)                                                                                  #
# ==================================================================================================================== #

@custom_define(slots=False)
class BasePositionDecoder(HDFMixin, AttrsBasedClassHelperMixin, ContinuousPeakLocationRepresentingMixin, PeakLocationRepresentingMixin, NeuronUnitSlicableObjectProtocol):
    """ 2023-04-06 - A simplified data-only version of the decoder that serves to remove all state related to specific computations to make each run independent 
    Stores only the raw inputs that are used to decode, with the user specifying the specifics for a given decoding (like time_time_sizes, etc.


    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder


    """
    pf: PfND = serialized_field()

    neuron_IDXs: np.ndarray = serialized_field(default=None, is_computable=True)
    neuron_IDs: np.ndarray = serialized_field(default=None, is_computable=True)
    F: np.ndarray = non_serialized_field(default=None)
    P_x: np.ndarray = non_serialized_field(default=None)

    setup_on_init:bool = True 
    post_load_on_init:bool = False
    debug_print: bool = False

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
    def pos_bin_size(self) -> Union[float, Tuple[float, float]]:
        """ extracts pos_bin_size: the size of the x_bin in [cm], from the decoder. 
        returns a tuple if 2D or a single float if 1D
        """
        return self.pf.pos_bin_size
    
    @property
    def original_position_data_shape(self):
        """The original_position_data_shape property."""
        return np.shape(self.pf.occupancy)    

    @property
    def flat_position_size(self):
        """The flat_position_size property."""
        return np.shape(self.F)[0] # like 288

    # PeakLocationRepresentingMixin + ContinuousPeakLocationRepresentingMixin conformances:
    @property
    def PeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        return self.ratemap.PeakLocationRepresentingMixin_peak_curves_variable
    
    @property
    def ContinuousPeakLocationRepresentingMixin_peak_curves_variable(self) -> NDArray:
        """ the variable that the peaks are calculated and returned for """
        return self.ratemap.ContinuousPeakLocationRepresentingMixin_peak_curves_variable
    
    
    
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
    def init_from_stateful_decoder(cls, stateful_decoder: "BayesianPlacemapPositionDecoder"):
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
        
        self._setup_computation_variables()

    @post_deserialize
    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        with WrappingMessagePrinter(f'post_load() called.', begin_line_ending='... ', finished_message='all rebuilding completed.', enable_print=self.debug_print):
            self._setup_computation_variables()
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
    
    @function_attributes(short_name=None, tags=['epoch', 'slice', 'restrict'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-29 19:08', related_items=[])
    def replacing_computation_epochs(self, epochs: Union[Epoch, pd.DataFrame]):
        """Implementors return a copy of themselves with their computation epochs (contained in their placefields at `self.pf`) replaced by the provided ones. The existing epochs are unrelated and do not need to be related to the new ones.
        """
        new_epochs_obj: Epoch = Epoch(ensure_dataframe(deepcopy(epochs)).epochs.get_valid_df()).get_non_overlapping()
        ## Restrict the PfNDs:
        epoch_replaced_pf1D: PfND = self.pf.replacing_computation_epochs(deepcopy(new_epochs_obj))
        ## apply the neuron_sliced_pf to the decoder:
        updated_decoder = BasePositionDecoder(epoch_replaced_pf1D, setup_on_init=self.setup_on_init, post_load_on_init=self.post_load_on_init, debug_print=self.debug_print)
        return updated_decoder


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
    def decode_specific_epochs(self, spikes_df: pd.DataFrame, filter_epochs, decoding_time_bin_size:float=0.05, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DecodedFilterEpochsResult:
        """ 
        Uses:
            BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...)
        """
        return self.perform_decode_specific_epochs(self, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print)

	# ==================================================================================================================== #
    # Non-Modifying Methods:                                                                                               #
    # ==================================================================================================================== #
    @function_attributes(short_name='decode', tags=['decode', 'pure'], input_requires=[], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['BayesianPlacemapPositionDecoder.perform_compute_most_likely_positions', 'ZhangReconstructionImplementation.neuropy_bayesian_prob'],
        used_by=['BayesianPlacemapPositionDecoder.perform_decode_specific_epochs'])
    def decode(self, unit_specific_time_binned_spike_counts, time_bin_size:float, output_flat_versions=False, debug_print=True):
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
        ## Capture inputs for debugging:
        computation_debug_mode = False
        if computation_debug_mode:
            debug_intermediates_mode=True
            use_flat_computation_mode=False
            # np.savez_compressed('test_parameters-neuropy_bayesian_prob_2023-04-07.npz', **{'tau':time_bin_size, 'P_x':self.P_x, 'F':self.F, 'n':unit_specific_time_binned_spike_counts})
        else:
            debug_intermediates_mode=False
            use_flat_computation_mode=True

        num_cells = np.shape(unit_specific_time_binned_spike_counts)[0]    
        num_time_windows = np.shape(unit_specific_time_binned_spike_counts)[1]
        if debug_print:
            print(f'num_cells: {num_cells}, num_time_windows: {num_time_windows}')
        with WrappingMessagePrinter(f'decode(...) called. Computing {num_time_windows} windows for final_p_x_given_n...', begin_line_ending='... ', finished_message='decode completed.', enable_print=(debug_print or self.debug_print)):
            if time_bin_size is None:
                print(f'time_bin_size is None, using internal self.time_bin_size.')
                time_bin_size = self.time_bin_size
            
            # Single sweep decoding:
            curr_flat_p_x_given_n = ZhangReconstructionImplementation.neuropy_bayesian_prob(time_bin_size, self.P_x, self.F, unit_specific_time_binned_spike_counts, debug_intermediates_mode=debug_intermediates_mode, use_flat_computation_mode=use_flat_computation_mode, debug_print=(debug_print or self.debug_print))
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
            
    @function_attributes(short_name='hyper_perform_decode', tags=['decode', 'pure'], input_requires=['self.neuron_IDs'], output_provides=[], creation_date='2023-03-23 19:10',
        uses=['self.decode', 'BayesianPlacemapPositionDecoder.perform_build_marginals', 'epochs_spkcount'],
        used_by=[])
    def hyper_perform_decode(self, spikes_df, decoding_time_bin_size=0.1, t_start=None, t_end=None, output_flat_versions=False, debug_print=False):
        """ Fully decodes the neural activity from its internal placefields, internally calling `self.decode(...)` and then in addition building the marginals and additional outputs.

        Does not alter the internal state of the decoder (doesn't change internal most_likely_positions or posterior, etc)

        Requires:
            self.neuron_IDs

        Uses:
            self.decode(...)
            BayesianPlacemapPositionDecoder.perform_build_marginals(...)
            epochs_spkcount(...)
        """
        # Range of the maze epoch (where position is valid):
        if t_start is None:
            t_maze_start = spikes_df[spikes_df.spikes.time_variable_name].loc[spikes_df.x.first_valid_index()] # 1048
            t_start = t_maze_start
    
        if t_end is None:
            t_maze_end = spikes_df[spikes_df.spikes.time_variable_name].loc[spikes_df.x.last_valid_index()] # 68159707
            t_end = t_maze_end
        
        epochs_df = pd.DataFrame({'start':[t_start],'stop':[t_end],'label':['epoch']})

        ## final step is to time_bin (relative to the start of each epoch) the time values of remaining spikes
        spkcount, nbins, time_bin_containers_list = epochs_spkcount(spikes_df, epochs_df, decoding_time_bin_size, slideby=decoding_time_bin_size, export_time_bins=True, included_neuron_ids=self.neuron_IDs, debug_print=debug_print) ## time_bins returned are not correct, they're subsampled at a rate of 1000
        spkcount = spkcount[0]
        nbins = nbins[0]
        time_bin_container = time_bin_containers_list[0] # neuropy.utils.mixins.binning_helpers.BinningContainer
        
        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = self.decode(spkcount, time_bin_size=decoding_time_bin_size, output_flat_versions=output_flat_versions, debug_print=debug_print)
        curr_unit_marginal_x, curr_unit_marginal_y = self.perform_build_marginals(p_x_given_n, most_likely_positions, debug_print=debug_print)
        return time_bin_container, p_x_given_n, most_likely_positions, curr_unit_marginal_x, curr_unit_marginal_y, flat_outputs_container


    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    @function_attributes(short_name='setup_computation_variables', tags=['pr'], input_requires=[], output_provides=[], uses=['self.pf', 'ZhangReconstructionImplementation.build_concatenated_F'], used_by=[], creation_date='2023-04-06 13:49')
    def _setup_computation_variables(self):
        """ Uses `self.pf` and sets up the computation variables F, P_x, neuron_IDs, neuron_IDXs

            maps: (40, 48, 6)
            np.shape(f_i[i]): (48, 6)
            np.shape(F_i[i]): (288, 1)
            np.shape(F): (288, 40)
            np.shape(P_x): (288, 1)
            
        History:
            Used to be named `_setup_concatenated_F` but renamed because it computes more than that.
        """
        self.neuron_IDXs, self.neuron_IDs, f_i, F_i, self.F, self.P_x = ZhangReconstructionImplementation.build_concatenated_F(self.pf, debug_print=self.debug_print) # fails when `self.pf.ratemap.n_neurons == 0` aka `self.pf.ratemap.ndim == 0`
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
    def perform_decode_specific_epochs(cls, active_decoder, spikes_df, filter_epochs, decoding_time_bin_size=0.05, use_single_time_bin_per_epoch: bool=False, debug_print=False) -> DecodedFilterEpochsResult:
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
        spkcount, nbins, time_bin_containers_list = epochs_spkcount(filter_epoch_spikes_df, filter_epochs, decoding_time_bin_size, slideby=decoding_time_bin_size, export_time_bins=True, included_neuron_ids=active_decoder.neuron_IDs, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=debug_print) ## time_bins returned are not correct, they're subsampled at a rate of 1000
        num_filter_epochs = len(nbins) # one for each epoch in filter_epochs

        filter_epochs_decoder_result.spkcount = spkcount
        filter_epochs_decoder_result.nbins = nbins
        filter_epochs_decoder_result.time_bin_containers = time_bin_containers_list
        filter_epochs_decoder_result.decoding_time_bin_size = decoding_time_bin_size
        filter_epochs_decoder_result.filter_epochs = filter_epochs
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
            if use_single_time_bin_per_epoch:
                assert curr_filter_epoch_time_bin_container.num_bins == 1
                curr_filter_epoch_time_bin_size: float = curr_filter_epoch_time_bin_container.edge_info.step # get the variable time_bin_size from the epoch object
            else:
                curr_filter_epoch_time_bin_size: float = decoding_time_bin_size
                
            most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = active_decoder.decode(curr_filter_epoch_spkcount, time_bin_size=curr_filter_epoch_time_bin_size, output_flat_versions=False, debug_print=debug_print)
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

    @function_attributes(short_name=None, tags=['filter', 'firing_rate', 'frate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-22 19:48', related_items=[])
    def filtered_by_frate(self, minimum_inclusion_fr_Hz: float = 5.0, debug_print=True):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)
        
        """
        return BasePositionDecoder.perform_filter_by_frate(self, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=debug_print)

    @classmethod
    def perform_filter_by_frate(cls, a_decoder, minimum_inclusion_fr_Hz: float = 5.0, debug_print=True):
        """ Filters the included neuron_ids by their `tuning_curve_unsmoothed_peak_firing_rates` (a property of their `.pf.ratemap`)
        minimum_inclusion_fr_Hz: float = 5.0
        modified_long_LR_decoder = filtered_by_frate(track_templates.long_LR_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True)
        
        Usage:
            minimum_inclusion_fr_Hz: float = 5.0
            filtered_decoder_list = [filtered_by_frate(a_decoder, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, debug_print=True) for a_decoder in (track_templates.long_LR_decoder, track_templates.long_RL_decoder, track_templates.short_LR_decoder, track_templates.short_RL_decoder)]

        """
        a_pf: PfND = a_decoder.pf # neuropy.analyses.placefields.PfND
        a_ratemap = a_pf.ratemap
        original_neuron_ids = deepcopy(a_ratemap.neuron_ids)
        is_aclu_included = (a_ratemap.tuning_curve_unsmoothed_peak_firing_rates >= minimum_inclusion_fr_Hz)
        included_aclus = np.array(a_ratemap.neuron_ids)[is_aclu_included]
        if debug_print:
            print(f'len(original_neuron_ids): {len(original_neuron_ids)}, len(included_aclus): {len(included_aclus)}')
        modified_decoder = a_decoder.get_by_id(included_aclus)
        return modified_decoder


    # ==================================================================================================================== #
    # HDF5 Methods:                                                                                                        #
    # ==================================================================================================================== #
    # def to_hdf(self, file_path, key: str, **kwargs):
    #         """ Saves the object to key in the hdf5 file specified by file_path
    #         Usage:
    #             hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
    #             _pfnd_obj: PfND = long_one_step_decoder_1D.pf
    #             _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
    #         """
    #         raise NotImplementedError # implementor must override!




# ==================================================================================================================== #
# Bayesian Decoder                                                                                                     #
# ==================================================================================================================== #
@custom_define(slots=False)
class BayesianPlacemapPositionDecoder(SerializedAttributesAllowBlockSpecifyingClass, BasePositionDecoder): # needs `HDFMixin, `
    """ Holds the placefields. Can be called on any spike data to compute the most likely position given the spike data.

    Used to try to decode everything in one go, meaning it took the parameters (like the time window) and the spikes to decode as well and did the computation internally, but the concept of a decoder is that it is a stateless object that can be called on any spike data to decode it, so this concept is depricated.

    Holds a PfND object in self.pf that is used for decoding.


    Call Hierarchy:
        Path 1:
            .decode_specific_epochs(...)
                BayesianPlacemapPositionDecoder.perform_decode_specific_epochs(...)
                    .decode(...)
        Path 2:
            .compute_all(...)
                .hyper_perform_decode(...)
                    .decode(...)
                .compute_corrected_positions(...)

    """
    ## Time Binning:
    time_bin_size: float = None # these are actually required, but not allowed to be missing default values because of the inheritance
    spikes_df: pd.DataFrame = None # these are actually required, but not allowed to be missing default values because of the inheritance

    time_binning_container: BinningContainer = None
    unit_specific_time_binned_spike_counts: np.ndarray = None
    total_spike_counts_per_window: np.ndarray = None

    ## Computed Results:
    flat_p_x_given_n: np.ndarray = None
    p_x_given_n: np.ndarray = None
    most_likely_position_flat_indicies: np.ndarray = None
    most_likely_position_indicies: np.ndarray = None
    marginal: DynamicContainer = None
    most_likely_positions: np.ndarray = None
    revised_most_likely_positions: np.ndarray = None


    # time_binning_container accessors ___________________________________________________________________________________ #
    @property
    def time_window_edges(self):
        return self.time_binning_container.edges
    @property
    def time_window_edges_binning_info(self):
        return self.time_binning_container.edge_info

    @property
    def time_window_centers(self):
        return self.time_binning_container.centers

    @property
    def time_window_center_binning_info(self):
        return self.time_binning_container.center_info

    @property
    def num_time_windows(self):
        """The num_time_windows property."""
        return self.time_window_center_binning_info.num_bins

    @property
    def active_time_windows(self):
        """The num_time_windows property."""        
        window_starts = self.time_window_centers - (self.time_bin_size / 2.0)
        window_ends = self.time_window_centers + (self.time_bin_size / 2.0)
        active_time_windows = [(window_starts[i], window_ends[i]) for i in self.time_window_center_binning_info.bin_indicies]
        return active_time_windows
    
    @property
    def active_time_window_centers(self):
        """The active_time_window_centers property are the center timepoints for each window. """        
        window_starts = self.time_window_centers - (self.time_bin_size / 2.0)
        window_ends = self.time_window_centers + (self.time_bin_size / 2.0)
        active_window_midpoints = window_starts + ((window_ends - window_starts) / 2.0)
        return active_window_midpoints
    
    @property
    def is_non_firing_time_bin(self):
        """A boolean array that indicates whether each time window has no spikes. Requires self.total_spike_counts_per_window"""
        return (self.total_spike_counts_per_window == 0)
        # return np.where(self.total_spike_counts_per_window == 0)[0]
            

    @classmethod
    def serialized_key_allowlist(cls):
        input_keys = ['time_bin_size', 'pf', 'spikes_df', 'debug_print']
        # intermediate_keys = ['unit_specific_time_binned_spike_counts', 'time_window_edges', 'time_window_edges_binning_info']
        saved_result_keys = ['flat_p_x_given_n']
        return input_keys + saved_result_keys
    
    @classmethod
    def from_dict(cls, val_dict):
        new_obj = BayesianPlacemapPositionDecoder(time_bin_size=val_dict.get('time_bin_size', 0.25), pf=val_dict.get('pf', None), spikes_df=val_dict.get('spikes_df', None), setup_on_init=val_dict.get('setup_on_init', True), post_load_on_init=val_dict.get('post_load_on_init', False), debug_print=val_dict.get('debug_print', False))
        return new_obj

    
    # ==================================================================================================================== #
    # Methods                                                                                                              #
    # ==================================================================================================================== #
    
    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        with WrappingMessagePrinter(f'post_load() called.', begin_line_ending='... ', finished_message='all rebuilding completed.', enable_print=self.debug_print):
            self._setup_computation_variables()
            self._setup_time_bin_spike_counts_N_i()
            # self._setup_time_window_centers()
            self.p_x_given_n = self._reshape_output(self.flat_p_x_given_n)
            self.compute_most_likely_positions()

    def setup(self):
        # This version should override the base class version to finish the more extended setup of the new properties
        self.neuron_IDXs = None
        self.neuron_IDs = None
        self.F = None
        self.P_x = None
        
        self._setup_computation_variables()
        # Could pre-filter the self.spikes_df by the 
        
        self.time_binning_container = None
        self.unit_specific_time_binned_spike_counts = None
        self.total_spike_counts_per_window = None
        
        self._setup_time_bin_spike_counts_N_i()
        
        # pre-allocate outputs:
        self.flat_p_x_given_n = None # np.zeros((self.flat_position_size, self.num_time_windows))
        self.p_x_given_n = None
        self.most_likely_position_flat_indicies = None
        self.most_likely_position_indicies = None
        self.marginal = None 
        
    def debug_dump_print(self):
        """ dumps the state for debugging purposes """
        variable_names_dict = dict(summary = ['ndim', 'num_neurons', 'num_time_windows'],
            time_variable_names = ['time_bin_size', 'time_window_edges', 'time_window_edges_binning_info', 'time_window_centers','time_window_center_binning_info'],
            binned_spikes = ['unit_specific_time_binned_spike_counts', 'total_spike_counts_per_window'],
            intermediate_computations = ['F', 'P_x'],
            posteriors = ['p_x_given_n'],
            most_likely = ['most_likely_positions'],
            marginals = ['marginal'],
            other_variables = ['neuron_IDXs', 'neuron_IDs']
        )
        for a_category_name, variable_names_list in variable_names_dict.items():
            print(f'# {a_category_name}:')
            # print(f'\t {variable_names_list}:')
            for a_variable_name in variable_names_list:
                a_var_value = getattr(self, a_variable_name)
                a_var_shape = safe_get_variable_shape(a_var_value)
                if a_var_shape is None:
                    if isinstance(a_var_value, (int, float, np.number)):
                        a_var_shape = a_var_value # display the value directly if we can
                    else:
                        a_var_shape = 'SCALAR' # otherwise just output the literal text "SCALAR"
                print(f'\t {a_variable_name}: {a_var_shape}')

        # TODO: Handle marginals:
        # self.marginal.x

    # External Updating __________________________________________________________________________________________________ #

    # for NeuronUnitSlicableObjectProtocol:
    def get_by_id(self, ids, defer_compute_all:bool=False):
        """Implementors return a copy of themselves with neuron_ids equal to ids
            Needs to update: neuron_sliced_decoder.pf, ... (much more)

            defer_compute_all: bool - should be set to False if you want to manually decode using custom epochs or something later. Otherwise it will compute for all spikes automatically.
        """
        neuron_sliced_decoder = super().get_by_id(ids, defer_compute_all=defer_compute_all)
        ## Recompute:
        if not defer_compute_all:
            neuron_sliced_decoder.compute_all() # does recompute, updating internal variables. TODO EFFICIENCY 2023-03-02 - This is overkill and I could filter the tuning_curves and etc directly, but this is easier for now. 
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
        self, did_recompute = super().conform_to_position_bins(target_one_step_decoder, force_recompute=force_recompute)
        if did_recompute or (self.p_x_given_n.shape[0] < target_one_step_decoder.p_x_given_n.shape[0]):
            self.setup() # re-setup the decoder
            self.compute_all()
            did_recompute = True
        return self, did_recompute


    def add_two_step_decoder_results(self, two_step_decoder_result):
        """ adds the results from the computed two_step_decoder to self (the one_step_decoder)
        ## In this new mode we'll add the two-step properties to the original one-step decoder:
        ## Adds the directly accessible properties to the active_one_step_decoder after they're computed in the active_two_step_decoder so that they can be plotted with the same functions/etc.

        Inputs:
            two_step_decoder_result: computation_result.computed_data[two_step_decoder_key]
 
        """
        
        # None initialize two-step properties on the one_step_decoder:
        self.p_x_given_n_and_x_prev = None
        self.two_step_most_likely_positions = None

        self.marginal.x.p_x_given_n_and_x_prev = None
        self.marginal.x.two_step_most_likely_positions_1D = None

        if self.marginal.y is not None:
            self.marginal.y.p_x_given_n_and_x_prev = None
            self.marginal.y.two_step_most_likely_positions_1D = None

        # Set the two-step properties on the one-step decoder:
        self.p_x_given_n_and_x_prev = two_step_decoder_result.p_x_given_n_and_x_prev.copy()
        self.two_step_most_likely_positions = two_step_decoder_result.most_likely_positions.copy()

        self.marginal.x.p_x_given_n_and_x_prev = two_step_decoder_result.marginal.x.p_x_given_n.copy()
        self.marginal.x.two_step_most_likely_positions_1D = two_step_decoder_result.marginal.x.most_likely_positions_1D.copy()

        if self.marginal.y is not None:
            self.marginal.y.p_x_given_n_and_x_prev = two_step_decoder_result.marginal.y.p_x_given_n.copy()
            self.marginal.y.two_step_most_likely_positions_1D = two_step_decoder_result.marginal.y.most_likely_positions_1D.copy()

    @function_attributes(short_name='to_1D_maximum_projection', tags=['updated'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-06 22:06')
    def to_1D_maximum_projection(self, defer_compute_all:bool=True):
        """ returns a copy of the decoder that is 1D """
        # Perform the projection. Can only be ran once.
        new_copy_decoder = super().to_1D_maximum_projection(defer_compute_all=defer_compute_all)
        # new_copy_decoder.setup()
        if not defer_compute_all:
            new_copy_decoder.compute_all() # Needed?
        return new_copy_decoder

    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    def _setup_time_bin_spike_counts_N_i(self, debug_print=False):
        """ updates: 
        
        .time_binning_container
        # .time_window_edges, 
        # .time_window_edges_binning_info
        
        .unit_specific_time_binned_spike_counts
        .total_spike_counts_per_window
        
        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        ## need to create new time_window_edges from the self.time_bin_size:
        if debug_print:
            print(f'WARNING: _setup_time_bin_spike_counts_N_i(): updating self.time_window_edges and self.time_window_edges_binning_info ...')

        self.unit_specific_time_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.spikes_df, time_bin_size=self.time_bin_size, debug_print=self.debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
        if debug_print:
            print(f'from ._setup_time_bin_spike_counts_N_i():')
            print(f'\ttime_window_edges.shape: {time_window_edges.shape}') #  (11882,)
            print(f'unit_specific_time_binned_spike_counts.shape: {self.unit_specific_time_binned_spike_counts.shape}') # (70, 11881)
        
        self.time_binning_container = BinningContainer(edges=time_window_edges, edge_info=time_window_edges_binning_info)
        
        # Here we should filter the outputs by the actual self.neuron_IDXs
        # assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        if np.shape(self.unit_specific_time_binned_spike_counts)[0] > len(self.neuron_IDXs):
            # Drop the irrelevant indicies:
            self.unit_specific_time_binned_spike_counts = self.unit_specific_time_binned_spike_counts[self.neuron_IDXs,:] # Drop the irrelevent indicies
        
        assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        self.total_spike_counts_per_window = np.sum(self.unit_specific_time_binned_spike_counts, axis=0) # gets the total number of spikes during each window (across all placefields)

    def _reshape_output(self, output_probability):
        return np.reshape(output_probability, (*self.original_position_data_shape, self.num_time_windows)) # changed for compatibility with 1D decoder
    
    def _flatten_output(self, output):
        """ the inverse of _reshape_output(output) to flatten the position coordinates into a single flat dimension """
        return np.reshape(output, (self.flat_position_size, self.num_time_windows)) # Trying to flatten the position coordinates into a single flat dimension
    
    
    
    # ==================================================================================================================== #
    # Main computation functions:                                                                                          #
    # ==================================================================================================================== #

    def compute_all(self, debug_print=False):
        """ computes all the outputs of the decoder, and stores them in the class instance 

        Uses:
            self.hyper_perform_decode(...)
            self.compute_corrected_positions(...)

        TODO 2023-05-16 - CHECK - CORRECTNESS - Figure out correct indexing and whether it returns binned data for edges or counts.
        
        """
        with WrappingMessagePrinter(f'compute_all final_p_x_given_n called. Computing windows...', begin_line_ending='... ', finished_message='compute_all completed.', enable_print=(debug_print or self.debug_print)):
            ## Single sweep decoding:

            ## 2022-09-23 - Epochs-style encoding (that works):
            self.time_binning_container, self.p_x_given_n, self.most_likely_positions, curr_unit_marginal_x, curr_unit_marginal_y, flat_outputs_container = self.hyper_perform_decode(self.spikes_df, decoding_time_bin_size=self.time_bin_size, output_flat_versions=True, debug_print=(debug_print or self.debug_print))
            self.marginal = DynamicContainer(x=curr_unit_marginal_x, y=curr_unit_marginal_y)
            assert isinstance(self.time_binning_container, BinningContainer) # Should be neuropy.utils.mixins.binning_helpers.BinningContainer
            self.compute_corrected_positions() ## this seems to fail for pf1D
            
            ## set flat properties for compatibility (I guess)
            self.flat_p_x_given_n = flat_outputs_container.flat_p_x_given_n
            self.most_likely_position_flat_indicies = flat_outputs_container.most_likely_position_flat_indicies

    def compute_most_likely_positions(self):
        """ Computes the most likely positions at each timestep from self.flat_p_x_given_n """        
        raise NotImplementedError
        self.most_likely_position_flat_indicies, self.most_likely_position_indicies = self.perform_compute_most_likely_positions(self.flat_p_x_given_n, self.original_position_data_shape)
        # np.shape(self.most_likely_position_flat_indicies) # (85841,)
        # np.shape(self.most_likely_position_indicies) # (2, 85841)

    def compute_corrected_positions(self):
        """ computes the revised most likely positions by taking into account the time-bins that had zero spikes and extrapolating position from the prior successfully decoded time bin
        
        Requires:
            .total_spike_counts_per_window
            .most_likely_positions
            
        Updates:
            .revised_most_likely_positions
            .marginal's .x & .y .revised_most_likely_positions_1D
        


        TODO: CRITICAL: CORRECTNESS: 2022-02-25: This was said not to be working for 1D somewhere else in the code, but I don't know if it's working or not. It doesn't seem to be.

        """
        ## Find the bins that don't have any spikes in them:
        # zero_bin_indicies = np.where(self.total_spike_counts_per_window == 0)[0]
        # is_non_firing_bin = self.is_non_firing_time_bin
        
        is_non_firing_bin = np.where(self.is_non_firing_time_bin)[0] # TEMP: do this to get around the indexing issue. TODO: IndexError: boolean index did not match indexed array along dimension 0; dimension is 11880 but corresponding boolean dimension is 11881
        self.revised_most_likely_positions = self.perform_compute_forward_filled_positions(self.most_likely_positions, is_non_firing_bin=is_non_firing_bin)
        
        if self.marginal is not None:
            _revised_marginals = self.perform_build_marginals(self.p_x_given_n, self.revised_most_likely_positions, debug_print=False) # Stupid way of doing this, but w/e
            if self.marginal.x is not None:
                # self.marginal.x.revised_most_likely_positions_1D = self.perform_compute_forward_filled_positions(self.marginal.x.most_likely_positions_1D, is_non_firing_bin=is_non_firing_bin)
                self.marginal.x.revised_most_likely_positions_1D = _revised_marginals[0].most_likely_positions_1D.copy()
            if self.marginal.y is not None:
                # self.marginal.y.revised_most_likely_positions_1D = self.perform_compute_forward_filled_positions(self.marginal.y.most_likely_positions_1D, is_non_firing_bin=is_non_firing_bin)
                self.marginal.y.revised_most_likely_positions_1D =  _revised_marginals[1].most_likely_positions_1D.copy()

        return self.revised_most_likely_positions


    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #


    # `perform_compute_spike_count_and_firing_rate_normalizations` is the only classmethod not ported over to the new stateless class
    @classmethod
    def perform_compute_spike_count_and_firing_rate_normalizations(cls, pho_custom_decoder):
        """ Computes several different normalizations of binned firing rate and spike counts
        
        Usage:
            pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
            unit_specific_time_binned_outputs = perform_compute_spike_count_and_firing_rate_normalizations(pho_custom_decoder)
            spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = unit_specific_time_binned_outputs # unwrap the output tuple
            
            
        TESTING CODE:
        
            pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
            print(f'most_likely_positions: {np.shape(pho_custom_decoder.most_likely_positions)}') # most_likely_positions: (3434, 2)
            unit_specific_time_binned_outputs = perform_compute_spike_count_and_firing_rate_normalizations(pho_custom_decoder)
            spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = unit_specific_time_binned_outputs # unwrap the output tuple:

            # pho_custom_decoder.unit_specific_time_binned_spike_counts.shape # (64, 1717)
            unit_specific_binned_spike_count_mean = np.nanmean(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
            unit_specific_binned_spike_count_var = np.nanvar(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
            unit_specific_binned_spike_count_median = np.nanmedian(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)

            unit_specific_binned_spike_count_mean
            unit_specific_binned_spike_count_median
            # unit_specific_binned_spike_count_mean.shape # (64, )

        """
        # produces a fraction which indicates which proportion of the window's firing belonged to each unit (accounts for global changes in firing rate (each window is scaled by the toial spikes of all cells in that window)
        unit_specific_time_binned_spike_proportion_global_fr_normalized = pho_custom_decoder.unit_specific_time_binned_spike_counts / pho_custom_decoder.total_spike_counts_per_window
        unit_specific_time_binned_firing_rate = pho_custom_decoder.unit_specific_time_binned_spike_counts / pho_custom_decoder.time_window_edges_binning_info.step
        # produces a unit firing rate for each window that accounts for global changes in firing rate (each window is scaled by the firing rate of all cells in that window
        unit_specific_time_binned_firing_rate_global_fr_normalized = unit_specific_time_binned_spike_proportion_global_fr_normalized / pho_custom_decoder.time_window_edges_binning_info.step
        # Return the computed values, leaving the original data unchanged.
        return unit_specific_time_binned_spike_proportion_global_fr_normalized, unit_specific_time_binned_firing_rate, unit_specific_time_binned_firing_rate_global_fr_normalized

