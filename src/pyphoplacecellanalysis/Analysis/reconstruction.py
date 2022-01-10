# methods of reconstruction/decoding

import numpy as np
import pandas as pd

from pyphocorehelpers.indexing_helpers import build_spanning_bins
from pyphocorehelpers.print_helpers import WrappingMessagePrinter

# cut_bins = np.linspace(59200, 60800, 9)
# pd.cut(df['column_name'], bins=cut_bins)

# # just want counts of number of occurences of each?
# df['column_name'].value_counts(bins=8, sort=False)



class ZhangReconstructionImplementation:
    def n_i(cell_idx_i, time_window):
        """ number of spikes fired by cell i within the time window """
        pass
    
    def phi_i(cell_idx_i, x):
        """ an arbitrary basis function or template function associated with this cell """
        pass

    def distribution(x):
        """ x is 2D """
        pass


    # Shared:    
    @staticmethod
    def compute_time_binned_spiking_activity(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
        """Given a spikes dataframe, this function temporally bins the spikes, counting the number that fall into each bin.

        Args:
            spikes_df ([type]): [description]
            time_bin_size ([type]): [description]
            debug_print (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
            
        Usage:
            time_bin_size=0.02

            curr_result_label = 'maze1'
            sess = curr_kdiba_pipeline.filtered_sessions[curr_result_label]
            pf = curr_kdiba_pipeline.computation_results[curr_result_label].computed_data['pf1D']

            spikes_df = sess.spikes_df.copy()
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = compute_time_binned_spiking_activity(spikes_df, time_bin_size)

        """
        time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'

        out_digitized_variable_bins, out_binning_info = build_spanning_bins(spikes_df[time_variable_name].to_numpy(), max_bin_size=max_time_bin_size, debug_print=debug_print) # np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]
        if debug_print:
            print(f'spikes_df[time_variable_name]: {np.shape(spikes_df[time_variable_name])}\nout_digitized_variable_bins: {np.shape(out_digitized_variable_bins)}')
            # assert (np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]), f'np.shape(out_digitized_variable_bins)[0]: {np.shape(out_digitized_variable_bins)[0]} should equal np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]}'
            print(out_binning_info)

        # any_unit_spike_counts = spikes_df[time_variable_name].value_counts(bins=out_binning_info.num_bins, sort=False) # fast way to get the binned counts across all cells

        spikes_df['binned_time'] = pd.cut(spikes_df[time_variable_name].to_numpy(), bins=out_digitized_variable_bins, include_lowest=True, labels=out_binning_info.bin_indicies[1:]) # same shape as the input data (time_binned_spikes_df: (69142,))

        # any_unit_spike_counts = spikes_df.groupby(['binned_time'])[time_variable_name].agg('count') # unused any cell spike counts
        
        unit_specific_bin_specific_spike_counts = spikes_df.groupby(['aclu','binned_time'])[time_variable_name].agg('count')
        active_aclu_binned_time_multiindex = unit_specific_bin_specific_spike_counts.index
        active_unique_aclu_values = np.unique(active_aclu_binned_time_multiindex.get_level_values('aclu'))
        unit_specific_binned_spike_counts = np.array([unit_specific_bin_specific_spike_counts[aclu].values for aclu in active_unique_aclu_values]).T # (85841, 40)
        if debug_print:
            print(f'np.shape(unit_specific_spike_counts): {np.shape(unit_specific_binned_spike_counts)}') # np.shape(unit_specific_spike_counts): (40, 85841)

        unit_specific_binned_spike_counts = pd.DataFrame(unit_specific_binned_spike_counts, columns=active_unique_aclu_values, index=out_binning_info.bin_indicies[1:])
        # unit_specific_spike_counts.get_group(2)

        # spikes_df.groupby(['binned_time']).agg('count')

        # for name, group in spikes_df.groupby(['aclu','binned_time']):
        #     print(f'name: {name}, group: {group}') 

        # neuron_ids, neuron_specific_spikes_dfs = partition(spikes_df, 'aclu')
        # spikes_df.groupby(['aclu','binned_time'])
        # groups.size().unstack()
        # spikes_df._obj.groupby(['aclu'])
        # neuron_ids, neuron_specific_spikes_dfs = partition(spikes_df, 'aclu')

        return unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info

    @staticmethod
    def build_concatenated_F(pf, debug_print=False):
        neuron_IDs = pf.ratemap.neuron_ids
        neuron_IDXs = np.arange(len(neuron_IDs))
        maps = pf.ratemap.normalized_tuning_curves  # (40, 48) for 1D, (40, 48, 10) for 2D
        if debug_print:
            print(f'maps: {np.shape(maps)}') # maps: (40, 48, 10)
        f_i = [np.squeeze(maps[i,:,:]) for i in neuron_IDXs] # produces (48 x 10) map
        if debug_print:
            print(f'np.shape(f_i[i]): {np.shape(f_i[0])}') # (48, 6)
        F_i = [np.reshape(f_i[i], (-1, 1)) for i in neuron_IDXs] # Convert each function to a column vector
        if debug_print:
            print(f'np.shape(F_i[i]): {np.shape(F_i[0])}') # (288, 1)
        F = np.hstack(F_i) # Concatenate each individual F_i to produce F
        if debug_print:
            print(f'np.shape(F): {np.shape(F)}') # (288, 40)
        P_x = np.reshape(pf.occupancy, (-1, 1)) # occupancy gives the P(x) in general.
        if debug_print:
            print(f'np.shape(P_x): {np.shape(P_x)}') # np.shape(P_x): (48, 6)

        return neuron_IDXs, neuron_IDs, f_i, F_i, F, P_x


    @staticmethod
    def time_bin_spike_counts_N_i(spikes_df, time_bin_size, debug_print=False):
        """ Returns the number of spikes that occured for each neuron in each time bin. """
        unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(spikes_df, time_bin_size);
        unit_specific_binned_spike_counts = unit_specific_binned_spike_counts.T # Want the outputs to have each time window as a column, with a single time window giving a column vector for each neuron
        if debug_print:
            print(f'unit_specific_binned_spike_counts.to_numpy(): {np.shape(unit_specific_binned_spike_counts.to_numpy())}') # (85841, 40)
        return unit_specific_binned_spike_counts.to_numpy(), out_digitized_variable_bins, out_binning_info


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


    # Bayesian Probabilistic Approach:
    @staticmethod
    def bayesian_prob(tau, P_x, F, n, debug_print=False):
        # n_i: the number of spikes fired by each cell during the time window of consideration
        assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}'

        # take n as a row vector, and repeat it vertically for each column.
        element_wise_n = np.tile(n, (np.shape(F)[0], 1)) # repeat n for each row (coresponding to a position x) in F.
        # repeats_array = np.tile(an_array, (repetitions, 1))
        if debug_print:
            print(f'np.shape(element_wise_n): {np.shape(element_wise_n)}') # np.shape(element_wise_n): (288, 40)

        # the inner expression np.power(F, element_wise_n) performs the element-wise exponentiation of F with the values in element_wise_n.
        # result = P_x * np.prod(np.power(F, element_wise_n), axis=1) # the product is over the neurons, so the second dimension
        term1 = np.squeeze(P_x) # np.shape(P_x): (48, 6)
        if debug_print:
            print(f'np.shape(term1): {np.shape(term1)}') # np.shape(P_x): (48, 6)
        term2 = np.prod(np.power(F, element_wise_n), axis=1) # np.shape(term2): (288,)
        if debug_print:
            print(f'np.shape(term2): {np.shape(term2)}') # np.shape(P_x): (48, 6)

        # result = C_tau_n * P_x
        term3 = np.exp(-tau * np.sum(F, axis=1)) # sum over all columns (corresponding to over all cells)
        if debug_print:
            print(f'np.shape(term3): {np.shape(term3)}') # np.shape(P_x): (48, 6)

        # each column_i of F, F[:,i] should be raised to the power of n_i[i]
        un_normalized_result = term1 * term2 * term3
        C_tau_n = 1.0 / np.sum(un_normalized_result) # normalize the result
        result = C_tau_n * un_normalized_result
        if debug_print:
            print(f'np.shape(result): {np.shape(result)}') # np.shape(P_x): (48, 6)
        """
            np.shape(term1): (288, 1)
            np.shape(term2): (288,)
            np.shape(term3): (288,)
            np.shape(result): (288, 288)
        """
        return result

    


class PlacemapPositionDecoder(object):
    """docstring for PlacemapPositionDecoder."""
    def __init__(self, time_bin_size: float, pf, spikes_df: pd.DataFrame, debug_print: bool=False):
        super(PlacemapPositionDecoder, self).__init__()
        self.time_bin_size = time_bin_size
        self.pf = pf
        self.spikes_df = spikes_df
        self.debug_print = debug_print
        # self.arg = arg
        
        # """Get binned spike counts

        # Parameters
        # ----------
        # bin_size : float, optional
        #     bin size in seconds, by default 0.25

        # Returns
        # -------
        # neuropy.core.BinnedSpiketrains

        # """

        # bins = np.arange(self.t_start, self.t_stop + bin_size, bin_size)
        # spike_counts = np.asarray([np.histogram(_, bins=bins)[0] for _ in self.spiketrains])
        
class BayesianPlacemapPositionDecoder(PlacemapPositionDecoder):
    """docstring for BayesianPlacemapPositionDecoder."""
    def __init__(self, *arg, **args):
        super(BayesianPlacemapPositionDecoder, self).__init__(*arg, **args)
        self.setup() # setup on run
        
    @property
    def flat_position_size(self):
        """The flat_position_size property."""
        return np.shape(self.F)[0] # like 288
    
    @property
    def num_time_windows(self):
        """The num_time_windows property."""
        return self.time_binning_info.num_bins
    
    
    def setup(self):        
        self.neuron_IDXs, self.neuron_IDs, f_i, F_i, self.F, self.P_x = ZhangReconstructionImplementation.build_concatenated_F(self.pf, debug_print=self.debug_print)
        """
            maps: (40, 48, 6)
            np.shape(f_i[i]): (48, 6)
            np.shape(F_i[i]): (288, 1)
            np.shape(F): (288, 40)
            np.shape(P_x): (288, 1)
        """
        self.unit_specific_time_binned_spike_counts, self.digitized_time_variable_bins, self.time_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.spikes_df, self.time_bin_size, debug_print=self.debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
        
        # pre-allocate:
        with WrappingMessagePrinter(f'pre-allocating final_p_x_given_n: np.shape(final_p_x_given_n) will be: ({self.flat_position_size} x {self.time_binning_info.num_bins})...', begin_line_ending='... ', enable_print=self.debug_print):
            # if self.debug_print:
            #     print(f'pre-allocating final_p_x_given_n: np.shape(final_p_x_given_n) will be: ({self.flat_position_size} x {self.time_binning_info.num_bins})...', end=' ') # np.shape(final_p_x_given_n): (288,)
            self.final_p_x_given_n = np.zeros((self.flat_position_size, self.time_binning_info.num_bins))
            # if self.debug_print:
            #     print('done.')

        
    def perform_compute_single_time_bin(self, time_window_idx):
        n = self.unit_specific_time_binned_spike_counts[:, time_window_idx]
        if self.debug_print:
            print(f'np.shape(n): {np.shape(n)}') # np.shape(n): (40,)
        final_p_x_given_n = ZhangReconstructionImplementation.bayesian_prob(self.time_bin_size, self.P_x, self.F, n, debug_print=self.debug_print) # np.shape(final_p_x_given_n): (288,)
        if self.debug_print:
            print(f'np.shape(final_p_x_given_n): {np.shape(self.final_p_x_given_n)}') # np.shape(final_p_x_given_n): (288,)
        return final_p_x_given_n
            
    def compute_all(self):
        with WrappingMessagePrinter(f'compute_all final_p_x_given_n called. Computing {np.shape(self.final_p_x_given_n)[0]} windows for self.final_p_x_given_n...', begin_line_ending='... ', finished_message='compute_all completed.', enable_print=self.debug_print):
            for bin_idx in self.time_binning_info.bin_indicies:
                with WrappingMessagePrinter(f'\t computing single final_p_x_given_n[:, {bin_idx}] for bin_idx {bin_idx}', begin_line_ending='... ', finished_message='', finished_line_ending='\n', enable_print=self.debug_print):
                    self.final_p_x_given_n[:, bin_idx] = self.perform_compute_single_time_bin(bin_idx)
            # # all computed
            # if self.debug_print:
            #     print('compute_all completed!')