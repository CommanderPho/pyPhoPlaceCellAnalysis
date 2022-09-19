from pathlib import Path
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.indexing_helpers import BinningInfo, compute_spanning_bins, get_bin_centers, build_spanning_grid_matrix
from pyphocorehelpers.print_helpers import WrappingMessagePrinter, SimplePrintable
from pyphocorehelpers.mixins.serialized import SerializedAttributesSpecifyingClass


from pyphocorehelpers.print_helpers import print_value_overview_only, print_keys_if_possible, debug_dump_object_member_shapes, safe_get_variable_shape



# cut_bins = np.linspace(59200, 60800, 9)
# pd.cut(df['column_name'], bins=cut_bins)

# # just want counts of number of occurences of each?
# df['column_name'].value_counts(bins=8, sort=False)

# methods of reconstruction/decoding:

""" 
occupancy gives the P(x) in general.
n_i: the number of spikes fired by each cell during the time window of consideration

"""

class ZhangReconstructionImplementation:

    # Shared:    
    @staticmethod
    def compute_time_bins(spikes_df, max_time_bin_size:float=0.02, debug_print=False):
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
        time_variable_name = spikes_df.spikes.time_variable_name # 't_rel_seconds'

        time_window_edges, time_window_edges_binning_info = compute_spanning_bins(spikes_df[time_variable_name].to_numpy(), bin_size=max_time_bin_size) # np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]
        
        if not (np.shape(time_window_edges)[0] < np.shape(spikes_df)[0]):
            print('WARNING: PREVIOUSLY ASSERT: ')
            print(f'\t spikes_df[time_variable_name]: {np.shape(spikes_df[time_variable_name])} should be less than time_window_edges: {np.shape(time_window_edges)}!')
        # assert np.shape(time_window_edges)[0] < np.shape(spikes_df)[0], f'spikes_df[time_variable_name]: {np.shape(spikes_df[time_variable_name])} should be less than time_window_edges: {np.shape(time_window_edges)}!'
        
        if debug_print:
            print(f'spikes_df[time_variable_name]: {np.shape(spikes_df[time_variable_name])}\ntime_window_edges: {np.shape(time_window_edges)}')
            # assert (np.shape(out_digitized_variable_bins)[0] == np.shape(spikes_df)[0]), f'np.shape(out_digitized_variable_bins)[0]: {np.shape(out_digitized_variable_bins)[0]} should equal np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]}'
            print(time_window_edges_binning_info)

        spikes_df['binned_time'] = pd.cut(spikes_df[time_variable_name].to_numpy(), bins=time_window_edges, include_lowest=True, labels=time_window_edges_binning_info.bin_indicies[1:]) # same shape as the input data (time_binned_spikes_df: (69142,))
        
        return time_window_edges, time_window_edges_binning_info, spikes_df
        
    
    @staticmethod
    def compute_unit_specific_bin_specific_spike_counts(spikes_df, time_bin_indicies, debug_print=False):
        """ 
        spikes_df: a dataframe with 
        time_bin_indicies = time_window_edges_binning_info.bin_indicies[1:]
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
        time_window_edges, time_window_edges_binning_info, spikes_df = ZhangReconstructionImplementation.compute_time_bins(spikes_df, max_time_bin_size=max_time_bin_size, debug_print=debug_print)
        unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies[1:], debug_print=debug_print)
        return unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info

    @staticmethod
    def build_concatenated_F(pf, debug_print=False):
        neuron_IDs = pf.ratemap.neuron_ids
        neuron_IDXs = np.arange(len(neuron_IDs))
        maps = pf.ratemap.normalized_tuning_curves  # (40, 48) for 1D, (40, 48, 10) for 2D
        
        ## 2022-09-19 - TODO: should this be the non-normalized tuning curves instead of the normalized ones?
        # e.g. maps = pf.ratemap.tuning_curves
        
        if debug_print:
            print(f'maps: {np.shape(maps)}') # maps: (40, 48, 10)
        f_i = [np.squeeze(maps[i,:,:]) for i in neuron_IDXs] # produces a list of (48 x 10) maps
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
    def time_bin_spike_counts_N_i(spikes_df, time_bin_size, time_window_edges=None, time_window_edges_binning_info=None, debug_print=False):
        """ Returns the number of spikes that occured for each neuron in each time bin.
        Example:
            unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
            
        """
        if time_window_edges is None or time_window_edges_binning_info is None:    
            time_window_edges, time_window_edges_binning_info, spikes_df = ZhangReconstructionImplementation.compute_time_bins(spikes_df, max_time_bin_size=time_bin_size, debug_print=debug_print)
            unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies[1:], debug_print=debug_print)
            ## Old Method:
            # unit_specific_binned_spike_counts, time_window_edges, time_window_edges_binning_info = ZhangReconstructionImplementation.compute_time_binned_spiking_activity(spikes_df, time_bin_size);
        else:
            # already have time bins:
            unit_specific_binned_spike_counts = ZhangReconstructionImplementation.compute_unit_specific_bin_specific_spike_counts(spikes_df, time_window_edges_binning_info.bin_indicies[1:], debug_print=debug_print)
        
        unit_specific_binned_spike_counts = unit_specific_binned_spike_counts.T # Want the outputs to have each time window as a column, with a single time window giving a column vector for each neuron
        if debug_print:
            print(f'unit_specific_binned_spike_counts.to_numpy(): {np.shape(unit_specific_binned_spike_counts.to_numpy())}') # (85841, 40)
        return unit_specific_binned_spike_counts.to_numpy(), time_window_edges, time_window_edges_binning_info


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
        assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'
        
        # total_number_spikes_n = np.sum(n) # the total number of spikes across all placecells during this timewindow
        
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



class Zhang_Two_Step:
    
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


    
    
# input_keys = ['pf']
# recomputed_input_keys = ['neuron_IDXs', 'neuron_IDs', 'F', 'P_x']
# intermediate_keys = ['original_position_data_shape']
# saved_result_keys = ['flat_p_x_given_n']
# recomputed_keys = ['p_x_given_n']

    
class PlacemapPositionDecoder(SerializedAttributesSpecifyingClass, SimplePrintable, object, metaclass=OrderedMeta):
    """docstring for PlacemapPositionDecoder.
    
    Call flow:
        ## Init/Setup Section:
        .__init__()
            .setup()
            .post_load() - Called after deserializing/loading saved result from disk to rebuild the needed computed variables. 
                self._setup_concatenated_F()
                self._setup_time_bin_spike_counts_N_i()
                self._setup_time_window_centers()
    
    
        ## Computation Section
        .compute_all()
            .perform_compute_single_time_bin(...)
            .perform_compute_most_likely_positions(...)
    """     
    
    def __init__(self, time_bin_size: float, pf, spikes_df: pd.DataFrame, manual_time_window_edges=None, manual_time_window_edges_binning_info:BinningInfo=None, setup_on_init:bool=True, post_load_on_init:bool=False, debug_print:bool=False):
        super(PlacemapPositionDecoder, self).__init__()
        self.time_bin_size = time_bin_size
        self.pf = pf
        self.spikes_df = spikes_df
        
        self.time_window_edges = manual_time_window_edges
        self.time_window_edges_binning_info = manual_time_window_edges_binning_info
        
        
        self.debug_print = debug_print
        if setup_on_init:
            self.setup() # setup on init
        if post_load_on_init:
            self.post_load()
            
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__.keys()};>"
    
    def setup(self):
        raise NotImplementedError
    
    def post_load(self):
        raise NotImplementedError
        
    @classmethod
    def serialized_keys(cls):
        input_keys = ['time_bin_size', 'pf', 'spikes_df', 'debug_print']
        return input_keys
    
    def to_dict(self):
        # return {member:self.__dict__[member] for member in PlacemapPositionDecoder._orderedKeys}  
        return self.__dict__
        # for member in PlacemapPositionDecoder._orderedKeys:
        #     if not getattr(PlacemapPositionDecoder, member):
        #         print(member)
        
    # @classmethod
    # def from_dict(cls, val_dict):
    #     return cls(val_dict.get('time_bin_size', 0.25), val_dict.get('pf', None), val_dict.get('spikes_df', None), setup_on_init=val_dict.get('setup_on_init', True), post_load_on_init=val_dict.get('post_load_on_init', False), debug_print=val_dict.get('debug_print', False))

        
    ## FileRepresentable protocol:
    @classmethod
    def from_file(cls, f):
        if f.is_file():
            dict_rep = None
            dict_rep = np.load(f, allow_pickle=True).item()
            if dict_rep is not None:
                # Convert to object
                dict_rep['setup_on_init'] = False
                dict_rep['post_load_on_init'] = False # set that to false too
                obj = cls.from_dict(dict_rep)
                post_load_dict = {k: v for k, v in dict_rep.items() if k in ['flat_p_x_given_n']}
                print(f'post_load_dict: {post_load_dict.keys()}')
                obj.flat_p_x_given_n = post_load_dict['flat_p_x_given_n']
                obj.post_load() # call the post_load function to update all the required variables
                return obj
            return dict_rep
        else:
            return None
        
    @classmethod
    def to_file(cls, data: dict, f, status_print=True):
        assert (f is not None), "WARNING: filename can not be None"
        if isinstance(f, str):
            f = Path(f) # conver to pathlib path
        assert isinstance(f, Path)
    
        with WrappingMessagePrinter(f'saving obj to file f: {str(f)}', begin_line_ending='... ', finished_message=f"{f.name} saved", enable_print=status_print):
            if not f.parent.exists():
                with WrappingMessagePrinter(f'parent path: {str(f.parent)} does not exist. Creating', begin_line_ending='... ', finished_message=f"{str(f.parent)} created.", enable_print=status_print):
                    f.parent.mkdir(parents=True, exist_ok=True) 
            np.save(f, data)

            
    def save(self, f, status_print=True, debug_print=False):
        active_serialization_keys = self.__class__.serialized_keys()
        all_data = self.to_dict()
        data = { a_serialized_key: all_data[a_serialized_key] for a_serialized_key in active_serialization_keys} # only get the serialized keys
        if debug_print:
            print(f'all_data.keys(): {all_data.keys()}, serialization_only_data.keys(): {data.keys()}')
        self.__class__.to_file(data, f, status_print=status_print)
    
        
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
    """docstring for BayesianPlacemapPositionDecoder.
    
    
    Called after deserializing/loading saved result from disk to rebuild the needed computed variables. 
    
    """
    
    
    @property
    def flat_position_size(self):
        """The flat_position_size property."""
        return np.shape(self.F)[0] # like 288
    @property
    def original_position_data_shape(self):
        """The original_position_data_shape property."""
        return np.shape(self.pf.occupancy)

   
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
    def most_likely_positions(self):
        """The most_likely_positions for each window."""
        return np.vstack((self.xbin_centers[self.most_likely_position_indicies[0,:]], self.ybin_centers[self.most_likely_position_indicies[1,:]])).T # much more efficient than the other implementation. Result is # (85844, 2)
    
    # placefield properties:
    @property
    def ratemap(self):
        return self.pf.ratemap
            
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
    
    @classmethod
    def serialized_keys(cls):
        input_keys = ['time_bin_size', 'pf', 'spikes_df', 'debug_print']
        # intermediate_keys = ['unit_specific_time_binned_spike_counts', 'time_window_edges', 'time_window_edges_binning_info']
        saved_result_keys = ['flat_p_x_given_n']
        return input_keys + saved_result_keys
    
    @classmethod
    def from_dict(cls, val_dict):
        new_obj = BayesianPlacemapPositionDecoder(val_dict.get('time_bin_size', 0.25), val_dict.get('pf', None), val_dict.get('spikes_df', None), setup_on_init=val_dict.get('setup_on_init', True), post_load_on_init=val_dict.get('post_load_on_init', False), debug_print=val_dict.get('debug_print', False))
        return new_obj
    
    
    
    # ==================================================================================================================== #
    # Methods                                                                                                              #
    # ==================================================================================================================== #
    
    def __init__(self, time_bin_size: float, pf, spikes_df: pd.DataFrame, manual_time_window_edges=None, manual_time_window_edges_binning_info:BinningInfo=None, setup_on_init:bool=True, post_load_on_init:bool=False, debug_print:bool=True):
        super(BayesianPlacemapPositionDecoder, self).__init__(time_bin_size, pf, spikes_df, manual_time_window_edges=manual_time_window_edges, manual_time_window_edges_binning_info=manual_time_window_edges_binning_info, setup_on_init=setup_on_init, post_load_on_init=post_load_on_init, debug_print=debug_print)
        

    def post_load(self):
        """ Called after deserializing/loading saved result from disk to rebuild the needed computed variables. """
        with WrappingMessagePrinter(f'post_load() called.', begin_line_ending='... ', finished_message='all rebuilding completed.', enable_print=self.debug_print):
            self._setup_concatenated_F()
            self._setup_time_bin_spike_counts_N_i()
            self._setup_time_window_centers()
            self.p_x_given_n = self._reshaped_output(self.flat_p_x_given_n)
            self.perform_compute_most_likely_positions()
    
    
    def setup(self):        
        self._setup_concatenated_F()
        # Could pre-filter the self.spikes_df by the 
        
        self._setup_time_bin_spike_counts_N_i()
        self._setup_time_window_centers()
        # pre-allocate outputs:
        self._setup_preallocate_outputs()
        
        
    def debug_dump_print(self):
        """ dumps the state for debugging purposes """
        variable_names_dict = dict(time_variable_names = ['time_bin_size', 'time_window_edges', 'time_window_edges_binning_info', 'total_spike_counts_per_window', 'time_window_centers','time_window_center_binning_info'],
            binned_spikes = ['unit_specific_time_binned_spike_counts', 'total_spike_counts_per_window'],
            intermediate_computations = ['F', 'P_x'],
            posteriors = ['p_x_given_n'],
            other_variables = ['neuron_IDXs', 'neuron_IDs']
        )
        for a_category_name, variable_names_list in variable_names_dict.items():
            print(f'# {a_category_name}:')
            # print(f'\t {variable_names_list}:')
            for a_variable_name in variable_names_list:
                a_var_value = getattr(self, a_variable_name)
                a_var_shape = safe_get_variable_shape(a_var_value) or 'SCALAR'
                print(f'\t {a_variable_name}: {a_var_shape}')

        
    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    def _setup_concatenated_F(self):
        """
            maps: (40, 48, 6)
            np.shape(f_i[i]): (48, 6)
            np.shape(F_i[i]): (288, 1)
            np.shape(F): (288, 40)
            np.shape(P_x): (288, 1)
        """
        self.neuron_IDXs, self.neuron_IDs, f_i, F_i, self.F, self.P_x = ZhangReconstructionImplementation.build_concatenated_F(self.pf, debug_print=self.debug_print)
        
    def _setup_time_bin_spike_counts_N_i(self):
        # Check if we already have self.time_window_edges and self.time_window_edges_binning_info to use
        if (self.time_window_edges is not None) and (self.time_window_edges_binning_info is not None):
            ## Already have time_window_edges to use, do not create new ones:
            assert self.time_window_edges_binning_info.step == self.time_bin_size
            self.unit_specific_time_binned_spike_counts, _, _ = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.spikes_df, self.time_bin_size, time_window_edges=self.time_window_edges, time_window_edges_binning_info=self.time_window_edges_binning_info, debug_print=self.debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
            
        else:
            ## need to create new time_window_edges from the self.time_bin_size:
            self.unit_specific_time_binned_spike_counts, self.time_window_edges, self.time_window_edges_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.spikes_df, self.time_bin_size, debug_print=self.debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
        
        # Here we should filter the outputs by the actual self.neuron_IDXs
        # assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        if np.shape(self.unit_specific_time_binned_spike_counts)[0] > len(self.neuron_IDXs):
            # Drop the irrelevant indicies:
            self.unit_specific_time_binned_spike_counts = self.unit_specific_time_binned_spike_counts[self.neuron_IDXs,:] # Drop the irrelevent indicies
        
        assert np.shape(self.unit_specific_time_binned_spike_counts)[0] == len(self.neuron_IDXs), f"in _setup_time_bin_spike_counts_N_i(): output should equal self.neuronIDXs but np.shape(self.unit_specific_time_binned_spike_counts)[0]: {np.shape(self.unit_specific_time_binned_spike_counts)[0]} and len(self.neuron_IDXs): {len(self.neuron_IDXs)}"
        self.total_spike_counts_per_window = np.sum(self.unit_specific_time_binned_spike_counts, axis=0) # gets the total number of spikes during each window (across all placefields)

    def _setup_preallocate_outputs(self):
        with WrappingMessagePrinter(f'pre-allocating final_p_x_given_n: np.shape(final_p_x_given_n) will be: ({self.flat_position_size} x {self.num_time_windows})...', begin_line_ending='... ', enable_print=self.debug_print):
            # if self.debug_print:
            #     print(f'pre-allocating final_p_x_given_n: np.shape(final_p_x_given_n) will be: ({self.flat_position_size} x {self.time_binning_info.num_bins})...', end=' ') # np.shape(final_p_x_given_n): (288,)
            self.flat_p_x_given_n = np.zeros((self.flat_position_size, self.num_time_windows))
            self.p_x_given_n = None
            self.most_likely_position_flat_indicies = None
            self.most_likely_position_indicies = None

    def _setup_time_window_centers(self):
        self.time_window_centers = get_bin_centers(self.time_window_edges)
        actual_time_window_size = self.time_window_centers[2] - self.time_window_centers[1]
        self.time_window_center_binning_info = BinningInfo(self.time_window_edges_binning_info.variable_extents, actual_time_window_size, len(self.time_window_centers), np.arange(len(self.time_window_centers)))
        
    def _reshaped_output(self, output_probability):
       return np.reshape(output_probability, (self.original_position_data_shape[0], self.original_position_data_shape[1], self.num_time_windows))

    
    # ==================================================================================================================== #
    # Main computation functions:                                                                                          #
    # ==================================================================================================================== #
    def perform_compute_single_time_bin(self, time_window_idx):
        """ the main computation function for a single time_window_idx """
        n = self.unit_specific_time_binned_spike_counts[:, time_window_idx] # this gets the specific n_t for this time window
        
        if self.debug_print:
            print(f'np.shape(n): {np.shape(n)}') # np.shape(n): (40,)
        final_p_x_given_n = ZhangReconstructionImplementation.bayesian_prob(self.time_bin_size, self.P_x, self.F, n, debug_print=self.debug_print) # np.shape(final_p_x_given_n): (288,)
        if self.debug_print:
            print(f'np.shape(final_p_x_given_n): {np.shape(self.flat_p_x_given_n)}') # np.shape(final_p_x_given_n): (288,)
        return final_p_x_given_n

            
    def compute_all(self):
        with WrappingMessagePrinter(f'compute_all final_p_x_given_n called. Computing {np.shape(self.flat_p_x_given_n)[0]} windows for self.final_p_x_given_n...', begin_line_ending='... ', finished_message='compute_all completed.', enable_print=self.debug_print):
            for bin_idx in np.arange(self.num_time_windows):
                with WrappingMessagePrinter(f'\t computing single final_p_x_given_n[:, {bin_idx}] for bin_idx {bin_idx}', begin_line_ending='... ', finished_message='', finished_line_ending='\n', enable_print=self.debug_print):
                    self.flat_p_x_given_n[:, bin_idx] = self.perform_compute_single_time_bin(bin_idx)
                    
            
            # all computed
            # Reshape the output variable:
            
            # np.shape(self.final_p_x_given_n) # (288, 85842)
            self.p_x_given_n = self._reshaped_output(self.flat_p_x_given_n)
            self.perform_compute_most_likely_positions()

            # self.p_x_given_n = np.reshape(self.flat_p_x_given_n, (self.original_position_data_shape[0], self.original_position_data_shape[1], self.num_time_windows))            
            # np.shape(rehsaped_final_p_x_given_n) # (48, 6, 85842) 
            # if self.debug_print:
            #     print('compute_all completed!')
            
    def perform_compute_most_likely_positions(self):
        """ Computes the most likely positions at each timestep from self.flat_p_x_given_n """
        self.most_likely_position_flat_indicies = np.argmax(self.flat_p_x_given_n, axis=0)
        # np.shape(self.most_likely_position_flat_indicies) # (85841,)
        self.most_likely_position_indicies = np.array(np.unravel_index(self.most_likely_position_flat_indicies, self.original_position_data_shape)) # convert back to an array
        # np.shape(self.most_likely_position_indicies) # (2, 85841)
        # self.most_likely_position_indicies
        