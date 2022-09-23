from copy import deepcopy
from pathlib import Path
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import factorial

from neuropy.utils.dynamic_container import DynamicContainer # for decode_specific_epochs
from neuropy.utils.mixins.time_slicing import add_epochs_id_identity # for decode_specific_epochs
from neuropy.analyses.decoders import epochs_spkcount # for decode_specific_epochs

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
        spikes_df = spikes_df.spikes.add_binned_time_column(time_window_edges, time_window_edges_binning_info, debug_print=debug_print)
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
        """ returns flattened versions of the occupancy (P_x), and the tuning_curves (F) """
        neuron_IDs = pf.ratemap.neuron_ids
        neuron_IDXs = np.arange(len(neuron_IDs))
        # maps = pf.ratemap.normalized_tuning_curves  # (40, 48) for 1D, (40, 48, 10) for 2D
        
        ## 2022-09-19 - TODO: should this be the non-normalized tuning curves instead of the normalized ones?
        # e.g. maps = pf.ratemap.tuning_curves
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
    # @staticmethod
    # def bayesian_prob(tau, P_x, F, n, debug_print=False):
    #     """ this seems to be broken as of 2022-09-20. Replaced by .neuropy_bayesian_prob(...) which does seem to work for 2D and was extracted from Neuropy """
    #     # n_i: the number of spikes fired by each cell during the time window of consideration
    #     assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'
        
    #     # total_number_spikes_n = np.sum(n) # the total number of spikes across all placecells during this timewindow
        
    #     # take n as a row vector, and repeat it vertically for each column.
    #     element_wise_n = np.tile(n, (np.shape(F)[0], 1)) # repeat n for each row (coresponding to a position x) in F.
    #     # repeats_array = np.tile(an_array, (repetitions, 1))
    #     if debug_print:
    #         print(f'np.shape(element_wise_n): {np.shape(element_wise_n)}') # np.shape(element_wise_n): (288, 40)

    #     # the inner expression np.power(F, element_wise_n) performs the element-wise exponentiation of F with the values in element_wise_n.
    #     # result = P_x * np.prod(np.power(F, element_wise_n), axis=1) # the product is over the neurons, so the second dimension
    #     term1 = np.squeeze(P_x) # np.shape(P_x): (48, 6)
    #     if debug_print:
    #         print(f'np.shape(term1): {np.shape(term1)}') # np.shape(P_x): (48, 6)
    #     term2 = np.prod(np.power(F, element_wise_n), axis=1) # np.shape(term2): (288,)
    #     if debug_print:
    #         print(f'np.shape(term2): {np.shape(term2)}') # np.shape(P_x): (48, 6)

    #     # result = C_tau_n * P_x
    #     term3 = np.exp(-tau * np.sum(F, axis=1)) # sum over all columns (corresponding to over all cells)
    #     if debug_print:
    #         print(f'np.shape(term3): {np.shape(term3)}') # np.shape(P_x): (48, 6)

    #     # each column_i of F, F[:,i] should be raised to the power of n_i[i]
    #     un_normalized_result = term1 * term2 * term3
    #     C_tau_n = 1.0 / np.sum(un_normalized_result) # normalize the result
    #     result = C_tau_n * un_normalized_result
    #     if debug_print:
    #         print(f'np.shape(result): {np.shape(result)}') # np.shape(P_x): (48, 6)
    #     """
    #         np.shape(term1): (288, 1)
    #         np.shape(term2): (288,)
    #         np.shape(term3): (288,)
    #         np.shape(result): (288, 288)
    #     """
    #     return result
    
    @staticmethod
    def neuropy_bayesian_prob(tau, P_x, F, n, debug_print=False):
        # n_i: the number of spikes fired by each cell during the time window of consideration
        assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'        
        if debug_print:
            print(f'np.shape(P_x): {np.shape(P_x)}, np.shape(F): {np.shape(F)}, np.shape(n): {np.shape(n)}')
        # np.shape(P_x): (1066, 1), np.shape(F): (1066, 66), np.shape(n): (66, 3530)
        
        # P_x = np.squeeze(P_x)
        nCells = n.shape[0]
        nTimeBins = n.shape[1] # many time_bins
        nFlatPositionBins = np.shape(P_x)[0]

        F = F.T # Transpose F so it's of the right form
        cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells))
        for cell in range(nCells):
            """ Comparing to the Zhang paper: the output posterior is P_n_given_x (Eqn 35)
                cell_ratemap: [f_{i}(x) for i in range(nCells)]
                cell_spkcnt: [n_{i} for i in range(nCells)]            
            """
            # cell_spkcnt = spkcount[cell, :][np.newaxis, :]
            # cell_ratemap = ratemaps[cell, :][:, np.newaxis]
            cell_spkcnt = n[cell, :][np.newaxis, :]
            cell_ratemap = F[cell, :][:, np.newaxis]
            coeff = 1 / (factorial(cell_spkcnt)) # 1/factorial(n_{i}) term
            # broadcasting
            cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (
                np.exp(-tau * cell_ratemap)
            )

        posterior = np.prod(cell_prob, axis=2)
        posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n

        return posterior
        
    
    



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
        return self.__dict__


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
        if self.ndim > 1:
            return np.vstack((self.xbin_centers[self.most_likely_position_indicies[0,:]], self.ybin_centers[self.most_likely_position_indicies[1,:]])).T # much more efficient than the other implementation. Result is # (85844, 2)
        else:
            # 1D Decoder case:
            # self.most_likely_position_indicies.shape # (1, 20717)
            return np.squeeze(self.xbin_centers[self.most_likely_position_indicies[0,:]]) # not sure if I actually want to squeeze the values # (20717,)
            
    # placefield properties:
    @property
    def ratemap(self):
        return self.pf.ratemap
    @property
    def ndim(self):
        return self.pf.ndim
    
            
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
            self.p_x_given_n = self._reshape_output(self.flat_p_x_given_n)
            self.compute_most_likely_positions()
    
    
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
        """ updates: 
        
        .unit_specific_time_binned_spike_counts
        .total_spike_counts_per_window
        
        """
        # Check if we already have self.time_window_edges and self.time_window_edges_binning_info to use
        if (self.time_window_edges is not None) and (self.time_window_edges_binning_info is not None):
            ## Already have time_window_edges to use, do not create new ones:
            assert self.time_window_edges_binning_info.step == self.time_bin_size
            self.unit_specific_time_binned_spike_counts, _, _ = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(self.spikes_df, self.time_bin_size, time_window_edges=self.time_window_edges, time_window_edges_binning_info=self.time_window_edges_binning_info, debug_print=self.debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)
            
        else:
            ## need to create new time_window_edges from the self.time_bin_size:
            print(f'WARNING: _setup_time_bin_spike_counts_N_i(): updating self.time_window_edges and self.time_window_edges_binning_info ...')
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
        
    def _reshape_output(self, output_probability):
        return np.reshape(output_probability, (*self.original_position_data_shape, self.num_time_windows)) # changed for compatibility with 1D decoder
    
    # ==================================================================================================================== #
    # Main computation functions:                                                                                          #
    # ==================================================================================================================== #
    # def perform_compute_single_time_bin(self, time_window_idx):
    #     """ the main computation function for a single time_window_idx """
    #     n = self.unit_specific_time_binned_spike_counts[:, time_window_idx] # this gets the specific n_t for this time window
        
    #     if self.debug_print:
    #         print(f'np.shape(n): {np.shape(n)}') # np.shape(n): (40,)
    #     # final_p_x_given_n = ZhangReconstructionImplementation.bayesian_prob(self.time_bin_size, self.P_x, self.F, n, debug_print=self.debug_print) # np.shape(final_p_x_given_n): (288,)
    #     # NeuroPy's decoder method:
    #     final_p_x_given_n = ZhangReconstructionImplementation.neuropy_bayesian_prob(self.time_bin_size, self.P_x, self.F, n, debug_print=self.debug_print)
    #     if self.debug_print:
    #         print(f'np.shape(final_p_x_given_n): {np.shape(self.flat_p_x_given_n)}') # np.shape(final_p_x_given_n): (288,)
    #     return final_p_x_given_n

            
    def compute_all(self):
        with WrappingMessagePrinter(f'compute_all final_p_x_given_n called. Computing {np.shape(self.flat_p_x_given_n)[0]} windows for self.final_p_x_given_n...', begin_line_ending='... ', finished_message='compute_all completed.', enable_print=self.debug_print):
            # for bin_idx in np.arange(self.num_time_windows):
            #     with WrappingMessagePrinter(f'\t computing single final_p_x_given_n[:, {bin_idx}] for bin_idx {bin_idx}', begin_line_ending='... ', finished_message='', finished_line_ending='\n', enable_print=self.debug_print):
            #         self.flat_p_x_given_n[:, bin_idx] = self.perform_compute_single_time_bin(bin_idx)

            # Single sweep decoding:
            self.flat_p_x_given_n[:, :] = ZhangReconstructionImplementation.neuropy_bayesian_prob(self.time_bin_size, self.P_x, self.F, self.unit_specific_time_binned_spike_counts, debug_print=self.debug_print)
            print(f'self.flat_p_x_given_n.shape: {self.flat_p_x_given_n.shape}')
                        
            # all computed
            # Reshape the output variable:
            
            # np.shape(self.final_p_x_given_n) # (288, 85842)
            self.p_x_given_n = self._reshape_output(self.flat_p_x_given_n)
            self.compute_most_likely_positions()

    def compute_most_likely_positions(self):
        """ Computes the most likely positions at each timestep from self.flat_p_x_given_n """        
        self.most_likely_position_flat_indicies, self.most_likely_position_indicies = self.perform_compute_most_likely_positions(self.flat_p_x_given_n, self.original_position_data_shape)
        # np.shape(self.most_likely_position_flat_indicies) # (85841,)
        # np.shape(self.most_likely_position_indicies) # (2, 85841)
        
        
            
    def decode(self, unit_specific_time_binned_spike_counts, time_bin_size, debug_print=True):
        """ decodes the neural activity from its internal placefields, returning its posterior and the predicted position 
        Does not alter the internal state of the decoder (doesn't change internal most_likely_positions or posterior, etc)
        
        unit_specific_time_binned_spike_counts: np.array of shape (num_cells, num_time_bins) - e.g. (69, 20717)
        
        
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

            if self.ndim > 1:
                most_likely_positions = np.vstack((self.xbin_centers[most_likely_position_indicies[0,:]], self.ybin_centers[most_likely_position_indicies[1,:]])).T # much more efficient than the other implementation. Result is # (85844, 2)
            else:
                # 1D Decoder case:
                most_likely_positions = np.squeeze(self.xbin_centers[most_likely_position_indicies[0,:]])
        
            return most_likely_positions, p_x_given_n, most_likely_position_indicies
            
    def decode_specific_epochs(self, spikes_df, filter_epochs, decoding_time_bin_size = 0.05, debug_print=False):
        """ TODO: CRITICAL: THIS IS THE ONLY VERSION OF THE DECODING THAT WORKS. The version perfomred by "compute_all" fails miserably! """
        return self.perform_decode_specific_epochs(self, spikes_df=spikes_df, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=debug_print)

    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #
        
    @classmethod
    def perform_decode_specific_epochs(cls, active_decoder, spikes_df, filter_epochs, decoding_time_bin_size = 0.05, debug_print=False):
        """Uses the decoder to decode the nerual activity (provided in spikes_df) for each epoch in filter_epochs

        NOTE: Uses active_decoder.decode(...) to actually do the decoding
        
        
        Args:
            new_2D_decoder (_type_): _description_
            spikes_df (_type_): _description_
            filter_epochs (_type_): _description_
            decoding_time_bin_size (float, optional): _description_. Defaults to 0.05.
            debug_print (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # build output result object:
        filter_epochs_decoder_result = DynamicContainer(most_likely_positions_list=[], p_x_given_n_list=[], marginal_x_list=[], most_likely_position_indicies_list=[])

        if debug_print:
            print(f'filter_epochs: {filter_epochs.n_epochs}')
        ## Get the spikes during these epochs to attempt to decode from:
        filter_epoch_spikes_df = deepcopy(spikes_df)
        ## Add the epoch ids to each spike so we can easily filter on them:
        filter_epoch_spikes_df = add_epochs_id_identity(filter_epoch_spikes_df, filter_epochs.to_dataframe(), epoch_id_key_name='temp_epoch_id', epoch_label_column_name=None, no_interval_fill_value=-1)
        if debug_print:
            print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')
        filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['temp_epoch_id'] != -1] # Drop all non-included spikes
        if debug_print:
            print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')

        ## final step is to time_bin (relative to the start of each epoch) the time values of remaining spikes
        spkcount, nbins, bad_time_bins = epochs_spkcount(filter_epoch_spikes_df, filter_epochs, decoding_time_bin_size, slideby=decoding_time_bin_size, export_time_bins=False, included_neuron_ids=active_decoder.neuron_IDs, debug_print=debug_print) ## time_bins returned are not correct, they're subsampled at a rate of 1000
        num_filter_epochs = len(nbins) # one for each epoch in filter_epochs

        filter_epochs_decoder_result.spkcount = spkcount
        filter_epochs_decoder_result.nbins = nbins
        # filter_epochs_decoder_result.time_bins = time_bins ## time_bins returned are not correct, they're subsampled at a rate of 1000
        filter_epochs_decoder_result.decoding_time_bin_size = decoding_time_bin_size
        filter_epochs_decoder_result.num_filter_epochs = num_filter_epochs
        if debug_print:
            print(f'num_filter_epochs: {num_filter_epochs}, nbins: {nbins}') # the number of time bins that compose each decoding epoch e.g. nbins: [7 2 7 1 5 2 7 6 8 5 8 4 1 3 5 6 6 6 3 3 4 3 6 7 2 6 4 1 7 7 5 6 4 8 8 5 2 5 5 8]

        # bins = np.arange(epoch.start, epoch.stop, 0.001)
        filter_epochs_decoder_result.most_likely_positions_list = []
        filter_epochs_decoder_result.p_x_given_n_list = []
        # filter_epochs_decoder_result.marginal_x_p_x_given_n_list = []
        filter_epochs_decoder_result.most_likely_position_indicies_list = []
        filter_epochs_decoder_result.time_bin_centers = []
        filter_epochs_decoder_result.time_bin_edges = []
        
        filter_epochs_decoder_result.marginal_x_list = []

        # half_decoding_time_bin_size = (decoding_time_bin_size/2.0)
        for i, curr_unit_spkcount, curr_unit_num_bins in zip(np.arange(num_filter_epochs), spkcount, nbins):
            # print(f'curr_unit_spkcount: {curr_unit_spkcount.shape}')
            curr_unit_correct_time_bin_edges, curr_binning_info = compute_spanning_bins(None, variable_start_value=filter_epochs.starts[i], variable_end_value=filter_epochs.stops[i], num_bins=curr_unit_num_bins)
            filter_epochs_decoder_result.time_bin_edges.append(curr_unit_correct_time_bin_edges)
            filter_epochs_decoder_result.time_bin_centers.append(get_bin_centers(curr_unit_correct_time_bin_edges))

            most_likely_positions, p_x_given_n, most_likely_position_indicies = active_decoder.decode(curr_unit_spkcount, time_bin_size=decoding_time_bin_size, debug_print=debug_print)
            filter_epochs_decoder_result.most_likely_positions_list.append(most_likely_positions)
            filter_epochs_decoder_result.p_x_given_n_list.append(p_x_given_n)
            filter_epochs_decoder_result.most_likely_position_indicies_list.append(most_likely_position_indicies)
            # Compute Marginal 1D Posterior:
            ## Build a container to hold the marginal distribution and its related values:
            curr_unit_marginal_x = DynamicContainer(p_x_given_n=None, most_likely_positions_1D=None)
            
            # Collapse the 2D position posterior into two separate 1D (X & Y) marginal posteriors. Be sure to re-normalize each marginal after summing
            curr_unit_marginal_x.p_x_given_n = np.squeeze(np.sum(p_x_given_n, 1)) # sum over all y. Result should be [x_bins x time_bins]
            curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n / np.sum(curr_unit_marginal_x.p_x_given_n, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
            ## Ensures that the marginal posterior is at least 2D:
            if curr_unit_marginal_x.p_x_given_n.ndim == 0:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n.reshape(1, 1)
            elif curr_unit_marginal_x.p_x_given_n.ndim == 1:
                curr_unit_marginal_x.p_x_given_n = curr_unit_marginal_x.p_x_given_n[:, np.newaxis]
                if debug_print:
                    print(f'\t added dimension to curr_posterior: {curr_unit_marginal_x.p_x_given_n.shape}')
                    
            ## Add the most-likely positions to the posterior_x container:
            if most_likely_positions.ndim < 2:
                curr_unit_marginal_x.most_likely_positions_1D = np.atleast_1d(most_likely_positions).T # already 1D positions, don't need to extract x-component
            else:
                curr_unit_marginal_x.most_likely_positions_1D = most_likely_positions[:,0].T
            
            # Add the marginal container to the list
            filter_epochs_decoder_result.marginal_x_list.append(curr_unit_marginal_x)
            # filter_epochs_decoder_result.marginal_x_p_x_given_n_list.append(curr_unit_marginal_x.p_x_given_n)

        return filter_epochs_decoder_result
        
    
        
    @classmethod
    def perform_compute_most_likely_positions(cls, flat_p_x_given_n, original_position_data_shape):
        """ Computes the most likely positions at each timestep from flat_p_x_given_n and the shape of the original position data """
        most_likely_position_flat_indicies = np.argmax(flat_p_x_given_n, axis=0)        
        most_likely_position_indicies = np.array(np.unravel_index(most_likely_position_flat_indicies, original_position_data_shape)) # convert back to an array
        # np.shape(most_likely_position_flat_indicies) # (85841,)
        # np.shape(most_likely_position_indicies) # (2, 85841)
        return most_likely_position_flat_indicies, most_likely_position_indicies