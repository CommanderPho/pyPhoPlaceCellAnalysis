from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING
from warnings import warn
from copy import deepcopy
from typing import Optional, Union
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from neuropy.core.epoch import Epoch, ensure_dataframe

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from attrs import define, field, Factory
import attrs # used for several things
import matplotlib.pyplot as plt
from matplotlib import cm
import nptyping as ND
from nptyping import NDArray # used for plot_kourosh_activity_style_figure version too to get a good colormap 
import numpy as np
import numpy.ma as ma # for masked array
import pandas as pd
import h5py
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots
from scipy.spatial.distance import cdist
# Distance metrics used by `_new_compute_surprise`
from scipy.spatial import distance # for Jensen-Shannon distance in `_subfn_compute_leave_one_out_analysis`
import random # for random.choice(mylist)
# from PendingNotebookCode import _scramble_curve
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr

# Neuropy:
from neuropy.core.position import build_position_df_resampled_to_time_windows # used in DecoderResultDisplayingPlot2D.setup()
from neuropy.analyses.placefields import PfND
# from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_subsession_neuron_differences, debug_print_ratemap, debug_print_spike_counts, debug_plot_2d_binning, print_aligned_columns
# from neuropy.utils.debug_helpers import parameter_sweeps, _plot_parameter_sweep, compare_placefields_info
from neuropy.core.epoch import Epoch
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.misc import shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.misc import split_array
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDFMixin

from pyphocorehelpers.indexing_helpers import find_neighbours
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.indexing_helpers import safe_np_vstack # for `_new_compute_surprise`
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult # perform_leave_one_aclu_out_decoding_analysis


# Plotting ___________________________________________________________________________________________________________ #




# ==================================================================================================================== #
# DecoderResultDisplaying* Classes                                                                                     #
# ==================================================================================================================== #

class DecoderResultDisplayingBaseClass:
    """ Initialize by passing in a decoder of type BayesianPlacemapPositionDecoder 
    Responsible for displaying the decoded positions. 
    
    """
    def __init__(self, decoder: BayesianPlacemapPositionDecoder):
        super(DecoderResultDisplayingBaseClass, self).__init__()
        self.decoder = decoder
        self.setup()

    def setup(self):
        raise NotImplementedError
        
    def display(self, i):
        raise NotImplementedError

    # decoder properties:
    @property
    def num_time_windows(self):
        """The num_time_windows property."""
        return self.decoder.num_time_windows

    @property
    def active_time_windows(self):
        return self.decoder.active_time_windows

    @property
    def p_x_given_n(self):
        return self.decoder.p_x_given_n
    
    # placefield result variable:
    @property
    def pf(self):
        return self.decoder.pf

    # placefield properties:
    @property
    def ratemap(self):
        return self.pf.ratemap
    
    @property
    def occupancy(self):
        return self.pf.occupancy
            
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
    def most_likely_positions(self):
        return self.decoder.most_likely_positions

@metadata_attributes(short_name=None, tags=['matplotlib', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-20 18:36', related_items=[])
class DecoderResultDisplayingPlot2D(DecoderResultDisplayingBaseClass):
    """ Displays the decoder for 2D position.

    Used by:
        _display_decoder_result

    """
    debug_print = False
    
    def __init__(self, decoder: BayesianPlacemapPositionDecoder, position_df):
        self.position_df = position_df
        super(DecoderResultDisplayingPlot2D, self).__init__(decoder)
        
    def setup(self):
        # make a new figure
        # self.fig = plt.figure(figsize=(15,15))
        self.fig, self.axs = plt.subplots(ncols=1, nrows=1, figsize=(15,15), clear=True, constrained_layout=True)
        # self.title_string = f'2D Decoded Positions'
        # self.fig.suptitle(self.title_string)
        self.index = 0
        active_window, active_p_x_given_n, active_most_likely_x_position, active_nearest_measured_position = self.get_data(self.index) # get the first item
        self.active_im = DecoderResultDisplayingPlot2D.plot_single_decoder_result(self.xbin, self.ybin, active_p_x_given_n, drop_below_threshold=None, final_string_components=[f'Decoder Result[i: {self.index}]: time window: {active_window}'], ax=self.axs)
        
        # self.active_most_likely_pos = self.axs.plot([], [], lw=2) # how to use a plot(...)
        
        if self.position_df is not None:
            # Setup data:
            self.position_df = build_position_df_resampled_to_time_windows(self.position_df, time_bin_size=self.decoder.time_bin_size)
            # Build Scatter plot when done:
            self.active_nearest_measured_pos_plot = self.axs.scatter([], [], label='actual_recorded_position', color='w')
    
        self.active_most_likely_pos_plot = self.axs.scatter([], [], label='most_likely_position', color='k') # How to initialize a scatter(...). see https://stackoverflow.com/questions/42722691/python-matplotlib-update-scatter-plot-from-a-function
        # line, = ax.plot([], [], lw=2)
        
        # self.fig.xticks()
        
        # Animation:
        # self.ani = FuncAnimation(self.fig, self.update, frames=2, interval=100, repeat=True)
        
        
    def get_data(self, window_idx):
        active_window = self.decoder.active_time_windows[window_idx] # a tuple with a start time and end time
        active_p_x_given_n = np.squeeze(self.decoder.p_x_given_n[:,:,window_idx]) # same size as occupancy
        
        active_most_likely_x_indicies = self.decoder.most_likely_position_indicies[:,window_idx]
        active_most_likely_x_position = (self.xbin_centers[active_most_likely_x_indicies[0]], self.ybin_centers[active_most_likely_x_indicies[1]])
        
        if self.position_df is not None:
            active_window_start = active_window[0]
            active_window_end = active_window[1]
            active_window_midpoint = active_window_start + ((active_window_end - active_window_start) / 2.0)
            [lowerneighbor_ind, upperneighbor_ind] = find_neighbours(active_window_midpoint, self.position_df, 't')
            active_nearest_measured_position = self.position_df.loc[lowerneighbor_ind, ['x','y']].to_numpy()
        else:
            active_nearest_measured_position = None
        
        return active_window, active_p_x_given_n, active_most_likely_x_position, active_nearest_measured_position

    @staticmethod
    def prepare_data_for_plotting(p_x_given_n, drop_below_threshold: float=0.0000001):
        curr_p_x_given_n = p_x_given_n.copy()
        if drop_below_threshold is not None:
            curr_p_x_given_n[np.where(curr_p_x_given_n < drop_below_threshold)] = np.nan # null out the p_x_given_n below certain values
            
        ## 2022-09-15 - Testing
        curr_p_x_given_n = np.nan_to_num(curr_p_x_given_n, nan=0.0, posinf=1.0, neginf=0.0) # IDK if this is smart or not
    
        ## Seems to work:
        curr_p_x_given_n = np.rot90(curr_p_x_given_n, k=-1)
        curr_p_x_given_n = np.fliplr(curr_p_x_given_n)
        return curr_p_x_given_n

    @staticmethod
    def plot_single_decoder_result(xbin, ybin, p_x_given_n, drop_below_threshold: float=0.0000001, final_string_components=[], ax=None):
        """Plots a single decoder posterior Heatmap
        """
        if ax is None:
            ax = plt.gca()

        curr_p_x_given_n = DecoderResultDisplayingPlot2D.prepare_data_for_plotting(p_x_given_n, drop_below_threshold=drop_below_threshold)
        
        """ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html """
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1])
        # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
        extent = (xmin, xmax, ymin, ymax)
        # print(f'extent: {extent}')
        # extent = None
        # We'll also create a black background into which the pixels will fade
        # background_black = np.full((*curr_p_x_given_n.shape, 3), 0, dtype=np.uint8)

        imshow_shared_kwargs = {
            'origin': 'lower',
            'extent': extent,
        }

        main_plot_kwargs = imshow_shared_kwargs | {
            # 'vmax': vmax,
            'vmin': 0,
            'vmax': 1,
            'cmap': 'jet',
        }

        # ax.imshow(background_black, **imshow_shared_kwargs) # add black background image
        im = ax.imshow(curr_p_x_given_n, **main_plot_kwargs) # add the curr_px_given_n image
        ax.axis("off")

        # conventional way:
        final_title = '\n'.join(final_string_components)
        ax.set_title(final_title) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"

        return im

    def update(self, i):
        if DecoderResultDisplayingPlot2D.debug_print:
            print(f'update(i: {i})')
            
        self.index = i
        active_window, active_p_x_given_n, active_most_likely_x_position, active_nearest_measured_position = self.get_data(self.index)
        if DecoderResultDisplayingPlot2D.debug_print:
            print(f'active_window: {active_window}, active_p_x_given_n: {active_p_x_given_n}, active_most_likely_x_position: {active_most_likely_x_position}, active_nearest_measured_position: {active_nearest_measured_position}')
        
        # Update only:
        self.active_im.set_array(DecoderResultDisplayingPlot2D.prepare_data_for_plotting(active_p_x_given_n, drop_below_threshold=None))
        self.active_most_likely_pos_plot.set_offsets(np.c_[active_most_likely_x_position[0], active_most_likely_x_position[1]]) # method for updating a scatter_plot
        
        if self.position_df is not None:
            self.active_nearest_measured_pos_plot.set_offsets(np.c_[active_nearest_measured_position[0], active_nearest_measured_position[1]]) # method for updating a scatter_plot
        
        self.axs.set_title(f'Decoder Result[i: {self.index}]: time window: {active_window}')  # update title
        return (self.active_im, self.active_most_likely_pos_plot)
        
        
    def display(self, i):
        updated_plots_tuple = self.update(i) # calls update
        # anim = animation.FuncAnimation(figure, func=update_figure, fargs=(bar_rects, iteration), frames=generator, interval=100, repeat=False)
        # return (self.active_im,)
        return self.fig # returns fig

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder

def _convert_dict_to_hdf_attrs_fn(f, key: str, value):
    """ value: dict-like """
    # if isinstance(f, h5py.File):
    with h5py.File(f, "a") as f:
        for sub_k, sub_v in value.items():
            f[f'{key}/{sub_k}'] = sub_v

        # with f.create_group(key) as g:
        #     for sub_k, sub_v in value.items():
        #         g[f'{key}/{sub_k}'] = sub_v

def _convert_optional_ndarray_to_hdf_attrs_fn(f, key: str, value):
    """ value: dict-like """
    # if isinstance(f, h5py.File):
    with h5py.File(f, "a") as f:
        if value is not None:
            f[f'{key}'] = value
        else:
            f[f'{key}'] = np.ndarray([])


@custom_define(slots=False, repr=False)
class LeaveOneOutDecodingResult(HDFMixin, AttrsBasedClassHelperMixin):
    """Newer things to merge into LeaveOneOutDecodingAnalysisResult
    
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import LeaveOneOutDecodingResult
        
        
    Seems to contain dictionaries for each value type with keys being the neuron_IDs that were left out and values usually being np.array

    ['one_left_out_to_global_surprises', 'one_left_out_posterior_to_pf_surprises', 'one_left_out_posterior_to_scrambled_pf_surprises', 'one_left_out_to_global_surprises_mean', 'shuffle_IDXs',
     'random_noise_curves', 'decoded_timebins_p_x_given_n', 'one_left_out_posterior_to_pf_surprises_mean', 'one_left_out_posterior_to_scrambled_pf_surprises_mean']
    
    random_noise_curves: Dict[int, np.ndarray] = Factory(dict) # LeaveOneOutDecodingResult
    decoded_timebins_p_x_given_n: Dict[int, np.ndarray]
    """
    one_left_out_to_global_surprises: Dict[int, np.ndarray] = serialized_field(default=Factory(dict), serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v))) # empty {}
    one_left_out_posterior_to_pf_surprises: Dict[int, np.ndarray] = serialized_field(default=Factory(dict), serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v)))
    one_left_out_posterior_to_pf_surprises_mean: Dict[int, float] = serialized_field(default=Factory(dict), is_computable=True, serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v)))
    one_left_out_posterior_to_scrambled_pf_surprises: Dict[int, np.ndarray] = serialized_field(default=Factory(dict), serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v)))
    one_left_out_posterior_to_scrambled_pf_surprises_mean: Dict[int, float] = serialized_field(default=Factory(dict), is_computable=True, serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v)))

    one_left_out_to_global_surprises_mean: dict = serialized_field(default=Factory(dict), serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v))) # empty {]
    shuffle_IDXs: np.array = serialized_field(default=None, serialization_fn=(lambda f, k, v: _convert_optional_ndarray_to_hdf_attrs_fn(f, k, v)))
    
    random_noise_curves: Dict[int, np.ndarray] = serialized_field(default=Factory(dict), serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v)))
    decoded_timebins_p_x_given_n: Dict[int, np.ndarray] = serialized_field(default=Factory(dict), serialization_fn=(lambda f, k, v: _convert_dict_to_hdf_attrs_fn(f, k, v)))

    
@custom_define(slots=False, repr=False)
class TimebinnedNeuronActivity(HDFMixin, AttrsBasedClassHelperMixin):
    """ 2023-04-18 - keeps track of which neurons are active and inactive in each decoded timebin
    
    TODO TimebinnedNeuronActivity is not HDF serializable, and it doesn't make sense to make it such. It should be a non_serialized_field
    
    """
    n_timebins: int = serialized_attribute_field()
    active_IDXs: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal)) # a ragged list of different length np.ndarrays, each containing the neuron_IDXs that are active in that timebin
    active_aclus: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    inactive_IDXs: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    inactive_aclus: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal))
    
    time_bin_centers: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal)) # the timebin center times that each time bin corresponds to

    # derived
    num_timebin_active_aclus: np.ndarray = non_serialized_field(default=None, is_computable=True, eq=attrs.cmp_using(eq=np.array_equal)) # int ndarray, the number of active aclus in each timebin
    is_timebin_valid: np.ndarray = non_serialized_field(default=None, is_computable=True, eq=attrs.cmp_using(eq=np.array_equal)) # bool ndarray, whether there is at least one aclu active in each timebin

    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        self.num_timebin_active_aclus = np.array([len(timebin_aclus) for timebin_aclus in self.active_aclus]) # .shape # (2917,)
        self.is_timebin_valid = (self.num_timebin_active_aclus > 0) # NEVERMIND: already is the leave-one-out result, so don't do TWO or more aclus in each timebin constraint due to leave-one-out-requirements

    @classmethod
    def init_from_results_obj(cls, results_obj: "LeaveOneOutDecodingAnalysisResult"):
        n_timebins = np.sum(results_obj.all_included_filter_epochs_decoder_result.nbins)
        # a list of lists where each list contains the aclus that are active during that timebin:
        timebins_active_neuron_IDXs = [np.array(results_obj.original_1D_decoder.neuron_IDXs)[a_timebin_is_cell_firing] for a_timebin_is_cell_firing in np.logical_not(results_obj.is_non_firing_time_bin).T]
        timebins_active_aclus = [np.array(results_obj.original_1D_decoder.neuron_IDs)[an_IDX] for an_IDX in timebins_active_neuron_IDXs]

        timebins_inactive_neuron_IDXs = [np.array(results_obj.original_1D_decoder.neuron_IDXs)[a_timebin_is_cell_firing] for a_timebin_is_cell_firing in results_obj.is_non_firing_time_bin.T]
        timebins_inactive_aclus = [np.array(results_obj.original_1D_decoder.neuron_IDs)[an_IDX] for an_IDX in timebins_inactive_neuron_IDXs]
        # timebins_p_x_given_n = np.hstack(results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)        

        assert np.shape(results_obj.flat_all_epochs_decoded_epoch_time_bins)[1] == n_timebins, f"the last dimension of long_results_obj.flat_all_epochs_decoded_epoch_time_bins should be equal to n_timebins but instead np.shape(results_obj.flat_all_epochs_decoded_epoch_time_bins): {np.shape(results_obj.flat_all_epochs_decoded_epoch_time_bins)} " 
        # long_results_obj.flat_all_epochs_decoded_epoch_time_bins[0].shape
        time_bin_centers = results_obj.flat_all_epochs_decoded_epoch_time_bins[0].copy()
        return cls(n_timebins=n_timebins, active_IDXs=timebins_active_neuron_IDXs, active_aclus=timebins_active_aclus, inactive_IDXs=timebins_inactive_neuron_IDXs, inactive_aclus=timebins_inactive_aclus,
                    time_bin_centers=time_bin_centers)


@custom_define(slots=False, repr=False)
class LeaveOneOutDecodingAnalysisResult(HDFMixin, AttrsBasedClassHelperMixin):
    """ 2023-03-27 - Holds the results from a surprise analysis

    Built with:
        from pyphocorehelpers.general_helpers import GeneratedClassDefinitionType, CodeConversion
        CodeConversion.convert_dictionary_to_class_defn(long_results_dict, class_name='LeaveOneOutDecodingAnalysisResult', class_definition_mode=GeneratedClassDefinitionType.DATACLASS)
    n_neurons, n_epochs * n_timebins_for_epoch_i
    {'n_neurons':67, 'n_epochs':625, 'n_total_time_bins':6855}
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import LeaveOneOutDecodingAnalysisResult
    """
    active_filter_epochs: Epoch = serialized_field()
    original_1D_decoder: BasePositionDecoder = serialized_field() # BayesianPlacemapPositionDecoder
    all_included_filter_epochs_decoder_result: DynamicContainer = non_serialized_field()
    
    flat_all_epochs_measured_cell_spike_counts: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins'), 'tags': ('firing_rate', 'measured')})
    flat_all_epochs_measured_cell_firing_rates: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins'), 'tags': ('firing_rate', 'measured')})
    flat_all_epochs_decoded_epoch_time_bins: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_computed_surprises: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_computed_expected_cell_firing_rates: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins'), 'tags': ('firing_rate', 'computed', 'expected')})
    flat_all_epochs_difference_from_expected_cell_spike_counts: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins'), 'tags': ('firing_rate', 'computed', 'expected')})
    flat_all_epochs_difference_from_expected_cell_firing_rates: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins'), 'tags': ('firing_rate', 'computed', 'expected')})
    all_epochs_decoded_epoch_time_bins_mean: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_computed_cell_surprises_mean: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_all_cells_computed_surprises_mean: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_epochs',)})
    flat_all_epochs_computed_one_left_out_to_global_surprises: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    all_epochs_computed_cell_one_left_out_to_global_surprises_mean: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_epochs',)})
    one_left_out_omitted_aclu_distance_df: pd.DataFrame = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons', 3)})
    most_contributing_aclus: np.ndarray = serialized_field(eq=attrs.cmp_using(eq=np.array_equal), metadata={'shape': ('n_neurons',)})
    
    result: LeaveOneOutDecodingResult = non_serialized_field(default=None)

    new_result: LeaveOneOutDecodingResult = non_serialized_field(default=None) # this is stupid, just done for compatibility with old `_new_compute_surprise` implementation
    
    timebinned_neuron_info: "TimebinnedNeuronActivity" = non_serialized_field(default=None, is_computable=True)
    result_df: pd.DataFrame = serialized_field(default=None)
    result_df_grouped: pd.DataFrame = serialized_field(default=None)

    def supplement_results(self):
        """ 2023-03-27 11:14pm - add the extra stuff to the surprise computation results
        TODO 2023-03-28 5:37pm [ ] - move into the `perform_full_session_leave_one_out_decoding_analysis` function or the supporting subfunctions

        Usage:

            short_results_obj = _supplement_results(short_results_obj)
            long_results_obj = _supplement_results(long_results_obj)

        """
        ## Flatten the measured spike counts over the time bins within all epochs to get something of the same shape as `flat_all_epochs_decoded_epoch_time_bins`:
        flat_all_epochs_measured_cell_spike_counts = np.hstack(self.all_included_filter_epochs_decoder_result.spkcount) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
        ## Get the time bins where each cell is firing (has more than one spike):
        is_cell_firing_time_bin = (flat_all_epochs_measured_cell_spike_counts > 0) # .shape (97, 5815)
        self.is_non_firing_time_bin = np.logical_not(is_cell_firing_time_bin) # .shape (97, 5815)
        ## Reshape to -for-each-epoch instead of -for-each-cell:
        n_epochs = self.active_filter_epochs.n_epochs
        reverse_epoch_indicies_array = [] # a flat array containing the epoch_idx required to access into the dictionary or list variables:
        self.all_epochs_num_epoch_time_bins = [] # one for each decoded epoch
        self.all_epochs_computed_one_left_out_posterior_to_pf_surprises = []
        self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises = []
        
        for decoded_epoch_idx in np.arange(n_epochs):
            num_curr_epoch_time_bins = [len(self.result.one_left_out_posterior_to_pf_surprises[aclu][decoded_epoch_idx]) for aclu in self.original_1D_decoder.neuron_IDs] # get the number of time bins in this epoch to build the reverse indexing array
            # all entries in `num_curr_epoch_time_bins` should be equal to the same value:
            num_curr_epoch_time_bins = num_curr_epoch_time_bins[0]
            self.all_epochs_num_epoch_time_bins.append(num_curr_epoch_time_bins)
            reverse_epoch_indicies_array.append(np.repeat(decoded_epoch_idx, num_curr_epoch_time_bins)) # for all time bins from this epoch, append the decoded_epoch_idx to the reverse array so that it can be recovered from the flattened arrays.    
            # all_epochs_decoded_epoch_time_bins.append(np.array([all_cells_decoded_epoch_time_bins[aclu][decoded_epoch_idx].centers for aclu in original_1D_decoder.neuron_IDs])) # these are duplicated (and the same) for each cell
            self.all_epochs_computed_one_left_out_posterior_to_pf_surprises.append(np.array([self.result.one_left_out_posterior_to_pf_surprises[aclu][decoded_epoch_idx] for aclu in self.original_1D_decoder.neuron_IDs]))
            self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises.append(np.array([self.result.one_left_out_posterior_to_scrambled_pf_surprises[aclu][decoded_epoch_idx] for aclu in self.original_1D_decoder.neuron_IDs]))
            
            
        self.all_epochs_computed_one_left_out_posterior_to_pf_surprises = np.hstack(self.all_epochs_computed_one_left_out_posterior_to_pf_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
        self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises = np.hstack(self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises)
        
        self.all_epochs_reverse_flat_epoch_indicies_array = np.hstack(reverse_epoch_indicies_array)
        self.all_epochs_num_epoch_time_bins = np.array(self.all_epochs_num_epoch_time_bins)

        n_neurons, n_all_epoch_timebins = self.all_epochs_computed_one_left_out_posterior_to_pf_surprises.shape # (n_neurons, n_epochs * n_timebins_for_epoch_i)
        print(f'({n_neurons = }, {n_all_epoch_timebins = })')

        assert np.sum(self.all_epochs_num_epoch_time_bins) == n_all_epoch_timebins
        assert len(self.all_epochs_num_epoch_time_bins) == n_epochs
        flattened_time_bin_indicies = np.arange(n_all_epoch_timebins)
        self.split_by_epoch_reverse_flattened_time_bin_indicies = split_array(flattened_time_bin_indicies, sub_element_lengths=self.all_epochs_num_epoch_time_bins)
        assert len(self.split_by_epoch_reverse_flattened_time_bin_indicies) == n_epochs
        
        # Specific advanced computations:
        self.all_epochs_computed_one_left_out_posterior_to_pf_surprises = ma.array(self.all_epochs_computed_one_left_out_posterior_to_pf_surprises, mask=self.is_non_firing_time_bin) # make sure mask doesn't need to be inverted
        self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises = ma.array(self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises, mask=self.is_non_firing_time_bin) # make sure mask doesn't need to be inverted
        
        ## Compute mean by averaging over bins within each epoch
        self.all_epochs_computed_cell_one_left_out_posterior_to_pf_surprises_mean = np.vstack([np.mean(self.all_epochs_computed_one_left_out_posterior_to_pf_surprises[:, flat_linear_indicies], axis=1) for decoded_epoch_idx, flat_linear_indicies in zip(np.arange(n_epochs), self.split_by_epoch_reverse_flattened_time_bin_indicies)]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
        self.all_epochs_computed_cell_one_left_out_posterior_to_scrambled_pf_surprises_mean = np.vstack([np.mean(self.all_epochs_computed_one_left_out_posterior_to_scrambled_pf_surprises[:, flat_linear_indicies], axis=1) for decoded_epoch_idx, flat_linear_indicies in zip(np.arange(n_epochs), self.split_by_epoch_reverse_flattened_time_bin_indicies)])
        
        ## Compute mean by averaging over each cell:
        self.all_epochs_all_cells_one_left_out_posterior_to_pf_surprises_mean = np.mean(self.all_epochs_computed_cell_one_left_out_posterior_to_pf_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)
        self.all_epochs_all_cells_one_left_out_posterior_to_scrambled_pf_surprises_mean = np.mean(self.all_epochs_computed_cell_one_left_out_posterior_to_scrambled_pf_surprises_mean, axis=1)

        self.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(self)

        try:
            active_surprise_metric_fn = lambda pf, p_x_given_n: distance.jensenshannon(pf, p_x_given_n)
            self._new_compute_surprise(active_surprise_metric_fn=active_surprise_metric_fn) # call self._new_compute_surprise()
        except Exception as e:
            print(f'encountered error running self._new_compute_surprise(...): {e}') # continue anyway

        return self

    @function_attributes(short_name='_new_compute_surprise', tags=[], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-14 00:00')
    def _new_compute_surprise(self, active_surprise_metric_fn):
        """ 2023-04-14 - To Finish factoring out
            long_results_obj.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(long_results_obj)
            short_results_obj.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(short_results_obj)
            assert long_results_obj.timebinned_neuron_info.n_timebins == short_results_obj.timebinned_neuron_info.n_timebins


        Usage:

            # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.jensenshannon(pf, p_x_given_n)
            # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.correlation(pf, p_x_given_n)
            # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.sqeuclidean(pf, p_x_given_n)
            # active_surprise_metric_fn = lambda pf, p_x_given_n: wasserstein_distance(pf, p_x_given_n) # Figure out the correct function for this, it's in my old notebooks
            active_surprise_metric_fn = lambda pf, p_x_given_n: pearsonr(pf, p_x_given_n)[0] # this returns just the correlation coefficient (R), not the p-value due to the [0]

        Sets/Updates self.result, self.result_df, self.result_df_grouped

        """
        # Extract important things from the decoded data, like the time bins which are the same for all:
        n_epochs = self.all_included_filter_epochs_decoder_result.num_filter_epochs
        n_timebins = np.sum(self.all_included_filter_epochs_decoder_result.nbins)
        shared_timebin_containers = self.all_included_filter_epochs_decoder_result.time_bin_containers

        # self.timebinned_neuron_info = TimebinnedNeuronActivity.init_from_results_obj(self) # should already have that from self.supplement_results()

        # @define(slots=False, repr=False)
        # class PlacefieldPosteriorComputationHelper:

        # 	def compute(self, curr_cell_pf_curve, curr_timebin_p_x_given_n):
        # 		result.one_left_out_posterior_to_pf_surprises[timebin_IDX].append(distance.jensenshannon(curr_cell_pf_curve, curr_timebin_p_x_given_n))
        # 		result.one_left_out_posterior_to_pf_correlations[timebin_IDX].append(distance.correlation(curr_cell_pf_curve, curr_timebin_p_x_given_n))


        if active_surprise_metric_fn is None:
            # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.jensenshannon(pf, p_x_given_n)
            # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.correlation(pf, p_x_given_n)
            # active_surprise_metric_fn = lambda pf, p_x_given_n: distance.sqeuclidean(pf, p_x_given_n)
            # active_surprise_metric_fn = lambda pf, p_x_given_n: wasserstein_distance(pf, p_x_given_n) # Figure out the correct function for this, it's in my old notebooks
            active_surprise_metric_fn = lambda pf, p_x_given_n: pearsonr(pf, p_x_given_n)[0] # this returns just the correlation coefficient (R), not the p-value due to the [0]


        timebinned_neuron_info = self.timebinned_neuron_info
        assert timebinned_neuron_info is not None
        self.new_result = LeaveOneOutDecodingResult(shuffle_IDXs=None)
        # assert self.result is not None

        pf_shape = (len(self.original_1D_decoder.pf.ratemap.xbin_centers),) # (59, )
        self.new_result.random_noise_curves = {}
        # result.random_noise_curves = np.random.uniform(low=0, high=1, size=(timebinned_neuron_info.n_timebins, *pf_shape))
        # result.random_noise_curves = (result.random_noise_curves.T / np.sum(result.random_noise_curves, axis=1)).T # normalize
        # result.random_noise_curves = (result.random_noise_curves.T / np.max(result.random_noise_curves, axis=1)).T # unit max normalization
        self.new_result.decoded_timebins_p_x_given_n = {}

        for index in np.arange(timebinned_neuron_info.n_timebins):
            # iterate through timebins
            ## Pre loop: add empty array for accumulation
            if index not in self.new_result.one_left_out_posterior_to_pf_surprises:
                self.new_result.one_left_out_posterior_to_pf_surprises[index] = []
            if index not in self.new_result.one_left_out_posterior_to_scrambled_pf_surprises:
                self.new_result.one_left_out_posterior_to_scrambled_pf_surprises[index] = []

            # curr_random_not_firing_cell_pf_curve = np.random.uniform(low=0, high=1, size=curr_cell_pf_curve.shape) # generate one at a time
            # curr_random_not_firing_cell_pf_curve = curr_random_not_firing_cell_pf_curve / np.sum(curr_random_not_firing_cell_pf_curve) # normalize
            # result.random_noise_curves.append(curr_random_not_firing_cell_pf_curve)

            # curr_random_not_firing_cell_pf_curve = result.random_noise_curves[index]

            self.new_result.random_noise_curves[index] = [] # list
            self.new_result.decoded_timebins_p_x_given_n[index] = []

            for neuron_IDX, aclu in zip(timebinned_neuron_info.active_IDXs[index], timebinned_neuron_info.active_aclus[index]):
                # iterate through only the active cells
                # 1. Get set of cells active in a given time bin, for each compute the surprise of its placefield with the leave-one-out decoded posterior.
                left_out_decoder_result = self.one_left_out_filter_epochs_decoder_result_dict[aclu]
                # curr_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.tuning_curves[neuron_IDX] # normalized pdf tuning curve
                # curr_cell_spike_curve = original_1D_decoder.pf.ratemap.spikes_maps[unit_IDX] ## not occupancy weighted... is this the right one to use for computing the expected spike rate? NO... doesn't seem like it
                curr_cell_pf_curve = self.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[neuron_IDX] # Unit max tuning curve

                _, _, curr_timebins_p_x_given_n = left_out_decoder_result.flatten()
                curr_timebin_p_x_given_n = curr_timebins_p_x_given_n[:, index] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
                assert curr_timebin_p_x_given_n.shape[0] == curr_cell_pf_curve.shape[0], f"{curr_timebin_p_x_given_n.shape = } == {curr_cell_pf_curve.shape = }"
                
                # if aclu not in result.one_left_out_posterior_to_pf_surprises:
                # 	result.one_left_out_posterior_to_pf_surprises[aclu] = []
                # result.one_left_out_posterior_to_pf_surprises[aclu].append(distance.jensenshannon(curr_cell_pf_curve, curr_timebin_p_x_given_n))

                self.new_result.one_left_out_posterior_to_pf_surprises[index].append(active_surprise_metric_fn(curr_cell_pf_curve, curr_timebin_p_x_given_n))
                # result.one_left_out_posterior_to_pf_correlations[timebin_IDX].append(distance.correlation(curr_cell_pf_curve, curr_timebin_p_x_given_n))

                # 2. From the remainder of cells (those not active), randomly choose one to grab the placefield of and compute the surprise with that and the same posterior.
                # shuffled_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.tuning_curves[shuffle_IDXs[i]]

                # a) Use a random non-firing cell's placefield:
                random_not_firing_neuron_IDX = random.choice(timebinned_neuron_info.inactive_IDXs[index])
                # random_not_firing_neuron_IDX = random.choices(timebinned_neuron_info.inactive_IDXs[index], k=)

                # random_not_firing_aclu = random.choice(timebinned_neuron_info.inactive_aclus[i])
                # curr_random_not_firing_cell_pf_curve = results_obj.original_1D_decoder.pf.ratemap.tuning_curves[random_not_firing_neuron_IDX] # normalized pdf tuning curve
                curr_random_not_firing_cell_pf_curve = self.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[random_not_firing_neuron_IDX] # Unit max tuning curve

                # b) Use a scrambled version of the real curve:
                # curr_random_not_firing_cell_pf_curve = _scramble_curve(curr_cell_pf_curve)


                ## Save the curve for this neuron
                self.new_result.random_noise_curves[index].append(curr_random_not_firing_cell_pf_curve)

                # Save the posteriors for this neuron:
                self.new_result.decoded_timebins_p_x_given_n[index].append(curr_timebin_p_x_given_n)

                # if aclu not in result.one_left_out_posterior_to_scrambled_pf_surprises:
                # 	result.one_left_out_posterior_to_scrambled_pf_surprises[aclu] = []
                # # The shuffled cell's placefield and the posterior from leaving a cell out:
                # result.one_left_out_posterior_to_scrambled_pf_surprises[aclu].append(distance.jensenshannon(curr_random_not_firing_cell_pf_curve, curr_timebin_p_x_given_n))
                
                # The shuffled cell's placefield and the posterior from leaving a cell out:
                self.new_result.one_left_out_posterior_to_scrambled_pf_surprises[index].append(active_surprise_metric_fn(curr_random_not_firing_cell_pf_curve, curr_timebin_p_x_given_n))
                # result.one_left_out_posterior_to_scrambled_pf_correlations[timebin_IDX].append(distance.correlation(curr_random_not_firing_cell_pf_curve, curr_timebin_p_x_given_n))

            # END Neuron Loop
            ## Post neuron loops: convert lists to np.arrays
            self.new_result.one_left_out_posterior_to_pf_surprises[index] = np.array(self.new_result.one_left_out_posterior_to_pf_surprises[index])
            self.new_result.one_left_out_posterior_to_scrambled_pf_surprises[index] = np.array(self.new_result.one_left_out_posterior_to_scrambled_pf_surprises[index])
            self.new_result.random_noise_curves[index] = safe_np_vstack(self.new_result.random_noise_curves[index]) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
            self.new_result.decoded_timebins_p_x_given_n[index] = safe_np_vstack(self.new_result.decoded_timebins_p_x_given_n[index]) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists

        # End Timebin Loop
        ## Post timebin loops compute mean variables:
        self.new_result.one_left_out_posterior_to_pf_surprises_mean = {k:np.mean(v) for k, v in self.new_result.one_left_out_posterior_to_pf_surprises.items() if np.size(v) > 0}
        self.new_result.one_left_out_posterior_to_scrambled_pf_surprises_mean = {k:np.mean(v) for k, v in self.new_result.one_left_out_posterior_to_scrambled_pf_surprises.items() if np.size(v) > 0}
        assert len(self.new_result.one_left_out_posterior_to_scrambled_pf_surprises_mean) == len(self.new_result.one_left_out_posterior_to_pf_surprises_mean)
        assert list(self.new_result.one_left_out_posterior_to_scrambled_pf_surprises_mean.keys()) == list(self.new_result.one_left_out_posterior_to_pf_surprises_mean.keys())

        valid_time_bin_indicies = np.array(list(self.new_result.one_left_out_posterior_to_pf_surprises_mean.keys()))
        one_left_out_posterior_to_pf_surprises_mean = np.array(list(self.new_result.one_left_out_posterior_to_pf_surprises_mean.values()))
        one_left_out_posterior_to_scrambled_pf_surprises_mean = np.array(list(self.new_result.one_left_out_posterior_to_scrambled_pf_surprises_mean.values()))
        
        # Build Output Dataframes:
        self.result_df = pd.DataFrame({'time_bin_indices': valid_time_bin_indicies, 'time_bin_centers': timebinned_neuron_info.time_bin_centers[timebinned_neuron_info.is_timebin_valid], 'epoch_IDX': self.all_epochs_reverse_flat_epoch_indicies_array[valid_time_bin_indicies],
            'posterior_to_pf_mean_surprise': one_left_out_posterior_to_pf_surprises_mean, 'posterior_to_scrambled_pf_mean_surprise': one_left_out_posterior_to_scrambled_pf_surprises_mean})
        self.result_df['surprise_diff'] = self.result_df['posterior_to_scrambled_pf_mean_surprise'] - self.result_df['posterior_to_pf_mean_surprise']
        # 24.9 seconds to compute

        ## Compute Aggregate Dataframe for Epoch means:
        # Group by 'epoch_IDX' and compute means of all columns
        self.result_df_grouped = self.result_df.groupby('epoch_IDX').mean()

    @classmethod
    def get_results_subset(cls, epochs_df: pd.DataFrame, epochs_decoder_result: DecodedFilterEpochsResult, included_epoch_indicies: Union[list, np.ndarray]):
        """ Subsets the passed epochs_df and epochs_decoder_result by the included_epoch_indicies
        
        Usage:
            subset_filter_epochs, subset = LeaveOneOutDecodingAnalysisResult.get_results_subset(epochs_df=curr_results_obj.active_filter_epochs.to_dataframe(),
                                            epochs_decoder_result=curr_results_obj.all_included_filter_epochs_decoder_result,
                                            included_epoch_indicies=good_epoch_indicies_L)

        """
        if not isinstance(included_epoch_indicies, np.ndarray):
            included_epoch_indicies = np.array(included_epoch_indicies)
            
        # Filter the active filter epochs:
        is_included_in_subset = np.isin(epochs_df.index, included_epoch_indicies)
        subset_filter_epochs = epochs_df[is_included_in_subset]
        subset = epochs_decoder_result.filtered_by_epochs(included_epoch_indicies)
        return subset_filter_epochs, subset
    




@function_attributes(short_name='one_aclu_loo_decoding_analysis', tags=['decoding', 'loo'], input_requires=[], output_provides=[], uses=['BasePositionDecoder'], creation_date='2023-03-03 00:00')
def perform_leave_one_aclu_out_decoding_analysis(spikes_df, active_pos_df, active_filter_epochs, original_all_included_decoder=None, filter_epoch_description_list=None, decoding_time_bin_size=0.025):
    """2023-03-03 - Performs a "leave-one-out" decoding analysis where we leave out each neuron one at a time and see how the decoding degrades (which serves as an indicator of the importance of that neuron on the decoding performance).

    
    1. Starts from an "all-included" decoder (which doesn't leave-one-out for any cells) and decodes all epochs normally to get that result.
    2. It then calls _build_one_left_out_decoders(...) which omits one aclu at a time and grabs the subset decoder (that has all the placefields of the all-included decoder minus the aclu being left out) that's later used to decode.
    
    
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_leave_one_aclu_out_decoding_analysis

    Called by:
        `perform_full_session_leave_one_out_decoding_analysis(...)`

    Restrictions:
        '1D_only'
    """

    def _build_one_left_out_decoders(original_all_included_decoder):
        """ From the "all-included" decoder, use '.get_by_id(...)' to get a copy of the decoder with one aclu left out. "Leave-one-out" decoding
        WARNING: this might suck up a ton of memory! 

        `defer_compute_all=True` is used to skip computations until the decoder is actually used. This is useful for when we want to build a bunch of decoders and then use them all at once. This is the case here, where we want to build a bunch of decoders and then use them all at once to compute the decoding performance of each one.

        """
        original_neuron_ids = np.array(original_all_included_decoder.pf.ratemap.neuron_ids) # original_pf.included_neuron_IDs
        one_left_out_decoder_dict = {}
        for aclu_to_omit in original_neuron_ids:
            subset_included_neuron_ids = np.array([aclu for aclu in original_neuron_ids if aclu != aclu_to_omit]) # get all but the omitted neuron
            one_left_out_decoder_dict[aclu_to_omit] = original_all_included_decoder.get_by_id(subset_included_neuron_ids, defer_compute_all=True) # skip computations
            
        return one_left_out_decoder_dict


    spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal') ## get only the pyramidal spikes
    
    ## Build placefield and all_included decoder to be used:
    if original_all_included_decoder is None:
        active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        original_decoder_pf1D = PfND(deepcopy(spikes_df), deepcopy(active_pos.linear_pos_obj)) # all other settings default
        ## Build the new decoder:
        # original_all_included_decoder = BayesianPlacemapPositionDecoder(decoding_time_bin_size, original_decoder_pf1D, original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)
        original_all_included_decoder = BasePositionDecoder(pf=original_decoder_pf1D, debug_print=False)

    else:
        print(f'USING EXISTING original_1D_decoder.')

    ## Decode all the epochs with the original decoder:
    all_included_filter_epochs_decoder_result = original_all_included_decoder.decode_specific_epochs(spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)

    # pretty dang inefficient, as there are 70 cells:
    one_left_out_decoder_dict = _build_one_left_out_decoders(original_all_included_decoder)
    ## `decode_specific_epochs` for each of the decoders:
    one_left_out_filter_epochs_decoder_result_dict = {}
    ### Loop through and perform the decoding for each epoch. This is the slow part.
    for left_out_aclu, curr_aclu_omitted_decoder in one_left_out_decoder_dict.items():
        filter_epochs_decoder_result = curr_aclu_omitted_decoder.decode_specific_epochs(spikes_df[spikes_df['aclu'] != left_out_aclu], filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False) # get the spikes_df except spikes for the left-out cell
        # if filter_epoch_description_list is None:
        #     filter_epoch_description_list = [f'Epoch {i}' for i in range(len(filter_epochs_decoder_result.epoch_description_list))]
        filter_epochs_decoder_result.epoch_description_list = deepcopy(filter_epoch_description_list) # PLOT_ONLY
        one_left_out_filter_epochs_decoder_result_dict[left_out_aclu] = filter_epochs_decoder_result

    """ Returns:
        original_1D_decoder: original decoder with all aclu values included
        all_included_filter_epochs_decoder_result: the decoder result for the original decoder with all aclu values included
        one_left_out_decoder_dict: a dictionary of decoders, where each decoder has one less aclu than the original decoder. The key is the aclu that was omitted from the decoder.
        one_left_out_filter_epochs_decoder_result_dict: a dictionary of decoder results for each of the decoders in one_left_out_decoder_dict. The key is the aclu that was omitted from the decoder.
    """
    return original_all_included_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict



@function_attributes(short_name='_analyze_leave_one_out_decoding_results', tags=['surprise', 'decoder', 'loo', 'BasePositionDecoder'], input_requires=[], output_provides=[], uses=['LeaveOneOutDecodingResult', 'scipy.spatial.distance', 'neuropy.utils.misc.shuffle_ids', 'neuropy.core.epoch.Epoch', 'LeaveOneOutDecodingResult'], used_by=['perform_full_session_leave_one_out_decoding_analysis'], creation_date='2023-03-23 00:00')
def _analyze_leave_one_out_decoding_results(active_pos_df, active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict):
    """ 2023-03-23 - Aims to generalize the `_analyze_leave_one_out_decoding_results`

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _subfn_compute_leave_one_out_analysis
        original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict = perform_leave_one_aclu_out_decoding_analysis(pyramidal_only_spikes_df, active_pos_df, active_filter_epochs)
        flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus = _subfn_compute_leave_one_out_analysis(active_pos_df, active_filter_epochs, original_1D_decoder, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict)

    """
    all_cells_decoded_epoch_time_bins = {}
    all_cells_computed_epoch_surprises = {}

    all_cells_computed_epoch_one_left_out_to_global_surprises = {}

    shuffled_aclus, shuffle_IDXs = shuffle_ids(original_1D_decoder.neuron_IDs)

    result = LeaveOneOutDecodingResult(shuffle_IDXs=shuffle_IDXs)


    # Secondary computations
    all_cells_decoded_expected_firing_rates = {}
    ## Compute the impact leaving each aclu out had on the average encoding performance:
    one_left_out_omitted_aclu_distance = {}

    ## for each cell:
    for i, left_out_aclu in enumerate(original_1D_decoder.neuron_IDs):
        # aclu = original_1D_decoder.neuron_IDs[i]
        left_out_neuron_IDX = original_1D_decoder.neuron_IDXs[i] # should just be i, but just to be safe
        ## TODO: only look at bins where the cell fires (is_cell_firing_time_bin[i])
        curr_cell_pf_curve = original_1D_decoder.pf.ratemap.tuning_curves[left_out_neuron_IDX]
        # curr_cell_spike_curve = original_1D_decoder.pf.ratemap.spikes_maps[unit_IDX] ## not occupancy weighted... is this the right one to use for computing the expected spike rate? NO... doesn't seem like it

        shuffled_cell_pf_curve = original_1D_decoder.pf.ratemap.tuning_curves[shuffle_IDXs[i]]

        left_out_decoder_result = one_left_out_filter_epochs_decoder_result_dict[left_out_aclu]
        ## single cell outputs:
        curr_cell_decoded_epoch_time_bins = [] # will be a list of the time bins in each epoch that correspond to each surprise in the corresponding list in curr_cell_computed_epoch_surprises 
        curr_cell_computed_epoch_surprises = [] # will be a list of np.arrays, with each array representing the surprise of each time bin in each epoch

        ## Must pre-allocate each with an empty list:
        all_cells_decoded_expected_firing_rates[left_out_aclu] = [] 
        all_cells_computed_epoch_one_left_out_to_global_surprises[left_out_aclu] = []
        result.one_left_out_posterior_to_scrambled_pf_surprises[left_out_aclu] = []
        result.one_left_out_posterior_to_pf_surprises[left_out_aclu] = []
        result.one_left_out_to_global_surprises[left_out_aclu] = []

        # have one list of posteriors p_x_given_n for each decoded epoch (active_filter_epochs.n_epochs):
        assert len(left_out_decoder_result.p_x_given_n_list) == active_filter_epochs.n_epochs == left_out_decoder_result.num_filter_epochs

        ## Compute the impact leaving each aclu out had on the average encoding performance:
        ### 1. The distance between the actual measured position and the decoded position at each timepoint for each decoder. A larger magnitude difference implies a stronger, more positive effect on the decoding quality.
        one_left_out_omitted_aclu_distance[left_out_aclu] = [] # list to hold the distance results from the epochs
        ## Iterate through each of the epochs for the given left_out_aclu (and its decoder), each of which has its own result
        for decoded_epoch_idx in np.arange(left_out_decoder_result.num_filter_epochs):
            curr_epoch_time_bin_container = left_out_decoder_result.time_bin_containers[decoded_epoch_idx]
            curr_time_bins = curr_epoch_time_bin_container.centers
            curr_epoch_p_x_given_n = left_out_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_cell_pf_curve.shape[0]
            
            ## Get the all-included values too for this decoded_epoch_idx:
            curr_epoch_all_included_p_x_given_n = all_included_filter_epochs_decoder_result.p_x_given_n_list[decoded_epoch_idx] # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)
            assert curr_epoch_p_x_given_n.shape[0] == curr_epoch_all_included_p_x_given_n.shape[0]

            ## Need to exclude estimates from bins that didn't have any spikes in them (in general these glitch around):
            curr_total_spike_counts_per_window = np.sum(left_out_decoder_result.spkcount[decoded_epoch_idx], axis=0) # left_out_decoder_result.spkcount[i].shape # (69, 222) - (nCells, nTimeWindowCenters)
            curr_is_time_bin_non_firing = (curr_total_spike_counts_per_window == 0) # this would mean that no cells fired in this time bin
            # curr_non_firing_time_bin_indicies = np.where(curr_is_time_bin_non_firing)[0] # TODO: could also filter on a minimum number of spikes larger than zero (e.g. at least 2 spikes are required).
            curr_posterior_container = left_out_decoder_result.marginal_x_list[decoded_epoch_idx]
            curr_posterior = curr_posterior_container.p_x_given_n # TODO: check the posteriors too!
            curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D

            ## Compute the distance metric for this epoch:

            # Interpolate the measured positions to the window center times:
            window_center_measured_pos_x = np.interp(curr_time_bins, active_pos_df.t, active_pos_df.lin_pos)
            
            ## Computed the distance metric finally:
            # is it fair to only compare the valid (windows containing at least one spike) windows?
            curr_omit_aclu_distance = distance.cdist(np.atleast_2d(window_center_measured_pos_x[~curr_is_time_bin_non_firing]), np.atleast_2d(curr_most_likely_positions[~curr_is_time_bin_non_firing]), 'sqeuclidean') # squared-euclidian distance between the two vectors
            # curr_omit_aclu_distance comes back double-wrapped in np.arrays for some reason (array([[659865.11994352]])), so .item() extracts the scalar value
            curr_omit_aclu_distance = curr_omit_aclu_distance.item()
            one_left_out_omitted_aclu_distance[left_out_aclu].append(curr_omit_aclu_distance)

            # Compute the expected firing rate for this cell during each bin by taking the computed position posterior and taking the sum of the element-wise product with the cell's placefield.
            curr_epoch_expected_fr = original_1D_decoder.pf.ratemap.tuning_curve_unsmoothed_peak_firing_rates[left_out_neuron_IDX] * np.array([np.sum(curr_cell_pf_curve * curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]) # * original_1D_decoder.pf.ratemap.
            all_cells_decoded_expected_firing_rates[left_out_aclu].append(curr_epoch_expected_fr)
            
            # Compute the Jensen-Shannon Distance as a measure of surprise between the placefield and the posteriors
            curr_cell_computed_epoch_surprises.append(np.array([distance.jensenshannon(curr_cell_pf_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T])) # works! Finite! [0.5839003679903784, 0.5839003679903784, 0.6997779781969289, 0.7725622595699131, 0.5992295785891731]
            curr_cell_decoded_epoch_time_bins.append(curr_epoch_time_bin_container)

            # Compute the Jensen-Shannon Distance as a measure of surprise between the all-included and the one-left-out posteriors:
            all_cells_computed_epoch_one_left_out_to_global_surprises[left_out_aclu].append(np.array([distance.jensenshannon(curr_all_included_p_x_given_n, curr_p_x_given_n) for curr_all_included_p_x_given_n, curr_p_x_given_n in zip(curr_epoch_all_included_p_x_given_n.T, curr_epoch_p_x_given_n.T)])) 

            # The shuffled cell's placefield and the posterior from leaving a cell out:
            result.one_left_out_posterior_to_scrambled_pf_surprises[left_out_aclu].append(np.array([distance.jensenshannon(shuffled_cell_pf_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]))
            result.one_left_out_posterior_to_pf_surprises[left_out_aclu].append(np.array([distance.jensenshannon(curr_cell_pf_curve, curr_p_x_given_n) for curr_p_x_given_n in curr_epoch_p_x_given_n.T]))

        ## End loop over decoded epochs
        assert len(curr_cell_decoded_epoch_time_bins) == len(curr_cell_computed_epoch_surprises)
        all_cells_decoded_epoch_time_bins[left_out_aclu] = curr_cell_decoded_epoch_time_bins
        all_cells_computed_epoch_surprises[left_out_aclu] = curr_cell_computed_epoch_surprises


    ## End loop over cells
    # build a dataframe version to hold the distances:
    one_left_out_omitted_aclu_distance_df = pd.DataFrame({'omitted_aclu':np.array(list(one_left_out_omitted_aclu_distance.keys())),
                                                        'distance': list(one_left_out_omitted_aclu_distance.values()),
                                                        'avg_dist': [np.mean(v) for v in one_left_out_omitted_aclu_distance.values()]}
                                                        )
    one_left_out_omitted_aclu_distance_df.sort_values(by='avg_dist', ascending=False, inplace=True) # this sort reveals the aclu values that when omitted had the largest performance decrease on decoding (as indicated by a larger distance)
    most_contributing_aclus = one_left_out_omitted_aclu_distance_df.omitted_aclu.values

    ## Reshape to -for-each-epoch instead of -for-each-cell
    all_epochs_decoded_epoch_time_bins = []
    all_epochs_computed_surprises = []
    all_epochs_computed_expected_cell_firing_rates = []
    all_epochs_computed_one_left_out_to_global_surprises = []
    for decoded_epoch_idx in np.arange(active_filter_epochs.n_epochs):
        all_epochs_decoded_epoch_time_bins.append(np.array([all_cells_decoded_epoch_time_bins[aclu][decoded_epoch_idx].centers for aclu in original_1D_decoder.neuron_IDs])) # these are duplicated (and the same) for each cell
        all_epochs_computed_surprises.append(np.array([all_cells_computed_epoch_surprises[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))
        all_epochs_computed_expected_cell_firing_rates.append(np.array([all_cells_decoded_expected_firing_rates[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))
        all_epochs_computed_one_left_out_to_global_surprises.append(np.array([all_cells_computed_epoch_one_left_out_to_global_surprises[aclu][decoded_epoch_idx] for aclu in original_1D_decoder.neuron_IDs]))

    assert len(all_epochs_computed_surprises) == active_filter_epochs.n_epochs
    assert len(all_epochs_computed_surprises[0]) == original_1D_decoder.num_neurons
    flat_all_epochs_decoded_epoch_time_bins = np.hstack(all_epochs_decoded_epoch_time_bins) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_surprises = np.hstack(all_epochs_computed_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_expected_cell_firing_rates = np.hstack(all_epochs_computed_expected_cell_firing_rates) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    flat_all_epochs_computed_one_left_out_to_global_surprises = np.hstack(all_epochs_computed_one_left_out_to_global_surprises) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs


    ## Could also do but would need to loop over all epochs for each of the three variables:
    # flat_all_epochs_computed_expected_cell_firing_rates, all_epochs_computed_expected_cell_firing_rates = _subfn_reshape_for_each_epoch_to_for_each_cell(all_cells_decoded_expected_firing_rates, epoch_IDXs=np.arange(active_filter_epochs.n_epochs), neuron_IDs=original_1D_decoder.neuron_IDs)

    ## Aggregates over all time bins in each epoch:
    all_epochs_decoded_epoch_time_bins_mean = np.vstack([np.mean(curr_epoch_time_bins, axis=1) for curr_epoch_time_bins in all_epochs_decoded_epoch_time_bins]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_cell_surprises_mean = np.vstack([np.mean(curr_epoch_surprises, axis=1) for curr_epoch_surprises in all_epochs_computed_surprises]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)
    all_epochs_computed_cell_one_left_out_to_global_surprises_mean = np.vstack([np.mean(curr_epoch_surprises, axis=1) for curr_epoch_surprises in all_epochs_computed_one_left_out_to_global_surprises]) # mean over all time bins in each epoch  # .shape (614, 65) - (n_epochs, n_neurons)

    ## Aggregates over all cells and all time bins in each epoch:
    all_epochs_all_cells_computed_surprises_mean = np.mean(all_epochs_computed_cell_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)
    all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean = np.mean(all_epochs_computed_cell_one_left_out_to_global_surprises_mean, axis=1) # average across all cells .shape (614,) - (n_epochs,)

    """ Returns:
        one_left_out_omitted_aclu_distance_df: a dataframe of the distance metric for each of the decoders in one_left_out_decoder_dict. The index is the aclu that was omitted from the decoder.
        most_contributing_aclus: a list of aclu values, sorted by the largest performance decrease on decoding (as indicated by a larger distance)
    """
    ## Output variables: flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean
    return flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus, result




@function_attributes(short_name='session_loo_decoding_analysis', tags=['decoding', 'loo', 'surprise'], input_requires=[], output_provides=[],
                      uses=['perform_leave_one_aclu_out_decoding_analysis', '_analyze_leave_one_out_decoding_results', 'LeaveOneOutDecodingAnalysisResult'], used_by=['_long_short_decoding_analysis_from_decoders'], creation_date='2023-03-17 00:00')
def perform_full_session_leave_one_out_decoding_analysis(sess, original_1D_decoder=None, decoding_time_bin_size = 0.02, cache_suffix = '', skip_cache_save:bool = True, perform_cache_load:bool = False) -> LeaveOneOutDecodingAnalysisResult:
    """ 2023-03-17 - Performs a leave one out decoding analysis for a full session

    Args:
        sess: a Session object
        original_1D_decoder: an optional `BayesianPlacemapPositionDecoder` object to decode with. If None, the decoder will be created from the session. If you pass a decoder, you want to provide the global session and not a particular filtered one.
        decoding_time_bin_size: the time bin size for the decoder
        cache_suffix: a suffix to add to the cache file name, or None to not cache
    Returns:
        decoder_result: the decoder result for the original decoder with all aclu values included

    Calls:
        perform_leave_one_aclu_out_decoding_analysis(...)
        _subfn_compute_leave_one_out_analysis(...)


    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_full_session_leave_one_out_decoding_analysis
    """
    from neuropy.core.epoch import Epoch
    # for caching/saving:
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData, saveData

    # if (cache_suffix is not None) and ((skip_cache_save is False) or (perform_cache_load is True)):
    ### Build a folder to store the temporary outputs:
    output_data_folder = sess.get_output_path()

    ## Get testing variables from `sess`
    spikes_df = sess.spikes_df
    pyramidal_only_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal') ## get only the pyramidal spikes
    active_pos = sess.position
    active_pos_df = active_pos.to_dataframe()

    # --------------
    # ### active_filter_epochs: sess.laps: 
    # laps_copy = deepcopy(sess.laps)
    # active_filter_epochs = laps_copy.filtered_by_lap_flat_index(np.arange(20)).as_epoch_obj() # epoch object

    ### active_filter_epochs: sess.replay: 
    active_filter_epochs = sess.replay.epochs.get_valid_df().epochs.get_epochs_longer_than(minimum_duration=5.0*decoding_time_bin_size).epochs.get_non_overlapping_df()
    assert len(active_filter_epochs) > 0, f'data frame empty after filtering!'
    if not 'stop' in active_filter_epochs.columns:
        # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
        active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
    if not 'label' in active_filter_epochs.columns:
        # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
        active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()
    active_filter_epochs = Epoch(active_filter_epochs)

    if original_1D_decoder is None:
        ## Build the new decoder (if not provided):
        active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        original_decoder_pf1D = PfND(deepcopy(pyramidal_only_spikes_df), deepcopy(active_pos.linear_pos_obj)) # all other settings default
        ## Build the new decoder:
        original_1D_decoder = BayesianPlacemapPositionDecoder(time_bin_size=decoding_time_bin_size, pf=original_decoder_pf1D, spikes_df=original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)
    else:
        print(f'reusing extant decoder.')

    leave_one_out_result_pickle_path = output_data_folder.joinpath(f'leave_one_out_results{cache_suffix}.pkl').resolve()
    if cache_suffix is not None and leave_one_out_result_pickle_path.exists() and perform_cache_load:
        # loading
        print(f'Loading leave_one_out_result_pickle_path: {leave_one_out_result_pickle_path}')
        active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict = loadData(leave_one_out_result_pickle_path)

    else:
        # -- Part 1 -- perform the decoding:
        original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict = perform_leave_one_aclu_out_decoding_analysis(pyramidal_only_spikes_df, active_pos_df, active_filter_epochs, original_all_included_decoder=original_1D_decoder, decoding_time_bin_size=decoding_time_bin_size)
        # one_left_out_omitted_aclu_distance_df, most_contributing_aclus = _temp_analyze(active_pos_df, one_left_out_filter_epochs_decoder_result_dict)
        # Save to file:
        if cache_suffix is not None and skip_cache_save is False:
            leave_one_out_result_pickle_path = output_data_folder.joinpath(f'leave_one_out_results{cache_suffix}.pkl').resolve()
            print(f'leave_one_out_result_pickle_path: {leave_one_out_result_pickle_path}')
            saveData(leave_one_out_result_pickle_path, (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict))

    
    # -- Part 2 -- perform the analysis on the decoder results:
    flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus, result = _analyze_leave_one_out_decoding_results(active_pos_df, active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict)

    ## Flatten the measured spike counts over the time bins within all epochs to get something of the same shape as `flat_all_epochs_decoded_epoch_time_bins`:
    flat_all_epochs_measured_cell_spike_counts = np.hstack(all_included_filter_epochs_decoder_result.spkcount) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    assert flat_all_epochs_computed_expected_cell_firing_rates.shape == flat_all_epochs_measured_cell_spike_counts.shape, f"{flat_all_epochs_measured_cell_spike_counts.shape = } != {flat_all_epochs_computed_expected_cell_firing_rates.shape =}"

    ## Get the time bins where each cell is firing (has more than one spike):
    is_cell_firing_time_bin = (flat_all_epochs_measured_cell_spike_counts > 0)

    ## Convert spike counts to firing rates by dividing by the time bin size:
    flat_all_epochs_measured_cell_firing_rates = flat_all_epochs_measured_cell_spike_counts / decoding_time_bin_size
    ## Convert the expected firing rates to spike counts by multiplying by the time bin size (NOTE: there can be fractional expected spikes):
    flat_all_epochs_computed_expected_cell_spike_counts = flat_all_epochs_computed_expected_cell_firing_rates * decoding_time_bin_size ## TODO: do some smarter sampling from a distribution or something?

    ## Compute the difference from the expected firing rate observed for each cell (in each time bin):
    flat_all_epochs_difference_from_expected_cell_spike_counts = flat_all_epochs_computed_expected_cell_spike_counts - flat_all_epochs_measured_cell_spike_counts
    flat_all_epochs_difference_from_expected_cell_firing_rates = flat_all_epochs_computed_expected_cell_firing_rates - flat_all_epochs_measured_cell_firing_rates
    # flat_all_epochs_difference_from_expected_cell_firing_rates

    if cache_suffix is not None and skip_cache_save is False:
        # Save to file to cache in case we crash:
        leave_one_out_surprise_result_pickle_path = output_data_folder.joinpath(f'leave_one_out_surprise_results{cache_suffix}.pkl').resolve()
        print(f'leave_one_out_surprise_result_pickle_path: {leave_one_out_surprise_result_pickle_path}')
        saveData(leave_one_out_surprise_result_pickle_path, (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, 
                                                            flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, 
                                                            flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates,
                                                            flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates,
                                                            all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean,
                                                            flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean,
                                                            one_left_out_omitted_aclu_distance_df, most_contributing_aclus))


    # (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus)


    result_tuple = (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, 
                                                            flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, 
                                                            flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates,
                                                            flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates,
                                                            all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean,
                                                            flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean,
                                                            one_left_out_omitted_aclu_distance_df, most_contributing_aclus, result)
    # build output object:
    results_obj = LeaveOneOutDecodingAnalysisResult(*result_tuple)
    ## Add in the one-left-out decoders:
    results_obj.one_left_out_decoder_dict = one_left_out_decoder_dict
    results_obj.one_left_out_filter_epochs_decoder_result_dict = one_left_out_filter_epochs_decoder_result_dict

    results_obj = results_obj.supplement_results() # compute the extra stuff
    return results_obj




# ==================================================================================================================== #
# 2023-05-25 - Radon Transform for Fitting Lines to Replays                                                            #
# ==================================================================================================================== #


import numpy as np
from neuropy.analyses.decoders import radon_transform, old_radon_transform

_allow_parallel_run_general:bool = True
# _allow_parallel_run_general:bool = False


def old_score_posterior(posterior, n_jobs:int=8):
    """Old version scoring of epochs that uses `old_radon_transform`

            score, slope = old_score_posterior(active_posterior)
            slope.shape # (120,)
            score.shape # (120,)
            pd.DataFrame({'score': score, 'slope': slope})

    Returns
    -------
    [type]
        [description]

    References
    ----------
    1) Kloosterman et al. 2012
    """
    run_parallel = _allow_parallel_run_general and (n_jobs > 1)
    if run_parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)( delayed(old_radon_transform)(epoch) for epoch in p )
    else:
        results = [old_radon_transform(epoch) for epoch in posterior]
    
    score = [res[0] for res in results]
    slope = [res[1] for res in results]
    return np.asarray(score), np.asarray(slope)



def get_radon_transform(posterior: Union[List, NDArray], decoding_time_bin_duration:float, pos_bin_size:float, nlines:int=5000, margin:Optional[float]=16.0, n_neighbours: Optional[int]=None, jump_stat=None, posteriors=None, n_jobs:int=8, enable_return_neighbors_arr: bool=False, debug_print=True,
                         t0: Optional[Union[float, List, Tuple, NDArray]]=None, x0: Optional[float]=None):
        """ 2023-05-25 - Radon Transform to fit line to decoded replay epoch posteriors. Gives score, velocity, and intercept. 

         t0: Optional[Union[float, List, Tuple, NDArray]] - usually a list of start times of the same length as posterior, with one for each decoded posterior


        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import get_radon_transform
            ## 2023-05-25 - Get the 1D Posteriors for each replay epoch so they can be analyzed via score_posterior(...) with a Radon Transform approch to find the line of best fit (which gives the velocity).
            active_epoch_decoder_result = long_results_obj.all_included_filter_epochs_decoder_result
            active_posterior = active_epoch_decoder_result.p_x_given_n_list # one for each epoch
            # the size of the x_bin in [cm]
            pos_bin_size = float(long_results_obj.original_1D_decoder.pf.bin_info['xstep'])
            ## compute the Radon transform to get the lines of best fit
            score, velocity, intercept = get_radon_transform(active_posterior, decoding_time_bin_duration=active_epoch_decoder_result.decoding_time_bin_size, pos_bin_size=pos_bin_size,
                                                            nlines=5000, margin=16, jump_stat=None, posteriors=None, n_jobs=1)
            pd.DataFrame({'score': score, 'velocity': velocity, 'intercept': intercept})
            
        """

        if posteriors is None:
            assert posterior is not None, "No posteriors found"
            if isinstance(posterior, (list, tuple)):
                posteriors = posterior # multiple posteriors, okay
            else:
                # a single posterior, wrap in a list:
                posteriors = [posterior,]

        if t0 is not None:
            if isinstance(t0, (list, tuple, NDArray)):
                t0s = t0 # multiple posteriors, okay
                assert len(t0s) == len(posteriors), f"len(t0s): {len(t0s)} == len(posteriors): {len(posteriors)}"
            else:
                # a single time bin, wrap in a list:
                t0s = [t0,]
        else:
            t0s = [None] * len(posteriors) # a list of all Nones

        if n_neighbours is None:
            # Set neighbors from margin, pos_bin_size
            assert margin is not None, f"both neighbours and margin are None!"
            n_neighbours = max(int(round(float(margin) / float(pos_bin_size))), 1) # neighbors must be at least one
            if debug_print:
                print(f'neighbours will be calculated from margin and pos_bin_size. n_neighbours: {n_neighbours} = int(margin: {margin} / pos_bin_size: {pos_bin_size})')
        else:
            # use existing neighbors
            n_neighbours = int(n_neighbours)
            if margin is not None:
                print(f'WARN: margin is not None but its value will not be used because n_neighbours is provided directly (n_neighbours: {n_neighbours}, margin: {margin})')

        run_parallel = _allow_parallel_run_general and (n_jobs > 1)
        if (n_jobs > 1) and (not _allow_parallel_run_general):
            print(f'WARNING: n_jobs > 1 (n_jobs: {n_jobs}) but _allow_parallel_run_general == False, so parallel computation will not be performed.')
        if run_parallel:
            from joblib import Parallel, delayed
            if enable_return_neighbors_arr:
                print(f'WARN: using enable_return_neighbors_arr=True in parallel mode seems to cause deadlocks. Setting `enable_return_neighbors_arr=False` and continuing.')
                enable_return_neighbors_arr = False
            results = Parallel(n_jobs=n_jobs)( delayed(radon_transform)(epoch, nlines=nlines, dt=decoding_time_bin_duration, dx=pos_bin_size, n_neighbours=n_neighbours, enable_return_neighbors_arr=enable_return_neighbors_arr, t0=a_t0, x0=x0) for epoch, a_t0 in zip(posteriors, t0s))

        else:
            results = [radon_transform(epoch, nlines=nlines, dt=decoding_time_bin_duration, dx=pos_bin_size, n_neighbours=n_neighbours, enable_return_neighbors_arr=enable_return_neighbors_arr, t0=a_t0, x0=x0) for epoch, a_t0 in zip(posteriors, t0s)]

        if enable_return_neighbors_arr:
            # score_velocity_intercept_tuple, (num_neighbours, neighbors_arr) = results # unpack
            score = []
            velocity = []
            intercept = []
            num_neighbours = []
            neighbors_arr = []
            debug_info = []

            for a_result_tuple in results:
                a_score, a_velocity, a_intercept, (a_num_neighbours, a_neighbors_arr, a_debug_info) = a_result_tuple
                score.append(a_score)
                velocity.append(a_velocity)
                intercept.append(a_intercept)
                num_neighbours.append(a_num_neighbours)
                neighbors_arr.append(a_neighbors_arr)
                debug_info.append(a_debug_info)
               
            score = np.array(score)
            velocity = np.array(velocity)
            intercept = np.array(intercept)
            num_neighbours = np.array(num_neighbours)
            # neighbors_arr = np.array(neighbors_arr)
            # score, velocity, intercept = np.asarray(score_velocity_intercept_tuple).T
        else:
            score, velocity, intercept = np.asarray(results).T

        # if jump_stat is not None:
        #     return score, velocity, intercept, self._get_jd(posteriors, jump_stat)
        # else:

        if enable_return_neighbors_arr:
            return score, velocity, intercept, (num_neighbours, neighbors_arr, debug_info)
        else:
            return score, velocity, intercept











# ==================================================================================================================== #
# Plotting                                                                                                             #
# ==================================================================================================================== #
from pyphocorehelpers.indexing_helpers import build_pairwise_indicies # used in plot_kourosh_activity_style_figure
if TYPE_CHECKING:
    ## typehinting only imports here
    import pyphoplacecellanalysis.External.pyqtgraph as pg # required for `DiagnosticDistanceMetricFigure`


@define(slots=False, repr=False)
class DiagnosticDistanceMetricFigure:
    """ 2023-04-14 - Metric Figure - Plots a vertical stack of 3 subplots with synchronized x-axes. 
    TOP: At the top is the placefield of the first firing cell in the current timebin.
    MID: The middle shows a placefield of a randomly chosen cell from the set that wasn't firing in this timebin.
    BOTTOM: The bottom shows the current timebin's decoded posterior (p_x_given_n)


    Usage: (for use in Jupyter Notebook)
        ```python
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import DiagnosticDistanceMetricFigure
        import ipywidgets as widgets
        from IPython.display import display

        def integer_slider(update_func):
            slider = widgets.IntSlider(description='Slider:', min=0, max=100, value=0)
            def on_slider_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    # Call the user-provided update function with the current slider index
                    update_func(change['new'])
            slider.observe(on_slider_change)
            display(slider)


        timebinned_neuron_info = long_results_obj.timebinned_neuron_info
        active_fig_obj, update_function = DiagnosticDistanceMetricFigure.build_interactive_diagnostic_distance_metric_figure(long_results_obj, timebinned_neuron_info, result)
        # Call the integer_slider function with the update function
        integer_slider(update_function)
        ```

    History:
        2023-04-17 - Refactored to class from standalone function `_build_interactive_diagnostic_distance_metric_figure`
    """

    results_obj: LeaveOneOutDecodingAnalysisResult
    timebinned_neuron_info: TimebinnedNeuronActivity
    result: LeaveOneOutDecodingResult
    hardcoded_sub_epoch_item_idx: int = 0

    ## derived
    plot_dict: dict = Factory(dict) # holds the pyqtgraph plot objects
    plot_data: dict = Factory(dict)
    is_valid: bool = False
    ## Graphics
    win: pg.GraphicsLayoutWidget = None

    @property
    def n_timebins(self):
        """The total number of timebins."""
        return np.sum(self.results_obj.all_epochs_num_epoch_time_bins)


    # ==================================================================================================================== #
    # Initializer                                                                                                          #
    # ==================================================================================================================== #

    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        import pyphoplacecellanalysis.External.pyqtgraph as pg # required for `DiagnosticDistanceMetricFigure`
        
        # Perform the primary setup to build the placefield
        self.win = pg.GraphicsLayoutWidget(show=True, title='diagnostic_plot')
        
        is_valid = False
        for index in np.arange(self.timebinned_neuron_info.n_timebins):
            # find the first valid index
            if not is_valid:
                self.plot_data, is_valid, (normal_surprise, random_surprise) = self._get_updated_plot_data(index)
                print(f'first valid index: {index}')

        self.plot_dict = self._initialize_plots()


    # Private Methods ____________________________________________________________________________________________________ #
    def _get_updated_plot_data(self, index):
        """ called to actually get the plot data for any given timebin index """
        curr_random_not_firing_cell_pf_curve = self.result.random_noise_curves[index]
        curr_decoded_timebins_p_x_given_n = self.result.decoded_timebins_p_x_given_n[index]
        neuron_IDX, aclu = self.timebinned_neuron_info.active_IDXs[index], self.timebinned_neuron_info.active_aclus[index]
        if len(neuron_IDX) > 0:
            # Get first index
            is_valid = True
            neuron_IDX = neuron_IDX[self.hardcoded_sub_epoch_item_idx]
            aclu = aclu[self.hardcoded_sub_epoch_item_idx]
            # curr_cell_pf_curve = long_results_obj.original_1D_decoder.pf.ratemap.tuning_curves[neuron_IDX]
            curr_cell_pf_curve = self.results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[neuron_IDX]

            if curr_random_not_firing_cell_pf_curve.ndim > 1:
                curr_random_not_firing_cell_pf_curve = curr_random_not_firing_cell_pf_curve[self.hardcoded_sub_epoch_item_idx]

            if curr_decoded_timebins_p_x_given_n.ndim > 1:
                curr_decoded_timebins_p_x_given_n = curr_decoded_timebins_p_x_given_n[self.hardcoded_sub_epoch_item_idx]

            # curr_timebin_p_x_given_n = curr_timebins_p_x_given_n[:, index]
            curr_timebin_p_x_given_n = curr_decoded_timebins_p_x_given_n
            normal_surprise, random_surprise = self.result.one_left_out_posterior_to_pf_surprises[index][self.hardcoded_sub_epoch_item_idx], self.result.one_left_out_posterior_to_scrambled_pf_surprises[index][self.hardcoded_sub_epoch_item_idx]
            updated_plot_data = {'curr_cell_pf_curve': curr_cell_pf_curve, 'curr_random_not_firing_cell_pf_curve': curr_random_not_firing_cell_pf_curve, 'curr_timebin_p_x_given_n': curr_timebin_p_x_given_n}
            
        else:
            # Invalid period
            is_valid = False
            normal_surprise, random_surprise = None, None
            updated_plot_data = {'curr_cell_pf_curve': None, 'curr_random_not_firing_cell_pf_curve': None, 'curr_timebin_p_x_given_n': None}

        return updated_plot_data, is_valid, (normal_surprise, random_surprise)


    @staticmethod 
    def _add_plot(win: pg.GraphicsLayoutWidget, data, name:str):
        plot = win.addPlot() # PlotItem has to be built first?
        curve = plot.plot(data, name=name, label=name)
        plot.setLabel('top', name)
        return plot, curve

    
    def _initialize_plots(self):
        for i, (name, data) in enumerate(self.plot_data.items()):
            plot_item, curve = self._add_plot(self.win, data=data, name=name)
            self.plot_dict[name] = {'plot_item':plot_item,'curve':curve}
            if i == 0:
                first_curve_name = name
            else:
                self.plot_dict[name]['plot_item'].setYLink(first_curve_name)  ## test linking by name
            self.win.nextRow()
        return self.plot_dict


    def _update_plots(self, updated_plot_data):
        """ updates the plots created with `_initialize_plots`"""
        for i, (name, data) in enumerate(updated_plot_data.items()):
            curr_plot = self.plot_dict[name]['plot_item']
            curr_curve = self.plot_dict[name]['curve']
            if data is not None:
                curr_curve.setData(data)
            else:
                curr_curve.setData([])

    # Public Functions ___________________________________________________________________________________________________ #
    def update_function(self, index):
        """ Define an update function that will be called with the current slider index 
        Captures plot_dict, and all data variables
        """
        import pyphoplacecellanalysis.External.pyqtgraph as pg # required for `DiagnosticDistanceMetricFigure`
        
        # print(f'Slider index: {index}')
        hardcoded_sub_epoch_item_idx = 0
        updated_plot_data, is_valid, (normal_surprise, random_surprise) = self._get_updated_plot_data(index)
        self.plot_data = updated_plot_data
        self.is_valid = is_valid

        if is_valid:
            if normal_surprise > random_surprise:
                # Set the pen color to green
                pen = pg.mkPen(color='g')
            else:
                pen = pg.mkPen(color='w')

            self._update_plots(updated_plot_data)
            self.plot_dict['curr_cell_pf_curve']['plot_item'].setLabel('bottom', f"{normal_surprise}")
            self.plot_dict['curr_random_not_firing_cell_pf_curve']['plot_item'].setLabel('bottom', f"{random_surprise}")
            curr_curve = self.plot_dict['curr_cell_pf_curve']['curve']
            curr_curve.setPen(pen)

        else:
            # Invalid period
            self.plot_dict['curr_cell_pf_curve']['plot_item'].setLabel('bottom', f"NO ACTIVITY")
            self.plot_dict['curr_random_not_firing_cell_pf_curve']['plot_item'].setLabel('bottom', f"NO ACTIVITY")
            self._update_plots(updated_plot_data)


    @classmethod
    def build_interactive_diagnostic_distance_metric_figure(cls, results_obj, timebinned_neuron_info, result):
        out_obj = cls(results_obj, timebinned_neuron_info, result)
        return out_obj, out_obj.update_function
    

    def export(self, **kwargs):
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot # works pretty well seemingly
        export_pyqtgraph_plot(self.win, **kwargs)
        
    # ==================================================================================================================== #
    # Public Jupyter-Lab Methods                                                                                           #
    # ==================================================================================================================== #
    @classmethod
    def integer_slider(cls, n_timebins:int, update_func):
        """ 2023-04-13 - Displays an integer slider in a jupyter-notebook (below the cell calling this code) that the user can adjust. WORKS!!!
        

        Args:
            update_func (function): A user-provided update function that will be called with the current slider index.

        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import DiagnosticDistanceMetricFigure
    
            n_timebins = np.sum(long_results_obj.all_epochs_num_epoch_time_bins)
            timebinned_neuron_info = long_results_obj.timebinned_neuron_info
            result = long_results_obj.new_result
            active_fig_obj, update_function = DiagnosticDistanceMetricFigure.build_interactive_diagnostic_distance_metric_figure(long_results_obj, timebinned_neuron_info, result)
            active_fig_obj.integer_slider(n_timebins=n_timebins, update_func=update_function)


        """
        import ipywidgets as widgets
        from IPython.display import display
        slider = widgets.IntSlider(description='Slider:', min=0, max=n_timebins-1, value=0)

        def on_slider_change(change):
            """Callback function for slider value change."""
            if change['type'] == 'change' and change['name'] == 'value':
                # Call the user-provided update function with the current slider index
                update_func(change['new'])

        slider.observe(on_slider_change)
        display(slider)


@function_attributes(short_name='plot_kourosh_activity_style_figure', tags=['plot', 'figure', 'heatmaps','pyqtgraph'], input_requires=[], output_provides=[], uses=['plot_raster_plot','visualize_heatmap_pyqtgraph','CustomLinearRegionItem'], used_by=[], creation_date='2023-04-04 09:03')
def plot_kourosh_activity_style_figure(results_obj: LeaveOneOutDecodingAnalysisResult, long_session, shared_aclus: np.ndarray, epoch_idx: int, callout_epoch_IDXs: list, unit_sort_order=None, unit_colors_list=None, skip_rendering_callouts:bool = False, debug_print=False):
    """ 2023-04-03 - plots a Kourosh-style figure that shows a top panel which displays the decoded posteriors and a raster plot of spikes for a single epoch 
    ## Requirements:
    # The goal is to produce a Kourosh-style figure that shows a top panel which displays the decoded posteriors and a raster plot of spikes for a given epoch.
        ## The example regions are indicated by linearRegions over the raster.
    # Below, several example time bins are pulled out and display: a single bin's decoded posterior, the active pf's for all active cells in the bin


    ## Inputs

    callout_epoch_IDXs: assumed to be a list of timebin indicies to be called-out relative to the start of the epoch (specified by epoch_idx)


    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import plot_kourosh_activity_style_figure
        from neuropy.core.neurons import NeuronType
        # Include only pyramidal aclus:
        print(f'all shared_aclus: {len(shared_aclus)}\nshared_aclus: {shared_aclus}')
        shared_aclu_neuron_type = long_session.neurons.neuron_type[np.isin(long_session.neurons.neuron_ids, shared_aclus)]
        assert len(shared_aclu_neuron_type) == len(shared_aclus)
        # Find only the aclus that are pyramidal:
        is_shared_aclu_pyramidal = (shared_aclu_neuron_type == NeuronType.PYRAMIDAL)
        pyramidal_only_shared_aclus = shared_aclus[is_shared_aclu_pyramidal]
        print(f'num pyramidal_only_shared_aclus: {len(pyramidal_only_shared_aclus)}\npyramidal_only_shared_aclus: {pyramidal_only_shared_aclus}')

        # app, win, plots, plots_data = plot_kourosh_activity_style_figure(long_results_obj, long_session, shared_aclus, epoch_idx=5, callout_epoch_IDXs=[0,1,2,3], skip_rendering_callouts=True)
        app, win, plots, plots_data = plot_kourosh_activity_style_figure(long_results_obj, long_session, pyramidal_only_shared_aclus, epoch_idx=5, callout_epoch_IDXs=[0,1,2,3], skip_rendering_callouts=False)

    """
    from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_raster_plot # used in `plot_kourosh_activity_style_figure`
    import pyphoplacecellanalysis.External.pyqtgraph as pg # used in `plot_kourosh_activity_style_figure`
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem # used in `plot_kourosh_activity_style_figure`

    ## Add linear regions to indicate the time bins
    def update_linear_regions(plots, plots_data):
        """ Updates the active time bin window indicators
            Uses no captured variables

            callout_flat_timebin_indicies=[59, 61]
        """
        # require plots has plots.linear_regions
        # assert plots.linear_regions
        # plots_data.callout_time_bins
        # plots_data.callout_flat_timebin_IDXs
        start_ts, end_ts = plots_data.callout_time_bins
        if debug_print:
            print(f'start_ts: {start_ts}, end_ts: {end_ts}')
        # for a_flat_timebin_idx in callout_flat_timebin_indicies:
        for start_t, end_t, a_flat_timebin_idx in zip(start_ts, end_ts, plots_data.callout_flat_timebin_IDXs):
            # Add the linear region overlay:
            scroll_window_region = CustomLinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=plots['scatter_plot'], movable=False) # bound the LinearRegionItem to the plotted data
            scroll_window_region.setObjectName(f'scroll_window_region[{a_flat_timebin_idx}]')
            scroll_window_region.setZValue(-11) # moves the linear regions to the back so the scatter points are clickable/hoverable
            # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
            plots['root_plot'].addItem(scroll_window_region, ignoreBounds=True)

            plots.linear_regions.append(scroll_window_region)
            # Set the position:
            # plots_data.callout_time_bins[a_flat_timebin_idx]
            if debug_print:
                print(f'setting region[{a_flat_timebin_idx}]: {start_t}, {end_t} :: end_t - start_t = {end_t - start_t}')
            scroll_window_region.setRegion([start_t, end_t]) # adjust scroll control


        return plots, plots_data

    # Get the colormap applied to the decoded posteriors:
    colormap = cm.get_cmap("viridis")  # "nipy_spectral" cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0-255 for Qt


    ## 0. Precompute the active neurons in each timebin, and the epoch-timebin-flattened decoded posteriors makes it easier to compute for a given time bin:
    # a list of lists where each list contains the aclus that are active during that timebin:
    timebins_active_neuron_IDXs = [np.array(results_obj.original_1D_decoder.neuron_IDXs)[a_timebin_is_cell_firing] for a_timebin_is_cell_firing in np.logical_not(results_obj.is_non_firing_time_bin).T]
    timebins_active_aclus = [np.array(results_obj.original_1D_decoder.neuron_IDs)[an_IDX] for an_IDX in timebins_active_neuron_IDXs]
    timebins_p_x_given_n = np.hstack(results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)
    
    ## Default Method: Directly Provided epoch_idx:
    active_epoch = results_obj.active_filter_epochs[epoch_idx]
    # Get a conversion between the epoch indicies and the flat indicies
    flat_bin_indicies = results_obj.split_by_epoch_reverse_flattened_time_bin_indicies[epoch_idx]
    
    ## New Timebin method:
    # assert len(results_obj.all_included_filter_epochs_decoder_result.time_bin_containers) == len(results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list) # num epochs
    active_epoch_time_bin_container = results_obj.all_included_filter_epochs_decoder_result.time_bin_containers[epoch_idx]
    active_epoch_time_bin_start_stops = active_epoch_time_bin_container.edges[build_pairwise_indicies(np.arange(active_epoch_time_bin_container.edge_info.num_bins))] # .shape # (4153, 2)
    # active_epoch_time_bin_start_stops # shape: (5, 2)
    assert np.shape(active_epoch_time_bin_start_stops)[1] == 2
    start_t = active_epoch_time_bin_start_stops[:,0] # shape: (5,)
    end_t = active_epoch_time_bin_start_stops[:,1] # shape: (5,)
    active_epoch_n_timebins = np.shape(active_epoch_time_bin_start_stops)[0] # get the number of timebins in the current epoch
    if debug_print:
        print(f'{active_epoch_n_timebins = }')

    ## Render Top (Epoch-level) panel:
    _active_epoch_spikes_df = deepcopy(long_session.spikes_df)
    # _active_epoch_spikes_df = long_results_obj.original_1D_decoder.spikes_df.copy()

    _active_epoch_spikes_df = _active_epoch_spikes_df[_active_epoch_spikes_df['aclu'].isin(shared_aclus)] ## restrict to only the shared aclus for both short and long
    _active_epoch_spikes_df, _temp_neuron_id_to_new_IDX_map = _active_epoch_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # I think this must be done prior to restricting to the current epoch, but after restricting to the shared_aclus
    _active_epoch_spikes_df = _active_epoch_spikes_df.spikes.time_sliced(*active_epoch[0]) # restrict to the active epoch
    if debug_print:
        print(f'{len(shared_aclus) = }')

    ## Create the raster plot:
    app, win, plots, plots_data = plot_raster_plot(_active_epoch_spikes_df, shared_aclus, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list, scatter_app_name=f"kourosh_activity_style_figure - Raster Epoch[{epoch_idx}]")
    
    ## Setup the aclu labels
    # Set the y range and ticks
    plots['root_plot'].setYRange(0, len(shared_aclus)-1)
    # plots['root_plot'].setYTicks([(i+1, f'{aclu}') for i, aclu in enumerate(shared_aclus)])
    # get the left y-axis:
    # ay = plots['root_plot'].getAxis('left')
    # ay.setTicks([(i+1, f'{aclu}') for i, aclu in enumerate(shared_aclus)])


    ## Set the x-grid to the time bins:
    # Add a grid on the x-axis
    # plots['root_plot'].showGrid(x=True, y=False)
    # Set the x range and divisions
    # plots['root_plot'].setXRange(0, 6)
    # plots['root_plot'].setXDivisions(active_epoch_n_timebins)

    
    ## Create the posterior plot for the decoded epoch
    win.nextRow()
    plots.epoch_posterior_plot = win.addPlot()
    active_epoch_p_x_given_n = results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list[epoch_idx] # all decoded posteriors for curent epoch
    # active_epoch_p_x_given_n.shape # (63, 13)
    epoch_posterior_win, epoch_posterior_img = visualize_heatmap_pyqtgraph(active_epoch_p_x_given_n, win=plots.epoch_posterior_plot, title=f"Epoch[{epoch_idx}]") # .T
    # Apply the colormap
    epoch_posterior_img.setLookupTable(lut)

    plots.root_plot.setXRange(*active_epoch[0])
    
    for a_plot in (plots.epoch_posterior_plot, plots.root_plot):
        # Disable Interactivity
        a_plot.setMouseEnabled(x=False, y=False)
        a_plot.setMenuEnabled(False)

    ## Render the linear regions for each callout:
    plots.linear_regions = []

    ## Render Callouts within the epoch:
    if callout_epoch_IDXs is None:
        if skip_rendering_callouts:
            callout_epoch_IDXs = np.array([]) # empty because it doesn't matter
        else:
            callout_epoch_IDXs = np.arange(active_epoch_n_timebins) # include all timebins if none are specified

    callout_flat_timebin_IDXs = np.array([flat_bin_indicies[an_epoch_relative_IDX] for an_epoch_relative_IDX in callout_epoch_IDXs]) # get absolute flat indicies
    callout_start_t = start_t[callout_epoch_IDXs]
    callout_end_t = end_t[callout_epoch_IDXs]

    if debug_print:
        print(f'{callout_flat_timebin_IDXs = }')
    plots_data.callout_flat_timebin_IDXs = callout_flat_timebin_IDXs
    plots_data.callout_time_bins = [callout_start_t, callout_end_t]

    if debug_print:
        print(f'plots_data.callout_time_bins: {plots_data.callout_time_bins}')

    def build_callout_subgraphic_pyqtgraph(callout_timebin_IDX = 6, axs=None):
        """ Builds a "callout" graphic for a single timebin within the epoch in question. 

        Captures:
            axs
            timebins_p_x_given_n
            long_results_obj (for long_results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves)
            timebins_active_neuron_IDXs
            timebins_active_aclus
        """
        # 1. Plot decoded posterior for this time bin
        curr_timebin_all_included_p_x_given_n = timebins_p_x_given_n[:, callout_timebin_IDX] 
        # use the existing axes:
        assert len(axs) == 2

        # turn into a single row:
        curr_timebin_all_included_p_x_given_n = np.reshape(curr_timebin_all_included_p_x_given_n, (1, -1)).T
        out_posterior_win, out_posterior_img = visualize_heatmap_pyqtgraph(curr_timebin_all_included_p_x_given_n, title=f"decoded posterior for timebin_IDX: {callout_timebin_IDX}", show_colorbar=False, win=axs[0])
        out_posterior_img.setLookupTable(lut)

        # 2. Get cells that were active during this time bin that contributed to this posterior, and get their placefields
        _temp_active_neuron_IDXs = timebins_active_neuron_IDXs[callout_timebin_IDX]
        _temp_n_active_neurons = _temp_active_neuron_IDXs.size
        if _temp_n_active_neurons > 0:
            _temp_active_neuron_aclus = timebins_active_aclus[callout_timebin_IDX]
            _temp_active_pfs = results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[_temp_active_neuron_IDXs,:].copy() 

            # 3. Plot their placefields as a column
            out_pfs_win, out_pfs_img = visualize_heatmap_pyqtgraph(_temp_active_pfs.T, title=f"Active Cell's pf1D during timebin_IDX: {callout_timebin_IDX}", show_yticks=False, show_xticks=True, show_colorbar=False, win=axs[1])
            aclu_x_ticks = [(float(i)+0.5, f'{aclu}') for i, aclu in enumerate(_temp_active_neuron_aclus)]
            ax_x = axs[1].getAxis('bottom')
            ax_x.setTicks((aclu_x_ticks, [])) # the second ones are the minor ticks, but why aren't they showing up?
            ax_x.setLabel(text='active placefields', units=None, unitPrefix=None, **{'font-size': '10pt', 'color': '#d8d8d8dd'})
            # [
            #     [ (majorTickValue1, majorTickString1), (majorTickValue2, majorTickString2), ... ],
            #     [ (minorTickValue1, minorTickString1), (minorTickValue2, minorTickString2), ... ],
            #     ...
            # ]

            # axs[1].showAxis('left')
            # # 4. Add label for each _temp_active_neuron_aclus centered on the bin
            # for i, aclu in enumerate(_temp_active_neuron_aclus):
            #     bin_aclu_label = pg.TextItem(text=f'{aclu}', anchor=(0.5, 0.5), color='w', fill=(0, 0, 0, 100), border=None)
            #     axs[1].addItem(bin_aclu_label)
            #     # bin_aclu_label.setPos(callout_timebin_IDX, i)

        else:
            pass

        return axs

    ## Add linear regions:
    plots, plots_data = update_linear_regions(plots, plots_data)

    if not skip_rendering_callouts:
        # Plot callout subgraphics in columns below:
        win.nextRow()
        callout_glw = win.addLayout()
        row_start_idx = 0

        plots.callouts = RenderPlots('callouts', layout=callout_glw, posteriors=[], placefields=[])
        
        ## Build callout subgraphics:
        for col_idx, a_callout_timebin_IDX in zip(np.arange(len(plots_data.callout_flat_timebin_IDXs)), plots_data.callout_flat_timebin_IDXs):
            top_ax = callout_glw.addPlot(row=row_start_idx, col=col_idx)
            bottom_ax = callout_glw.addPlot(row=row_start_idx+1, col=col_idx)
            curr_callout_axs = build_callout_subgraphic_pyqtgraph(callout_timebin_IDX=a_callout_timebin_IDX, axs=[top_ax, bottom_ax])
            plots.callouts.posteriors.append(top_ax)
            plots.callouts.placefields.append(bottom_ax)

    return app, win, plots, plots_data
    



