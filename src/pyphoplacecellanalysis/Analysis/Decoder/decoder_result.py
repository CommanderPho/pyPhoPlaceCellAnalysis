from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import cm # used for plot_kourosh_activity_style_figure version too to get a good colormap 
import numpy as np
import pandas as pd
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots
from scipy.spatial.distance import cdist

# Neuropy:
from neuropy.core.position import build_position_df_resampled_to_time_windows # used in DecoderResultDisplayingPlot2D.setup()
from neuropy.analyses.placefields import PfND
# from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_subsession_neuron_differences, debug_print_ratemap, debug_print_spike_counts, debug_plot_2d_binning, print_aligned_columns
# from neuropy.utils.debug_helpers import parameter_sweeps, _plot_parameter_sweep, compare_placefields_info
from neuropy.core.epoch import Epoch
from neuropy.utils.dynamic_container import DynamicContainer

from pyphocorehelpers.indexing_helpers import find_neighbours
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # perform_leave_one_aclu_out_decoding_analysis

# Plotting ___________________________________________________________________________________________________________ #
import matplotlib
import matplotlib.pyplot as plt
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap, visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
from pyphoplacecellanalysis.GUI.PyQtPlot.Examples.pyqtplot_RasterPlot import plot_raster_plot # used in `plot_kourosh_activity_style_figure`
import pyphoplacecellanalysis.External.pyqtgraph as pg # used in `plot_kourosh_activity_style_figure`
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem # used in `plot_kourosh_activity_style_figure`
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget # used in `plot_kourosh_activity_style_figure` # for embedded matplotlib figure



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

from pyphoplacecellanalysis.Analysis.Decoder.decoder_stateless import BasePositionDecoder


@function_attributes(short_name='one_aclu_loo_decoding_analysis', tags=['decoding', 'loo'], input_requires=[], output_provides=[], creation_date='2023-03-03 00:00')
def perform_leave_one_aclu_out_decoding_analysis(spikes_df, active_pos_df, active_filter_epochs, original_all_included_decoder=None, filter_epoch_description_list=None, decoding_time_bin_size=0.025):
    """2023-03-03 - Performs a "leave-one-out" decoding analysis where we leave out each neuron one at a time and see how the decoding degrades (which serves as an indicator of the importance of that neuron on the decoding performance).

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_leave_one_aclu_out_decoding_analysis

    Called by:
        `perform_full_session_leave_one_out_decoding_analysis(...)`

    Restrictions:
        '1D_only'
    """

    def _build_one_left_out_decoders(original_all_included_decoder):
        """ "Leave-one-out" decoding
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
    
 
    ## Build placefield for the decoder to use:
    if original_all_included_decoder is None:
        active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        original_decoder_pf1D = PfND(deepcopy(spikes_df), deepcopy(active_pos.linear_pos_obj)) # all other settings default
        ## Build the new decoder:
        # original_all_included_decoder = BayesianPlacemapPositionDecoder(decoding_time_bin_size, original_decoder_pf1D, original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)
        original_all_included_decoder = BasePositionDecoder(pf=original_decoder_pf1D, debug_print=False)

        # original_decoder_pf1D.filtered_spikes_df.copy()


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
        filter_epochs_decoder_result = curr_aclu_omitted_decoder.decode_specific_epochs(spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
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


# ==================================================================================================================== #
# 2023-03-17 Surprise Analysis                                                                                         #
# ==================================================================================================================== #
from attrs import define, field
# import cattrs 
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import LeaveOneOutDecodingResult
from neuropy.utils.misc import split_array
import numpy.ma as ma # for masked array

@define(slots=False)
class SurpriseAnalysisResult:
    """ 

    Built with:
        from pyphocorehelpers.general_helpers import GeneratedClassDefinitionType, CodeConversion
        CodeConversion.convert_dictionary_to_class_defn(long_results_dict, class_name='SurpriseAnalysisResult', class_definition_mode=GeneratedClassDefinitionType.DATACLASS)
    n_neurons, n_epochs * n_timebins_for_epoch_i
    {'n_neurons':67, 'n_epochs':625, 'n_total_time_bins':6855}
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import SurpriseAnalysisResult
    """
    active_filter_epochs: Epoch
    original_1D_decoder: BasePositionDecoder # BayesianPlacemapPositionDecoder
    all_included_filter_epochs_decoder_result: DynamicContainer
    flat_all_epochs_measured_cell_spike_counts: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_measured_cell_firing_rates: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_decoded_epoch_time_bins: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_computed_surprises: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_computed_expected_cell_firing_rates: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_difference_from_expected_cell_spike_counts: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_difference_from_expected_cell_firing_rates: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    all_epochs_decoded_epoch_time_bins_mean: np.ndarray = field(metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_computed_cell_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_all_cells_computed_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs',)})
    flat_all_epochs_computed_one_left_out_to_global_surprises: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    all_epochs_computed_cell_one_left_out_to_global_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs',)})
    one_left_out_omitted_aclu_distance_df: pd.core.frame.DataFrame = field(metadata={'shape': ('n_neurons', 3)})
    most_contributing_aclus: np.ndarray = field(metadata={'shape': ('n_neurons',)})
    result: LeaveOneOutDecodingResult = None
    

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
        return self


    # def __attrs_post_init__(self):
    #     self.z = self.x + self.y

    # def sliced_by_aclus(self, aclus):
    #     """ returns a copy of itself sliced by the aclus provided. """
    #     from attrs import asdict, fields, evolve
    #     aclu_is_included = np.isin(self.original_1D_decoder.neuron_IDs, aclus)  #.shape # (104, 63)
    #     def _filter_obj_attribute(an_attr, attr_value):
    #         """ return attributes only if they have n_neurons in their shape metadata """
    #         return ('n_neurons' in an_attr.metadata.get('shape', ()))            
    #     _temp_obj_dict = asdict(self, filter=_filter_obj_attribute)
    #     # Find all fields that contain a 'n_neurons':
    #     neuron_indexed_attributes = [a_field for a_field in fields(type(self)) if ('n_neurons' in a_field.metadata.get('shape', ()))]
    #     # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
    #     neuron_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_neurons index location
    #     _temp_obj_dict = {k:v.take(indices=aclu_is_included, axis=neuron_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_neurons axis containing items to get a reduced dictionary
    #     return evolve(self, **_temp_obj_dict)



    # @staticmethod
    # def _build_results_dict(a_results_tuple):
    #     active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean = a_results_tuple
    #     return {'active_filter_epochs':active_filter_epochs, 'original_1D_decoder':original_1D_decoder, 'all_included_filter_epochs_decoder_result':all_included_filter_epochs_decoder_result, 'flat_all_epochs_measured_cell_spike_counts':flat_all_epochs_measured_cell_spike_counts, 'flat_all_epochs_measured_cell_firing_rates':flat_all_epochs_measured_cell_firing_rates, 'flat_all_epochs_decoded_epoch_time_bins':flat_all_epochs_decoded_epoch_time_bins, 'flat_all_epochs_computed_surprises':flat_all_epochs_computed_surprises, 'flat_all_epochs_computed_expected_cell_firing_rates':flat_all_epochs_computed_expected_cell_firing_rates, 'flat_all_epochs_difference_from_expected_cell_spike_counts':flat_all_epochs_difference_from_expected_cell_spike_counts, 'flat_all_epochs_difference_from_expected_cell_firing_rates':flat_all_epochs_difference_from_expected_cell_firing_rates, 'all_epochs_decoded_epoch_time_bins_mean':all_epochs_decoded_epoch_time_bins_mean, 'all_epochs_computed_cell_surprises_mean':all_epochs_computed_cell_surprises_mean, 'all_epochs_all_cells_computed_surprises_mean':all_epochs_all_cells_computed_surprises_mean}


@function_attributes(short_name='session_loo_decoding_analysis', tags=['decoding', 'loo'], input_requires=[], output_provides=[], creation_date='2023-03-17 00:00')
def perform_full_session_leave_one_out_decoding_analysis(sess, original_1D_decoder=None, decoding_time_bin_size = 0.02, cache_suffix = '', skip_cache_save:bool = True, perform_cache_load:bool = False) -> SurpriseAnalysisResult:
    """ 2023-03-17 - Performs a full session leave one out decoding analysis.

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
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _analyze_leave_one_out_decoding_results

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
        original_1D_decoder = BayesianPlacemapPositionDecoder(decoding_time_bin_size, original_decoder_pf1D, original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)
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
    flat_all_epochs_computed_expected_cell_spike_counts = flat_all_epochs_computed_expected_cell_firing_rates * decoding_time_bin_size

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
    results_obj = SurpriseAnalysisResult(*result_tuple)
    results_obj = results_obj.supplement_results() # compute the extra stuff
    ## Add in the one-left-out decoders:
    results_obj.one_left_out_decoder_dict = one_left_out_decoder_dict
    results_obj.one_left_out_filter_epochs_decoder_result_dict = one_left_out_filter_epochs_decoder_result_dict

    return results_obj





# ==================================================================================================================== #
# Plotting                                                                                                             #
# ==================================================================================================================== #

@function_attributes(short_name='plot_kourosh_activity_style_figure', tags=['plot', 'figure', 'heatmaps'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-04 09:03')
def plot_kourosh_activity_style_figure(long_results_obj: SurpriseAnalysisResult, long_session, shared_aclus: np.ndarray, epoch_idx: int, callout_epoch_IDXs: list, skip_rendering_callouts:bool = False):
    """ 2023-04-03 - plots a single epoch 
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
        
        # for a_flat_timebin_idx in callout_flat_timebin_indicies:
        for start_t, end_t, a_flat_timebin_idx in zip(start_ts, end_ts, plots_data.callout_flat_timebin_IDXs):
            # Add the linear region overlay:
            scroll_window_region = CustomLinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=plots['scatter_plot'], movable=False) # bound the LinearRegionItem to the plotted data
            scroll_window_region.setObjectName(f'scroll_window_region[{a_flat_timebin_idx}]')
            scroll_window_region.setZValue(10)
            # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
            plots['root_plot'].addItem(scroll_window_region, ignoreBounds=True)

            plots.linear_regions.append(scroll_window_region)
            # Set the position:
            # plots_data.callout_time_bins[a_flat_timebin_idx]
            scroll_window_region.setRegion([start_t, end_t]) # adjust scroll control


        return plots, plots_data

    # Get the colormap applied to the decoded posteriors:
    colormap = cm.get_cmap("viridis")  # "nipy_spectral" cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt


    ## 0. Precompute the active neurons in each timebin, and the epoch-timebin-flattened decoded posteriors makes it easier to compute for a given time bin:
    # a list of lists where each list contains the aclus that are active during that timebin:
    timebins_active_neuron_IDXs = [np.array(long_results_obj.original_1D_decoder.neuron_IDXs)[a_timebin_is_cell_firing] for a_timebin_is_cell_firing in np.logical_not(long_results_obj.is_non_firing_time_bin).T]
    timebins_active_aclus = [np.array(long_results_obj.original_1D_decoder.neuron_IDs)[an_IDX] for an_IDX in timebins_active_neuron_IDXs]
    timebins_p_x_given_n = np.hstack(long_results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list) # # .shape: (239, 5) - (n_x_bins, n_epoch_time_bins)  --TO-->  .shape: (63, 4146) - (n_x_bins, n_flattened_all_epoch_time_bins)
    
    ## Default Method: Directly Provided epoch_idx:
    active_epoch = long_results_obj.active_filter_epochs[epoch_idx]
    # Get a conversion between the epoch indicies and the flat indicies
    flat_bin_indicies = long_results_obj.split_by_epoch_reverse_flattened_time_bin_indicies[epoch_idx]
    # print(f'long_results_obj.flat_all_epochs_decoded_epoch_time_bins.shape: {np.shape(long_results_obj.flat_all_epochs_decoded_epoch_time_bins)}') # (97, 5815)
    flat_time_bin_center_times = np.array([long_results_obj.flat_all_epochs_decoded_epoch_time_bins[0, a_flat_time_bin_idx] for a_flat_time_bin_idx in flat_bin_indicies]) # flat_time_bin_times: [43.415 43.435 43.455 43.475 43.495]
    print(f'flat_time_bin_times: {flat_time_bin_center_times}')
    # the bins are centered, so we need to offset them (transfrom them into start/end)
    time_step = np.diff(flat_time_bin_center_times).mean()
    half_time_step = time_step / 2.0
    start_t = flat_time_bin_center_times - half_time_step
    end_t = flat_time_bin_center_times + half_time_step

    active_epoch_n_timebins = len(flat_time_bin_center_times) # get the number of timebins in the current epoch
    print(f'{active_epoch_n_timebins = }')

    ## Alternative Method: reverse-determine the epoch from the timebin provided:
    # example_timebin_containing_Epoch_IDX = long_results_obj.all_epochs_reverse_flat_epoch_indicies_array[callout_timebin_IDX]
    # example_epoch = long_results_obj.active_filter_epochs[example_timebin_containing_Epoch_IDX]
    # print(f'{callout_timebin_IDX = }, epoch: {example_timebin_containing_Epoch_IDX = }')

    ## Render Top (Epoch-level) panel:
    _active_epoch_spikes_df = deepcopy(long_session.spikes_df)
    # _active_epoch_spikes_df = long_results_obj.original_1D_decoder.spikes_df.copy()

    _active_epoch_spikes_df = _active_epoch_spikes_df[_active_epoch_spikes_df['aclu'].isin(shared_aclus)] ## restrict to only the shared aclus for both short and long
    _active_epoch_spikes_df, _temp_neuron_id_to_new_IDX_map = _active_epoch_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # I think this must be done prior to restricting to the current epoch, but after restricting to the shared_aclus
    _active_epoch_spikes_df = _active_epoch_spikes_df.spikes.time_sliced(*active_epoch[0]) # restrict to the active epoch
    # _temp_active_spikes_df = _temp_active_spikes_df[_temp_active_spikes_df['aclu'].isin(_temp_active_neuron_aclus)] ## restrict to active neurons only	
    
    print(f'{len(shared_aclus) = }')

    ## Create the raster plot:
    app, win, plots, plots_data = plot_raster_plot(_active_epoch_spikes_df, shared_aclus, scatter_app_name=f"Raster Epoch[{epoch_idx}]")

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

    # Manual Method:
    # for a_bin_start_t in start_t:
    #     # time_bin_edge_pen_color = pg.mkColor((1.0, 1.0, 1.0, 0.5)) # white with 0.5 alpha
    #     # time_bin_edge_pen = pg.mkPen(time_bin_edge_pen_color, width=1.5)
    #     time_bin_edge_pen = 'w'
    #     plots['root_plot'].addLine(x=a_bin_start_t, pen=time_bin_edge_pen)


    ## Create the posterior plot for the decoded epoch
    win.nextRow()
    plots.epoch_posterior_plot = win.addPlot()
    active_epoch_p_x_given_n = long_results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list[epoch_idx] # all decoded posteriors for curent epoch
    # active_epoch_p_x_given_n.shape # (63, 13)
    epoch_posterior_win, epoch_posterior_img = visualize_heatmap_pyqtgraph(active_epoch_p_x_given_n.T, win=plots.epoch_posterior_plot, title=f"Epoch[{epoch_idx}]")
    # Apply the colormap
    epoch_posterior_img.setLookupTable(lut)

    plots.root_plot.setXRange(*active_epoch[0])
    # plots.scatter_plot.setXRange(*active_epoch[0])
    # plots.epoch_posterior_plot.setXRange(*active_epoch[0]) # This does not work for the posterior plot, probably because it's an image, but this plot is aligned correctly anyway
    # plots.root_plot.setXLink(plots.epoch_posterior_plot) # bind/link the two axes

    for a_plot in (plots.epoch_posterior_plot, plots.root_plot):
        # Disable Interactivity
        a_plot.setMouseEnabled(x=False, y=False)
        a_plot.setMenuEnabled(False)

    # ## TODO: disable interactivity for callout plots too:
    # placefield_axes_list = plots.callouts.placefields
    # posteriors_axes_list = plots.callouts.posteriors

    ## Render the linear regions for each callout:
    plots.linear_regions = []

    ## Render Callouts within the epoch:
    callout_flat_timebin_IDXs = np.array([flat_bin_indicies[an_epoch_relative_IDX] for an_epoch_relative_IDX in callout_epoch_IDXs]) # get absolute flat indicies
    callout_time_bin_center_times = np.array([flat_time_bin_center_times[an_epoch_relative_IDX] for an_epoch_relative_IDX in callout_epoch_IDXs]) # [43.415 43.435 43.455 43.475 43.495]
    callout_start_t = start_t[callout_epoch_IDXs]
    callout_end_t = end_t[callout_epoch_IDXs]
    print(f'{callout_flat_timebin_IDXs = }, {callout_time_bin_center_times = }')	
    plots_data.callout_flat_timebin_IDXs = callout_flat_timebin_IDXs
    plots_data.callout_time_bins = [callout_start_t, callout_end_t]
    print(f'time_step: {time_step}, plots_data.callout_time_bins: {plots_data.callout_time_bins}')

    def build_callout_subgraphic(callout_timebin_IDX = 6, axs=None):
        """ Builds a "callout" graphic for a single timebin within the epoch in question. 

        Captures: 
            timebins_p_x_given_n
            long_results_obj (for long_results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves)
            timebins_active_neuron_IDXs
            timebins_active_aclus
            
        """
        # 1. Plot decoded posterior for this time bin
        # len(long_results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list)
        curr_timebin_all_included_p_x_given_n = timebins_p_x_given_n[:, callout_timebin_IDX] 
        # curr_timebin_all_included_p_x_given_n #.shape # (63, ) (n_x_bins, )

        
        if axs is None:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 8)) # nrows, ncolumns
        else:
            # use the existing axes:
            assert len(axs) == 2

        fig, ax, im = visualize_heatmap(curr_timebin_all_included_p_x_given_n, title=f"decoded posterior for example_timebin_IDX: {callout_timebin_IDX}", show_colorbar=False, ax=axs[0])
        # 2. Get cells that were active during this time bin that contributed to this posterior, and get their placefields
        _temp_active_neuron_IDXs = timebins_active_neuron_IDXs[callout_timebin_IDX]
        _temp_active_neuron_aclus = timebins_active_aclus[callout_timebin_IDX]
        _temp_active_pfs = long_results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[_temp_active_neuron_IDXs,:].copy() 

        # 3. Plot their placefields as a column
        ## Plot a stacked heatmap for all place cells, with each row being a different cell:
        fig, ax, im = visualize_heatmap(_temp_active_pfs, title=f"1D Placefields for active aclus during example_timebin_IDX: {callout_timebin_IDX}", show_colorbar=False, ax=axs[1])

        # Set y-ticks to show the unit IDs
        ax.set_yticks(np.arange(len(_temp_active_neuron_aclus)))
        ax.set_yticklabels(_temp_active_neuron_aclus)
        # Rotate the y-tick labels and set their alignment
        plt.xticks(rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), fontsize=10)

    def build_callout_subgraphic_pyqtgraph(callout_timebin_IDX = 6, axs=None):
        """ Builds a "callout" graphic for a single timebin within the epoch in question. 

        Captures:
            timebins_p_x_given_n
            long_results_obj (for long_results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves)
            timebins_active_neuron_IDXs
            timebins_active_aclus
        """
        # 1. Plot decoded posterior for this time bin
        # len(long_results_obj.all_included_filter_epochs_decoder_result.p_x_given_n_list)
        curr_timebin_all_included_p_x_given_n = timebins_p_x_given_n[:, callout_timebin_IDX] 
        # curr_timebin_all_included_p_x_given_n #.shape # (63, ) (n_x_bins, )
        
        # use the existing axes:
        assert len(axs) == 2

        # turn into a single row:
        curr_timebin_all_included_p_x_given_n = np.reshape(curr_timebin_all_included_p_x_given_n, (1, -1)).T
        out_posterior_win, out_posterior_img = visualize_heatmap_pyqtgraph(curr_timebin_all_included_p_x_given_n, title=f"decoded posterior for example_timebin_IDX: {callout_timebin_IDX}", show_colorbar=False, win=axs[0])
        out_posterior_img.setLookupTable(lut)

        # 2. Get cells that were active during this time bin that contributed to this posterior, and get their placefields
        _temp_active_neuron_IDXs = timebins_active_neuron_IDXs[callout_timebin_IDX]
        _temp_n_active_neurons = _temp_active_neuron_IDXs.size
        if _temp_n_active_neurons > 0:
            _temp_active_neuron_aclus = timebins_active_aclus[callout_timebin_IDX]
            _temp_active_pfs = long_results_obj.original_1D_decoder.pf.ratemap.unit_max_tuning_curves[_temp_active_neuron_IDXs,:].copy() 

            # 3. Plot their placefields as a column
            ## Plot a stacked heatmap for all place cells, with each row being a different cell:
            out_pfs_win, out_pfs_img = visualize_heatmap_pyqtgraph(_temp_active_pfs.T, title=f"1D Placefields for active aclus during example_timebin_IDX: {callout_timebin_IDX}", show_yticks=True, show_colorbar=False, win=axs[1])
            
            # # Set y-ticks to show the unit IDs
            aclu_y_ticks = [(float(i)+0.5, f'{aclu}') for i, aclu in enumerate(_temp_active_neuron_aclus)] # offset by +0.5 to center each tick on the row
            # print(f'aclu_y_ticks: {aclu_y_ticks}')
            ## Setup the aclu labels
            # Set the y range and ticks
            # plots['root_plot'].setYRange(0, len(shared_aclus)-1)
            # plots['root_plot'].setYTicks([(i+1, f'{aclu}') for i, aclu in enumerate(shared_aclus)])
            # get the left y-axis:
            ay = axs[1].getAxis('left')
            # ay.setTicks(aclu_y_ticks)
            ay.setTicks((aclu_y_ticks, [])) # add list of major ticks; no minor ticks
            axs[1].showAxis('left') # show the axis
        else:
            # empty bin with no firing
            # TODO 2023-04-06 - Clear anything on ax[1]?
            pass

        return axs

    ## Add linear regions:
    plots, plots_data = update_linear_regions(plots, plots_data)

    if not skip_rendering_callouts:
        # Plot callout subgraphics in columns below:
        win.nextRow()

        ## CustomMatplotlibWidget:
        # plots.mw = CustomMatplotlibWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=False, scrollAreaContents_MinimumHeight=200)
        # subplot = plots.mw.getFigure().add_subplot(2, active_epoch_n_timebins, 1)

        # PyQtGraph Version:
        # top_axis_indicies = np.arange(active_epoch_n_timebins)+1 # must be 1-based, not zero based
        # bottom_axis_indicies = top_axis_indicies + active_epoch_n_timebins

        # MATPLOTLIB version:
        # top_axis_indicies = np.arange(active_epoch_n_timebins)+1 # must be 1-based, not zero based
        # bottom_axis_indicies = top_axis_indicies + active_epoch_n_timebins
        # top_axis_objs = [plots.mw.getFigure().add_subplot(2, active_epoch_n_timebins, an_idx) for an_idx in top_axis_indicies]
        # bottom_axis_objs = [plots.mw.getFigure().add_subplot(2, active_epoch_n_timebins, an_idx) for an_idx in bottom_axis_indicies]
        
        # win.addItem(plots.mw)
        # subplot.plot(x,y)
        # mw.draw()

        # callout_glw = pg.GraphicsLayoutWidget(show=True, title="test")
        callout_glw = win.addLayout()
        row_start_idx = 0

        plots.callouts = RenderPlots('callouts', layout=callout_glw, posteriors=[], placefields=[])
        
        # callout_glw = win
        # row_start_idx = 2

        ## Build callout subgraphics:
        # for top_ax, bottom_ax, a_callout_timebin_IDX in zip(top_axis_objs, bottom_axis_objs, plots_data.callout_flat_timebin_IDXs):
        # for top_axis_idx, bottom_axis_idx, a_callout_timebin_IDX in zip(top_axis_indicies, top_axis_indicies, plots_data.callout_flat_timebin_IDXs):
        for col_idx, a_callout_timebin_IDX in zip(np.arange(len(plots_data.callout_flat_timebin_IDXs)), plots_data.callout_flat_timebin_IDXs):
            # print(f'top_axis_idx: {top_axis_idx}, bottom_axis_idx: {bottom_axis_idx}')
            # top_ax = mw.getFigure().add_subplot(2, active_epoch_n_timebins, top_axis_idx)
            # bottom_ax = mw.getFigure().add_subplot(2, active_epoch_n_timebins, bottom_axis_idx)
            top_ax = callout_glw.addPlot(row=row_start_idx, col=col_idx)
            bottom_ax = callout_glw.addPlot(row=row_start_idx+1, col=col_idx)
            curr_callout_axs = build_callout_subgraphic_pyqtgraph(callout_timebin_IDX=a_callout_timebin_IDX, axs=[top_ax, bottom_ax])
            plots.callouts.posteriors.append(top_ax)
            plots.callouts.placefields.append(bottom_ax)

        # win.addItem(callout_glw)
        # plots.mw.show()
        # plots.mw.draw()

    return app, win, plots, plots_data
    



