from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Neuropy:
from neuropy.core.position import build_position_df_resampled_to_time_windows # used in DecoderResultDisplayingPlot2D.setup()

from neuropy.analyses.placefields import PfND
from .reconstruction import BayesianPlacemapPositionDecoder
from pyphocorehelpers.indexing_helpers import find_neighbours

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # perform_leave_one_aclu_out_decoding_analysis

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
    """ Displays the decoder for 2D position. """
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



def perform_leave_one_aclu_out_decoding_analysis(spikes_df, active_pos_df, active_filter_epochs, filter_epoch_description_list=None, decoding_time_bin_size=0.025):
    """2023-03-03 - Performs a "leave-one-out" decoding analysis where we leave out each neuron one at a time and see how the decoding degrades (which serves as an indicator of the importance of that neuron on the decoding performance).

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_leave_one_aclu_out_decoding_analysis
    """

    def _compute_leave_one_out_decoding(original_decoder):
        """ "Leave-one-out" decoding
        WARNING: this might suck up a ton of memory! 
        """
        original_neuron_ids = np.array(original_decoder.pf.ratemap.neuron_ids) # original_pf.included_neuron_IDs
        one_left_out_decoder_dict = {}
        for i, aclu_to_omit in enumerate(original_neuron_ids):
            subset_included_neuron_ids = np.array([aclu for aclu in original_neuron_ids if aclu != aclu_to_omit]) # get all but the omitted neuron
            one_left_out_decoder_dict[aclu_to_omit] = original_decoder.get_by_id(subset_included_neuron_ids, defer_compute_all=True) # skip computations
            
        return one_left_out_decoder_dict


    spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal') ## get only the pyramidal spikes
    active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object

    # spikes_df = curr_active_pipeline.sess.spikes_df
    # active_pos = curr_active_pipeline.sess.position
    # active_pos_df = sess.position.to_dataframe()

    


    ## Build placefield for the decoder to use:
    original_decoder_pf1D = PfND(deepcopy(spikes_df), deepcopy(active_pos.linear_pos_obj)) # all other settings default
    ## Build the new decoder:
    original_1D_decoder = BayesianPlacemapPositionDecoder(decoding_time_bin_size, original_decoder_pf1D, original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)

    # pretty dang inefficient, as there are 70 cells:
    one_left_out_decoder_dict = _compute_leave_one_out_decoding(original_1D_decoder)

    ## `decode_specific_epochs` for each of the decoders:
    one_left_out_filter_epochs_decoder_result_dict = {}

    ### Loop through and perform the decoding for each epoch. This is the slow part.
    for left_out_aclu, curr_aclu_omitted_decoder in one_left_out_decoder_dict.items():
        filter_epochs_decoder_result = curr_aclu_omitted_decoder.decode_specific_epochs(spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)

        if filter_epoch_description_list is None:
            filter_epoch_description_list = [f'Epoch {i}' for i in range(len(filter_epochs_decoder_result.epoch_description_list))]

        filter_epochs_decoder_result.epoch_description_list = deepcopy(filter_epoch_description_list) # PLOT_ONLY
        one_left_out_filter_epochs_decoder_result_dict[left_out_aclu] = filter_epochs_decoder_result

    one_left_out_omitted_aclu_distance = {}

    for left_out_aclu, left_out_decoder_result in one_left_out_filter_epochs_decoder_result_dict.items():
        ## Compute the impact leaving each aclu out had on the average encoding performance:
        ### 1. The distance between the actual measured position and the decoded position at each timepoint for each decoder. A larger magnitude difference implies a stronger, more positive effect on the decoding quality.

        one_left_out_omitted_aclu_distance[left_out_aclu] = [] # list to hold the distance results from the epochs
        ## Iterate through each of the epochs for the given left_out_aclu (and its decoder), each of which has its own result
        for i in np.arange(left_out_decoder_result.num_filter_epochs):
            curr_time_bin_container = left_out_decoder_result.time_bin_containers[i]
            curr_time_bins = curr_time_bin_container.centers
            ## Need to exclude estimates from bins that didn't have any spikes in them (in general these glitch around):
            curr_total_spike_counts_per_window = np.sum(left_out_decoder_result.spkcount[i], axis=0) # left_out_decoder_result.spkcount[i].shape # (69, 222) - (nCells, nTimeWindowCenters)
            curr_is_time_bin_non_firing = (curr_total_spike_counts_per_window == 0)
            # curr_non_firing_time_bin_indicies = np.where(curr_is_time_bin_non_firing)[0] # TODO: could also filter on a minimum number of spikes larger than zero (e.g. at least 2 spikes are required).
            curr_posterior_container = left_out_decoder_result.marginal_x_list[i]
            curr_posterior = curr_posterior_container.p_x_given_n # TODO: check the posteriors too!
            curr_most_likely_positions = curr_posterior_container.most_likely_positions_1D

            ## Compute the distance metric for this epoch:

            # Interpolate the measured positions to the window center times:
            window_center_measured_pos_x = np.interp(curr_time_bins, active_pos_df.t, active_pos_df.lin_pos)
            # ## PLOT_ONLY: NaN out the most_likely_positions that don't have spikes.
            # curr_most_likely_valid_positions = deepcopy(curr_most_likely_positions)
            # curr_most_likely_valid_positions[curr_non_firing_time_bin_indicies] = np.nan
            
            ## Computed the distance metric finally:
            # is it fair to only compare the valid (windows containing at least one spike) windows?
            curr_omit_aclu_distance = cdist(np.atleast_2d(window_center_measured_pos_x[~curr_is_time_bin_non_firing]), np.atleast_2d(curr_most_likely_positions[~curr_is_time_bin_non_firing]), 'sqeuclidean') # squared-euclidian distance between the two vectors
            # curr_omit_aclu_distance comes back double-wrapped in np.arrays for some reason (array([[659865.11994352]])), so .item() extracts the scalar value
            curr_omit_aclu_distance = curr_omit_aclu_distance.item()
            
            one_left_out_omitted_aclu_distance[left_out_aclu].append(curr_omit_aclu_distance)
    
    # build a dataframe version to hold the distances:
    one_left_out_omitted_aclu_distance_df = pd.DataFrame({'omitted_aclu':np.array(list(one_left_out_omitted_aclu_distance.keys())),
                                                        'distance': list(one_left_out_omitted_aclu_distance.values()),
                                                        'avg_dist': [np.mean(v) for v in one_left_out_omitted_aclu_distance.values()]}
                                                        )
    one_left_out_omitted_aclu_distance_df.sort_values(by='avg_dist', ascending=False, inplace=True) # this sort reveals the aclu values that when omitted had the largest performance decrease on decoding (as indicated by a larger distance)
    most_contributing_aclus = one_left_out_omitted_aclu_distance_df.omitted_aclu.values

    return one_left_out_omitted_aclu_distance_df, most_contributing_aclus