import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# Neuropy:
# from neuropy.utils.mixins.binning_helpers import BinnedPositionsMixin, bin_pos_nD, build_df_discretized_binned_position_columns

from .reconstruction import BayesianPlacemapPositionDecoder
from pyphocorehelpers.indexing_helpers import find_neighbours


def build_position_df_time_window_idx(active_pos_df, curr_active_time_windows, debug_print=False):
    """ adds the time_window_idx column to the active_pos_df
    Usage:
        curr_active_time_windows = np.array(pho_custom_decoder.active_time_windows)
        active_pos_df = build_position_df_time_window_idx(sess.position.to_dataframe(), curr_active_time_windows)
    """
    active_pos_df['time_window_idx'] = np.full_like(active_pos_df['t'], -1, dtype='int')
    starts = curr_active_time_windows[:,0]
    stops = curr_active_time_windows[:,1]
    num_slices = len(starts)
    if debug_print:
        print(f'starts: {np.shape(starts)}, stops: {np.shape(stops)}, num_slices: {num_slices}')
    for i in np.arange(num_slices):
        active_pos_df.loc[active_pos_df[active_pos_df.position.time_variable_name].between(starts[i], stops[i], inclusive='both'), ['time_window_idx']] = int(i) # set the 'time_window_idx' identifier on the object
    active_pos_df['time_window_idx'] = active_pos_df['time_window_idx'].astype(int) # ensure output is the correct datatype
    return active_pos_df

# def build_position_df_discretized_binned_positions(active_pos_df, active_computation_config, xbin_values=None, ybin_values=None, debug_print=False):
#     """ Adds the 'binned_x' and 'binned_y' columns to the position dataframe 
    
#     - [ ] TODO: CORRECTNESS: POTENTIAL_BUG: I notice that the *bin_centers are being passed here from its call as opposed to the .xbin, .ybin themselves. Is this an issue?
    
    
#     NOTE: This is independently re-implemented from a static version in neuropy.analyses.placefields.PfND with the same name
#         https://github.com/CommanderPho/NeuroPy/blob/feature%2Fpho_variant/neuropy/analyses/placefields.py#L813
        
        
    
#     """
    
#     active_pos_df, (xbin, ybin), bin_infos = build_df_discretized_binned_position_columns(active_pos_df, bin_values=(xbin_values, ybin_values), active_computation_config=active_computation_config, force_recompute=False, debug_print=debug_print)
#     return active_pos_df, xbin, ybin, bin_infos


def build_position_df_resampled_to_time_windows(active_pos_df, time_bin_size=0.02):
    """ Note that this returns a TimedeltaIndexResampler, not a dataframe proper. To get the real dataframe call .nearest() on output. """
    position_time_delta = pd.to_timedelta(active_pos_df[active_pos_df.position.time_variable_name], unit="sec")
    active_pos_df['time_delta_sec'] = position_time_delta
    active_pos_df = active_pos_df.set_index('time_delta_sec')
    window_resampled_pos_df = active_pos_df.resample(f'{time_bin_size}S', base=0)#.nearest() # '0.02S' 0.02 second bins
    return window_resampled_pos_df



    
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
        # self.active_im = self.axs.imshow(active_p_x_given_n, interpolation='none', aspect='auto', vmin=0, vmax=1)
        # self.active_im = plot_single_tuning_map_2D(self.xbin, self.ybin, active_p_x_given_n, self.occupancy, drop_below_threshold=None, ax=self.axs)
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


