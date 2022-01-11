import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


from ...Analysis.reconstruction import BayesianPlacemapPositionDecoder


class DecoderResultDisplayingBaseClass:
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
    
    

class DecoderResultDisplayingPlot2D(DecoderResultDisplayingBaseClass):
    debug_print = False
    
    def __init__(self, decoder: BayesianPlacemapPositionDecoder):
        super(DecoderResultDisplayingPlot2D, self).__init__(decoder)
        
    def setup(self):
        # make a new figure
        # self.fig = plt.figure(figsize=(15,15))
        self.fig, self.axs = plt.subplots(ncols=1, nrows=1, figsize=(15,15), clear=True, constrained_layout=True)
        # self.title_string = f'2D Decoded Positions'
        # self.fig.suptitle(self.title_string)
        self.index = 0
        active_window, active_p_x_given_n, active_most_likely_x_position = self.get_data(self.index) # get the first item
        # self.active_im = self.axs.imshow(active_p_x_given_n, interpolation='none', aspect='auto', vmin=0, vmax=1)
        # self.active_im = plot_single_tuning_map_2D(self.xbin, self.ybin, active_p_x_given_n, self.occupancy, drop_below_threshold=None, ax=self.axs)
        self.active_im = DecoderResultDisplayingPlot2D.plot_single_decoder_result(self.xbin, self.ybin, active_p_x_given_n, drop_below_threshold=None, final_string_components=[f'Decoder Result[i: {self.index}]: time window: {active_window}'], ax=self.axs)
        
        # self.active_most_likely_pos = self.axs.plot([], [], lw=2)
        
        self.active_most_likely_pos_plot = self.axs.scatter([], [], label='most_likely_position', color='k') # see https://stackoverflow.com/questions/42722691/python-matplotlib-update-scatter-plot-from-a-function
        # line, = ax.plot([], [], lw=2)
        
        # self.fig.xticks()
        

    def get_data(self, window_idx):
        active_window = self.decoder.active_time_windows[window_idx] # a tuple with a start time and end time
        active_p_x_given_n = np.squeeze(self.decoder.p_x_given_n[:,:,window_idx]) # same size as occupancy
        
        active_most_likely_x_indicies = self.decoder.most_likely_position_indicies[:,window_idx]
        active_most_likely_x_position = (self.xbin_centers[active_most_likely_x_indicies[0]], self.ybin_centers[active_most_likely_x_indicies[1]])
        
        return active_window, active_p_x_given_n, active_most_likely_x_position

    @staticmethod
    def plot_single_decoder_result(xbin, ybin, p_x_given_n, drop_below_threshold: float=0.0000001, final_string_components=[], ax=None):
        """Plots a single decoder posterior Heatmap
        """
        use_special_overlayed_title = True

        if ax is None:
            ax = plt.gca()

        curr_p_x_given_n = p_x_given_n.copy()
        if drop_below_threshold is not None:
            curr_p_x_given_n[np.where(curr_p_x_given_n < drop_below_threshold)] = np.nan # null out the p_x_given_n below certain values

        ## Seems to work:
        curr_p_x_given_n = np.rot90(curr_p_x_given_n, k=-1)
        curr_p_x_given_n = np.fliplr(curr_p_x_given_n)

        """ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html """
        """ Use the brightness to reflect the confidence in the outcome. Could also use opacity. """
        # mesh_X, mesh_Y = np.meshgrid(xbin, ybin)
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1])
        # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
        extent = (xmin, xmax, ymin, ymax)
        # print(f'extent: {extent}')
        # extent = None
        # We'll also create a black background into which the pixels will fade
        background_black = np.full((*curr_p_x_given_n.shape, 3), 0, dtype=np.uint8)

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

        ax.imshow(background_black, **imshow_shared_kwargs) # add black background image
        im = ax.imshow(curr_p_x_given_n, **main_plot_kwargs) # add the curr_px_given_n image
        ax.axis("off")

        # ax.vlines(200, 'ymin'=0, 'ymax'=1, 'r')
        # ax.set_xticks([25, 50])
        # ax.vline(50, 'r')
        # ax.vlines([50], 0, 1, transform=ax.get_xaxis_transform(), colors='r')
        # ax.vlines([50], 0, 1, colors='r')
        # brev_mode = PlotStringBrevityModeEnum.MINIMAL
        # final_string_components = [full_extended_id_string, pf_firing_rate_string]

        # conventional way:
        final_title = '\n'.join(final_string_components)
        ax.set_title(final_title) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"

        return im

    def display(self, i):
        if DecoderResultDisplayingPlot2D.debug_print:
            print(f'display(i: {i})')
            
        self.index = i
        # curr_ax = self.axs
        # active_window = pho_custom_decoder.active_time_windows[i] # a tuple with a start time and end time
        # active_p_x_given_n = np.squeeze(pho_custom_decoder.p_x_given_n[:,:,i]) # same size as occupancy
        active_window, active_p_x_given_n, active_most_likely_x_position = self.get_data(self.index)
        if DecoderResultDisplayingPlot2D.debug_print:
            print(f'active_window: {active_window}, active_p_x_given_n: {active_p_x_given_n}, active_most_likely_x_position: {active_most_likely_x_position}')

        # Plot the main heatmap for this pfmap:
        # im = plot_single_tuning_map_2D(self.xbin, self.ybin, active_p_x_given_n, self.occupancy, neuron_extended_id=self.ratemap.neuron_extended_ids[cell_idx], drop_below_threshold=drop_below_threshold, brev_mode=brev_mode, plot_mode=plot_mode, ax=curr_ax)
        # self.active_im = plot_single_tuning_map_2D(self.xbin, self.ybin, active_p_x_given_n, self.occupancy, drop_below_threshold=None, ax=self.axs)
        self.active_im = DecoderResultDisplayingPlot2D.plot_single_decoder_result(self.xbin, self.ybin, active_p_x_given_n, drop_below_threshold=None, final_string_components=[f'Decoder Result[i: {self.index}]: time window: {active_window}'], ax=self.axs);
        # self.active_im.set_array(active_p_x_given_n)
        
        # display the most likely position:
        # self.active_most_likely_pos = self.axs.scatter(active_most_likely_x_position[0], active_most_likely_x_position[1])
        # self.active_most_likely_pos.set_data(active_most_likely_x_position[0], active_most_likely_x_position[1])
        self.active_most_likely_pos_plot.set_offsets(np.c_[active_most_likely_x_position[0], active_most_likely_x_position[1]]) # method for updating a scatter_plot
        
        
        # anim = animation.FuncAnimation(figure, func=update_figure, fargs=(bar_rects, iteration), frames=generator, interval=100, repeat=False)
        # return (self.active_im,)
        return self.fig


