from copy import deepcopy
import param
import numpy as np
import pandas as pd
from attrs import define, field, Factory
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)
from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.indexing_helpers import get_dict_subset
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import NeuronConfigOwningMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_placefields2D, update_plotColorsPlacefield2D

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


# ==================================================================================================================== #
# 2024-04-12 - Decoded Trajectory Plotting on Maze (1D & 2D) - Posteriors and Most Likely Position Paths               #
# ==================================================================================================================== #

from itertools import islice
from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import LapsVisualizationMixin, LineCollection, _plot_helper_add_arrow # plot_lap_trajectories_2d

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def adjust_saturation(rgb, saturation_factor: float):
    """ adjusts the rgb colors by the saturation_factor by converting to HSV space.
    
    """
    import matplotlib.colors as mcolors
    import colorsys
    # Convert RGB to HSV
    hsv = mcolors.rgb_to_hsv(rgb)

    if np.ndim(hsv) < 3:
        # Multiply the saturation by the saturation factor
        hsv[:, 1] *= saturation_factor
        
        # Clip the saturation value to stay between 0 and 1
        hsv[:, 1] = np.clip(hsv[:, 1], 0, 1)
        
    else: 
        # Multiply the saturation by the saturation factor
        hsv[:, :, 1] *= saturation_factor
        # Clip the saturation value to stay between 0 and 1
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
    
    # Convert back to RGB
    return mcolors.hsv_to_rgb(hsv)


# Define a function to create the colormap
def make_saturating_red_cmap(time: float, N_colors:int=256, debug_print:bool=False):
    """ time is between 0.0 and 1.0 
    
    Usage: Test Example:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import make_saturating_red_cmap

        n_time_bins = 5
        cmaps = [make_saturating_red_cmap(float(i) / float(n_time_bins - 1)) for i in np.arange(n_time_bins)]
        for cmap in cmaps:
            cmap
            
    Usage:
        # Example usage
        # You would replace this with your actual data and timesteps
        data = np.random.rand(10, 10)  # Sample data
        n_timesteps = 5  # Number of timesteps

        # Plot data with increasing red for each timestep
        fig, axs = plt.subplots(1, n_timesteps, figsize=(15, 3))
        for i in range(n_timesteps):
            time = i / (n_timesteps - 1)  # Normalize time to be between 0 and 1
            # cmap = make_timestep_cmap(time)
            cmap = make_red_cmap(time)
            axs[i].imshow(data, cmap=cmap)
            axs[i].set_title(f'Timestep {i+1}')
        plt.show()

    """
    colors = np.array([(0, 0, 0), (1, 0, 0)]) # np.shape(colors): (2, 3)
    if debug_print:
        print(f'np.shape(colors): {np.shape(colors)}')
    # Apply a saturation change
    saturation_factor = float(time) # 0.5  # Increase saturation by 1.5 times
    adjusted_colors = adjust_saturation(colors, saturation_factor)
    if debug_print:
        print(f'np.shape(adjusted_colors): {np.shape(adjusted_colors)}')
    adjusted_colors = adjusted_colors.tolist()
    ## Set the alpha of the first color to 0.0 and of the final color to 0.82
    adjusted_colors = [[*v, 0.82] for v in adjusted_colors]
    adjusted_colors[0][-1] = 0.0

    # n_bins = [2]  # Discretizes the interpolation into bins
    return LinearSegmentedColormap.from_list('CustomMap', adjusted_colors, N=N_colors)

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult


@define(slots=False)
class DecodedTrajectoryPlotter:
    """ plots a decoded 1D or 2D trajectory. 
    
    """
    curr_epoch_idx: int = field(default=None)
    a_result: DecodedFilterEpochsResult = field(default=None)
    xbin_centers: NDArray = field(default=None)
    ybin_centers: Optional[NDArray] = field(default=None)
    xbin: NDArray = field(default=None)
    ybin: Optional[NDArray] = field(default=None)

    @property
    def num_filter_epochs(self) -> int:
        """The num_filter_epochs: int property."""
        return self.a_result.num_filter_epochs
    
    @property
    def curr_n_time_bins(self) -> int:
        """The num_filter_epochs: int property."""
        return len(self.a_result.time_bin_containers[self.curr_epoch_idx].centers)




@define(slots=False)
class DecodedTrajectoryMatplotlibPlotter(DecodedTrajectoryPlotter):
    """ plots a decoded 1D or 2D trajectory using matplotlib. 

    Usage:    
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryMatplotlibPlotter

        num_filter_epochs: int = a_result.num_filter_epochs
        a_decoded_traj_plotter = DecodedTrajectoryMatplotlibPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)
        fig, axs, laps_pages = a_decoded_traj_plotter.plot_decoded_trajectories_2d(global_session, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False, use_theoretical_tracks_instead=True)

        integer_slider = a_decoded_traj_plotter.plot_epoch_with_slider_widget(an_epoch_idx=6)
        integer_slider

    """
    prev_heatmaps: List = field(default=Factory(list))
    a_line = field(default=None)
    _out_markers = field(default=None)
    fig = field(default=None)
    axs: NDArray = field(default=None)
    laps_pages: List = field(default=Factory(list))

    ## MAIN PLOT FUNCTION:
    def plot_epoch(self, an_epoch_idx: int, include_most_likely_pos_line: Optional[bool]=None, time_bin_index: Optional[int]=None):
        """ 
        """
        self.curr_epoch_idx = an_epoch_idx

        an_ax = self.axs[0][1]

        assert len(self.xbin_centers) == np.shape(self.a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(self.a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(self.xbin_centers)}"

        a_p_x_given_n = self.a_result.p_x_given_n_list[an_epoch_idx]
        a_most_likely_positions = self.a_result.most_likely_positions_list[an_epoch_idx]
        a_time_bin_edges = self.a_result.time_bin_edges[an_epoch_idx]
        a_time_bin_centers = self.a_result.time_bin_containers[an_epoch_idx].centers

        # n_time_bins: int = len(self.a_result.time_bin_containers[an_epoch_idx].centers)

        assert len(a_time_bin_centers) == len(a_most_likely_positions)

        # heatmaps, a_line, _out_markers, _slider_tuple = add_decoded_posterior_and_trajectory(an_ax, xbin_centers=xbin_centers, a_p_x_given_n=a_p_x_given_n,
        #                                                                      a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=ybin_centers) # , allow_time_slider=True

        # removing existing:
        for a_heatmap in self.prev_heatmaps:
            a_heatmap.remove()
        self.prev_heatmaps.clear()

        if self._out_markers is not None:
            self._out_markers.remove()

        if self.a_line is not None:
            self.a_line.remove()

        ## Perform the plot:
        self.prev_heatmaps, self.a_line, self._out_markers = self._perform_add_decoded_posterior_and_trajectory(an_ax, xbin_centers=self.xbin_centers, a_p_x_given_n=a_p_x_given_n,
                                                                            a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=self.ybin_centers, include_most_likely_pos_line=include_most_likely_pos_line, time_bin_index=time_bin_index) # , allow_time_slider=True

        self.fig.canvas.draw_idle()

    def plot_epoch_with_slider_widget(self, an_epoch_idx: int, include_most_likely_pos_line: Optional[bool]=None):
        import ipywidgets as widgets
        from IPython.display import display

        def integer_slider(update_func):
            """ Captures: valid_aclus
            """
            # slider = widgets.IntSlider(description='epoch_IDX:', min=0, max=a_decoded_traj_plotter.num_filter_epochs-1, value=0)
            slider = widgets.IntSlider(description='time bin:', min=0, max=(self.curr_n_time_bins-1), value=0)

            def on_slider_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    # Call the user-provided update function with the current slider index
                    update_func(change['new'])
            slider.observe(on_slider_change)
            display(slider)


        def update_function(index):
            """ Define an update function that will be called with the current slider index 
            Captures a_result, an_ax, an_epoch_idx
            """
            # print(f'Slider index: {index}')
            a_time_bin_index: int = int(index)
            self.plot_epoch(an_epoch_idx=an_epoch_idx, include_most_likely_pos_line=include_most_likely_pos_line, time_bin_index=a_time_bin_index)

        ## Start by doing a plot:
        self.plot_epoch(an_epoch_idx=an_epoch_idx)
        # n_time_bins: int = a_decoded_traj_plotter.curr_n_time_bins
        # Call the integer_slider function with the update function
        return integer_slider(update_function)
    
    # fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=22, active_page_index=0)
    @classmethod
    def _helper_add_gradient_line(cls, ax, t, x, y, add_markers=False):
        """ Adds a gradient line representing a timeseries of (x, y) positions.

        add_markers (bool): if True, draws points at each (x, y) position colored the same as the underlying line.
        
        
        _helper_add_gradient_line(ax=axs[curr_row][curr_col]],
            t=np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
            x=laps_position_traces[curr_lap_id][0,:],
            y=laps_position_traces[curr_lap_id][1,:]
        )

        """
        # Create a continuous norm to map from data points to colors
        assert len(t) == len(x), f"len(t): {len(t)} != len(x): {len(x)}"
        norm = plt.Normalize(t.min(), t.max())
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(t)
        lc.set_linewidth(2)
        lc.set_alpha(0.85)
        line = ax.add_collection(lc)

        if add_markers:
            # Builds scatterplot markers (points) along the path
            colors_arr = line.get_colors() # (17, 4)
            # segments_arr = line.get_segments() # (16, 2, 2)
            # len(a_most_likely_positions) # 17
            _out_markers = ax.scatter(x=x, y=y, c=colors_arr)
            return line, _out_markers
        else:
            return line

    def plot_decoded_trajectories_2d(self, sess, curr_num_subplots=10, active_page_index=0, plot_actual_lap_lines:bool=False, fixed_columns: int = 2, use_theoretical_tracks_instead: bool = True ):
        """ Plots a MatplotLib 2D Figure with each lap being shown in one of its subplots
        
        Great plotting for laps.
        Plots in a paginated manner.
        
        use_theoretical_tracks_instead: bool = True - # if False, renders all positions the animal traversed over the entire session. Otherwise renders the theoretical (idaal) track.

        ISSUE: `fixed_columns: int = 1` doesn't work due to indexing


        History: based off of plot_lap_trajectories_2d

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_decoded_trajectories_2d
        
            fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)

        
        """

        if use_theoretical_tracks_instead:
            from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance, _perform_plot_matplotlib_2D_tracks
            long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(deepcopy(sess.config))


        def _subfn_chunks(iterable, size=10):
            iterator = iter(iterable)
            for first in iterator:    # stops when iterator is depleted
                def chunk():          # construct generator for next chunk
                    yield first       # yield element from for loop
                    for more in islice(iterator, size - 1):
                        yield more    # yield more elements from the iterator
                yield chunk()         # in outer generator, yield next chunk
            
        def _subfn_build_laps_multiplotter(nfields, linear_plot_data=None):
            """ captures: fixed_columns, (long_track_inst, short_track_inst)
            
            """
            linear_plotter_indicies = np.arange(nfields)
            needed_rows = int(np.ceil(nfields / fixed_columns))
            row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
            mp, axs = plt.subplots(needed_rows, fixed_columns, sharex=True, sharey=True) #ndarray (5,2)
            axs = np.atleast_2d(axs)
            # mp.set_size_inches(18.5, 26.5)

            background_track_shadings = {}
            for a_linear_index in linear_plotter_indicies:
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                if not use_theoretical_tracks_instead:
                    background_track_shadings[a_linear_index] = axs[curr_row][curr_col].plot(linear_plot_data[a_linear_index][0,:], linear_plot_data[a_linear_index][1,:], c='k', alpha=0.2)
                else:
                    # active_config = curr_active_pipeline.sess.config
                    an_ax = axs[curr_row][curr_col]
                    background_track_shadings[a_linear_index] = _perform_plot_matplotlib_2D_tracks(long_track_inst=long_track_inst, short_track_inst=short_track_inst, ax=an_ax)
                
            return mp, axs, linear_plotter_indicies, row_column_indicies, background_track_shadings
        
        def _subfn_add_specific_lap_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, laps_position_traces, lap_time_ranges, use_time_gradient_line=True):
            # Add the lap trajectory:
            for a_linear_index in linear_plotter_indicies:
                curr_lap_id = active_page_laps_ids[a_linear_index]
                curr_row = row_column_indicies[0][a_linear_index]
                curr_col = row_column_indicies[1][a_linear_index]
                curr_lap_time_range = lap_time_ranges[curr_lap_id]
                curr_lap_label_text = 'Lap[{}]: t({:.2f}, {:.2f})'.format(curr_lap_id, curr_lap_time_range[0], curr_lap_time_range[1])
                curr_lap_num_points = len(laps_position_traces[curr_lap_id][0,:])
                if use_time_gradient_line:
                    # Create a continuous norm to map from data points to colors
                    curr_lap_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
                    norm = plt.Normalize(curr_lap_timeseries.min(), curr_lap_timeseries.max())
                    # needs to be (numlines) x (points per line) x 2 (for x and y)
                    points = np.array([laps_position_traces[curr_lap_id][0,:], laps_position_traces[curr_lap_id][1,:]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap='viridis', norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(curr_lap_timeseries)
                    lc.set_linewidth(2)
                    lc.set_alpha(0.85)
                    a_line = axs[curr_row][curr_col].add_collection(lc)
                    # add_arrow(line)
                else:
                    a_line = axs[curr_row][curr_col].plot(laps_position_traces[curr_lap_id][0,:], laps_position_traces[curr_lap_id][1,:], c='k', alpha=0.85)
                    # curr_lap_endpoint = curr_lap_position_traces[curr_lap_id][:,-1].T
                    a_start_arrow = _plot_helper_add_arrow(a_line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                    a_middle_arrow = _plot_helper_add_arrow(a_line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                    a_end_arrow = _plot_helper_add_arrow(a_line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
                    # add_arrow(line[0], position=curr_lap_endpoint, position_mode='abs', direction='right', size=50, color='blue')
                    # add_arrow(line[0], position=None, position_mode='rel', direction='right', size=50, color='blue')
                # add lap text label
                a_lap_label_text = axs[curr_row][curr_col].text(250, 126, curr_lap_label_text, horizontalalignment='right', size=12)
                # PhoWidgetHelper.perform_add_text(p[curr_row, curr_col], curr_lap_label_text, name='lblLapIdIndicator')

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

        # Compute required data from session:
        curr_position_df, lap_specific_position_dfs = LapsVisualizationMixin._compute_laps_specific_position_dfs(sess)
        
        # lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in session.laps.lap_id]

        laps_position_traces_list = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
        laps_time_range_list = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in lap_specific_position_dfs]
        
        num_laps = len(sess.laps.lap_id)
        linear_lap_index = np.arange(num_laps)
        lap_time_ranges = dict(zip(sess.laps.lap_id, laps_time_range_list))
        lap_position_traces = dict(zip(sess.laps.lap_id, laps_position_traces_list))
        
        all_maze_positions = curr_position_df[['x','y']].to_numpy().T # (2, 59308)
        # np.shape(all_maze_positions)
        all_maze_data = [all_maze_positions for i in np.arange(curr_num_subplots)] # repeat the maze data for each subplot. (2, 593080)
        p, axs, linear_plotter_indicies, row_column_indicies, background_track_shadings = _subfn_build_laps_multiplotter(curr_num_subplots, all_maze_data)
        # generate the pages
        laps_pages = [list(chunk) for chunk in _subfn_chunks(sess.laps.lap_id, curr_num_subplots)]
        
        if plot_actual_lap_lines:
            active_page_laps_ids = laps_pages[active_page_index]
            _subfn_add_specific_lap_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, lap_position_traces, lap_time_ranges, use_time_gradient_line=True)
            # plt.ylim((125, 152))
            
        self.fig = p
        self.axs = axs
        self.laps_pages = laps_pages

        return p, axs, laps_pages

    @classmethod
    def _perform_add_decoded_posterior_and_trajectory(cls, an_ax, xbin_centers, a_p_x_given_n, a_time_bin_centers, a_most_likely_positions, ybin_centers=None,
                                                        include_most_likely_pos_line: Optional[bool]=None, time_bin_index: Optional[int]=None, debug_print=False):
        """ Plots the 1D or 2D posterior and most likely position trajectory over the top of an axes created with `fig, axs, laps_pages = plot_decoded_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=8, active_page_index=0, plot_actual_lap_lines=False)`
        
        np.shape(a_time_bin_centers) # 1D & 2D: (12,)
        np.shape(a_most_likely_positions) # 2D: (12, 2)
        np.shape(posterior): 1D: (56, 27);    2D: (12, 6, 57)

        
        time_bin_index: if time_bin_index is not None, only a single time bin will be plotted. Provide this to plot using a slider or programmatically animating.


        Usage:

        # for 1D need to set `ybin_centers = None`
        an_ax = axs[0][0]
        heatmaps, a_line, _out_markers = add_decoded_posterior_and_trajectory(an_ax, xbin_centers=xbin_centers, a_p_x_given_n=a_p_x_given_n,
                                                                            a_time_bin_centers=a_time_bin_centers, a_most_likely_positions=a_most_likely_positions, ybin_centers=ybin_centers)


        """
        posterior_masking_value: float = 0.01
        # full_posterior_opacity: float = 0.92
        full_posterior_opacity: float = 1.0
        
        ## INPUTS: xbin_centers, a_p_x_given_n, a_time_bin_centers, a_most_likely_positions, ybin_centers=None
        posterior = deepcopy(a_p_x_given_n).T # np.shape(posterior): 1D: (56, 27);    2D: (12, 6, 57)
        if debug_print:
            print(f'np.shape(posterior): {np.shape(posterior)}')
        # Create a masked array where all values < 0.25 are masked
        masked_posterior = np.ma.masked_less(posterior, posterior_masking_value)
        # Define a normalization instance which scales data values to the [0, 1] range
        # norm = mcolors.Normalize(vmin=np.nanmin(masked_posterior), vmax=np.nanmax(masked_posterior))
        is_2D: bool = False
        if np.ndim(posterior) >= 3:
            # 2D case
            is_2D = True

        x_values = deepcopy(xbin_centers)  # Replace with your x axis values

        if not is_2D: # 1D case
            # 1D Case:    
            if include_most_likely_pos_line is None:
                include_most_likely_pos_line = True # default to True for 2D
            # Build fake 2D data out of 1D posterior
                
            ## Build the fake y-values from the current axes ylims, which are set when the track graphic is plotted:
            y_min, y_max = an_ax.get_ylim()
            fake_y_width = (y_max - y_min)
            fake_y_center: float = y_min + (fake_y_width / 2.0)
            fake_y_lower_bound: float = (fake_y_center - fake_y_width)
            fake_y_upper_bound: float = (fake_y_center + fake_y_width)

            # ## Build the fake-y values from scratch using hardcoded values:
            # fake_y_width: float = 2.5
            # fake_y_center: float = 140.0
            # fake_y_lower_bound: float = (fake_y_center - fake_y_width)
            # fake_y_upper_bound: float = (fake_y_center + fake_y_width)

            fake_y_num_samples: int = 5
            # y_values = np.linspace(0, 5, 50)    # Replace with your y axis values
            y_values = np.linspace(fake_y_lower_bound, fake_y_upper_bound, fake_y_num_samples) # Replace with your y axis value
            # posterior = np.repeat(a_p_x_given_n, repeats=fake_y_num_samples, axis=0)
            fake_y_num_samples: int = len(a_time_bin_centers)
            fake_y_arr = np.linspace(fake_y_lower_bound, fake_y_upper_bound, fake_y_num_samples)
        else:
            # 2D case:
            assert ybin_centers is not None
            y_values = deepcopy(ybin_centers)
            if include_most_likely_pos_line is None:
                include_most_likely_pos_line = False # default to False for 2D

        # Plot the posterior heatmap _________________________________________________________________________________________ #
        # Note: origin='lower' makes sure that the [0, 0] index is at the bottom left corner.
        n_time_bins = len(a_time_bin_centers)
        assert n_time_bins == np.shape(masked_posterior)[0]

        if not is_2D: # 1D case
            # 1D Case:    
            if time_bin_index is not None:
                assert (time_bin_index < n_time_bins)
                a_heatmap = an_ax.imshow(masked_posterior[time_bin_index, :], aspect='auto', cmap='viridis', alpha=full_posterior_opacity,
                                    extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
                                    origin='lower', interpolation='none') # , norm=norm
            else:
                a_heatmap = an_ax.imshow(masked_posterior, aspect='auto', cmap='viridis', alpha=full_posterior_opacity,
                                    extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
                                    origin='lower', interpolation='none') # , norm=norm
                
            heatmaps = [a_heatmap]

        else:
            # 2D case:
            heatmaps = []
            vmin_global = np.nanmin(posterior)
            vmax_global = np.nanmax(posterior)
            if debug_print:
                print(f'vmin_global: {vmin_global}, vmax_global: {vmax_global}')
            if time_bin_index is not None:
                assert (time_bin_index < n_time_bins)
                cmap='viridis'
                a_heatmap = an_ax.imshow(np.squeeze(masked_posterior[time_bin_index,:,:]), aspect='auto', cmap=cmap, alpha=full_posterior_opacity,
                                extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
                                origin='lower', interpolation='none',
                                vmin=vmin_global, vmax=vmax_global) # , norm=norm
                heatmaps.append(a_heatmap)
            else:
                # plot all of them in a loop:
                time_step_opacity: float = full_posterior_opacity/float(n_time_bins)
                time_step_opacity = max(time_step_opacity, 0.2) # no less than 0.2
                if debug_print:
                    print(f'time_step_opacity: {time_step_opacity}')

                for i in np.arange(n_time_bins):
                    # time = float(i) / (float(n_time_bins) - 1.0)  # Normalize time to be between 0 and 1
                    # cmap = make_timestep_cmap(time)
                    # cmap = make_red_cmap(time)
                    # viridis_obj = mpl.colormaps['viridis'].resampled(8)
                    # cmap = viridis_obj
                    cmap='viridis'
                    a_heatmap = an_ax.imshow(np.squeeze(masked_posterior[i,:,:]), aspect='auto', cmap=cmap, alpha=time_step_opacity,
                                    extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
                                    origin='lower', interpolation='none',
                                    vmin=vmin_global, vmax=vmax_global) # , norm=norm
                    heatmaps.append(a_heatmap)

        # # Add colorbar
        # cbar = plt.colorbar(a_heatmap, ax=an_ax)
        # cbar.set_label('Posterior Probability Density')

        # Add Gradient Most Likely Position Line _____________________________________________________________________________ #
        if include_most_likely_pos_line:
            if not is_2D:
                x = np.atleast_1d([a_most_likely_positions[time_bin_index]])
                y = np.atleast_1d([fake_y_arr[time_bin_index]])
            else:
                # 2D:
                x = np.squeeze(a_most_likely_positions[:,0])
                y = np.squeeze(a_most_likely_positions[:,1])
                
            if time_bin_index is not None:
                ## restrict to single time bin if time_bin_index is not None:
                assert (time_bin_index < n_time_bins)
                a_time_bin_centers = np.atleast_1d([a_time_bin_centers[time_bin_index]])
                x = np.atleast_1d([x[time_bin_index]])
                y = np.atleast_1d([y[time_bin_index]])
                
            if not is_2D: # 1D case
                # a_line = _helper_add_gradient_line(an_ax, t=a_time_bin_centers, x=a_most_likely_positions, y=np.full_like(a_time_bin_centers, fake_y_center))
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, x=x, y=y, add_markers=True)
            else:
                # 2D case
                a_line, _out_markers = cls._helper_add_gradient_line(an_ax, t=a_time_bin_centers, x=x, y=y, add_markers=True)
        else:
            a_line, _out_markers = None, None

        return heatmaps, a_line, _out_markers


from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter

@define(slots=False)
class DecodedTrajectoryPyVistaPlotter(DecodedTrajectoryPlotter):
    """ plots a decoded 3D using pyqtgraph. 
    
    Usage:
    from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins import DecodedTrajectoryPyVistaPlotter
    from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveCustomDataExplorer import InteractiveCustomDataExplorer

    
    curr_active_pipeline.prepare_for_display()
    _out = curr_active_pipeline.display(display_function='_display_3d_interactive_custom_data_explorer', active_session_configuration_context=global_epoch_context,
                                        params_kwargs=dict(should_use_linear_track_geometry=True, **{'t_start': t_start, 't_delta': t_delta, 't_end': t_end}),
                                        )
    iplapsDataExplorer: InteractiveCustomDataExplorer = _out['iplapsDataExplorer']
    pActiveInteractiveLapsPlotter = _out['plotter']
    a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = DecodedTrajectoryPyVistaPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, p=iplapsDataExplorer.p)
    a_decoded_trajectory_pyvista_plotter.build_ui()

    """
    p = field(default=None)
    curr_time_bin_index: int = field(default=0)
    enable_point_labels: bool = field(default=False)
    enable_plot_all_time_bins_in_epoch_mode: bool = field(default=False)


    slider_epoch = field(default=None)
    slider_epoch_time_bin = field(default=None)
    slider_epoch_time_bin_playback_checkbox = field(default=None)
    
    interactive_plotter: PhoInteractivePlotter = field(default=None)
    plotActors = field(default=None)
    data_dict = field(default=None)
    plotActors_CenterLabels = field(default=None)
    data_dict_CenterLabels = field(default=None)


    def build_ui(self):
        """ builds the slider vtk widgets 
        """

        assert self.p is not None
        if self.curr_epoch_idx is None:
            self.curr_epoch_idx = 0
        
        num_filter_epochs: int = self.num_filter_epochs
        curr_num_epoch_time_bins: int = self.curr_n_time_bins

        if self.slider_epoch is None:
            self.slider_epoch = self.p.add_slider_widget(
                callback=lambda value: self.on_update_slider_epoch_idx(int(value)), #storage_engine('epoch', int(value)), # triggering .__call__(self, param='epoch', value)....
                rng=[0, num_filter_epochs-1],
                value=0,
                title="Epoch Idx",
                pointa=(0.35, 0.1),
                pointb=(0.64, 0.1),
                style='modern',
                fmt='%0.0f',
            )


        if not self.enable_plot_all_time_bins_in_epoch_mode:
            if self.slider_epoch_time_bin is None:
                self.slider_epoch_time_bin = self.p.add_slider_widget(
                    callback=lambda value: self.on_update_slider_epoch_time_bin(int(value)), #storage_engine('time_bin', value),
                    rng=[0, curr_num_epoch_time_bins-1],
                    value=0,
                    title="Timebin IDX",
                    pointa=(0.67, 0.1),
                    pointb=(0.98, 0.1),
                    style='modern',
                    # fmt="%d",
                    event_type="always",
                    fmt='%0.0f',
                )

            if (self.interactive_plotter is None) or (self.slider_epoch_time_bin_playback_checkbox is None):
                self.interactive_plotter = PhoInteractivePlotter.init_from_plotter_and_slider(pyvista_plotter=self.p, interactive_timestamp_slider_actor=self.slider_epoch_time_bin, step_size=1, animation_callback_interval_ms=500) # 500ms per time bin
                self.slider_epoch_time_bin_playback_checkbox = self.interactive_plotter.interactive_checkbox_actor




    def update_ui(self):
        """ called to update the epoch_time_bin slider when the epoch_index slider is changed. 
        """
        if (self.slider_epoch_time_bin is not None) and (self.curr_n_time_bins is not None):
            self.slider_epoch_time_bin.GetRepresentation().SetMaximumValue((self.curr_n_time_bins-1))
            self.slider_epoch_time_bin.GetRepresentation().SetValue(self.slider_epoch_time_bin.GetRepresentation().GetMinimumValue()) # set to 0


    def on_update_slider_epoch_idx(self, value: int):
        """ called when the epoch_idx slider changes. 
        """
        # print(f'.on_update_slider_epoch(value: {value})')
        self.curr_epoch_idx = int(value) ## Update `curr_epoch_idx`
        if not self.enable_plot_all_time_bins_in_epoch_mode:
            self.curr_time_bin_index = 0 # change to 0
        else:
            ## otherwise default to a range
            self.curr_time_bin_index = np.arange(self.curr_n_time_bins)

        self.update_ui() # called to update the dependent time_bin slider

        if not self.enable_plot_all_time_bins_in_epoch_mode:
            self.perform_update_plot_single_epoch_time_bin(self.curr_time_bin_index)
        else:
            ## otherwise default to a range
            self.perform_update_plot_epoch_time_bin_range(self.curr_time_bin_index)


    def on_update_slider_epoch_time_bin(self, value: int):
        """ called when the epoch_time_bin within a given epoch_idx slider changes 
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        self.perform_update_plot_single_epoch_time_bin(value=value)
        


    @function_attributes(short_name=None, tags=['main_plot_update'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:03', related_items=[])
    def perform_update_plot_single_epoch_time_bin(self, value: int):
        """ single-time-bin plotting:
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        assert self.p is not None
        self.curr_time_bin_index = int(value) # update `self.curr_time_bin_index` 
        a_posterior_p_x_given_n, a_time_bin_centers = self.get_curr_posterior(an_epoch_idx=self.curr_epoch_idx, time_bin_index=self.curr_time_bin_index)

        ## remove existing actors if they exist and are needed:
        self.perform_clear_existing_decoded_trajectory_plots()

        (self.plotActors, self.data_dict), (self.plotActors_CenterLabels, self.data_dict_CenterLabels) = DecoderRenderingPyVistaMixin.perform_plot_posterior_bars(self.p,
                                                                                                xbin=self.xbin, ybin=self.ybin, xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers,
                                                                                                posterior_p_x_given_n=a_posterior_p_x_given_n, enable_point_labels=self.enable_point_labels)
        

    @function_attributes(short_name=None, tags=['main_plot_update'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-04-25 02:04', related_items=[])
    def perform_update_plot_epoch_time_bin_range(self, value: Optional[NDArray]=None):
        """ multi-time-bin plotting:
        """
        # print(f'.on_update_slider_epoch_time_bin(value: {value})')
        assert self.p is not None
        if value is None:
            value = np.arange(self.curr_n_time_bins)
        self.curr_time_bin_index = value # update `self.curr_time_bin_index` 
        a_posterior_p_x_given_n, a_time_bin_centers = self.get_curr_posterior(an_epoch_idx=self.curr_epoch_idx, time_bin_index=value)

        ## remove existing actors if they exist and are needed:
        self.perform_clear_existing_decoded_trajectory_plots()

        (self.plotActors, self.data_dict), (self.plotActors_CenterLabels, self.data_dict_CenterLabels) = DecoderRenderingPyVistaMixin.perform_plot_posterior_bars(self.p,
                                                                                                xbin=self.xbin, ybin=self.ybin, xbin_centers=self.xbin_centers, ybin_centers=self.ybin_centers,
                                                                                                time_bin_centers=a_time_bin_centers, posterior_p_x_given_n=a_posterior_p_x_given_n, enable_point_labels=self.enable_point_labels)

    def perform_clear_existing_decoded_trajectory_plots(self):
        ## remove existing actors
        from pyphoplacecellanalysis.Pho3D.PyVista.graphs import clear_3d_binned_bars_plots

        if self.plotActors is not None:
            clear_3d_binned_bars_plots(p=self.p, plotActors=self.plotActors)
            self.plotActors.clear()
        if self.data_dict is not None:
            self.data_dict.clear()
            
        if self.plotActors_CenterLabels is not None:
            self.plotActors_CenterLabels.clear()
        if self.data_dict_CenterLabels is not None:
            self.data_dict_CenterLabels.clear()




    def get_curr_posterior(self, an_epoch_idx: int = 0, time_bin_index:Union[int, NDArray]=0):
        a_posterior_p_x_given_n, a_time_bin_centers = self._perform_get_curr_posterior(a_result=self.a_result, an_epoch_idx=an_epoch_idx, time_bin_index=time_bin_index)
        n_epoch_timebins: int = len(a_time_bin_centers)

        if np.ndim(a_posterior_p_x_given_n) > 2:
            assert np.ndim(a_posterior_p_x_given_n) == 3, f"np.ndim(a_posterior_p_x_given_n) should be either 2 or 3, but it is {np.ndim(a_posterior_p_x_given_n)}"
            n_xbins, n_ybins, actual_n_epoch_timebins = np.shape(a_posterior_p_x_given_n) # (5, 312)
            assert n_epoch_timebins == actual_n_epoch_timebins, f"n_epoch_timebins: {n_epoch_timebins} != actual_n_epoch_timebins: {actual_n_epoch_timebins} from np.shape(a_posterior_p_x_given_n) ({np.shape(a_posterior_p_x_given_n)})"
        else:
            n_xbins, n_ybins = np.shape(a_posterior_p_x_given_n) # (5, 312)

        assert n_xbins == np.shape(self.xbin_centers)[0], f"n_xbins: {n_xbins} != np.shape(xbin_centers)[0]: {np.shape(self.xbin_centers)}"
        assert n_ybins == np.shape(self.ybin_centers)[0], f"n_ybins: {n_ybins} != np.shape(ybin_centers)[0]: {np.shape(self.ybin_centers)}"
        # assert len(xbin_centers) == np.shape(a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(xbin_centers)}"
        return a_posterior_p_x_given_n, a_time_bin_centers
    
    @classmethod
    def _perform_get_curr_posterior(cls, a_result, an_epoch_idx: int = 0, time_bin_index: Union[int, NDArray]=0, desired_max_height: float = 50.0):
        """ gets the current posterior for the specified epoch_idx and time_bin_index within the epoch."""
        # a_result.time_bin_containers
        a_posterior_p_x_given_n_all_t = a_result.p_x_given_n_list[an_epoch_idx]
        # assert len(xbin_centers) == np.shape(a_result.p_x_given_n_list[an_epoch_idx])[0], f"np.shape(a_result.p_x_given_n_list[an_epoch_idx]): {np.shape(a_result.p_x_given_n_list[an_epoch_idx])}, len(xbin_centers): {len(xbin_centers)}"
        # a_p_x_given_n = a_result.p_x_given_n_list[an_epoch_idx]
        a_most_likely_positions = a_result.most_likely_positions_list[an_epoch_idx]
        # a_time_bin_edges = a_result.time_bin_edges[an_epoch_idx]
        a_time_bin_centers = a_result.time_bin_containers[an_epoch_idx].centers
        # n_time_bins: int = len(self.a_result.time_bin_containers[an_epoch_idx].centers)
        assert len(a_time_bin_centers) == len(a_most_likely_positions), f"len(a_time_bin_centers): {len(a_time_bin_centers)} != len(a_most_likely_positions): {len(a_most_likely_positions)}"
        # print(f'np.shape(a_posterior_p_x_given_n): {np.shape(a_posterior_p_x_given_n)}') # : (58, 5, 312) - (n_xbins, n_ybins, n_epoch_timebins)
        n_xbins, n_ybins, n_epoch_timebins = np.shape(a_posterior_p_x_given_n_all_t)

        min_v = np.nanmin(a_posterior_p_x_given_n_all_t)
        max_v = np.nanmax(a_posterior_p_x_given_n_all_t)
        # print(f'min_v: {min_v}, max_v: {max_v}')
        multiplier_factor: float = desired_max_height / (float(max_v) - float(min_v))
        # print(f'multiplier_factor: {multiplier_factor}')

        ## get the specific time_bin_index posterior:
        a_posterior_p_x_given_n = np.squeeze(a_posterior_p_x_given_n_all_t[:, :, time_bin_index])
        a_posterior_p_x_given_n = a_posterior_p_x_given_n * multiplier_factor # multiply by the desired multiplier factor
        return a_posterior_p_x_given_n, a_time_bin_centers



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
@metadata_attributes(short_name=None, tags=['pyvista', 'mixin', 'decoder', '3D', 'position'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-27 14:38', related_items=[])
class DecoderRenderingPyVistaMixin:
    """ Implementors render decoded positions and decoder info with PyVista 
    
    Requires:
        self.params
        
    Provides:
    
        Adds:
            ... More?
            
            
    Known Uses:
        InteractivePlaceCellTuningCurvesDataExplorer
    """

    def add_nearest_decoded_position_indicator_circle(self, active_one_step_decoder, debug_print=False):
        """ Adds a red position indicator callback for the current decoded position

        Usage:
            active_one_step_decoder = global_results.pf2D_Decoder
            _update_nearest_decoded_most_likely_position_callback, _conn = add_nearest_decoded_position_indicator_circle(self, active_one_step_decoder, _debug_print = False)

        """
        def _update_nearest_decoded_most_likely_position_callback(start_t, end_t):
            """ Only uses end_t
            Implicitly captures: self, _get_nearest_decoded_most_likely_position_callback
            
            Usage:
                _update_nearest_decoded_most_likely_position_callback(0.0, self.t[0])
                _conn = self.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

            """
            def _get_nearest_decoded_most_likely_position_callback(t):
                """ A callback that when passed a visualization timestamp (the current time to render) returns the most likely predicted position provided by the active_two_step_decoder
                Implicitly captures:
                    active_one_step_decoder, active_two_step_decoder
                Usage:
                    _get_nearest_decoded_most_likely_position_callback(9000.1)
                """
                active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
                active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
                # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
                assert np.shape(active_time_window_variable)[0] == np.shape(active_most_likely_positions)[1], f"timestamps and num positions must be the same but np.shape(active_time_window_variable): {np.shape(active_time_window_variable)} and np.shape(active_most_likely_positions): {np.shape(active_most_likely_positions)}!"
                last_window_index = np.searchsorted(active_time_window_variable, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
                # TODO: CORRECTNESS: why is it returning an index that corresponds to a time later than the current time?
                # for current time t=9000.0
                #     last_window_index: 1577
                #     last_window_time: 9000.5023
                # EH: close enough
                last_window_time = active_time_window_variable[last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
                displayed_time_offset = t - last_window_time # negative value if the window time being displayed is in the future
                if debug_print:
                    print(f'for current time t={t}\n\tlast_window_index: {last_window_index}\n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}')
                return (last_window_time, *list(np.squeeze(active_most_likely_positions[:, last_window_index]).copy()))

            t = end_t # the t under consideration should always be the end_t. This is written this way just for compatibility with the self.sigOnUpdateMeshes (float, float) signature
            curr_t, curr_x, curr_y = _get_nearest_decoded_most_likely_position_callback(t)
            curr_debug_point = [curr_x, curr_y, self.z_fixed[-1]]
            if debug_print:
                print(f'tcurr_debug_point: {curr_debug_point}') # \n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}
            self.perform_plot_location_point('decoded_position_point_plot', curr_debug_point, color='r', render=True)
            return curr_debug_point

        _update_nearest_decoded_most_likely_position_callback(0.0, self.t[0]) # initialize by calling the callback with the current time
        # _conn = pg.SignalProxy(self.sigOnUpdateMeshes, rateLimit=14, slot=_update_nearest_decoded_most_likely_position_callback)
        _conn = self.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

        # TODO: need to assign these results to somewhere in self. Not sure if I need to retain a reference to `active_one_step_decoder`
        # self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor']

        return _update_nearest_decoded_most_likely_position_callback, _conn # return the callback and the connection

    
    @property
    def decoded_trajectory_pyvista_plotter(self) -> DecodedTrajectoryPyVistaPlotter:
        """The decoded_trajectory_pyvista_plotter property."""
        return self.params['decoded_trajectory_pyvista_plotter']


    def add_decoded_posterior_bars(self, a_result: DecodedFilterEpochsResult, xbin: NDArray, xbin_centers: NDArray, ybin: Optional[NDArray], ybin_centers: Optional[NDArray], enable_plot_all_time_bins_in_epoch_mode:bool=True):
        """ adds the decoded posterior to the PyVista plotter
         
          
        Usage:

            a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = iplapsDataExplorer.add_decoded_posterior_bars(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers)

        """
        
        a_decoded_trajectory_pyvista_plotter: DecodedTrajectoryPyVistaPlotter = DecodedTrajectoryPyVistaPlotter(a_result=a_result, xbin=xbin, xbin_centers=xbin_centers, ybin=ybin, ybin_centers=ybin_centers, p=self.p, curr_epoch_idx=0, curr_time_bin_index=0, enable_plot_all_time_bins_in_epoch_mode=enable_plot_all_time_bins_in_epoch_mode)
        a_decoded_trajectory_pyvista_plotter.build_ui()
        self.params['decoded_trajectory_pyvista_plotter'] = a_decoded_trajectory_pyvista_plotter
        return a_decoded_trajectory_pyvista_plotter
    

    @classmethod
    def perform_plot_posterior_bars(cls, p, xbin, ybin, xbin_centers, ybin_centers, posterior_p_x_given_n, time_bin_centers=None, enable_point_labels: bool = True, point_labeling_function=None, point_masking_function=None, posterior_name='P_x_given_n'):
        """ called to perform the mesh generation and add_mesh calls
        
        """
        
        is_single_time_bin_posterior_plot: bool = (np.ndim(posterior_p_x_given_n) < 3)

        if is_single_time_bin_posterior_plot:
            from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars, plot_point_labels

            plotActors, data_dict = plot_3d_binned_bars(p, xbin, ybin, posterior_p_x_given_n, drop_below_threshold=1E-6, name=posterior_name, opacity=0.75)
            # , **({'drop_below_threshold': 1e-06, 'name': 'Occupancy', 'opacity': 0.75} | kwargs)

            if point_labeling_function is None:
                # The full point shown:
                # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
                # Only the z-values
                point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'

            if point_masking_function is None:
                # point_masking_function = lambda points: points[:, 2] > 20.0
                point_masking_function = lambda points: points[:, 2] > 1E-6

            if enable_point_labels:
                plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(p, xbin_centers, ybin_centers, posterior_p_x_given_n, 
                                                                                    point_labels=point_labeling_function, 
                                                                                    point_mask=point_masking_function,
                                                                                    shape='rounded_rect', shape_opacity= 0.5, show_points=False, name=f'{posterior_name}Labels')
            else:
                plotActors_CenterLabels, data_dict_CenterLabels = None, None

        else:
            ## multi-time bin plot:
            from pyphoplacecellanalysis.Pho3D.PyVista.graphs import plot_3d_binned_bars_timeseries

            plotActors, data_dict = plot_3d_binned_bars_timeseries(p=p, xbin=xbin, ybin=ybin, t_bins=time_bin_centers, data=posterior_p_x_given_n,
                                           drop_below_threshold=1E-6, name=posterior_name, opacity=0.75)
            
            if enable_point_labels:
                print(f'WARN: enable_point_labels is not currently implemented for multi-time-bin plotting mode.')

            plotActors_CenterLabels, data_dict_CenterLabels = None, None



        return (plotActors, data_dict), (plotActors_CenterLabels, data_dict_CenterLabels)
    

