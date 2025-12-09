from neuropy.core.user_annotations import function_attributes
import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
from pyphocorehelpers.assertion_helpers import Assert
from neuropy.utils.mixins.dict_representable import overriding_dict_with

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.AnimalTrajectoryPlottingMixin import AnimalTrajectoryPlottingMixin
from attrs import define, field, Factory
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.UserEditableROIMixin import UserEditableROIMixin, Rois

""" Renders a `6 x n_cols` grid of subplots, each showing a heatmap of 2D place cells colored according to that' cells identity color, sorted according to their peak linearized 1D position (`lin_pos`) along the track.
"""

# @metadata_attributes(short_name=None, tags=['debug', 'window', 'visualization', 'activity', 'spikes', 'syncrhonized'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-12-03 15:41', related_items=[])
class TimeSynchronizedPlacefieldActivityDebugPlotter(UserEditableROIMixin, AnimalTrajectoryPlottingMixin, TimeSynchronizedPlotterBase):
    """ Renders a `6 x n_cols` grid of subplots, each showing a heatmap of 2D place cells colored according to that' cells identity color, sorted according to their peak linearized 1D position (`lin_pos`) along the track.
    All cell heatmaps start black, but light up when the cell fires, fading out back to black gradually over the following 3 seconds. 

    """
    # Application/Window Configuration Options:
    applicationName = 'TimeSynchronizedPlacefieldActivityDebugPlotterApp'
    windowName = 'TimeSynchronizedPlacefieldActivityDebugPlotterWindow'
    
    enable_debug_print = True
    
    
    @property
    def time_window_centers(self):
        """The time_window_centers property."""
        return self.active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,)
    

    


    @property
    def last_t(self) -> float:
        """for AnimalTrajectoryPlottingMixin"""
        return (self.last_window_time) or 0.0
    
    @property
    def curr_recent_trajectory(self):
        """The animal's most recent trajectory preceding self.active_time_dependent_placefields.last_t"""
        # Fixed time ago backward:
        earliest_trajectory_start_time = self.last_t - self.params.recent_position_trajectory_max_seconds_ago # gets the earliest start time for the current trajectory to display
        return self.AnimalTrajectoryPlottingMixin_all_time_pos_df.position.time_sliced(earliest_trajectory_start_time, self.last_t)[['t','x','y']] # Get all rows within the most recent time
    
    @property
    def curr_position(self):
        return self.AnimalTrajectoryPlottingMixin_filtered_pos_df.iloc[-1:][['t','x','y']] # Get only the most recent row

    
    def __init__(self, active_one_step_decoder, active_two_step_decoder=None, drop_below_threshold: float=0.0000001, posterior_variable_to_render='p_x_given_n', application_name=None, window_name=None, parent=None, **param_kwargs):
        """Initialize the Placefield Activity Debug Plotter.
        
        Args:
            active_one_step_decoder: Decoder object containing ratemap and placefield data
            active_two_step_decoder: Two-step decoder (optional, kept for compatibility, not used in new implementation)
            drop_below_threshold: Threshold below which placefield values are set to NaN
            posterior_variable_to_render: Kept for compatibility, not used in new implementation
            application_name: Optional application name
            window_name: Optional window name
            parent: Optional parent widget
            **param_kwargs: Additional parameters
        """
        super().__init__(application_name=application_name, window_name=(window_name or TimeSynchronizedPlacefieldActivityDebugPlotter.windowName), parent=parent) # Call the inherited classes __init__ method
        
        self.last_window_index = None
        self.last_window_time = None
        self.active_one_step_decoder = active_one_step_decoder
        self.active_two_step_decoder = active_two_step_decoder
        
        self.setup()
        self.params.debug_print = True # self.enable_debug_print
        self.params.needs_background_image = param_kwargs.pop('needs_background_image', False) # creates `self.ui.bg_imv` IFF this is true. Useful for creating the track shapes.
        
        if self.params.debug_print:
            print(f'TimeSynchronizedPlacefieldActivityDebugPlotter: params.debug_print is True, so debugging info will be printed!')
        self.params.drop_below_threshold = drop_below_threshold
        
        self.buildUI()
        self._update_plots()
        
    def setup(self):
        """Initialize parameters for grid layout, fading duration, and cell colors."""
        from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
        
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName, debug_view_mode=False)
        self.params.shared_axis_order = 'col-major'
        
        # Grid layout parameters: 6 rows × N columns
        self.params.grid_rows = 6
        ratemap = self.active_one_step_decoder.ratemap
        n_neurons = ratemap.n_neurons
        self.params.grid_cols = int(np.ceil(n_neurons / self.params.grid_rows))
        
        # Fading duration: 3 seconds
        self.params.fade_duration_seconds = 3.0
        
        # Build cell colors using existing color mapping utilities
        # colormap_source=None uses pyqtgraph's built-in colormaps (PAL-relaxed_bright)
        neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(
            n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None, return_255_array=False  # type: ignore[arg-type]
        )
        # Store colors as both QColor list and numpy array (normalized 0-1)
        self.params.neuron_qcolors_list = neuron_qcolors_list
        self.params.neuron_colors_ndarray = neuron_colors_ndarray  # Shape: (4, n_neurons) RGBA normalized
        
        # Create mapping from neuron_id to color index
        self.params.neuron_id_to_color_idx = {neuron_id: idx for idx, neuron_id in enumerate(ratemap.neuron_ids)}
        
        # Image bounds and ranges
        self.params.image_margins = 0.0
        self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(
            self.active_one_step_decoder.xbin, self.active_one_step_decoder.ybin, 
            margin=self.params.image_margins, debug_print=self.enable_debug_print
        )
        
        # Base colormap for grayscale placefields (will be tinted by cell colors)
        self.params.cmap = pg.colormap.get('gray', 'matplotlib')
        self.params.debug_view_mode = True
        
        self.AnimalTrajectoryPlottingMixin_on_setup()
    
    def _compute_peak_linearized_positions(self):
        """Compute peak linearized position for each cell and return sorted cell indices.
        
        Returns:
            sorted_cell_indices: Array of cell indices sorted by peak linearized position
            peak_lin_positions: Array of peak linearized positions for each cell
        """
        ratemap = self.active_one_step_decoder.ratemap
        position_df = self.active_one_step_decoder.pf.filtered_pos_df
        
        # Check if lin_pos column exists
        if 'lin_pos' not in position_df.columns:
            if self.params.debug_print:
                print('WARN: lin_pos column not found in position data. Using x coordinate as proxy.')
            lin_pos_data = position_df['x'].values
        else:
            lin_pos_data = position_df['lin_pos'].values
        
        x_coords = position_df['x'].values
        y_coords = position_df['y'].values
        
        # Get placefield data: shape (n_neurons, y_bins, x_bins)
        tuning_curves = ratemap.normalized_tuning_curves
        xbin = self.active_one_step_decoder.xbin
        ybin = self.active_one_step_decoder.ybin
        
        # Compute bin centers
        xbin_centers = (xbin[:-1] + xbin[1:]) / 2
        ybin_centers = (ybin[:-1] + ybin[1:]) / 2
        
        peak_lin_positions = []
        n_neurons = tuning_curves.shape[0]
        
        for neuron_idx in range(n_neurons):
            # Find peak location in 2D placefield
            placefield = tuning_curves[neuron_idx, :, :]
            peak_y_idx, peak_x_idx = np.unravel_index(np.nanargmax(placefield), placefield.shape)
            
            # Get 2D coordinates of peak
            peak_x = xbin_centers[peak_x_idx]
            peak_y = ybin_centers[peak_y_idx]
            
            # Find nearest position sample to peak location
            distances = np.sqrt((x_coords - peak_x)**2 + (y_coords - peak_y)**2)
            nearest_idx = np.nanargmin(distances)
            
            # Get linearized position at nearest sample
            peak_lin_pos = lin_pos_data[nearest_idx]
            peak_lin_positions.append(peak_lin_pos)
        
        peak_lin_positions = np.array(peak_lin_positions)
        
        # Sort by peak linearized position
        sorted_indices = np.argsort(peak_lin_positions)
        
        return sorted_indices, peak_lin_positions
    
    def _get_cell_activity_levels(self, current_time: float):
        """Compute fading activity levels for all cells based on spike times.
        
        Args:
            current_time: Current time in seconds
            
        Returns:
            activity_levels: Array of activity levels (0-1) for each cell
        """
        try:
            spikes_df = self.active_one_step_decoder.pf.filtered_spikes_df
        except AttributeError:
            # Fallback if filtered_spikes_df doesn't exist
            if self.params.debug_print:
                print('WARN: filtered_spikes_df not found. Using empty activity levels.')
            ratemap = self.active_one_step_decoder.ratemap
            return np.zeros(ratemap.n_neurons)
        
        ratemap = self.active_one_step_decoder.ratemap
        n_neurons = ratemap.n_neurons
        neuron_ids = ratemap.neuron_ids
        activity_levels = np.zeros(n_neurons)
        
        # Time window for recent spikes
        time_window_start = current_time - self.params.fade_duration_seconds
        
        # Get spikes within the time window
        recent_spikes = spikes_df[(spikes_df['t'] >= time_window_start) & (spikes_df['t'] <= current_time)]
        
        if len(recent_spikes) > 0:
            # Group by neuron_id and find most recent spike time for each
            for neuron_idx, neuron_id in enumerate(neuron_ids):
                neuron_spikes = recent_spikes[recent_spikes['aclu'] == neuron_id]
                # print(f'aclu: {neuron_id} active.')
                if len(neuron_spikes) > 0:
                    # Get most recent spike time
                    last_spike_time = neuron_spikes['t'].max()
                    # Compute activity level: fade from 1.0 to 0.0 over fade_duration_seconds
                    time_since_spike = current_time - last_spike_time
                    activity_level = max(0.0, 1.0 - (time_since_spike / self.params.fade_duration_seconds))
                    activity_levels[neuron_idx] = activity_level
        
        return activity_levels
    
    def _apply_cell_colors(self, placefield_image: np.ndarray, neuron_idx: int, activity_level: float):
        """Apply cell identity color to a placefield image with activity level.
        
        Args:
            placefield_image: 2D array of placefield values (normalized 0-1)
            neuron_idx: Index of the neuron
            activity_level: Activity level (0-1) for fading effect
            
        Returns:
            colored_image: RGB image array with cell color applied
        """
        # Get cell color (RGBA normalized 0-1)
        cell_color = self.params.neuron_colors_ndarray[:, neuron_idx]  # Shape: (4,) RGBA
        
        # Create RGB image from grayscale placefield
        # Start with black (all zeros)
        h, w = placefield_image.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.float32)
        
        # Apply cell color scaled by activity level and placefield intensity
        for c in range(3):  # RGB channels
            rgb_image[:, :, c] = placefield_image * cell_color[c] * activity_level
        
        return rgb_image

    def _buildGraphics(self):
        """Create 6×N grid layout with ImageItems for each cell placefield."""
        from pyphocorehelpers.print_helpers import generate_html_string
        
        # Initialize arrays to store plot components
        self.ui.img_item_array = []
        self.ui.plot_array = []
        
        # Create root graphics layout widget
        self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
        
        # Get placefield data
        ratemap = self.active_one_step_decoder.ratemap
        tuning_curves = ratemap.normalized_tuning_curves.copy()  # Shape: (n_neurons, y_bins, x_bins)
        occupancy = ratemap.occupancy
        
        # Compute sorted cell indices by peak linearized position
        sorted_cell_indices, _ = self._compute_peak_linearized_positions()
        
        # Create grid of subplots
        for grid_idx, neuron_idx in enumerate(sorted_cell_indices):
            # Calculate grid position (6 rows × N cols)
            row = grid_idx % self.params.grid_rows
            col = grid_idx // self.params.grid_rows
            
            # Get cell ID and create identifier strings
            cell_ID = ratemap.neuron_ids[neuron_idx]
            curr_cell_identifier_string = f'Cell[{cell_ID}]'
            curr_plot_identifier_string = f'TimeSynchronizedPlacefieldActivityDebugPlotter.{curr_cell_identifier_string}'
            
            # Get placefield image for this cell
            image = np.squeeze(tuning_curves[neuron_idx, :, :]).copy()
            
            # Normalize and filter
            with np.errstate(divide='ignore', invalid='ignore'):
                image_max = np.nanmax(image)
                if image_max > 0:
                    image = image / image_max
                else:
                    image = np.zeros_like(image)
                
                if self.params.drop_below_threshold is not None:
                    image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan
            
            # Create plot for this cell
            curr_plot = self.ui.root_graphics_layout_widget.addPlot(
                row=row, col=col, 
                title=generate_html_string(input_str=curr_cell_identifier_string, font_size=2, color='grey')
            )
            curr_plot.setObjectName(curr_plot_identifier_string)
            
            # Create image item (will be updated with RGB data in _update_plots)
            img_item = pg.ImageItem(border='w')
            curr_plot.addItem(img_item, defaultPadding=0.0)
            
            # Configure axes visibility
            is_first_column = (col == 0)
            is_last_row = (row == self.params.grid_rows - 1)
            
            curr_plot.showAxes(False)
            if is_last_row:
                curr_plot.showAxes('x', True)
                curr_plot.showAxis('bottom', show=True)
            if is_first_column:
                curr_plot.showAxes('y', True)
                curr_plot.showAxis('left', show=True)
            
            curr_plot.hideButtons()
            curr_plot.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0, update=False, disableAutoRange=True)
            curr_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])
            curr_plot.setMouseEnabled(x=False, y=False)
            curr_plot.setMenuEnabled(enableMenu=False)
            
            # Link axes to first plot for synchronized viewing
            if grid_idx > 0:
                prev_plot_item = self.ui.plot_array[0]
                curr_plot.setXLink(prev_plot_item)
                curr_plot.setYLink(prev_plot_item)
            
            # Store components
            self.ui.img_item_array.append(img_item)
            self.ui.plot_array.append(curr_plot)
        
        # Add root graphics layout widget to main layout
        self.ui.layout.addWidget(self.ui.root_graphics_layout_widget, 0, 0)
        
        # Note: AnimalTrajectoryPlottingMixin is not used in grid layout (would need per-cell plots)
        # self.AnimalTrajectoryPlottingMixin_on_buildUI()

    
    @function_attributes(short_name=None, tags=['track_shapes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 11:31', related_items=[])
    def add_track_shapes(self, loaded_track_limits=None, override_ax=None, debug_print:bool=True, defer_draw:bool=False):
        """ Adds the Long and Short track shapes to the plotter:
    
        Usage:
            from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.CreateLinkedWidget_MenuProvider import CreateNewTimeSynchronizedPlotterCommand, CreateNewTimeSynchronizedCombinedPlotterCommand
            from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlacefieldActivityDebugPlotter import TimeSynchronizedPlacefieldActivityDebugPlotter

            active_config_name = None # kwargs.get('active_config_name', None)
            active_config_name = global_any_name
            active_context = None
            display_output = {}
            active_pf_2D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_dt', None)
            active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None)
            active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
            _out = CreateNewTimeSynchronizedPlotterCommand(spike_raster_window, active_pf_2D_dt=active_pf_2D_dt, plotter_type='decoder', curr_active_pipeline=curr_active_pipeline, active_context=active_context, active_config_name=active_config_name, display_output=display_output, action_identifier='actionTimeSynchronizedDecoderPlotter')
            _out.execute()
            a_plotter_obj, _a_conn = display_output['synchronizedPlotter_decoder']
            active_ax = a_plotter_obj.ui.root_plot
            a_plotter_obj: TimeSynchronizedPlacefieldActivityDebugPlotter = a_plotter_obj # TimeSynchronizedPlacefieldActivityDebugPlotter 
            (long_rects_outputs, short_rects_outputs) = a_plotter_obj.add_track_shapes() ## add the static track shapes
            a_plotter_obj.update(t=active_2d_plot.active_window_start_time, defer_render=False)

        """
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager, long_short_display_config_manager
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance
        from neuropy.utils.mixins.dict_representable import overriding_dict_with

        if loaded_track_limits is None:
            loaded_track_limits = {'long_xlim': np.array([59.0774, 228.69]),
                'long_unit_xlim': np.array([0.205294, 0.794698]),
                'short_xlim': np.array([94.0156, 193.757]),
                'short_unit_xlim': np.array([0.326704, 0.673304]),
                'long_ylim': np.array([138.164, 146.12]),
                'long_unit_ylim': np.array([0.48012, 0.507766]),
                'short_ylim': np.array([138.021, 146.263]),
                'short_unit_ylim': np.array([0.479622, 0.508264]),
            }

        ## INPUTS: active_ax

        long_track_inst, short_track_inst = LinearTrackInstance.init_LS_tracks_from_loaded_track_limits(loaded_track_limits=loaded_track_limits)
        # # Centered above and below the y=0.0 line:
        # long_offset = (long_track_inst.grid_bin_bounds.center_point[0], 0.75)
        # short_offset = (short_track_inst.grid_bin_bounds.center_point[0], -0.75)

        if override_ax is None:
            # Use first plot in grid as default (or could use root_graphics_layout_widget)
            if hasattr(self, 'ui') and hasattr(self.ui, 'plot_array') and len(self.ui.plot_array) > 0:
                active_ax = self.ui.plot_array[0]
            else:
                raise ValueError("No plots available for track shapes. Ensure _buildGraphics() has been called.")
        else:
            active_ax = override_ax

        ## INPUTS: track_ax, rotate_to_vertical, perform_autoscale

        # long_track_combined_collection, long_rect_items, long_rects = long_track_inst.plot_rects(active_ax)
        # short_track_combined_collection, short_rect_items, short_rects = short_track_inst.plot_rects(active_ax)

        # long_epoch_matplotlib_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
        # long_kwargs = deepcopy(long_epoch_matplotlib_config)
        long_kwargs = dict(edgecolor="#0000FFC3", facecolor="#0000FF8D")
        # long_kwargs = dict(edgecolor='#000000ff', facecolor='#000000ff')
        # long_rects_outputs = long_track_inst.plot_rects(active_ax, offset=long_offset, matplotlib_rect_kwargs_override=overriding_dict_with(lhs_dict=long_kwargs, **dict(linewidth=2, zorder=-99)))
        long_rects_outputs = long_track_inst.plot_rects(active_ax, matplotlib_rect_kwargs_override=overriding_dict_with(lhs_dict=long_kwargs, **dict(linewidth=2, zorder=-99)))

        # short_epoch_matplotlib_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
        # short_kwargs = deepcopy(short_epoch_matplotlib_config)
        short_kwargs = dict(edgecolor="#FF0000C3", facecolor="#FF00008D")
        # short_kwargs = dict(edgecolor='#000000ff', facecolor='#000000ff')
        # short_rects_outputs = short_track_inst.plot_rects(active_ax, offset=short_offset, matplotlib_rect_kwargs_override=overriding_dict_with(lhs_dict=short_kwargs, **dict(linewidth=2, zorder=-99)))
        short_rects_outputs = short_track_inst.plot_rects(active_ax, matplotlib_rect_kwargs_override=overriding_dict_with(lhs_dict=short_kwargs, **dict(linewidth=2, zorder=-99)))
            
        # if not defer_draw:
        #     if override_ax is None:
        #         self.redraw()
        #     else:
        #         override_ax.get_figure().canvas.draw_idle()

        return (long_rects_outputs, short_rects_outputs)



    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #

    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.params.debug_print:
            print(f'TimeSynchronizedPlacefieldActivityDebugPlotter.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        # if self.enable_debug_print:
        #     profiler = pg.debug.Profiler(disabled=True, delayed=True)
        # self.update(end_t, defer_render=False)
        self.update(start_t, defer_render=False)
        if self.params.debug_print:
            print('\tFinished calling _update_plots()')


    def update(self, t, defer_render=False):
        # Finds the nearest previous decoded position for the time t:
        self.last_window_index = np.searchsorted(self.time_window_centers, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
        self.last_window_time = self.time_window_centers[self.last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
        # Update the plots:
        if not defer_render:
            self._update_plots()


    def _update_plots(self):
        """Update all cell displays with fading activity visualization."""
        if self.params.debug_print:
            print(f'TimeSynchronizedPlacefieldActivityDebugPlotter._update_plots()')
        
        curr_t = self.last_window_time
        
        if curr_t is None:
            if self.params.debug_print:
                print(f'WARN: TimeSynchronizedPlacefieldActivityDebugPlotter._update_plots: curr_t is None')
            return
        
        # Get placefield data
        ratemap = self.active_one_step_decoder.ratemap
        tuning_curves = ratemap.normalized_tuning_curves.copy()  # Shape: (n_neurons, y_bins, x_bins)
        occupancy = ratemap.occupancy
        
        # Get sorted cell indices (computed once, cached if needed)
        if not hasattr(self, '_cached_sorted_indices'):
            self._cached_sorted_indices, _ = self._compute_peak_linearized_positions()
        sorted_cell_indices = self._cached_sorted_indices
        
        # Get activity levels for all cells
        activity_levels = self._get_cell_activity_levels(curr_t)
        
        # Update each cell's display
        for grid_idx, neuron_idx in enumerate(sorted_cell_indices):
            if grid_idx >= len(self.ui.img_item_array):
                continue  # Safety check
            
            # Get placefield image for this cell
            image = np.squeeze(tuning_curves[neuron_idx, :, :]).copy()
            
            # Normalize and filter
            with np.errstate(divide='ignore', invalid='ignore'):
                image_max = np.nanmax(image)
                if image_max > 0:
                    image = image / image_max
                else:
                    image = np.zeros_like(image)
                
                if self.params.drop_below_threshold is not None:
                    image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan
            
            # Apply cell color and activity level
            activity_level = activity_levels[neuron_idx]
            colored_image = self._apply_cell_colors(image, neuron_idx, activity_level)
            
            # Convert to format suitable for pyqtgraph ImageItem
            # pyqtgraph expects (H, W) for grayscale or (H, W, 3) for RGB color
            # We need to transpose and handle axis order, and convert to uint8
            if self.params.shared_axis_order == 'col-major':
                # Image is (y_bins, x_bins), need to transpose for display
                colored_image_display = np.transpose(colored_image, (1, 0, 2))
            else:
                colored_image_display = colored_image
            
            # Convert to uint8 format (0-255) for pyqtgraph RGB display
            colored_image_display = np.clip(colored_image_display * 255.0, 0, 255).astype(np.uint8)
            
            # Update image item
            img_item = self.ui.img_item_array[grid_idx]
            # pyqtgraph ImageItem can handle RGB images as (H, W, 3) uint8 arrays
            img_item.setImage(colored_image_display, rect=self.params.image_bounds_extent, autoLevels=False)
        
        # Update window title
        self.setWindowTitle(f'TimeSynchronizedPlacefieldActivityDebugPlotter - t = {curr_t:.2f}')
        
        # Note: AnimalTrajectoryPlottingMixin_update_plots() not called as trajectory is not shown in grid layout


    @function_attributes(short_name=None, tags=['video', 'export', 'mp4', 'avi', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-11-24 23:09', related_items=[])
    def export_video(self, output_path: str, start_t: Optional[float] = None, end_t: Optional[float] = None, fps: float = 30.0, width: Optional[int] = None, height: Optional[int] = None, progress_print: bool = True, debug_print: bool = False):
        """Efficiently export a video from the TimeSynchronizedPlacefieldActivityDebugPlotter instance (faster than real-time playback)
        
        This method iterates through time points, updates the plotter, captures frames using
        pyqtgraph's ImageExporter, and saves them as a video using OpenCV.
        
        Args:
            output_path: Path to save the output video file (e.g., 'output/videos/decoder_video.avi')
            start_t: Start time for video export. If None, uses the first available time window center.
            end_t: End time for video export. If None, uses the last available time window center.
            fps: Frames per second for the output video (default: 30.0)
            width: Width of exported frames in pixels. If None, uses current widget width.
            height: Height of exported frames in pixels. If None, uses current widget height.
            progress_print: Whether to print progress messages (default: True)
            debug_print: Whether to print debug information (default: False)
            
        Returns:
            Path: Path to the saved video file
            
        Usage:
            plotter = TimeSynchronizedPlacefieldActivityDebugPlotter(...)
            video_path = plotter.export_video('output/videos/decoder.avi', start_t=100.0, end_t=200.0, fps=30.0)
        """
        from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
        from pyphoplacecellanalysis.External.pyqtgraph import functions as fn
        from pathlib import Path
        import cv2
        import sys
        
        # Get time window centers
        time_window_centers = self.time_window_centers
        if len(time_window_centers) == 0:
            raise ValueError("No time window centers available for video export")
        
        # Determine time range
        if start_t is None:
            start_t = float(time_window_centers[0])
        if end_t is None:
            end_t = float(time_window_centers[-1])
        
        # Find valid time indices
        start_idx = np.searchsorted(time_window_centers, start_t, side='left')
        end_idx = np.searchsorted(time_window_centers, end_t, side='right')
        
        if start_idx >= end_idx:
            raise ValueError(f"Invalid time range: start_t={start_t}, end_t={end_t}. No valid frames found.")
        
        # Subsample frame indices based on fps to reduce processing
        # Calculate desired time step between frames (in seconds)
        desired_time_step = 1.0 / fps if fps > 0 else float('inf')
        
        # Get all candidate frame indices
        all_frame_indices = np.arange(start_idx, end_idx)
        all_frame_times = time_window_centers[all_frame_indices]
        
        # Subsample frames based on desired time step
        if desired_time_step < float('inf') and len(all_frame_indices) > 1:
            # Start with the first frame
            subsampled_indices = [all_frame_indices[0]]
            last_selected_time = all_frame_times[0]
            
            # Select frames that are at least desired_time_step apart
            for i in range(1, len(all_frame_indices)):
                current_time = all_frame_times[i]
                time_since_last = current_time - last_selected_time
                
                if time_since_last >= desired_time_step:
                    subsampled_indices.append(all_frame_indices[i])
                    last_selected_time = current_time
            
            frame_indices = np.array(subsampled_indices)
        else:
            # If fps is 0 or invalid, use all frames
            frame_indices = all_frame_indices
        
        n_frames: int = len(frame_indices)
        if n_frames == 0:
            raise ValueError(f"No frames to export after subsampling at {fps} fps")
        
        if progress_print:
            total_available_frames = len(all_frame_indices)
            print(f'Exporting video: {n_frames} frames (from {total_available_frames} available) from t={time_window_centers[start_idx]:.2f} to t={time_window_centers[end_idx-1]:.2f} at {fps} fps')
        
        # Get widget dimensions
        if width is None or height is None:
            widget_size = self.ui.root_graphics_layout_widget.size()
            if width is None:
                width = widget_size.width()
            if height is None:
                height = widget_size.height()
        
        # Disable debug printing during export for performance
        original_debug_print = self.params.debug_print
        self.params.debug_print = debug_print
        
        # Helper to convert QImage to BGR array for OpenCV (contiguous uint8 for compatibility)
        def qimage_to_bgr(qimage):
            img_array = fn.ndarray_from_qimage(qimage)
            # Handle ARGB32 format conversion based on byte order
            if img_array.shape[2] == 4:
                # ARGB32 format - extract RGB channels based on byte order
                if sys.byteorder == 'little':
                    # Little-endian: channels are [B, G, R, A] in memory
                    bgr = img_array[:, :, :3]  # B, G, R (first 3 channels)
                else:
                    # Big-endian: channels are [A, R, G, B] in memory
                    bgr = img_array[:, :, [3, 2, 1]]  # B, G, R from indices 3,2,1
            elif img_array.shape[2] == 3:
                # Already RGB format, convert to BGR for OpenCV
                bgr = img_array[:, :, ::-1]
            else:
                raise ValueError(f"Unexpected image format with {img_array.shape[2]} channels")
            # Ensure contiguous uint8 array for OpenCV compatibility
            return np.ascontiguousarray(bgr, dtype=np.uint8)
        
        out = None  # Initialize to None for proper cleanup
        try:
            # Create ImageExporter for the entire GraphicsLayoutWidget (grid layout)
            exporter = ImageExporter(self.ui.root_graphics_layout_widget)
            exporter.parameters()['width'] = width
            exporter.parameters()['height'] = height
            exporter.parameters()['antialias'] = True
            
            # Process events to ensure widget is rendered
            QtWidgets.QApplication.processEvents()
            
            # Capture first frame to get actual output dimensions (may differ from requested)
            first_frame_idx = frame_indices[0]
            self.update(time_window_centers[first_frame_idx], defer_render=False)
            QtWidgets.QApplication.processEvents()
            first_qimage = exporter.export(toBytes=True)
            first_bgr = qimage_to_bgr(first_qimage)
            actual_height, actual_width = first_bgr.shape[:2]
            del first_qimage  # Free memory
            
            # Set up output path and directory
            video_filepath = Path(output_path).resolve()
            video_parent_path = video_filepath.parent
            if not video_parent_path.exists():
                if progress_print:
                    print(f'Creating output directory: {video_parent_path}')
                video_parent_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize video writer with actual frame dimensions
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(str(video_filepath), fourcc, fps, (actual_width, actual_height), isColor=True)
            
            if not out.isOpened():
                raise RuntimeError(f"Failed to open video writer for {video_filepath}")
            
            # Write the first frame we already captured
            out.write(first_bgr)
            del first_bgr  # Free memory
            
            # Streaming capture and write: process remaining frames one at a time
            progress_print_every_n_frames = max(1, n_frames // 20)
            for i, frame_idx in enumerate(frame_indices):
                if i == 0:
                    # First frame already written
                    if progress_print:
                        print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                    continue
                    
                if progress_print and (i % progress_print_every_n_frames == 0 or i == n_frames - 1):
                    print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                
                # Update plotter to current time
                t = time_window_centers[frame_idx]
                self.update(t, defer_render=False)
                
                # Process events to ensure rendering
                QtWidgets.QApplication.processEvents()
                
                # Capture frame and write directly to video
                qimage = exporter.export(toBytes=True)
                bgr_array = qimage_to_bgr(qimage)
                out.write(bgr_array)
        finally:
            # Always close video writer (if opened) and restore debug print setting
            if out is not None:
                out.release()
            self.params.debug_print = original_debug_print
        
        if progress_print:
            print(f'Video exported successfully to: {video_filepath}')
        
        return video_filepath


# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedPlacefieldActivityDebugPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()



