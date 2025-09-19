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


class TimeSynchronizedPositionDecoderPlotter(AnimalTrajectoryPlottingMixin, TimeSynchronizedPlotterBase):
    """ Plots the decoded position posteriors at a given moment in time. 
    Uses pyqtgraph to render the decoded posteriors
    Its inherited `self.on_window_changed_rate_limited(...)` is called to perform updates

    TODO: refactor, these plotters are all supposed to be for the PfND_TimeDependent class usage I think. 
    
    
    Usage:
    
        TODO: Document

    """
    # Application/Window Configuration Options:
    applicationName = 'TimeSynchronizedPositionDecoderPlotterApp'
    windowName = 'TimeSynchronizedPositionDecoderPlotterWindow'
    
    enable_debug_print = True
    
    
    @property
    def time_window_centers(self):
        """The time_window_centers property."""
        return self.active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,)
    

    @property
    def posterior_variable_to_render(self):
        """The occupancy_mode_to_render property."""
        return self.params.posterior_variable_to_render
    @posterior_variable_to_render.setter
    def posterior_variable_to_render(self, value):
        self.params.posterior_variable_to_render = value
        # on update, be sure to call self._update_plots()
        self._update_plots()
    


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

    
    def __init__(self, active_one_step_decoder, active_two_step_decoder, drop_below_threshold: float=0.0000001, posterior_variable_to_render='p_x_given_n', application_name=None, window_name=None, parent=None):
        """_summary_
        
        ## allows toggling between the various computed occupancies: such as raw counts,  normalized location, and seconds_occupancy
            occupancy_mode_to_render: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        
        """
        super().__init__(application_name=application_name, window_name=(window_name or TimeSynchronizedPositionDecoderPlotter.windowName), parent=parent) # Call the inherited classes __init__ method
    
        self.last_window_index = None
        self.last_window_time = None
        self.active_one_step_decoder = active_one_step_decoder
        self.active_two_step_decoder = active_two_step_decoder
        
        self.setup()
        self.params.debug_print = True # self.enable_debug_print
        if self.params.debug_print:
            print(f'TimeSynchronizedPositionDecoderPlotter: params.debug_print is True, so debugging info will be printed!')
        self.params.posterior_variable_to_render = posterior_variable_to_render
        self.params.drop_below_threshold = drop_below_threshold
        
        self.buildUI()
        self._update_plots()
        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName, debug_view_mode=False)
        # self.params.shared_axis_order = 'row-major'
        self.params.shared_axis_order = 'col-major'
        # self.params.shared_axis_order = None # #TODO 2025-06-30 17:42: - [ ] was like this, but posteriors plotted seem wrong
        
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        self.params.cmap = pg.colormap.get('viridis','matplotlib')
        self.params.debug_view_mode = True
        
        self.params.image_margins = 0.0
        self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(self.active_one_step_decoder.xbin, self.active_one_step_decoder.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        
        self.AnimalTrajectoryPlottingMixin_on_setup()
        

    def _buildGraphics(self):
        ## More Involved Mode:
        self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
        # self.ui.root_view = self.ui.root_graphics_layout_widget.addViewBox()
        ## lock the aspect ratio so pixels are always square
        # self.ui.root_view.setAspectLocked(True)

        ## Create image item
        
        self.ui.imv = pg.ImageItem(border='w')
        # self.ui.root_view.addItem(self.ui.imv)
        # self.ui.root_view.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

        self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=f'PositionDecoder -  t = {self.last_window_time}') # , name=f'PositionDecoder'
        self.ui.root_plot.setObjectName('PositionDecoder')
        self.ui.root_plot.addItem(self.ui.imv, defaultPadding=0.0)  # add ImageItem to PlotItem
        self.ui.root_plot.showAxes(True)
        self.ui.root_plot.hideButtons() # Hides the auto-scale button
        
        # self.ui.root_plot.showAxes(False)        
        self.ui.root_plot.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0)
        # Sets only the panning limits:
        self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])

        ## Sets all limits:
        # _x, _y, _width, _height = self.params.image_bounds_extent # [23.923329354140844, 123.85967782096927, 241.7178791533281, 30.256480996256016]
        # self.ui.root_plot.setLimits(minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1],
        #                             minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        
        debug_view_mode: bool = self.params.debug_view_mode
        print(f'debug_view_mode: {debug_view_mode}')
        
        # debug_view_mode: bool = False
        self.ui.root_plot.setMouseEnabled(x=debug_view_mode, y=debug_view_mode)
        self.ui.root_plot.setMenuEnabled(enableMenu=debug_view_mode)
        
        ## Optional Animal Trajectory Path Plot:
        self.AnimalTrajectoryPlottingMixin_on_buildUI()
        
        # ## Optional Interactive Color Bar:
        # bar = pg.ColorBarItem(values= (0, 1), colorMap=self.params.cmap, width=5, interactive=False) # prepare interactive color bar
        # # Have ColorBarItem control colors of img and appear in 'plot':
        # bar.setImageItem(self.ui.imv, insert_in=self.ui.root_plot)
        
        self.ui.layout.addWidget(self.ui.root_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        # Set the color map:
        self.ui.imv.setColorMap(self.params.cmap)
        ## Set initial view bounds
        # self.ui.root_view.setRange(QtCore.QRectF(0, 0, 600, 600))

    
    @function_attributes(short_name=None, tags=['track_shapes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 11:31', related_items=[])
    def add_track_shapes(self, loaded_track_limits=None, override_ax=None, debug_print:bool=True, defer_draw:bool=False):
        """ Adds the Long and Short track shapes to the plotter:
    
        Usage:
            from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.CreateLinkedWidget_MenuProvider import CreateNewTimeSynchronizedPlotterCommand, CreateNewTimeSynchronizedCombinedPlotterCommand
            from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPositionDecoderPlotter import TimeSynchronizedPositionDecoderPlotter

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
            a_plotter_obj: TimeSynchronizedPositionDecoderPlotter = a_plotter_obj # TimeSynchronizedPositionDecoderPlotter 
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
            active_ax = self.ui.root_plot
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
            print(f'TimeSynchronizedPositionDecoderPlotter.on_window_changed(start_t: {start_t}, end_t: {end_t})')
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
        if self.params.debug_print:
            print(f'TimeSynchronizedPositionDecoderPlotter._update_plots()')
            
        # Update the existing one:
        
        # Update the plots:
        curr_time_window_index = self.last_window_index
        curr_t = self.last_window_time
        
        if (curr_time_window_index is None) or (curr_t is None):
            print(f'WARN: TimeSynchronizedPositionDecoderPlotter._update_plots: curr_time_window_index: {curr_time_window_index}')
            return # return without updating
        
        Assert.is_in(self.posterior_variable_to_render, allowed_variable_list=['p_x_given_n', 'p_x_given_n_and_x_prev'])
        # self.posterior_variable_to_render: allowed values: ['p_x_given_n', 'p_x_given_n_and_x_prev', ...]
        if self.posterior_variable_to_render == 'p_x_given_n':
            image = np.squeeze(self.active_one_step_decoder.p_x_given_n[:, :, curr_time_window_index]).copy()
            image_title = f'p_x_given_n'
        elif self.posterior_variable_to_render == 'p_x_given_n_and_x_prev':
            image = np.squeeze(self.active_two_step_decoder.p_x_given_n_and_x_prev[:, :, curr_time_window_index]).copy()
            image_title = f'p_x_given_n_and_x_prev'
        # elif self.posterior_variable_to_render == 'num_pos_samples_smoothed_occupancy':
        #     image = self.active_time_dependent_placefields.curr_num_pos_samples_smoothed_occupancy_map.copy()
        #     image_title = 'curr_num_pos_samples_occupancy map (smoothed)'
        # elif self.posterior_variable_to_render == 'normalized_occupancy':
        #     image = self.active_time_dependent_placefields.curr_normalized_occupancy.copy()
        #     image_title = 'curr_normalized_occupancy map'
        else:
            raise NotImplementedError
        
        if self.params.drop_below_threshold is not None:
            # image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan # null out the occupancy
            image[np.where(image < self.params.drop_below_threshold)] = np.nan # null out the occupancy
        
        # self.ui.imv.setImage(image, xvals=self.active_time_dependent_placefields.xbin)
        if self.params.shared_axis_order is None:
            self.ui.imv.setImage(image, rect=self.params.image_bounds_extent)
        else:
            self.ui.imv.setImage(image, rect=self.params.image_bounds_extent, axisOrder=self.params.shared_axis_order)
        
        
        # self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
        self.setWindowTitle(f'TimeSynchronizedPositionDecoderPlotter - {image_title} t = {curr_t}')
    
        self.AnimalTrajectoryPlottingMixin_update_plots()

# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedPositionDecoderPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()