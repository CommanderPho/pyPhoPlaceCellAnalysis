# from neuropy.core.user_annotations import function_attributes
import numpy as np
import signal
import threading
import pandas as pd
from qtpy import QtCore, QtWidgets
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import imageio

# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
from pyphocorehelpers.assertion_helpers import Assert
from neuropy.utils.mixins.dict_representable import overriding_dict_with
from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch, EpochHelpers
from neuropy.utils.mixins.time_slicing import TimePointEventAccessor

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.AnimalTrajectoryPlottingMixin import AnimalTrajectoryPlottingMixin
from attrs import define, field, Factory
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.UserEditableROIMixin import UserEditableROIMixin, Rois

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedGenericPlotterLayer import TimeSynchronizedGenericPlotterLayer, LayerDisplayConfig


def _decoder_bin_left_right_edges(active_one_step_decoder, centers: NDArray) -> Tuple[NDArray, NDArray]:
    """Per-bin (left, right) edges aligned with decoded frames. Uses time_bin_container when present, else midpoint inference from centers."""
    centers = np.asarray(centers, dtype=float)
    n: int = len(centers)
    if n == 0:
        return np.array([]), np.array([])
    if hasattr(active_one_step_decoder, 'time_bin_container') and active_one_step_decoder.time_bin_container is not None:
        tbc = active_one_step_decoder.time_bin_container
        left = np.asarray(tbc.left_edges, dtype=float)
        right = np.asarray(tbc.right_edges, dtype=float)
        if len(left) >= n and len(right) >= n:
            return left[:n], right[:n]
    if n == 1:
        d = 1.0
    else:
        d = float(np.median(np.diff(centers)))
    mid = (centers[:-1] + centers[1:]) / 2.0
    left = np.empty(n, dtype=float)
    left[0] = centers[0] - (mid[0] - centers[0]) if n > 1 else centers[0] - d / 2.0
    left[1:] = mid
    right = np.empty(n, dtype=float)
    right[-1] = centers[-1] + (centers[-1] - mid[-1]) if n > 1 else centers[-1] + d / 2.0
    right[:-1] = mid
    return left, right


def _included_posterior_bin_indices_for_viewport(active_one_step_decoder, centers: NDArray, viewport_start: float, viewport_end: float) -> NDArray:
    """Decoded bin indices with [left,right] strictly inside [viewport_start, viewport_end]; if none, single bin from clamped searchsorted(viewport_start). Used for viewport aggregation when TimeSynchronizedPositionDecoderPlotter.use_all_active_viewport_timebins is True."""
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    if n == 0:
        return np.array([], dtype=np.intp)
    left, right = _decoder_bin_left_right_edges(active_one_step_decoder, centers)
    strict = [i for i in range(n) if left[i] >= viewport_start and right[i] <= viewport_end]
    if strict:
        return np.asarray(strict, dtype=np.intp)
    i0 = int(np.searchsorted(centers, viewport_start, side='left'))
    i0 = max(0, min(i0, n - 1))
    return np.array([i0], dtype=np.intp)


class TimeSynchronizedPositionDecoderPlotter(UserEditableROIMixin, AnimalTrajectoryPlottingMixin, TimeSynchronizedPlotterBase):
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

    
    def __init__(self, active_one_step_decoder, active_two_step_decoder, drop_below_threshold: float=0.0000001, posterior_variable_to_render='p_x_given_n', application_name=None, window_name=None, parent=None, **param_kwargs):
        """_summary_
        
        ## allows toggling between the various computed occupancies: such as raw counts,  normalized location, and seconds_occupancy
            occupancy_mode_to_render: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        
        """
        super().__init__(application_name=application_name, window_name=(window_name or TimeSynchronizedPositionDecoderPlotter.windowName), parent=parent) # Call the inherited classes __init__ method
    
        self.last_window_index = None
        self.last_window_time = None
        self.last_included_posterior_bin_indices: Optional[NDArray] = None
        self.active_one_step_decoder = active_one_step_decoder
        self.active_two_step_decoder = active_two_step_decoder
        
        self.setup()
        self.params.debug_print = True # self.enable_debug_print
        self.params.needs_background_image = param_kwargs.pop('needs_background_image', False) # creates `self.ui.bg_imv` IFF this is true. Useful for creating the track shapes.
        self.params.show_posteriors = param_kwargs.pop('show_posteriors', True)
        self.params.decoded_time_bins_info_df = param_kwargs.pop('decoded_time_bins_info_df', None) ## 
        self.use_all_active_viewport_timebins = param_kwargs.pop('use_all_active_viewport_timebins', False)

        if self.params.debug_print:
            print(f'TimeSynchronizedPositionDecoderPlotter: params.debug_print is True, so debugging info will be printed!')
        self.params.posterior_variable_to_render = posterior_variable_to_render
        self.params.drop_below_threshold = drop_below_threshold
        
        
        self.buildUI() # calls `self._buildGraphics()`
        self._update_plots()
        

        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName, debug_view_mode=False)
        # self.params.shared_axis_order = 'row-major'
        self.params.shared_axis_order = 'col-major'
        # self.params.shared_axis_order = None # #TODO 2025-06-30 17:42: - [ ] was like this, but posteriors plotted seem wrong
        self.params.decoded_time_bins_info_df = None
        
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        # self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        self.params.cmap = pg.colormap.get('viridis','matplotlib')
        # self.params.cmap_PBE = pg.colormap.get('magma','matplotlib')
        self.params.cmap_PBE = pg.colormap.get('brg','matplotlib')
        
        self.params.image_visual_display_config = {'default': dict(border=pg.mkPen('white', width=1), cmap=pg.colormap.get('viridis','matplotlib')),
                                                   'PBE': dict(border=pg.mkPen('r', width=2), cmap=pg.colormap.get('brg','matplotlib')),
        }

        # curr_time_bin_type: str = 'default'
        # curr_time_bin_type: str = 'PBE'        
        # curr_time_bin_config: Dict = self.params.image_visual_display_config[curr_time_bin_type]
        
        # # plot_item.getAxis('left').setPen(curr_time_bin_config['axis']) # Change left axis color
        # # plot_item.getAxis('bottom').setPen(curr_time_bin_config['axis']) # Change bottom axis color
        # self.ui.imv.setBorder(curr_time_bin_config['border'])

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

        ## Create the new plot_stack to hold the render hierarchy
        self.ui.plot_stack = {} ## initialize

        ## Background-only image item
        if self.params.needs_background_image:
            self.ui.bg_imv = pg.ImageItem()
            self.ui.plot_stack['bg_imv'] = self.ui.bg_imv
            # bg_imv_layer: TimeSynchronizedGenericPlotterLayer = TimeSynchronizedGenericPlotterLayer(name='bg_imv', parent=self, contents={'main': self.ui.bg_imv}, data={'time_window_centers': deepcopy(self.time_window_centers),
            #                                                                                                                                         'main': deepcopy(a_moving_avg),
            #                                                                                                                                         })
            # self.ui.plot_stack['bg_imv'] = bg_imv_layer
            # self.ui.root_view.addItem(self.ui.bg_imv)


        ## Create image item
        self.ui.imv = pg.ImageItem(border='w')
        ## Build layer with appropriate controls:
        Assert.is_in(self.posterior_variable_to_render, allowed_variable_list=['p_x_given_n', 'p_x_given_n_and_x_prev'])
        # self.posterior_variable_to_render: allowed values: ['p_x_given_n', 'p_x_given_n_and_x_prev', ...]
        if self.posterior_variable_to_render == 'p_x_given_n':
            main_data = self.active_one_step_decoder.p_x_given_n
            main_data_title = f'p_x_given_n'
        elif self.posterior_variable_to_render == 'p_x_given_n_and_x_prev':
            main_data = self.active_two_step_decoder.p_x_given_n_and_x_prev
            main_data_title = f'p_x_given_n_and_x_prev'
        # elif self.posterior_variable_to_render == 'num_pos_samples_smoothed_occupancy':
        #     image = self.active_time_dependent_placefields.curr_num_pos_samples_smoothed_occupancy_map.copy()
        #     main_data_title = 'curr_num_pos_samples_occupancy map (smoothed)'
        # elif self.posterior_variable_to_render == 'normalized_occupancy':
        #     image = self.active_time_dependent_placefields.curr_normalized_occupancy.copy()
        #     main_data_title = 'curr_normalized_occupancy map'
        else:
            raise NotImplementedError        

        a_layer_key: str = f'imv[{main_data_title}]'
        imv_layer: TimeSynchronizedGenericPlotterLayer = TimeSynchronizedGenericPlotterLayer(name=a_layer_key, parent=self, contents={'main': self.ui.imv}, data={'time_window_centers': self.time_window_centers,
                                                                                                                                                            'main': main_data,
                                                                                                                                                            })
        # self.ui.plot_stack['imv'] = self.ui.imv
        if a_layer_key not in self.ui.plot_stack:
            self.ui.plot_stack[a_layer_key] = imv_layer
        

        # self.ui.root_view.addItem(self.ui.imv)
        # self.ui.root_view.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

        self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=f'PositionDecoder -  t = {self.last_window_time}') # , name=f'PositionDecoder'
        self.ui.root_plot.setObjectName('PositionDecoder')

        if self.params.needs_background_image:
            self.ui.root_plot.addItem(self.ui.bg_imv, defaultPadding=0.0)  # add ImageItem to PlotItem
            
        self.ui.root_plot.addItem(self.ui.imv, defaultPadding=0.0)  # add ImageItem to PlotItem
        if not self.params.show_posteriors:
            self.ui.imv.setVisible(False)
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
        self.enable_user_editable_rois(parent_plot_item=self.ui.root_plot)

    
    # @function_attributes(short_name=None, tags=['track_shapes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-17 11:31', related_items=[])
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


    def add_decoded_time_bin_info(self, curr_session):
        """ adds a dataframe containing the decoded time bin windows and add informationa bout whether they occur during a lap or PBE so we can change the cmap potentially: 

            curr_session = curr_active_pipeline.filtered_sessions['roam']

        """
        assert self.active_one_step_decoder is not None
        ## Take the decoded time bin windows and add informationa bout whether they occur during a lap or PBE so we can change the cmap potentially:
        decoded_time_bins_df: pd.DataFrame = pd.DataFrame(dict(t=self.active_one_step_decoder.time_window_centers))
        decoded_time_bins_df['label'] = decoded_time_bins_df.index.astype(int)
        decoded_time_bins_df['time_bin_id'] = decoded_time_bins_df.index.astype(int)
        # decoded_time_bins_df

        ## PBE ids:
        pbe_epochs_df: pd.DataFrame = ensure_dataframe(curr_session.pbe)
        pbe_epochs_df['label'] = pbe_epochs_df.index.astype(int)
        # pbe_epochs_df
        decoded_time_bins_df = decoded_time_bins_df.time_point_event.adding_epochs_identity_column(epochs_df=pbe_epochs_df, epoch_id_key_name='pbe_id', epoch_label_column_name='label', override_time_variable_name='t',
                                                                                no_interval_fill_value=np.nan, should_replace_existing_column=True, drop_non_epoch_events=False)

        ## Lap id:
        laps_epochs_df: pd.DataFrame = curr_session.laps.to_dataframe()		   
        laps_epochs_df['label'] = laps_epochs_df.index.astype(int)
        # laps_epochs_df
        decoded_time_bins_df = decoded_time_bins_df.time_point_event.adding_epochs_identity_column(epochs_df=laps_epochs_df, epoch_id_key_name='lap_id', epoch_label_column_name='lap_id', override_time_variable_name='t',
                                                                                no_interval_fill_value=np.nan, should_replace_existing_column=True, drop_non_epoch_events=False)


        ## OUTPUTS: decoded_time_bins_df
        self.params.decoded_time_bins_info_df = decoded_time_bins_df

        return decoded_time_bins_df

    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #

    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.params.debug_print:
            print(f'TimeSynchronizedPositionDecoderPlotter.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        centers = np.asarray(self.time_window_centers, dtype=float)
        n = len(centers)
        if n == 0:
            self.last_included_posterior_bin_indices = np.array([], dtype=np.intp)
            self.last_window_index = None
            self.last_window_time = None
        elif self.use_all_active_viewport_timebins:
            self.last_included_posterior_bin_indices = _included_posterior_bin_indices_for_viewport(self.active_one_step_decoder, centers, float(start_t), float(end_t))
            if len(self.last_included_posterior_bin_indices) == 0:
                self.last_window_index = None
                self.last_window_time = None
            else:
                fi = int(self.last_included_posterior_bin_indices[0])
                self.last_window_index = fi
                self.last_window_time = centers[fi]
        else:
            idx = int(np.searchsorted(centers, float(start_t), side='left'))
            idx = max(0, min(idx, n - 1))
            self.last_included_posterior_bin_indices = np.array([idx], dtype=np.intp)
            self.last_window_index = idx
            self.last_window_time = centers[idx]

        ## update any additional image layers in the stack
        for z_idx, (a_stack_item_key, a_stack_item) in enumerate(self.ui.plot_stack.items()):
            if self.enable_debug_print:
                print(f'on_window_changed: z_idx: {z_idx}, a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
            try:
                if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
                    a_stack_item.on_window_changed(start_t=start_t, end_t=end_t, defer_render=True) ## call update plots on the child item
                    if self.enable_debug_print:
                        print(f'\ton_window_changed successful.')
                else:
                    if self.enable_debug_print:
                        print(f'\tskipped!')
            except (KeyError, AttributeError) as e:
                print(f'\t encountered error "{e}" while trying to on_window_changed item. Skipping.')
            except Exception as e:
                ## Unexpected exception!
                raise e

        self._update_plots()
        if self.params.debug_print:
            print('\tFinished calling _update_plots()')


    def update(self, t, defer_render=False):
        # Finds the nearest previous decoded position for the time t:
        centers = np.asarray(self.time_window_centers, dtype=float)
        n = len(centers)
        if n == 0:
            self.last_window_index = None
            self.last_window_time = None
            self.last_included_posterior_bin_indices = np.array([], dtype=np.intp)
        else:
            idx = int(np.searchsorted(centers, t, side='left')) # side='left' ensures that no future values (later than 't') are ever returned
            idx = max(0, min(idx, n - 1))
            self.last_window_index = idx
            self.last_window_time = centers[idx]
            self.last_included_posterior_bin_indices = np.array([idx], dtype=np.intp)
        
        ## update any additional image layers in the stack
        for z_idx, (a_stack_item_key, a_stack_item) in enumerate(self.ui.plot_stack.items()):
            if self.enable_debug_print:
                print(f'Update: z_idx: {z_idx}, a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
            try:
                if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
                    a_stack_item.update(t=t, defer_render=True) ## call update plots on the child item
                    if self.enable_debug_print:
                        print(f'\tupdate successful.')
                else:
                    if self.enable_debug_print:
                        print(f'\tskipped!')
            except (KeyError, AttributeError) as e:
                print(f'\t encountered error "{e}" while trying to update item. Skipping.')
            except Exception as e:
                ## Unexpected exception!
                raise e
            

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

        curr_time_bin_type: str = 'default'
        # curr_cmap = self.params.cmap
        if self.params.decoded_time_bins_info_df is not None:
            curr_time_window_info = self.params.decoded_time_bins_info_df.iloc[curr_time_window_index]
            if (not pd.isna(curr_time_window_info['pbe_id'])):
                ## is PBE
                curr_time_bin_type = 'PBE'
                if self.params.debug_print:
                    print(f'\tis_pbe!')
                # curr_cmap = self.params.cmap_PBE
                # self.ui.imv.setColorMap(self.params.cmap)


        ## update the current colormap and border:
        curr_time_bin_config: Dict = self.params.image_visual_display_config[curr_time_bin_type]
        # plot_item.getAxis('left').setPen(curr_time_bin_config['axis']) # Change left axis color
        # plot_item.getAxis('bottom').setPen(curr_time_bin_config['axis']) # Change bottom axis color
        self.ui.imv.setBorder(curr_time_bin_config['border'])
        self.ui.imv.setColorMap(curr_time_bin_config['cmap'])

        # self.params.cmap = pg.colormap.get('viridis','matplotlib')

        # Assert.is_in(self.posterior_variable_to_render, allowed_variable_list=['p_x_given_n', 'p_x_given_n_and_x_prev'])
        # # self.posterior_variable_to_render: allowed values: ['p_x_given_n', 'p_x_given_n_and_x_prev', ...]
        # if self.posterior_variable_to_render == 'p_x_given_n':
        #     image = np.squeeze(self.active_one_step_decoder.p_x_given_n[:, :, curr_time_window_index]).copy()
        #     image_title = f'p_x_given_n'
        # elif self.posterior_variable_to_render == 'p_x_given_n_and_x_prev':
        #     image = np.squeeze(self.active_two_step_decoder.p_x_given_n_and_x_prev[:, :, curr_time_window_index]).copy()
        #     image_title = f'p_x_given_n_and_x_prev'
        # # elif self.posterior_variable_to_render == 'num_pos_samples_smoothed_occupancy':
        # #     image = self.active_time_dependent_placefields.curr_num_pos_samples_smoothed_occupancy_map.copy()
        # #     image_title = 'curr_num_pos_samples_occupancy map (smoothed)'
        # # elif self.posterior_variable_to_render == 'normalized_occupancy':
        # #     image = self.active_time_dependent_placefields.curr_normalized_occupancy.copy()
        # #     image_title = 'curr_normalized_occupancy map'
        # else:
        #     raise NotImplementedError
        
        # if self.params.drop_below_threshold is not None:
        #     # image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan # null out the occupancy
        #     image[np.where(image < self.params.drop_below_threshold)] = np.nan # null out the occupancy
        
        # # self.ui.imv.setImage(image, xvals=self.active_time_dependent_placefields.xbin)
        # if self.params.shared_axis_order is None:
        #     self.ui.imv.setImage(image, rect=self.params.image_bounds_extent)
        # else:
        #     self.ui.imv.setImage(image, rect=self.params.image_bounds_extent, axisOrder=self.params.shared_axis_order)
        
        # self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
        # self.setWindowTitle(f'TimeSynchronizedPositionDecoderPlotter - {image_title} t = {curr_t}')
    
        self.AnimalTrajectoryPlottingMixin_update_plots()

        ## update any additional image layers in the stack
        for z_idx, (a_stack_item_key, a_stack_item) in enumerate(self.ui.plot_stack.items()):
            if self.enable_debug_print:
                print(f'Update: z_idx: {z_idx}, a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
            try:
                if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
                    a_stack_item._update_plots() ## call update plots on the child item
                    if self.enable_debug_print:
                        print(f'\tupdate successful.')
                else:
                    if self.enable_debug_print:
                        print(f'\tskipped!')
            except (KeyError, AttributeError) as e:
                print(f'\t encountered error "{e}" while trying to update item. Skipping.')
            except Exception as e:
                ## Unexpected exception!
                raise e

        if not self.params.show_posteriors:
            self.ui.imv.setVisible(False)
            
        ## Update the plot title if needed:
        curr_window_title_str: str = f'PositionDecoder -  '
        if (curr_time_window_index is not None) and (curr_time_window_index >= 0):
            curr_window_title_str = f"{curr_window_title_str} t_idx: {curr_time_window_index}"
            
        if (self.last_window_time is not None):
            curr_window_title_str = f"{curr_window_title_str} | t: {self.last_window_time}"
        if self.use_all_active_viewport_timebins and (self.last_included_posterior_bin_indices is not None):
            curr_window_title_str = f"{curr_window_title_str} | n_bins: {len(self.last_included_posterior_bin_indices)}"

        # self.ui.root_plot.setTitle(f'PositionDecoder -  t = {self.last_window_time}')
        self.ui.root_plot.setTitle(curr_window_title_str)
        


    def _export_video_spike_raster_duration_seconds(self, spike_raster_plt_2d: Any) -> float:
        """Visible window duration for sliding the spike raster during export (seconds)."""
        rd = getattr(spike_raster_plt_2d, 'render_window_duration', None)
        if rd is not None:
            d = float(rd)
            if d > 0:
                return d
        pd = getattr(spike_raster_plt_2d, 'plot_data', None)
        if pd is not None:
            try:
                te = float(pd.time_window_end)
                ts = float(pd.time_window_start)
                if te > ts:
                    return te - ts
            except (TypeError, ValueError, AttributeError):
                pass
        return 15.0



    def _export_video_apply_frame_time(self, t: float, spike_raster_plt_2d: Any, additional_decoder_plotters: Optional[List[Any]]) -> None:
        """Advance time for export one frame: either scroll+emit the driver raster (updates connected decoders) or update() on each plotter.
        
        Never called?!?
        
        """
        if spike_raster_plt_2d is not None:
            duration = self._export_video_spike_raster_duration_seconds(spike_raster_plt_2d)
            win_start = float(t)
            win_end = win_start + duration
            spike_raster_plt_2d.update_scroll_window_region(win_start, win_end, block_signals=True)
            spike_raster_plt_2d.window_scrolled.emit(win_start, win_end)
        else:
            self.update(t, defer_render=False)
            if additional_decoder_plotters:
                for p in additional_decoder_plotters:
                    p.update(t, defer_render=False)


    # @function_attributes(short_name=None, tags=['video', 'export', 'mp4', 'avi', 'gif', 'output'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-11-24 23:09', related_items=[])
    def export_video(self, output_path: str, start_t: Optional[float] = None, end_t: Optional[float] = None, fps: float = 30.0, width: Optional[int] = None, height: Optional[int] = None, progress_print: bool = True, debug_print: bool = False):
        """Efficiently export a video or animated GIF from the TimeSynchronizedPositionDecoderPlotter instance (faster than real-time playback)
        
        Export format is determined automatically from output_path extension: .avi, .mp4, .mov for video (OpenCV);
        .gif for animated GIF (imageio).
        
        This method iterates through time points, updates the plotter, captures frames using
        pyqtgraph's ImageExporter, and saves them as video (OpenCV) or animated GIF (imageio).
        
        Args:
            output_path: Path to save the output file. Extension selects format: .avi/.mp4/.mov (video) or .gif (animated GIF).
            start_t: Start time for export. If None, uses the first available time window center.
            end_t: End time for export. If None, uses the last available time window center.
            fps: Frames per second for the output (default: 30.0)
            width: Width of exported frames in pixels. If None, uses current widget width.
            height: Height of exported frames in pixels. If None, uses current widget height.
            progress_print: Whether to print progress messages (default: True)
            debug_print: Whether to print debug information (default: False)
            
        Returns:
            Path: Path to the saved file
            
        Usage:
            plotter = TimeSynchronizedPositionDecoderPlotter(...)
            plotter.export_video('output/videos/decoder.avi', start_t=100.0, end_t=200.0, fps=30.0)
            plotter.export_video('output/videos/decoder.gif', start_t=100.0, end_t=200.0, fps=10.0)
        """
        from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
        from pyphoplacecellanalysis.External.pyqtgraph_extensions.export_helpers import ExportHelpers
        from pathlib import Path
        import sys
        
        # Disable debug printing during export for performance
        original_debug_print: bool = self.params.debug_print
        desired_debug_print: bool = debug_print
        self.params.debug_print = desired_debug_print
        ## update any additional image layers in the stack
        for (a_stack_item_key, a_stack_item) in self.ui.plot_stack.items():
            if desired_debug_print:
                print(f'Update: a_stack_item_key: "{a_stack_item_key}", a_stack_item: {a_stack_item}')
            try:
                if (hasattr(a_stack_item, 'is_layer') and getattr(a_stack_item, 'is_layer', False)):
                    a_stack_item.params.debug_print = desired_debug_print
                    if desired_debug_print:
                        print(f'\tupdate successful.')
                else:
                    if desired_debug_print:
                        print(f'\tskipped!')
            except (KeyError, AttributeError) as e:
                print(f'\t encountered error "{e}" while trying to update item. Skipping.')
            except Exception as e:
                raise
        ## END for (a_stack_item_key, a_stack_item) in self.ui.plot_stack.items()...

        video_filepath: Path = Path(output_path).resolve()
        suffix = video_filepath.suffix.lower()
        VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov'}
        GIF_EXTENSIONS = {'.gif'}
        if suffix in GIF_EXTENSIONS:
            export_format = 'gif'
        elif suffix in VIDEO_EXTENSIONS:
            export_format = 'video'
        else:
            raise ValueError(f"Unsupported output extension '{suffix}'. Supported: video {VIDEO_EXTENSIONS}, animated GIF {GIF_EXTENSIONS}.")
        
        if export_format == 'video':
            import cv2
        
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
        desired_time_step: float = 1.0 / fps if fps > 0 else float('inf')
        
        # Get all candidate frame indices
        all_frame_indices = np.arange(start_idx, end_idx)
        all_frame_times = time_window_centers[all_frame_indices]
        
        # Subsample frames based on desired time step
        if (desired_time_step < float('inf')) and (len(all_frame_indices) > 1):
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
            total_available_frames: int = len(all_frame_indices)
            kind = 'animated GIF' if export_format == 'gif' else 'video'
            print(f'Exporting {kind}: {n_frames} frames (from {total_available_frames} available) from t={time_window_centers[start_idx]:.2f} to t={time_window_centers[end_idx-1]:.2f} at {fps} fps')
        
        # Get widget dimensions
        if width is None or height is None:
            widget_size = self.ui.root_graphics_layout_widget.size()
            if width is None:
                width = widget_size.width()
            if height is None:
                height = widget_size.height()
        


        out = None  # Initialize to None for proper cleanup
        try:
            # Create ImageExporter for the root plot
            exporter: ImageExporter = ImageExporter(self.ui.root_plot)
            exporter.parameters()['width'] = width
            exporter.parameters()['height'] = height
            exporter.parameters()['antialias'] = True
            
            # Process events to ensure widget is rendered
            QtWidgets.QApplication.processEvents()
            
            # Capture first frame to get actual output dimensions (may differ from requested)
            first_frame_idx: int = frame_indices[0]
            self.update(time_window_centers[first_frame_idx], defer_render=False)
            QtWidgets.QApplication.processEvents()
            first_qimage = exporter.export(toBytes=True)
            if export_format == 'gif':
                first_frame = ExportHelpers.qimage_to_rgb(first_qimage)
            else:
                first_bgr = ExportHelpers.qimage_to_bgr(first_qimage)
                first_frame = first_bgr
            actual_height, actual_width = first_frame.shape[:2]
            del first_qimage  # Free memory
            
            # Set up output path and directory
            video_parent_path: Path = video_filepath.parent
            if not video_parent_path.exists():
                if progress_print:
                    print(f'Creating output directory: {video_parent_path}')
                video_parent_path.mkdir(parents=True, exist_ok=True)
            
            if export_format == 'gif':
                
                frames_list = [first_frame]
                progress_print_every_n_frames = max(1, n_frames // 20)
                
                # Main export loop ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                for i, frame_idx in enumerate(frame_indices):
                    if i == 0:
                        if progress_print:
                            print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                        continue
                    if progress_print and (i % progress_print_every_n_frames == 0 or i == n_frames - 1):
                        print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                    t = time_window_centers[frame_idx]
                    self.update(t, defer_render=False) ## Update call
                    QtWidgets.QApplication.processEvents()
                    qimage = exporter.export(toBytes=True)
                    frames_list.append(ExportHelpers.qimage_to_rgb(qimage))
                ## END: for i, frame_idx in enumerate(frame_indices)


                duration_sec = 1.0 / fps if fps > 0 else 0.1
                imageio.mimsave(str(video_filepath), frames_list, format='GIF', duration=duration_sec, loop=0)
            else:
                # Initialize video writer with actual frame dimensions
                fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                out = cv2.VideoWriter(str(video_filepath), fourcc, fps, (actual_width, actual_height), isColor=True)
                if not out.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {video_filepath}")
                out.write(first_bgr)
                del first_bgr  # Free memory
                progress_print_every_n_frames = max(1, n_frames // 20)
                
                # Main export loop ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                for i, frame_idx in enumerate(frame_indices):
                    if i == 0:
                        if progress_print:
                            print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                        continue
                    if progress_print and (i % progress_print_every_n_frames == 0 or i == n_frames - 1):
                        print(f'Processing frame {i+1}/{n_frames} (t={time_window_centers[frame_idx]:.2f})')
                    t = time_window_centers[frame_idx]
                    self.update(t, defer_render=False) ## Update call
                    
                    QtWidgets.QApplication.processEvents()
                    qimage = exporter.export(toBytes=True)
                    bgr_array = ExportHelpers.qimage_to_bgr(qimage)
                    out.write(bgr_array)
                ## END: for i, frame_idx in enumerate(frame_indices)


        finally:
            # Always close video writer (if opened) and restore debug print setting
            if out is not None:
                out.release()
            self.params.debug_print = original_debug_print
        
        if progress_print:
            kind = 'Animated GIF' if export_format == 'gif' else 'Video'
            print(f'{kind} exported successfully to: {video_filepath}')
        
        return video_filepath


# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedPositionDecoderPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()



