from copy import deepcopy
import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import qtawesome as qta

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider

# import qdarkstyle

from pyphoplacecellanalysis.General.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterBase import SpikeRasterBase

from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_spikes_raster_2D import SpikesRasterItem


def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug

                

class Spike2DRaster(SpikeRasterBase):
    """ Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Spike2DRaster import Spike2DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike2DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    temporal_mapping_changed = QtCore.pyqtSignal() # signal emitted when the mapping from the temporal window to the spatial layout is changed
    close_signal = QtCore.pyqtSignal() # Called when the window is closing. 
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
    

    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""    
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL']
        lines_dict[af.AlignTop | af.AlignRight] = ['TR', 
                                                   f"n_cells : {self.n_cells}",
                                                   f'render_window_duration: {self.render_window_duration}',
                                                   f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'playback_rate_multiplier: {self.playback_rate_multiplier}']
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    ######  Get/Set Properties ######:
    
    
    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, **kwargs):
        super(Spike2DRaster, self).__init__(spikes_df, *args, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, **kwargs)
        # super(Spike2DRaster, self).__init__(*args, **kwargs)
        # Initialize member variables:
            
        # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        # self.on_window_duration_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.show()


    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp("Spike2DRaster")
        
        # Configure pyqtgraph config:
        try:
            import OpenGL
            pg.setConfigOption('useOpenGL', True)
            pg.setConfigOption('enableExperimental', True)
        except Exception as e:
            print(f"Enabling OpenGL failed with {e}. Will result in slow rendering. Try installing PyOpenGL.")
            
        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', "#1B1B1B")
        pg.setConfigOption('foreground', "#727272")
    
        # Config
        # self.params.center_mode = 'zero_centered'
        self.params.center_mode = 'starting_at_zero'
        self.params.bin_position_mode = 'bin_center'
        # self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        # self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)
        self.params.temporal_zoom_factor = 1.0        
        
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        
        # Build Required SpikesDf fields:
        # print(f'unit_ids: {self.unit_ids}, n_cells: {self.n_cells}')
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self.y_unit_id_map = dict(zip(self.unit_ids, self.y))

        # Compute the y for all windows, not just the current one:
        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            print('Spike2DRaster.setup(): adding "visualization_raster_y_location" column to spikes_df...')
            # all_y = [y[i] for i, a_cell_id in enumerate(curr_spikes_df['unit_id'].to_numpy())]
            all_y = [self.y_unit_id_map[a_cell_id] for a_cell_id in self.spikes_df['unit_id'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes
            print('done.')
        # self.spikes_df
        
        
    def _build_spikes_data_values(self, spikes_df):
        # All units at once approach:
        # Filter the dataframe using that column and value from the list
        curr_spike_t = spikes_df[spikes_df.spikes.time_variable_name].to_numpy() # this will map
        curr_spike_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (0.0, +self.temporal_axis_length))
        curr_spike_y = spikes_df['visualization_raster_y_location'].to_numpy() # this will map
        curr_spike_pens = [self.config_unit_id_map[a_cell_id][2] for a_cell_id in spikes_df['unit_id'].to_numpy()] # get the pens for each spike from the configs map
        curr_n = len(curr_spike_t) # curr number of spikes
        return curr_spike_x, curr_spike_y, curr_spike_pens, curr_n
    
    
    def _build_all_spikes_data_values(self):
        """ build global spikes for entire dataframe (not just the current window) """
        # All units at once approach:
        # Filter the dataframe using that column and value from the list
        curr_spike_t = self.spikes_window.df[self.spikes_window.df.spikes.time_variable_name].to_numpy() # this will map
        # curr_spike_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (0.0, +self.temporal_axis_length))
        curr_spike_y = self.spikes_window.df['visualization_raster_y_location'].to_numpy() # this will map
        curr_spike_pens = [self.config_unit_id_map[a_cell_id][2] for a_cell_id in self.spikes_window.df['unit_id'].to_numpy()] # get the pens for each spike from the configs map
        curr_n = len(curr_spike_t) # curr number of spikes
        return curr_spike_t, curr_spike_y, curr_spike_pens, curr_n
    
    
    def _buildScrollRasterPreviewWindowGraphics(self):
        # Common Tick Label
        vtick = QtGui.QPainterPath()
        vtick.moveTo(0, -0.5)
        vtick.lineTo(0, 0.5)
        
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        # self.params.config_items = []
        # for i, cell_id in enumerate(self.unit_ids):
        #     curr_color = pg.mkColor((i, self.n_cells*1.3))
        #     curr_color.setAlphaF(0.5)
        #     curr_pen = pg.mkPen(curr_color)
        #     curr_config_item = (i, cell_id, curr_pen, self.lower_y[i], self.upper_y[i])            
        #     self.params.config_items.append(curr_config_item)    
        #     # s2 = pg.ScatterPlotItem(pxMode=True, symbol=vtick, size=1, pen=curr_pen)
        #     # self.ui.main_plot_widget.addItem(s2)
    
        # self.config_unit_id_map = dict(zip(self.unit_ids, self.params.config_items))
    
        #############################
        ## Bottom Windowed Scroll Plot/Widget:
        self.ui.main_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0)
        # ALL Spikes in the preview window:
        curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_all_spikes_data_values()        
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
        
        self.ui.preview_overview_scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1})
        self.ui.preview_overview_scatter_plot.opts['useCache'] = True
        self.ui.preview_overview_scatter_plot.addPoints(self.all_spots)
        self.ui.main_scroll_window_plot.addItem(self.ui.preview_overview_scatter_plot)
        
        # Add the linear region overlay:
        self.ui.scroll_window_region = pg.LinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.ui.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
    
        self.ui.scroll_window_region.setZValue(10)
        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
        self.ui.main_scroll_window_plot.addItem(self.ui.scroll_window_region, ignoreBounds=True)
        
        # Setup axes bounds for the bottom windowed plot:
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        self.ui.main_scroll_window_plot.hideAxis('left')
        self.ui.main_scroll_window_plot.hideAxis('bottom')
        # self.ui.main_scroll_window_plot.setLabel('bottom', 'Time', units='s')
        self.ui.main_scroll_window_plot.setMouseEnabled(x=False, y=False)
        self.ui.main_scroll_window_plot.disableAutoRange('xy')
        # self.ui.main_scroll_window_plot.enableAutoRange(x=False, y=False)
        self.ui.main_scroll_window_plot.setAutoVisible(x=False, y=False)
        self.ui.main_scroll_window_plot.setAutoPan(x=False, y=False)
        self.ui.main_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
        self.ui.main_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)
        
  
    def _buildGraphics(self):
        ##### Main Raster Plot Content Top ##########
        
        self.ui.main_graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.ui.main_graphics_layout_widget.useOpenGL(True)
        self.ui.main_graphics_layout_widget.resize(1000,600)
        # Add the main widget to the layout in the (0, 0) location:
        self.ui.layout.addWidget(self.ui.main_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        # self.ui.main_gl_widget.clicked.connect(self.play_pause)
        # self.ui.main_gl_widget.doubleClicked.connect(self.toggle_full_screen)
        # self.ui.main_gl_widget.wheel.connect(self.wheel_handler)
        # self.ui.main_gl_widget.keyPressed.connect(self.key_handler)
        
        #### Build Graphics Objects ##### 
        # Add debugging widget:
        
        # Custom 2D raster plot:    
        self.ui.main_plot_widget = self.ui.main_graphics_layout_widget.addPlot(row=1, col=0)
        # self.ui.plots = [] # create an empty array for each plot, of which there will be one for each unit.
        # # build the position range for each unit along the y-axis:
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self.lower_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        self.upper_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
       
        # Common Tick Label
        vtick = QtGui.QPainterPath()
        vtick.moveTo(0, -0.5)
        vtick.lineTo(0, 0.5)

        self.ui.main_plot_widget.setLabel('left', 'Cell ID', units='')
        self.ui.main_plot_widget.setLabel('bottom', 'Time', units='s')
        self.ui.main_plot_widget.setMouseEnabled(x=False, y=False)
        self.ui.main_plot_widget.enableAutoRange(x=False, y=False)
        self.ui.main_plot_widget.setAutoVisible(x=False, y=False)
        self.ui.main_plot_widget.setAutoPan(x=False, y=False)
        self.ui.main_plot_widget.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
        
        # self.ui.main_plot_widget.disableAutoRange()
        self._update_plot_ranges()
             
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        self.params.config_items = []
        for i, cell_id in enumerate(self.unit_ids):
            curr_color = pg.mkColor((i, self.n_cells*1.3))
            curr_color.setAlphaF(0.5)
            curr_pen = pg.mkPen(curr_color)
            curr_config_item = (i, cell_id, curr_pen, self.lower_y[i], self.upper_y[i])
            self.params.config_items.append(curr_config_item)    
    
        self.config_unit_id_map = dict(zip(self.unit_ids, self.params.config_items))
    
        # self.ui.spikes_raster_item_plot = SpikesRasterItem(self.params.config_items)
        self.ui.scatter_plot = pg.ScatterPlotItem(name='spikeRasterScatterPlotItem', pxMode=True, symbol=vtick, size=10, pen={'color': 'w', 'width': 2})
        self.ui.scatter_plot.opts['useCache'] = True
        self.ui.main_plot_widget.addItem(self.ui.scatter_plot)

        
        self._buildScrollRasterPreviewWindowGraphics()
                
        # All units at once approach:
        # curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_spikes_data_values(self.active_windowed_df)
        # # print(f'np.shape(curr_spike_t): {np.shape(curr_spike_t)}, np.shape(curr_spike_x): {np.shape(curr_spike_x)}, np.shape(curr_spike_y): {np.shape(curr_spike_y)}, curr_n: {curr_n}')
        # pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        # # print(f'np.shape(pos): {np.shape(pos)}') # should be 2xN # np.shape(pos): (2, 11)
        # # spots = [{'pos': pos[:,i], 'data': 1, 'brush':pg.intColor(i, n), 'symbol': i%10, 'size': 5+i/10.} for i in range(n)]
        # spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
        # self.ui.scatter_plot.addPoints(spots)
        self.ui.scatter_plot.addPoints(self.all_spots)


        # Connect the signals for the zoom region and the LinearRegionItem
        self.ui.scroll_window_region.sigRegionChanged.connect(self.update_zoom_plotter)
        self.ui.main_plot_widget.sigRangeChanged.connect(self.update_region)
        
        
    ###################################
    #### EVENT HANDLERS
    ##################################
    
    def _update_plot_ranges(self):
        # self.ui.main_plot_widget.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
        # self.ui.main_plot_widget.setXRange(0.0, +self.temporal_axis_length, padding=0)
        # self.ui.main_plot_widget.setYRange(self.y[0], self.y[-1], padding=0)
        # self.ui.main_plot_widget.disableAutoRange()
        self.ui.main_plot_widget.disableAutoRange('xy')
        self.ui.main_plot_widget.setRange(xRange=[0.0, +self.temporal_axis_length], yRange=[self.y[0], self.y[-1]])
    
    @QtCore.pyqtSlot()
    def on_adjust_temporal_spatial_mapping(self):
        """ called when the spatio-temporal mapping property is changed.
        
        Should change whenever any of the following change:
            self.temporal_zoom_factor
            self.render_window_duration
            
        """
        # self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        # self.lower_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        # self.upper_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        # self.y_unit_id_map = dict(zip(self.unit_ids, self.y))
        # all_y = [self.y_unit_id_map[a_cell_id] for a_cell_id in self.spikes_df['unit_id'].to_numpy()]
        # self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes
        
        # print(f'lower_y: {lower_y}\n upper_y: {upper_y}')
        pass


    def _update_plots(self):
        """ performance went:
        FROM:
            > Entering Spike2DRaster.on_window_changed
            Finished calling _update_plots(): 1179.6892 ms
            < Exiting Spike2DRaster.on_window_changed, total time: 1179.7600 ms

        TO:
            > Entering Spike2DRaster.on_window_changed
            Finished calling _update_plots(): 203.8840 ms
            < Exiting Spike2DRaster.on_window_changed, total time: 203.9544 ms

        Just by removing the lines that initialized the color. Conclusion is that pg.mkColor((cell_id, self.n_cells*1.3)) must be VERY slow.
    
        """
        if self.enable_debug_print:
            print(f'Spike2DRaster._update_plots()')
        # assert (len(self.ui.plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.plots): {len(self.ui.plots)} and self.n_cells: {self.n_cells}!"
        # build the position range for each unit along the y-axis:
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
        
        # Get updated time window
        updated_time_window = self.spikes_window.active_time_window # (30.0, 930.0)
        # update the current scroll region:
        # self.ui.scroll_window_region.setRegion(updated_time_window)
        
        
        # All units at once approach:
        # Filter the dataframe using that column and value from the list
        # curr_spike_t = self.active_windowed_df[self.active_windowed_df.spikes.time_variable_name].to_numpy() # this will map
        # curr_spike_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (0.0, +self.temporal_axis_length))
        # curr_spike_y = self.active_windowed_df['visualization_raster_y_location'].to_numpy() # this will map
        # curr_spike_pens = [self.config_unit_id_map[a_cell_id][2] for a_cell_id in self.active_windowed_df['unit_id'].to_numpy()] # get the pens for each spike from the configs map
        # curr_n = len(curr_spike_t) # curr number of spikes
        # curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_spikes_data_values(self.active_windowed_df)
        # self.ui.scatter_plot.setData(x=curr_spike_x, y=curr_spike_y, pen=curr_spike_pens)
        
        # TODO: just scroll since we set all spike data at once.
        # pass
        

    @QtCore.pyqtSlot()
    def update_zoom_plotter(self) -> None:
        """self when the region moves.zoom_Change plotter area"""
        self.ui.scroll_window_region.setZValue(10)
        min_x, max_x = self.ui.scroll_window_region.getRegion()
        self.ui.main_plot_widget.setXRange(min_x, max_x, padding=0)
        # self.render_window_duration = (max_x - min_x) # update the render_window_duration from the slider width
        scroll_window_width = max_x - min_x
        # print(f'min_x: {min_x}, max_x: {max_x}, scroll_window_width: {scroll_window_width}') # min_x: 59.62061245756003, max_x: 76.83228787177144, scroll_window_width: 17.211675414211413

        # Update the active time window to match the scroll window:
        # old_time_window = spike_raster_plt.spikes_window.active_time_window # (30.0, 930.0)

        # spike_raster_plt.render_window_duration = 
        # spike_raster_plt.spikes_window
        # spike_raster_plt.temporal_zoom_factor
        # # self.spikes_window.update_window_start(next_start_timestamp)

        self.ui.spinTemporalZoomFactor.setValue(1.0)
        self.ui.spinRenderWindowDuration.setValue(scroll_window_width)
        self.spikes_window.update_window_start(min_x)
                
        
        # self.temporal_axis_length

    @QtCore.pyqtSlot()
    def update_region(self) -> None:
        """self.zoom_Change the region of the region when the plotter moves
            viewRange returns the display range of the graph. The type is
            [[Xmin, Xmax], [Ymin, Ymax]]
        """
        rgn = self.ui.main_plot_widget.viewRange()[0]
        self.ui.scroll_window_region.setRegion(rgn)
        # self.render_window_duration = (max_x - min_x) # update the render_window_duration from the slider width


    # Slider Functions:
    # def _compute_window_transform(self, relative_offset):
    #     """ computes the transform from 0.0-1.0 as the slider would provide to the offset given the current information. """
    #     earliest_t, latest_t = self.spikes_window.total_df_start_end_times
    #     total_spikes_df_duration = latest_t - earliest_t # get the duration of the entire spikes df
    #     render_window_offset = (total_spikes_df_duration * relative_offset) + earliest_t
    #     return render_window_offset
    
    # def increase_slider_val(self):
    #     slider_val = self.ui.slider.value() # integer value between 0-100
    #     if self.enable_debug_print:
    #         print(f'Spike2DRaster.increase_slider_val(): slider_val: {slider_val}')
    #     if slider_val < 100:
    #         self.ui.slider.setValue(slider_val + 1)
    #     else:
    #         print("thread ended..")
    #         self.ui.btn_slide_run.setText(">")
    #         self.ui.btn_slide_run.tag = "paused"
    #         self.sliderThread.terminate()

    # def slider_val_changed(self, val):
    #     self.slidebar_val = val / 100
    #     # Gets the transform from relative (0.0 - 1.0) to absolute timestamp offset
    #     curr_t = self._compute_window_transform(self.slidebar_val)
        
    #     if self.enable_debug_print:
    #         print(f'Spike2DRaster.slider_val_changed(): self.slidebar_val: {self.slidebar_val}, curr_t: {curr_t}')
    #         print(f'BEFORE: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    #      # set the start time which will trigger the update cascade and result in on_window_changed(...) being called
    #     self.spikes_window.update_window_start(curr_t)
    #     if self.enable_debug_print:
    #         print(f'AFTER: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    

    # #### from pyqtgraph_animated3Dplot_pairedLines's animation style ###:
    # def start(self):
    #     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #         QtGui.QApplication.instance().exec_()
      
            
    # # def set_plotdata(self, name, points, color, width):
    # #     # self.traces in the original
    # #     self.ui.gl_line_plots[name].setData(pos=points, color=color, width=width, mode='lines', antialias=True)
        
    # def update(self):
    #     """ called on timer timeout """
    #     self._update_plots()
    #     self.shift_animation_frame_val(1)
        
    # def animation(self):
    #     timer = QtCore.QTimer()
    #     timer.timeout.connect(self.update)
    #     # timer.start(20)
    #     timer.start(50)
    #     self.start()
        
    # def computeTransform(self, x, y, t = None):
    #     if t == None:
    #         v1_x = (1 * (1 - self.slidebar_val)) + (self.v1_x * self.slidebar_val)
    #         v1_y = (0 * (1 - self.slidebar_val)) + (self.v1_y * self.slidebar_val)

    #         v2_y = (1 * (1 - self.slidebar_val)) + (self.v2_y * self.slidebar_val)
    #         v2_x = (0 * (1 - self.slidebar_val)) + (self.v2_x * self.slidebar_val)
    #     else:
    #         v1_x = self.v1_x
    #         v1_y = self.v1_y
    #         v2_x = self.v2_x
    #         v2_y = self.v2_y
    #     return ((v1_x * x) + (v2_x * y), (v1_y * x) + (v2_y * y))


    # Speed Burst Features:
    # def toggle_speed_burst(self):
    #     curr_is_speed_burst_enabled = self.is_speed_burst_mode_active
    #     updated_speed_burst_enabled = (not curr_is_speed_burst_enabled)
    #     if (updated_speed_burst_enabled):
    #         self.engage_speed_burst()
    #     else:
    #         self.disengage_speed_burst()

    # # Engages a temporary speed burst 
    # def engage_speed_burst(self):
    #     print("Speed burst enabled!")
    #     self.is_speed_burst_mode_active = True
    #     # Set the playback speed temporarily to the burst speed
    #     self.media_player.set_rate(self.speedBurstPlaybackRate)

    #     self.ui.toolButton_SpeedBurstEnabled.setEnabled(True)
    #     self.ui.doubleSpinBoxPlaybackSpeed.setEnabled(False)
    #     self.ui.button_slow_down.setEnabled(False)
    #     self.ui.button_speed_up.setEnabled(False)
        
    # def disengage_speed_burst(self):
    #     print("Speed burst disabled!")
    #     self.is_speed_burst_mode_active = False
    #     # restore the user specified playback speed
    #     self.media_player.set_rate(self.ui.doubleSpinBoxPlaybackSpeed.value)

    #     self.ui.toolButton_SpeedBurstEnabled.setEnabled(False)
    #     self.ui.doubleSpinBoxPlaybackSpeed.setEnabled(True)
    #     self.ui.button_slow_down.setEnabled(True)
    #     self.ui.button_speed_up.setEnabled(True)







# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike2DRaster()
#     v.animation()
