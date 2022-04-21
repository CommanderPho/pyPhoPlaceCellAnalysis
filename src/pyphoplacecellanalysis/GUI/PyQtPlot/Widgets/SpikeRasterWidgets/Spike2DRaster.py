from copy import deepcopy
import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

# import qdarkstyle

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin


class Spike2DRaster(Render2DScrollWindowPlotMixin, SpikeRasterBase):
    """ Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike2DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    # Application/Window Configuration Options:
    applicationName = 'Spike2DRaster'
    windowName = 'Spike2DRaster'
    
    # GUI Configuration Options:
    WantsRenderWindowControls = False
    WantsPlaybackControls = False
    
    ## Scrollable Window Signals
    # window_scrolled = QtCore.pyqtSignal(float, float) # signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    

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
    


    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, application_name=None, **kwargs):
        super(Spike2DRaster, self).__init__(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, **kwargs)
         # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.timeWindow.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        # self.on_window_duration_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.show()
        


    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        # self.app = pg.mkQApp("Spike2DRaster")
        self.app = pg.mkQApp(self.applicationName)
        
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
    
    
    def _build_cell_configs(self):
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        self.params.config_items = []
        for i, cell_id in enumerate(self.unit_ids):
            curr_color = pg.mkColor((i, self.n_cells*1.3))
            curr_color.setAlphaF(0.5)
            curr_pen = pg.mkPen(curr_color)
            curr_config_item = (i, cell_id, curr_pen, self.lower_y[i], self.upper_y[i])
            self.params.config_items.append(curr_config_item)    
    
        self.config_unit_id_map = dict(zip(self.unit_ids, self.params.config_items))
        
        
  
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
             
        self._build_cell_configs()    
    
        # self.ui.spikes_raster_item_plot = SpikesRasterItem(self.params.config_items)
        self.ui.scatter_plot = pg.ScatterPlotItem(name='spikeRasterScatterPlotItem', pxMode=True, symbol=vtick, size=10, pen={'color': 'w', 'width': 2})
        self.ui.scatter_plot.opts['useCache'] = True
        self.ui.main_plot_widget.addItem(self.ui.scatter_plot)

        
        # From Render2DScrollWindowPlotMixin:
        self.ui.main_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0)
        self.ui.main_scroll_window_plot = self._buildScrollRasterPreviewWindowGraphics(self.ui.main_scroll_window_plot)

        # self._buildScrollRasterPreviewWindowGraphics()
            
        self.ui.scatter_plot.addPoints(self.all_spots)
        # self.Render2DScrollWindowPlot_on_window_update # register with the animation time window for updates for the scroller.
        # Connect the signals for the zoom region and the LinearRegionItem
        # self.ui.scroll_window_region.sigRegionChanged.connect(self.update_zoom_plotter)
        self.window_scrolled.connect(self.update_zoomed_plot)
        # self.ui.main_plot_widget.sigRangeChanged.connect(self.update_region)
        
        
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
        # print(f'lower_y: {lower_y}\n upper_y: {upper_y}')
        pass


    def _update_plots(self):
        """
        
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
        
        
        

    @QtCore.pyqtSlot(float, float)
    def update_zoomed_plot(self, min_t, max_t):
        # Update the main_plot_widget:
        self.ui.main_plot_widget.setXRange(min_t, max_t, padding=0)
        # self.render_window_duration = (max_x - min_x) # update the render_window_duration from the slider width
        scroll_window_width = max_t - min_t
        # print(f'min_x: {min_x}, max_x: {max_x}, scroll_window_width: {scroll_window_width}') # min_x: 59.62061245756003, max_x: 76.83228787177144, scroll_window_width: 17.211675414211413

        # Update GUI if we have one:
        if self.WantsRenderWindowControls:
            self.ui.spinTemporalZoomFactor.setValue(1.0)
            self.ui.spinRenderWindowDuration.setValue(scroll_window_width)
            
        # Finally, update the actual spikes_window. This is the part that updates the 3D Raster plot because we bind to this window's signal
        self.spikes_window.update_window_start(min_t)
        
    @QtCore.pyqtSlot(float, float)
    def update_scroll_window_region(self, new_start, new_end, block_signals: bool=True):
        """ called to update the interactive scrolling window control """
        if block_signals:
            self.ui.scroll_window_region.blockSignals(True) # Block signals so it doesn't recurrsively update
        self.ui.scroll_window_region.setRegion([new_start, new_end]) # adjust scroll control
        if block_signals:
            self.ui.scroll_window_region.blockSignals(False)
        
        
# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike2DRaster()
#     v.animation()
# dfsd