from copy import deepcopy
import time
import sys
from indexed import IndexedOrderedDict

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

# import qdarkstyle

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DNeuronIdentityLinesMixin import Render2DNeuronIdentityLinesMixin

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import Specific2DRenderTimeEpochsHelper
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves3D.Render3DTimeCurvesMixin import PyQtGraphSpecificTimeCurvesMixin


class Spike2DRaster(PyQtGraphSpecificTimeCurvesMixin, EpochRenderingMixin, Render2DScrollWindowPlotMixin, SpikeRasterBase):
    """ Displays a 2D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike2DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
        
        
    TODO: FATAL: The Spike3DRaster doesn't make use of the colors set in params or anything where the 3D does! Instead it's unique in that it stores a list of configs for each neuron. While this is a neat idea, it should be scrapped entirely for consistency.
    # self.params.config_items and self._build_cell_configs(...) called from self._buildGraphics(...)
    
    """
    
    # Application/Window Configuration Options:
    applicationName = 'Spike2DRaster'
    windowName = 'Spike2DRaster'
    
    # GUI Configuration Options:
    WantsRenderWindowControls = False
    WantsPlaybackControls = False
    Includes2DActiveWindowScatter = True # Includes2DActiveWindowScatter: if True, it displays the main scatter plot for the active window.
    
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
                                                #    f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'playback_rate_multiplier: {self.playback_rate_multiplier}']
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    ## FOR EpochRenderingMixin
    @property    
    def interval_rendering_plots(self):
        """ returns the list of child subplots/graphics (usually PlotItems) that participate in rendering intervals """
        return [self.plots.background_static_scroll_window_plot, self.plots.main_plot_widget] # for spike_raster_plt_2d
    
    
    ######  Get/Set Properties ######:

    ## FOR TimeCurvesViewMixin
    @property
    def floor_z(self):
        """The offset of the floor in the ordinate-axis. Which is actually the y-axis for a 2D plot """
        return 0
        


    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, application_name=None, **kwargs):
        super(Spike2DRaster, self).__init__(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, **kwargs)
        
        # Init the TimeCurvesViewMixin for 3D Line plots:
        ### No plots will actually be added until self.add_3D_time_curves(plot_dataframe) is called with a valid dataframe.
        self.TimeCurvesViewMixin_on_init()
        
         # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.timeWindow.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        # self.on_window_duration_changed.connect(self.on_adjust_temporal_spatial_mapping)
        
        self.EpochRenderingMixin_on_init()
        
        if self.enable_show_on_init:
            self.show()
            
        # NOTE: It looks like this didn't work when called before self.show(), but worked when called from the Notebook. Might just be a timeing thing.
        ## Make sure to set the initial linear scroll region size/location to something reasonable and not cut-off so the user can adjust it:
        self._fix_initial_linearRegionLocation() # Implemented in Render2DScrollWindowPlotMixin, since it's the one that creates the Scrollwindow anyways
        
        ## Starts the delayed_gui_itemer which will run after 1-second to update the GUI:
        self._delayed_gui_timer = QtCore.QTimer(self)
        self._delayed_gui_timer.timeout.connect(self._run_delayed_gui_load_code)
        #Set the interval and start the timer.
        self._delayed_gui_timer.start(1000)
        
    
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
        # print(f'fragile_linear_neuron_IDXs: {self.fragile_linear_neuron_IDXs}, n_cells: {self.n_cells}')
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, self.y))

        # Compute the y for all windows, not just the current one:
        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            print('Spike2DRaster.setup(): adding "visualization_raster_y_location" column to spikes_df...')
            # all_y = [y[i] for i, a_cell_id in enumerate(curr_spikes_df['fragile_linear_neuron_IDX'].to_numpy())]
            all_y = [self.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in self.spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes. BUG? NO, RESOLVED: actually, this should be updated when anything that would change .y_fragile_linear_neuron_IDX_map would change, right? Meaning: .y, ... oh, I see. self.y doesn't change because self.params.center_mode, self.params.bin_position_mode, and self.params.side_bin_margins aren't expected to change. 
            print('done.')
            
        self.EpochRenderingMixin_on_setup()

        # Required for Time Curves:        
        self.params.time_curves_datasource = None # required before calling self._update_plot_ranges()
    
    def _build_cell_configs(self):
        """ Adds the neuron/cell configurations that are used to color and format the scatterplot spikes and such. 
        Requires:
            self.lower_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
            self.upper_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        
        NOTE: on self.y vs (self.lower_y, self.upper_y): two ndarrays of the same length as self.y but they each express the start/end edges of each series as a ratio of the total.
            this means for example: 
                y:       [0.5, 1.5, 2.5, ..., 65.5, 66.5, 67.5]
                lower_y: [0.0, 0.0147059, 0.0294118, ..., 0.955882, 0.970588, 0.985294]
                upper_y: [0.0147059, 0.0294118, 0.0441176, ..., 0.970588, 0.985294, 1.0]

        Adds:
            self.params.config_items: list
            self.config_fragile_linear_neuron_IDX_map: dict<self.fragile_linear_neuron_IDXs, self.params.config_items>
        
        Known Calls:
            From self._buildGraphics()
        """
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        self.params.config_items = IndexedOrderedDict()
        curr_neuron_ids_list = self.find_cell_ids_from_neuron_IDXs(self.fragile_linear_neuron_IDXs)
        
        for i, fragile_linear_neuron_IDX in enumerate(self.fragile_linear_neuron_IDXs):
            curr_neuron_id = curr_neuron_ids_list[i] # aclu value
            
            curr_color = self.params.neuron_qcolors_map[fragile_linear_neuron_IDX]
            curr_color.setAlphaF(0.5)
            curr_pen = pg.mkPen(curr_color)
            curr_config_item = (i, fragile_linear_neuron_IDX, curr_pen, self.lower_y[i], self.upper_y[i])
            self.params.config_items[curr_neuron_id] = curr_config_item # add the current config item to the config items 
            
    
        self.config_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, self.params.config_items.values()))
        
        
  
    def _buildGraphics(self):
        """ 
        plots.main_plot_widget: 2D display 
            self.plots.scatter_plot: the active 2D display of the current window
        
        plots.background_static_scroll_window_plot: the static plot of the entire data (always shows the entire time range)
            Presents a linear scroll region over the top to allow the user to select the active window.
            
            
        """
        ##### Main Raster Plot Content Top ##########
        
        self.ui.main_graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.ui.main_graphics_layout_widget.setObjectName('main_graphics_layout_widget')
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
        self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self.lower_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        self.upper_y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        self._build_cell_configs()   
        
        # Custom 2D raster plot:
        self.params.main_graphics_plot_widget_rowspan = 3 # how many rows the main graphics PlotItems should span
        
        curr_plot_row = 1
        if self.Includes2DActiveWindowScatter:
            self.plots.main_plot_widget = self.ui.main_graphics_layout_widget.addPlot(row=curr_plot_row, col=0, rowspan=self.params.main_graphics_plot_widget_rowspan, colspan=1) # , name='main_plot_widget'
            self.plots.main_plot_widget.setObjectName('main_plot_widget') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
            curr_plot_row += (1 * self.params.main_graphics_plot_widget_rowspan)
            # self.ui.plots = [] # create an empty array for each plot, of which there will be one for each unit.
            # # build the position range for each unit along the y-axis:
            
            # Common Tick Label
            vtick = QtGui.QPainterPath()
            vtick.moveTo(0, -0.5)
            vtick.lineTo(0, 0.5)

            
            self.plots.main_plot_widget.setLabel('left', 'Cell ID', units='')
            self.plots.main_plot_widget.setLabel('bottom', 'Time', units='s')
            self.plots.main_plot_widget.setMouseEnabled(x=False, y=False)
            self.plots.main_plot_widget.enableAutoRange(x=False, y=False)
            self.plots.main_plot_widget.setAutoVisible(x=False, y=False)
            self.plots.main_plot_widget.setAutoPan(x=False, y=False)
            self.plots.main_plot_widget.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
            
            # self.plots.main_plot_widget.disableAutoRange()
            self._update_plot_ranges()
            
            ## This scatter plot is the dynamic raster that "zooms" on adjustment of the lienar slider region. It is NOT static background raster that's rendered at the bottom of the window!
            self.plots.scatter_plot = pg.ScatterPlotItem(name='spikeRasterScatterPlotItem', pxMode=True, symbol=vtick, size=10, pen={'color': 'w', 'width': 2})
            self.plots.scatter_plot.setObjectName('scatter_plot')
            self.plots.scatter_plot.opts['useCache'] = True
            self.plots.main_plot_widget.addItem(self.plots.scatter_plot)
            _v_axis_item = Render2DNeuronIdentityLinesMixin.setup_custom_neuron_identity_axis(self.plots.main_plot_widget, self.n_cells)
                
        else:
            self.plots.main_plot_widget = None
            self.plots.scatter_plot = None

        
        # From Render2DScrollWindowPlotMixin:
        self.plots.background_static_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=curr_plot_row, col=0, rowspan=self.params.main_graphics_plot_widget_rowspan, colspan=1) # , name='background_static_scroll_window_plot'  curr_plot_row: 2 if  self.Includes2DActiveWindowScatter
        self.plots.background_static_scroll_window_plot.setObjectName('background_static_scroll_window_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend  AND drastically slows down the plotting

        # print(f'main_plot_widget.objectName(): {main_plot_widget.objectName()}')

        self.plots.background_static_scroll_window_plot = self.ScrollRasterPreviewWindow_on_BuildUI(self.plots.background_static_scroll_window_plot)

        # self.ScrollRasterPreviewWindow_on_BuildUI()
        if self.Includes2DActiveWindowScatter:
            self.plots.scatter_plot.addPoints(self.plots_data.all_spots)
    
        self.EpochRenderingMixin_on_buildUI()
        
        # self.Render2DScrollWindowPlot_on_window_update # register with the animation time window for updates for the scroller.
        # Connect the signals for the zoom region and the LinearRegionItem        
        self.rate_limited_signal_scrolled_proxy = pg.SignalProxy(self.window_scrolled, rateLimit=60, slot=self.update_zoomed_plot_rate_limited) # Limit updates to 60 Signals/Second
    

        # For this 2D Implementation of TimeCurvesViewMixin/PyQtGraphSpecificTimeCurvesMixin
        self.ui.main_time_curves_view_widget = None
    
        
    def _run_delayed_gui_load_code(self):
        """ called when the self._delayed_gui_timer QTimer fires. """
        #Stop the timer.
        self._delayed_gui_timer.stop()
        print(f'_run_delayed_gui_load_code() called!')
        ## Make sure to set the initial linear scroll region size/location to something reasonable and not cut-off so the user can adjust it:
        self._fix_initial_linearRegionLocation() # Implemented in Render2DScrollWindowPlotMixin, since it's the one that creates the Scrollwindow anyways

        
    ###################################
    #### EVENT HANDLERS
    ##################################
    
    def _update_plot_ranges(self):
        """
        I believe this runs only once to setup the bounds of the plot.
        TODO: TODO-DOC: Figure out when this is called and what its purpose is
        
        """
        # self.plots.main_plot_widget.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
        # self.plots.main_plot_widget.setXRange(0.0, +self.temporal_axis_length, padding=0)
        # self.plots.main_plot_widget.setYRange(self.y[0], self.y[-1], padding=0)
        # self.plots.main_plot_widget.disableAutoRange()
        if self.Includes2DActiveWindowScatter:
            self.plots.main_plot_widget.disableAutoRange('xy')
            ## TODO: BUG: CONFIRMED: This is for-sure a problem. In the .ScrollRasterPreviewWindow_on_BuildUI(...) where the linear region widget (scroll_window_region) is built, those x-values are definintely timestamps and start slightly negative. This is why the widget is getting cut-off
            """ From the first setup:
                # Setup range for plot:
                earliest_t, latest_t = self.spikes_window.total_df_start_end_times
                background_static_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
                background_static_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)

            Here it looks like I'm trying to use some sort of reletive x-coordinates (as I noted that I did in self.lower_y, self.upper_y?)
            
            OOPS, back-up, this is the main_plot_widget (that should be displaying the contents of the window above), not the same as the static background plot that displays all time.
            """
                    
            # # Get updated time window
            # updated_time_window = self.spikes_window.active_time_window # (30.0, 930.0) ## CHECKL this might actually be invalid at this timepoint, idk
            # earliest_t, latest_t = updated_time_window
            # resolved_start_x = np.nanmin(earliest_t, 0.0)
            # print(f'resolved_start_x: {resolved_start_x}')
            # resolved_end_x = (resolved_start_x+self.temporal_axis_length) # only let it go to the start_x + its appropriate length, otherwise it'll be too long?? Maybe I should actually use the window's end
            # print(f'resolved_end_x: {resolved_end_x}')
            # self.plots.main_plot_widget.setRange(xRange=[resolved_start_x, resolved_end_x], yRange=[self.y[0], self.y[-1]])
            # ## NOW I THINK THIS IS JUST THE ZOOMED PLOT AND NOT THE REASON THE LINEAR SCROLL REGION is cut off
                        
            self.plots.main_plot_widget.setRange(xRange=[0.0, +self.temporal_axis_length], yRange=[self.y[0], self.y[-1]]) # After all this, I've concluded that it was indeed correct!
            _v_axis_item = Render2DNeuronIdentityLinesMixin.setup_custom_neuron_identity_axis(self.plots.main_plot_widget, self.n_cells)
    
    
        # Update 3D Curves if we have them: TODO: figure out where this goes!
        self.TimeCurvesViewMixin_on_window_update() # Don't think this does much here
        
    
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
        
        # update the current scroll region:
        # self.ui.scroll_window_region.setRegion(updated_time_window)
        
        # Update 3D Curves if we have them: TODO: figure out where this goes!
        self.TimeCurvesViewMixin_on_window_update() # Don't think this does much here
        

    @QtCore.pyqtSlot(object)
    def update_zoomed_plot_rate_limited(self, evt):
        min_t, max_t = evt ## using signal proxy turns original arguments into a tuple
        self.update_zoomed_plot(min_t, max_t)


    @QtCore.pyqtSlot(float, float)
    def update_zoomed_plot(self, min_t, max_t):
        # Update the main_plot_widget:
        if self.Includes2DActiveWindowScatter:
            self.plots.main_plot_widget.setXRange(min_t, max_t, padding=0)

        # self.render_window_duration = (max_x - min_x) # update the render_window_duration from the slider width
        scroll_window_width = max_t - min_t
        # print(f'min_x: {min_x}, max_x: {max_x}, scroll_window_width: {scroll_window_width}') # min_x: 59.62061245756003, max_x: 76.83228787177144, scroll_window_width: 17.211675414211413

        # Update GUI if we have one:
        if self.WantsRenderWindowControls:
            self.ui.spinTemporalZoomFactor.setValue(1.0)
            self.ui.spinRenderWindowDuration.setValue(scroll_window_width)
            
        # Finally, update the actual spikes_window. This is the part that updates the 3D Raster plot because we bind to this window's signal
        # self.spikes_window.update_window_start(min_t)
        
        # Here is the main problem: The duration and window end-time aren't being updated
        self.spikes_window.update_window_start_end(new_start=min_t, new_end=max_t)
        
        
        # Update 3D Curves if we have them: TODO: figure out where this goes!
        self.TimeCurvesViewMixin_on_window_update()
        
        
        
    @QtCore.pyqtSlot(float, float)
    def update_scroll_window_region(self, new_start, new_end, block_signals: bool=True):
        """ called to update the interactive scrolling window control """
        if block_signals:
            self.ui.scroll_window_region.blockSignals(True) # Block signals so it doesn't recurrsively update
        self.ui.scroll_window_region.setRegion([new_start, new_end]) # adjust scroll control
        if block_signals:
            self.ui.scroll_window_region.blockSignals(False)
        
        
    @QtCore.pyqtSlot(object)
    def on_neuron_colors_changed(self, neuron_id_color_update_dict):
        """ Called when the neuron colors have finished changing (changed) to update the rendered elements.
        
        Inputs:
            neuron_id_color_update_dict: a neuron_id:QColor dictionary
        Updates:
            self.plots_data.all_spots
            
        """
        print(f'Spike2DRaster.neuron_id_color_update_dict: {neuron_id_color_update_dict}')
        ## Rebuild Raster Plot Points:
        self._build_cell_configs()

        # ALL Spikes in the preview window:
        curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_all_spikes_data_values()        
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.plots_data.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)] # update self.plots_data.all_spots
        # Update preview_overview_scatter_plot
        self.plots.preview_overview_scatter_plot.setData(self.plots_data.all_spots)
        if self.Includes2DActiveWindowScatter:
            self.plots.scatter_plot.setData(self.plots_data.all_spots)
        
    
    
    ######################################################
    # EpochRenderingMixin Convencince methods:
    #####################################################
    def _perform_add_render_item(self, a_plot, a_render_item):
        """Performs the operation of adding the render item from the plot specified

        Args:
            a_render_item (_type_): _description_
            a_plot (_type_): _description_
        """
        a_plot.addItem(a_render_item) # 2D (PlotItem)
        
        
    def _perform_remove_render_item(self, a_plot, a_render_item):
        """Performs the operation of removing the render item from the plot specified

        Args:
            a_render_item (IntervalRectsItem): _description_
            a_plot (PlotItem): _description_
        """
        a_plot.removeItem(a_render_item) # 2D (PlotItem)
        
        
    def add_laps_intervals(self, sess):
        """ Convenince method to add the Laps rectangles to the 2D Plots 
            NOTE: sess can be a DataSession, a Laps object, or an Epoch object containing Laps directly.
            active_2d_plot.add_PBEs_intervals(sess)
        """
        laps_interval_datasource = Specific2DRenderTimeEpochsHelper.build_Laps_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=42.0, series_height=1.0)
        self.add_rendered_intervals(laps_interval_datasource, name='Laps', debug_print=False) # removes the rendered intervals
        
    def remove_laps_intervals(self):
        self.remove_rendered_intervals('Laps', debug_print=False)
        
    def add_PBEs_intervals(self, sess):
        """ Convenince method to add the PBE rectangles to the 2D Plots 
            NOTE: sess can be a DataSession, or an Epoch object containing PBEs directly.
        """
        new_PBEs_interval_datasource = Specific2DRenderTimeEpochsHelper.build_PBEs_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=43.0, series_height=1.0) # new_PBEs_interval_datasource
        self.add_rendered_intervals(new_PBEs_interval_datasource, name='PBEs', debug_print=False) # adds the rendered intervals

    def remove_PBEs_intervals(self):
        self.remove_rendered_intervals('PBEs', debug_print=False)
        
        
        


    ######################################################
    # TimeCurvesViewMixin/PyQtGraphSpecificTimeCurvesMixin specific overrides for 2D:
    """ 
    As soon as the first 2D Time Curve plot is needed, it creates:
        self.ui.main_time_curves_view_widget - PlotItem by calling add_separate_render_time_curves_plot_item(...)
    
    main_time_curves_view_widget creates new PlotDataItems by calling self.ui.main_time_curves_view_widget.plot(...)
        This .plot(...) command can take either: 
            .plot(x=x, y=y)
            .plot(ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]
            
     
    
    """
    
    #####################################################
    def clear_all_3D_time_curves(self):
        for (aUID, plt) in self.plots.time_curves.items():
            self.ui.main_time_curves_view_widget.removeItem(plt) # this should automatically work for 2D curves as well
            # plt.delete_later() #?
        # Clear the dict
        self.plots.time_curves.clear()
        ## This part might be 3D only, but we do have a working 2D version so maybe just bring that in?
        self.remove_3D_time_curves_baseline_grid_mesh() # from Render3DTimeCurvesBaseGridMixin
        
    def update_3D_time_curves(self):
        """ initialize the graphics objects if needed, or update them if they already exist. """
        if self.params.time_curves_datasource is None:
            return
        elif self.params.time_curves_no_update:
            # don't update because we're in no_update mode
            print(f'')
            return
        else:
            # Common to both:
            # Get current plot items:
            curr_plot3D_active_window_data = self.params.time_curves_datasource.get_updated_data_window(self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time) # get updated data for the active window from the datasource # if we want the data from the whole time, we aren't getting that here unfortunately
            
            is_data_series_mode = self.params.time_curves_datasource.has_data_series_specs # True for SpikeRaster2D
            if is_data_series_mode:
                data_series_spaital_values_list = self.params.time_curves_datasource.data_series_specs.get_data_series_spatial_values(curr_plot3D_active_window_data)
                num_data_series = len(data_series_spaital_values_list)
            else:
                # old compatibility mode:
                num_data_series = 1

            # curr_data_series_index = 0
            # Loop through the active data series:                
            for curr_data_series_index in np.arange(num_data_series):
                # Data series mode:
                if is_data_series_mode:
                    # Get the current series:
                    curr_data_series_dict = data_series_spaital_values_list[curr_data_series_index]
                    
                    curr_plot_column_name = curr_data_series_dict.get('name', f'series[{curr_data_series_index}]') # get either the specified name or the generic 'series[i]' name otherwise
                    curr_plot_name = self.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
                    # points for the current plot:
                    pts = np.column_stack([curr_data_series_dict['x'], curr_data_series_dict['y'], curr_data_series_dict['z']])
                    
                    # Extra options:
                    # color_name = curr_data_series_dict.get('color_name','white')
                    extra_plot_options_dict = {'color_name':curr_data_series_dict.get('color_name', 'white'),
                                               'color':curr_data_series_dict.get('color', None),
                                               'line_width':curr_data_series_dict.get('line_width', 0.5),
                                               'z_scaling_factor':curr_data_series_dict.get('z_scaling_factor', 0.5)}
                    
                else:
                    # TODO: currently only gets the first data_column. (doesn't yet support multiple)
                    curr_plot_column_name = self.params.time_curves_datasource.data_column_names[curr_data_series_index]
                    curr_plot_name = self.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
                    
                    curve_y_value = -self.n_half_cells
                    
                    # Get y-values:
                    curr_x = self.temporal_to_spatial(curr_plot3D_active_window_data['t'].to_numpy())
                    pts = np.column_stack([curr_x, np.full_like(curr_x, curve_y_value), curr_plot3D_active_window_data[curr_plot_column_name].to_numpy()])
                    
                    extra_plot_options_dict = {}
                
                # outputs of either mode are curr_plot_name, pts
                curr_plt = self._build_or_update_plot(curr_plot_name, pts, **extra_plot_options_dict)
                # end for curr_data_series_index in np.arange(num_data_series)

            self.add_3D_time_curves_baseline_grid_mesh() # from Render3DTimeCurvesBaseGridMixin



        
        
    def _build_or_update_plot(self, plot_name, points, **kwargs):
        """ For 3D
        uses or builds a new self.ui.main_time_curves_view_widget, which the item is added to
        
        """
        if self.ui.main_time_curves_view_widget is None:
            # needs to build the primary 2D time curves plotItem:
            print(f'Spike2DRaster created a new self.ui.main_time_curves_view_widget for TimeCurvesViewMixin plots!')
            # row=0 adds above extant plot
            row_index = (self.params.main_graphics_plot_widget_rowspan * 2)+1 # row 2 if they were all rowspan 2
            self.ui.main_time_curves_view_widget = self.create_separate_render_plot_item(row=row_index, col=0, rowspan=1, colspan=1, name='new_curves_separate_plot') # PlotItem
        
        
        # build the plot arguments (color, line thickness, etc)        
        plot_args = ({'color_name':'white','line_width':0.5,'z_scaling_factor':1.0} | kwargs)
        
        ## Drop the y-value from the 3D version to get the appropriate 2D coordinates (x,y)
        if np.shape(points)[1] == 3:
            # same data from 3D version, drop the y-value accordingly:
            """
                points: (N, 3)
                # t/x, _, 'y' 
                array([[-7.47296, -35, 0.931493],
                    [-7.43977, -35, 0.931998],
                    ...
            """
            points = points[:, [0, 2]]

        assert np.shape(points)[1] == 2, f"points must be (N, 2) but it instead {np.shape(points)}"


        if plot_name in self.plots.time_curves:
            # Plot already exists, update it instead.
            plt = self.plots.time_curves[plot_name]
            plt.setData(pos=points)
        else:
            # plot doesn't exist, built it fresh.
            
            line_color = plot_args.get('color', None)
            if line_color is None:
                # if no explicit color value is provided, build a new color from the 'color_name' key, or if that's missing just use white.
                line_color = pg.mkColor(plot_args.setdefault('color_name', 'white'))
                line_color.setAlphaF(0.8)

            # Note .plot(...) seems to allow more options than .addLine(...)
            # curr_plt = self.ui.main_time_curves_view_widget.addLine(x=curr_data_series_dict['x'], y=curr_data_series_dict['y'])
            plt = self.ui.main_time_curves_view_widget.plot(points, pen=line_color, name=plot_name) # TODO: is this the slow version of name =?
            # end for curr_data_series_index in np.arange(num_data_series)
            self.plots.time_curves[plot_name] = plt # add it to the dictionary.

            # TODO: set line_width?
            # TODO: scaling like the 3D version?
            
            # plt = gl.GLLinePlotItem(pos=points, color=line_color, width=plot_args.setdefault('line_width',0.5), antialias=True)
            # plt.scale(1.0, 1.0, plot_args.setdefault('z_scaling_factor',1.0)) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.            
            # # plt.scale(1.0, 1.0, self.data_z_scaling_factor) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.
            # self.ui.main_gl_widget.addItem(plt)
            # self.plots.time_curves[plot_name] = plt # add it to the dictionary.
        return plt
    
    
    def create_separate_render_plot_item(self, row=2, col=0, rowspan=1, colspan=1, name='new_curves_separate_plot'):
        """ Adds a separate independent plot for epoch time rects to the 2D plot above the others:
        
        Requires:
            active_2d_plot.ui.main_graphics_layout_widget <GraphicsLayoutWidget>
            
        Returns:
         new_curves_separate_plot: a PlotItem
            
        """
        main_graphics_layout_widget = self.ui.main_graphics_layout_widget # GraphicsLayoutWidget
        new_curves_separate_plot = main_graphics_layout_widget.addPlot(row=row, col=col, rowspan=rowspan, colspan=colspan) # PlotItem
        new_curves_separate_plot.setObjectName(name)

        # Setup axes bounds for the bottom windowed plot:
        # new_curves_separate_plot.hideAxis('left')
        new_curves_separate_plot.showAxis('left')
        new_curves_separate_plot.hideAxis('bottom')

        # # setup the new_curves_separate_plot to have a linked X-axis to the other scroll plot:
        # main_plot_widget = self.plots.main_plot_widget # PlotItem
        # new_curves_separate_plot.setXLink(main_plot_widget) # works to synchronize the main zoomed plot (current window) with the epoch_rect_separate_plot (rectangles plotter)
        
        return new_curves_separate_plot

        
        
    # Overrides for Render3DTimeCurvesBaseGridMixin, since this 2D class can't draw a 3D background grid _________________ #
    def init_3D_time_curves_baseline_grid_mesh(self):
        self.params.setdefault('time_curves_enable_baseline_grid', False) # this is False for this class (until it's implemented at least)
        self.params.setdefault('time_curves_baseline_grid_color', 'White')
        self.params.setdefault('time_curves_baseline_grid_alpha', 0.5)
        # BaseGrid3DTimeCurvesHelper.init_3D_time_curves_baseline_grid_mesh(self)
        pass

    def add_3D_time_curves_baseline_grid_mesh(self):
        # TODO: needs to be updated on .on_adjust_temporal_spatial_mapping(...)
        # return BaseGrid3DTimeCurvesHelper.add_3D_time_curves_baseline_grid_mesh(self)
        return False

    def update_3D_time_curves_baseline_grid_mesh(self):
        # BaseGrid3DTimeCurvesHelper.update_3D_time_curves_baseline_grid_mesh(self)
        pass

    def remove_3D_time_curves_baseline_grid_mesh(self):
        return False # nothing to remove
        # return BaseGrid3DTimeCurvesHelper.remove_3D_time_curves_baseline_grid_mesh(self)
    
    
# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike2DRaster()
#     v.animation()
# dfsd