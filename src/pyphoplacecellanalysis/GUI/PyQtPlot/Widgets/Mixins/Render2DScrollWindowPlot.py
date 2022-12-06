import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem

from pyphoplacecellanalysis.General.Model.Datasources.Datasources import DataframeDatasource


class Render2DScrollWindowPlotMixin:
    """ Adds a LinearRegionItem to the plot that represents the entire data timerange which defines a user-adjustable window into the data. Finally, also adds a plot that shows only the zoomed-in data within the window. 
    
    Known Uses:
        Implemented by Spike2DRaster
    
    Requires:
        a Datasource to fetch the spiking data from.
        TimeWindow: a Active Window to synchronize the LinearRegionItem (2D Scroll Widget) with.
    
    
    Provides:
        window_scrolled (float, float) signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    """
    
    ## Scrollable Window Signals
    window_scrolled = QtCore.pyqtSignal(float, float) # signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    
    

    # def ScrollRasterPreviewWindow_on_BuildUI(self, graphics_layout_widget: pg.GraphicsLayoutWidget=None, layout_row=0, layout_col=0):
    def ScrollRasterPreviewWindow_on_BuildUI(self, background_static_scroll_window_plot):
        """ Note that this doesn't need to update because the background is static (it shows all time) 
        
        Inputs:
        
        background_static_scroll_window_plot: the plot to add to. For example created with `graphics_layout_widget.addPlot(row=layout_row, col=layout_col)`
         
        Requires:
        
            self.plots
            self.ui

            self.spikes_window.total_df_start_end_times # to get the current start/end times to set the linear region to
        Creates:
            self.plots_data.all_spots # data for all spikes to be rendered on a scatter plot
            self.ui.scroll_window_region # a pg.LinearRegionItem                        
            self.plots.preview_overview_scatter_plot # a pg.ScatterPlotItem
        
        Usage:
            self.plots.background_static_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0)
            self.plots.background_static_scroll_window_plot = self.ScrollRasterPreviewWindow_on_BuildUI(self.plots.background_static_scroll_window_plot)
        
        
        """
        # Common Tick Label
        vtick = QtGui.QPainterPath()
        vtick.moveTo(0, -0.5)
        vtick.lineTo(0, 0.5)
        
        #############################
        ## Bottom Windowed Scroll Plot/Widget:

        # ALL Spikes in the preview window:
        curr_spike_x, curr_spike_y, curr_spike_pens, self.plots_data.all_spots, curr_n = self._build_all_spikes_data_values()
        
        self.plots.preview_overview_scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1})
        self.plots.preview_overview_scatter_plot.setObjectName('preview_overview_scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
        self.plots.preview_overview_scatter_plot.opts['useCache'] = True
        self.plots.preview_overview_scatter_plot.addPoints(self.plots_data.all_spots) # , hoverable=True
        background_static_scroll_window_plot.addItem(self.plots.preview_overview_scatter_plot)
        
        # Add the linear region overlay:
        # self.ui.scroll_window_region = pg.LinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
        
        self.ui.scroll_window_region = CustomLinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
        self.ui.scroll_window_region.setObjectName('scroll_window_region')
        self.ui.scroll_window_region.setZValue(10)
        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
        background_static_scroll_window_plot.addItem(self.ui.scroll_window_region, ignoreBounds=True)
        self.ui.scroll_window_region.sigRegionChanged.connect(self._Render2DScrollWindowPlot_on_linear_region_item_update)

        
        # Setup axes bounds for the bottom windowed plot:
        background_static_scroll_window_plot.hideAxis('left')
        background_static_scroll_window_plot.hideAxis('bottom')
        # background_static_scroll_window_plot.setLabel('bottom', 'Time', units='s')
        background_static_scroll_window_plot.setMouseEnabled(x=False, y=False)
        background_static_scroll_window_plot.disableAutoRange('xy')
        # background_static_scroll_window_plot.enableAutoRange(x=False, y=False)
        background_static_scroll_window_plot.setAutoVisible(x=False, y=False)
        background_static_scroll_window_plot.setAutoPan(x=False, y=False)
        
        # Setup range for plot:
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        background_static_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
        background_static_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)
        
        return background_static_scroll_window_plot

    
    def _fix_initial_linearRegionLocation(self, debug_print=False):
        """ Hopefully finally resolves the issue where the linear scroll region window was always cut-off initially. 
        Note that when this is called, self.temporal_axis_length and self.spikes_window.window_duration were both observed to be 0.0, which is why they couldn't be used.
        """
        confirmed_valid_window_start_t = self.spikes_window.total_data_start_time
        if (self.spikes_window.window_duration == 0.0):
            # invalid window length, just choose something reasonable the user can grab, say 5% of the total window data
            total_data_duration = self.spikes_window.total_data_end_time - self.spikes_window.total_data_start_time
            reasonable_active_window_duration = float(total_data_duration) * 0.05 # 5%
            ## UGHH, it works but please note that the final window is actually going to be MORE than 5% of the total data duration because of the temporal_zoom_factor > 1.0. 
        else:
            reasonable_active_window_duration = float(self.spikes_window.window_duration)
            
        # Compute the final reasonable window end_t:
        confirmed_valid_window_end_t = confirmed_valid_window_start_t + reasonable_active_window_duration
        if debug_print:
            print(f'_fix_initial_linearRegionLocation():')
            print(f'\tconfirmed_valid_window: (start_t: {confirmed_valid_window_start_t}, end_t: {confirmed_valid_window_end_t})')
        ## THIS SHOULD FIX THE INITIAL SCROLLWINDOW ISSUE, preventing it from being outside the window it's rendered on top of, unless the active window is set wrong.
        # self.update_scroll_window_region(confirmed_valid_window_start_t, confirmed_valid_window_end_t, block_signals=False)
        self.Render2DScrollWindowPlot_on_window_update(confirmed_valid_window_start_t, confirmed_valid_window_end_t)
        
        
    def update_rasters(self):
        """ updates all rasters (which are scatter plots) from the self.plot_data.all_spots variable """
        # Update preview_overview_scatter_plot
        self.plots.preview_overview_scatter_plot.setData(self.plots_data.all_spots)
        if self.Includes2DActiveWindowScatter:
            self.plots.scatter_plot.setData(self.plots_data.all_spots)
            
            

    @QtCore.pyqtSlot()
    def _Render2DScrollWindowPlot_on_linear_region_item_update(self) -> None:
        """self when the region moves.zoom_Change plotter area"""
        # self.ui.scroll_window_region.setZValue(10) # bring to the front
        min_x, max_x = self.ui.scroll_window_region.getRegion() # get the current region
        self.window_scrolled.emit(min_x, max_x) # emit this mixin's own window_scrolled function
        
        

    @QtCore.pyqtSlot()
    def Render2DScrollWindowPlot_on_init():
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.pyqtSlot()
    def Render2DScrollWindowPlot_on_setup():
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # Connect the signals for the zoom region and the LinearRegionItem
        # self.ui.scroll_window_region.sigRegionChanged.connect(self.update_zoom_plotter)
        pass

    @QtCore.pyqtSlot()
    def Render2DScrollWindowPlot_on_destroy():
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass

    @QtCore.pyqtSlot(float, float)
    def Render2DScrollWindowPlot_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # Make sure that the scroller isn't too tiny to grab.
        self.ui.scroll_window_region.setRegion([new_start, new_end]) # adjust scroll control
    
    
    # ==================================================================================================================== #
    # Private/Internal Methods                                                                                             #
    # ==================================================================================================================== #
    
    def _build_all_spikes_data_values(self, is_included_indicies=None, **kwargs):
        """ build global spikes for entire dataframe (not just the current window) 
        
        Called ONLY by ScrollRasterPreviewWindow_on_BuildUI(self, background_static_scroll_window_plot)

        Implementation:
            Uses the df['visualization_raster_y_location'] field added to the spikes dataframe to get the y-value for the spike
            Note that the colors are built using the self.config_fragile_linear_neuron_IDX_map property
            is_included_indicies: Optional np.array of bools indicating whether each spike is included in the generated points
            Internally calls `cls.build_spikes_data_values_from_df(...)
            
        """
        # All units at once approach:
        return Render2DScrollWindowPlotMixin.build_spikes_data_values_from_df(self.spikes_window.df, self.config_fragile_linear_neuron_IDX_map, is_spike_included=is_included_indicies, **kwargs)
        
    def _build_all_spikes_all_spots(self, is_included_indicies=None, **kwargs):
        """ build the all_spots from the global spikes for entire dataframe (not just the current window) 
        

        Called by:
            on_unit_sort_order_changed
            on_neuron_colors_changed
            reset_spike_emphasis
            update_spike_emphasis


        Example:
            ## Rebuild Raster Plot Points:
            self._build_cell_configs()

            # ALL Spikes in the preview window:
            self.plots_data.all_spots = self._build_all_spikes_all_spots()
            # Update preview_overview_scatter_plot
            self.plots.preview_overview_scatter_plot.setData(self.plots_data.all_spots)
            if self.Includes2DActiveWindowScatter:
                self.plots.scatter_plot.setData(self.plots_data.all_spots)     
        """
        return Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(self.spikes_window.df, self.config_fragile_linear_neuron_IDX_map, is_spike_included=is_included_indicies, **kwargs)
    


    
    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #
    @classmethod
    def build_spikes_data_values_from_df(cls, spikes_df, config_fragile_linear_neuron_IDX_map, is_spike_included=None, **kwargs):
        """ build global spikes for entire dataframe (not just the current window) 
        
        Called by:
            self.build_spikes_all_spots_from_df(...)
            cls.build_spikes_all_spots_from_df(...)

        Uses the df['visualization_raster_y_location'] field added to the spikes dataframe to get the y-value for the spike
        
        Note that the colors are built using the self.config_fragile_linear_neuron_IDX_map property
        
        config_fragile_linear_neuron_IDX_map: a map from fragile_linear_neuron_IDX to config (tuple) values
        is_included_indicies: Optional np.array of bools indicating whether each spike is included in the generated points
        
        """
        # All units at once approach:
        active_time_variable_name = spikes_df.spikes.time_variable_name
        # Copy only the relevent columns so filtering is easier:
        filtered_spikes_df = spikes_df[[active_time_variable_name, 'visualization_raster_y_location',  'visualization_raster_emphasis_state', 'fragile_linear_neuron_IDX']].copy()
        
        spike_emphasis_states = kwargs.get('spike_emphasis_state', None)
        if spike_emphasis_states is not None:
            assert len(spike_emphasis_states) == np.shape(spikes_df)[0], f"if specified, spike_emphasis_states must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]} and len(is_included_indicies): {len(spike_emphasis_states)}"
            # Can set it on the dataframe:
            # 'visualization_raster_y_location'
        
        if is_spike_included is not None:
            assert len(is_spike_included) == np.shape(spikes_df)[0], f"if specified, is_included_indicies must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]} and len(is_included_indicies): {len(is_spike_included)}"
            ## filter them by the is_included_indicies:
            filtered_spikes_df = filtered_spikes_df[is_spike_included]
        
        # Filter the dataframe using that column and value from the list
        curr_spike_t = filtered_spikes_df[active_time_variable_name].to_numpy() # this will map
        curr_spike_y = filtered_spikes_df['visualization_raster_y_location'].to_numpy() # this will map
        
        # config_fragile_linear_neuron_IDX_map values are of the form: (i, fragile_linear_neuron_IDX, curr_pen, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i])
        # Emphasis/Deemphasis-Dependent Pens:
        curr_spike_pens = [config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][2][a_spike_emphasis_state] for a_fragile_linear_neuron_IDX, a_spike_emphasis_state in zip(filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # get the pens for each spike from the configs map
        
        curr_n = len(curr_spike_t) # curr number of spikes
        # builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes
        pos = np.vstack((curr_spike_t, curr_spike_y))
        all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
        return curr_spike_t, curr_spike_y, curr_spike_pens, all_spots, curr_n
    
    @classmethod
    def build_spikes_all_spots_from_df(cls, spikes_df, config_fragile_linear_neuron_IDX_map, is_spike_included=None, **kwargs):
        """ builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes 
        Internally calls `cls.build_spikes_data_values_from_df(...)
        """
        curr_spike_x, curr_spike_y, curr_spike_pens, all_spots, curr_n = cls.build_spikes_data_values_from_df(spikes_df, config_fragile_linear_neuron_IDX_map, is_spike_included=is_spike_included, **kwargs)
        return all_spots
    