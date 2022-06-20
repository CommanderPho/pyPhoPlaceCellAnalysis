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
    
    def _build_all_spikes_data_values(self):
        """ build global spikes for entire dataframe (not just the current window) 
        
        Uses the df['visualization_raster_y_location'] field added to the spikes dataframe to get the y-value for the spike
        
        """
        # All units at once approach:
        # Filter the dataframe using that column and value from the list
        curr_spike_t = self.spikes_window.df[self.spikes_window.df.spikes.time_variable_name].to_numpy() # this will map
        # curr_spike_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (0.0, +self.temporal_axis_length))
        curr_spike_y = self.spikes_window.df['visualization_raster_y_location'].to_numpy() # this will map
        curr_spike_pens = [self.config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][2] for a_fragile_linear_neuron_IDX in self.spikes_window.df['fragile_linear_neuron_IDX'].to_numpy()] # get the pens for each spike from the configs map
        curr_n = len(curr_spike_t) # curr number of spikes
        return curr_spike_t, curr_spike_y, curr_spike_pens, curr_n
    
    
    
    # def _buildScrollRasterPreviewWindowGraphics(self, graphics_layout_widget: pg.GraphicsLayoutWidget=None, layout_row=0, layout_col=0):
    def _buildScrollRasterPreviewWindowGraphics(self, background_static_scroll_window_plot):
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
            self.plots.background_static_scroll_window_plot = self._buildScrollRasterPreviewWindowGraphics(self.plots.background_static_scroll_window_plot)
        
        
        """
        # Common Tick Label
        vtick = QtGui.QPainterPath()
        vtick.moveTo(0, -0.5)
        vtick.lineTo(0, 0.5)
        
        #############################
        ## Bottom Windowed Scroll Plot/Widget:

        # ALL Spikes in the preview window:
        curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_all_spikes_data_values()        
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.plots_data.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
        
        self.plots.preview_overview_scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1})
        self.plots.preview_overview_scatter_plot.opts['useCache'] = True
        self.plots.preview_overview_scatter_plot.addPoints(self.plots_data.all_spots) # , hoverable=True
        background_static_scroll_window_plot.addItem(self.plots.preview_overview_scatter_plot)
        
        # Add the linear region overlay:
        # self.ui.scroll_window_region = pg.LinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
        
        self.ui.scroll_window_region = CustomLinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
                
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
    
    