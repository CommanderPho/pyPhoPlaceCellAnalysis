import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np

from pyphoplacecellanalysis.GUI.PyQtPlot.Model.Datasources import DataframeDatasource




class SpikesDatasource(DataframeDatasource):
    """ Provides the list of values, 'v' and the timestamps at which they occur 't'.
    Externally should 
    
    Contains a dataframe.
    
    Signals:
    	source_data_changed_signal = QtCore.pyqtSignal() # signal emitted when the internal model data has changed.
     
     Slots:
        @QtCore.pyqtSlot(float, float) 
        def get_updated_data_window(self, new_start, new_end):
    """

    @property
    def time_column_name(self):
        """ the name of the relevant time column. Gets the values from the spike dataframe """
        return self.df.spikes.time_variable_name
    
    
    def __init__(self, df, datasource_name='default_spikes_datasource'):
        # Initialize the datasource as a QObject
        DataframeDatasource.__init__(self, df, datasource_name=datasource_name)

    """
        ## It seems the relevant columns are:
        'visualization_raster_y_location' # for y-offsets in 3D plot
        'unit_id' # for colors
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
    """ 
    


class Render2DScrollWindowPlotMixin:
    """ 
    
    Requires:
        a Datasource to fetch the spiking data from.
        TimeWindow: a Active Window to synchronize the LinearRegionItem (2D Scroll Widget) with.
    
    """
    
    ## Scrollable Window Signals
    window_scrolled = QtCore.pyqtSignal(float, float) # signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    
    
    # def _buildScrollRasterPreviewWindowGraphics(self, graphics_layout_widget: pg.GraphicsLayoutWidget=None, layout_row=0, layout_col=0):
    def _buildScrollRasterPreviewWindowGraphics(self, main_scroll_window_plot):
        """ Note that this doesn't need to update because the background is static (it shows all time) 
        
        Inputs:
        
        main_scroll_window_plot: the plot to add to. For example created with `graphics_layout_widget.addPlot(row=layout_row, col=layout_col)`
         
        Requires:
        
            self.plots
            self.ui
        
        Creates:
        
        self.ui.scroll_window_region
        
        
        self.plots.preview_overview_scatter_plot
        
        Usage:
            self.ui.main_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0)
            self.ui.main_scroll_window_plot = self._buildScrollRasterPreviewWindowGraphics(self.ui.main_scroll_window_plot)
        
        
        """
        
        # if graphics_layout_widget is None:
        #     raise NotImplementedError
        #     self.ui.main_graphics_layout_widget = pg.GraphicsLayoutWidget()
        #     self.ui.main_graphics_layout_widget.useOpenGL(True)
        #     self.ui.main_graphics_layout_widget.resize(1000,600)
        #     # Add the main widget to the layout in the (0, 0) location:
        #     self.ui.layout.addWidget(self.ui.main_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
            
        
        # Common Tick Label
        vtick = QtGui.QPainterPath()
        vtick.moveTo(0, -0.5)
        vtick.lineTo(0, 0.5)
        
        #############################
        ## Bottom Windowed Scroll Plot/Widget:

        # ALL Spikes in the preview window:
        curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_all_spikes_data_values()        
        pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
        self.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
        
        self.plots.preview_overview_scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1})
        self.plots.preview_overview_scatter_plot.opts['useCache'] = True
        self.plots.preview_overview_scatter_plot.addPoints(self.all_spots)
        main_scroll_window_plot.addItem(self.plots.preview_overview_scatter_plot)
        
        # Add the linear region overlay:
        self.ui.scroll_window_region = pg.LinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
        self.ui.scroll_window_region.setZValue(10)
        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
        main_scroll_window_plot.addItem(self.ui.scroll_window_region, ignoreBounds=True)
        self.ui.scroll_window_region.sigRegionChanged.connect(self._Render2DScrollWindowPlot_on_linear_region_item_update)

        
        # Setup axes bounds for the bottom windowed plot:
        main_scroll_window_plot.hideAxis('left')
        main_scroll_window_plot.hideAxis('bottom')
        # main_scroll_window_plot.setLabel('bottom', 'Time', units='s')
        main_scroll_window_plot.setMouseEnabled(x=False, y=False)
        main_scroll_window_plot.disableAutoRange('xy')
        # main_scroll_window_plot.enableAutoRange(x=False, y=False)
        main_scroll_window_plot.setAutoVisible(x=False, y=False)
        main_scroll_window_plot.setAutoPan(x=False, y=False)
        
        # Setup range for plot:
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        main_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
        main_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)
        
        return main_scroll_window_plot


    @QtCore.pyqtSlot()
    def _Render2DScrollWindowPlot_on_linear_region_item_update(self) -> None:
        """self when the region moves.zoom_Change plotter area"""
        self.ui.scroll_window_region.setZValue(10) # bring to the front
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
        pass
    
    
     
    # def _buildScrollRasterPreviewWindowGraphics(self):
    #     # Common Tick Label
    #     vtick = QtGui.QPainterPath()
    #     vtick.moveTo(0, -0.5)
    #     vtick.lineTo(0, 0.5)
        
    #     #############################
    #     ## Bottom Windowed Scroll Plot/Widget:
    #     self.ui.main_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0)
    #     # ALL Spikes in the preview window:
    #     curr_spike_x, curr_spike_y, curr_spike_pens, curr_n = self._build_all_spikes_data_values()        
    #     pos = np.vstack((curr_spike_x, curr_spike_y)) # np.shape(curr_spike_t): (11,), np.shape(curr_spike_x): (11,), np.shape(curr_spike_y): (11,), curr_n: 11
    #     self.all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)]
        
    #     self.plots.preview_overview_scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1})
    #     self.plots.preview_overview_scatter_plot.opts['useCache'] = True
    #     self.plots.preview_overview_scatter_plot.addPoints(self.all_spots)
    #     self.ui.main_scroll_window_plot.addItem(self.plots.preview_overview_scatter_plot)
        
    #     # Add the linear region overlay:
    #     self.ui.scroll_window_region = pg.LinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
    #     self.ui.scroll_window_region.setZValue(10)
    #     # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
    #     self.ui.main_scroll_window_plot.addItem(self.ui.scroll_window_region, ignoreBounds=True)
        
    #     # Setup axes bounds for the bottom windowed plot:
    #     earliest_t, latest_t = self.spikes_window.total_df_start_end_times
    #     self.ui.main_scroll_window_plot.hideAxis('left')
    #     self.ui.main_scroll_window_plot.hideAxis('bottom')
    #     # self.ui.main_scroll_window_plot.setLabel('bottom', 'Time', units='s')
    #     self.ui.main_scroll_window_plot.setMouseEnabled(x=False, y=False)
    #     self.ui.main_scroll_window_plot.disableAutoRange('xy')
    #     # self.ui.main_scroll_window_plot.enableAutoRange(x=False, y=False)
    #     self.ui.main_scroll_window_plot.setAutoVisible(x=False, y=False)
    #     self.ui.main_scroll_window_plot.setAutoPan(x=False, y=False)
    #     self.ui.main_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
    #     self.ui.main_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)