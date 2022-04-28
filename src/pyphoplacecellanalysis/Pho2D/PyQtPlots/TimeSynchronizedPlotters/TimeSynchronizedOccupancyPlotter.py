import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import _pyqtplot_build_image_bounds_extent


class TimeSynchronizedOccupancyPlotter(TimeSynchronizedPlotterBase):
    """
    
    Usage:
    
        included_epochs = None
        computation_config = active_session_computation_configs[0]
        print('Recomputing active_epoch_placefields2D...', end=' ')
        # PfND version:
        t_list = []
        ratemaps_list = []
        active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        curr_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_time_dependent_placefields2D)
        curr_occupancy_plotter.show()

    """
    # Application/Window Configuration Options:
    applicationName = 'TimeSynchronizedOccupancyPlotterApp'
    windowName = 'TimeSynchronizedOccupancyPlotterWindow'
    
    enable_debug_print = False
    
    def __init__(self, active_time_dependent_placefields2D, drop_below_threshold: float=0.0000001, application_name=None, parent=None):
        """_summary_
        """
        super().__init__(application_name=application_name, parent=parent) # Call the inherited classes __init__ method
        
        self.active_time_dependent_placefields = active_time_dependent_placefields2D
        
        self.setup()
        self.params.drop_below_threshold = drop_below_threshold
        
        self.buildUI()
        self._update_plots()
        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName)
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        self.params.image_margins = 0.0
        self.params.image_bounds_extent, self.params.x_range, self.params.y_range = _pyqtplot_build_image_bounds_extent(self.active_time_dependent_placefields.xbin, self.active_time_dependent_placefields.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        
        self.params.recent_position_trajectory_max_seconds_ago = 7.0
        
        
    def _buildGraphics(self):
        # Build a single image view to display the image:
        # ## Single pg.ImageView Mode:
        # self.ui.imv = pg.ImageView()
        # self.ui.layout.addWidget(self.ui.imv, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        ## More Involved Mode:
        self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
        # self.ui.root_view = self.ui.root_graphics_layout_widget.addViewBox()
        ## lock the aspect ratio so pixels are always square
        # self.ui.root_view.setAspectLocked(True)

        ## Create image item
        
        self.ui.imv = pg.ImageItem(border='w')
        # self.ui.root_view.addItem(self.ui.imv)
        # self.ui.root_view.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

        self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, name=f'Occupancy', title=f'Occupancy -  t = {self.active_time_dependent_placefields.last_t}')
        self.ui.root_plot.addItem(self.ui.imv)  # add ImageItem to PlotItem
        self.ui.root_plot.showAxes(True)
        self.ui.root_plot.setXRange(*self.params.x_range)
        self.ui.root_plot.setYRange(*self.params.y_range)

        ## Optional Animal Trajectory Path Plot:            
        # Note that pg.PlotDataItem is a combination of pg.PlotCurveItem and pg.ScatterPlotItem
        self.ui.trajectory_curve = pg.PlotDataItem(pen=({'color': 'white', 'width': 2}), symbol='o', symbolBrush=(50,50,50), pxMode=True, symbolSize=6.0, antialias=True, name='recent trajectory') #downsample=20, downsampleMethod='peak', autoDownsample=True, skipFiniteCheck=True, clipToView=True
        
        
        # curr_occupancy_plotter.ui.trajectory_curve = pg.PlotCurveItem(pen=({'color': 'white', 'width': 3}), skipFiniteCheck=True)
        self.ui.root_plot.addItem(self.ui.trajectory_curve)

        # ## Optional Interactive Color Bar:
        # bar = pg.ColorBarItem(values= (0, 1), colorMap=self.params.cmap, width=5, interactive=False) # prepare interactive color bar
        # # Have ColorBarItem control colors of img and appear in 'plot':
        # bar.setImageItem(self.ui.imv, insert_in=self.ui.root_plot)
        
        self.ui.layout.addWidget(self.ui.root_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        # Set the color map:
        self.ui.imv.setColorMap(self.params.cmap)
        ## Set initial view bounds
        # self.ui.root_view.setRange(QtCore.QRectF(0, 0, 600, 600))

    
    def update(self, t, defer_render=False):
        # Compute the updated placefields/occupancy for the time t:
        with np.errstate(divide='ignore', invalid='ignore'):
            self.active_time_dependent_placefields.update(t)
        # Update the plots:
        if not defer_render:
            self._update_plots()


    def _update_plots(self):
        if self.enable_debug_print:
            print(f'TimeSynchronizedOccupancyPlotter._update_plots()')
            
        # Update the existing one:
        
        # Update the plots:
        curr_t = self.active_time_dependent_placefields.last_t
        
        # image = curr_ratemap.occupancy
        # image = self.active_time_dependent_placefields.curr_normalized_occupancy
        # image_title = 'curr_normalized_occupancy'
        
        image = self.active_time_dependent_placefields.curr_seconds_occupancy.copy()
        image_title = 'curr_seconds_occupancy'
        
        if self.params.drop_below_threshold is not None:
            # image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan # null out the occupancy
            image[np.where(image < self.params.drop_below_threshold)] = np.nan # null out the occupancy
        
        # self.ui.imv.setImage(image, xvals=self.active_time_dependent_placefields.xbin)
        
        self.ui.imv.setImage(image, rect=self.params.image_bounds_extent)
        
        # Update most recent trajectory plot:
        curr_trajectory_rows = self.curr_recent_trajectory
        self.ui.trajectory_curve.setData(x=curr_trajectory_rows.x.to_numpy(), y=curr_trajectory_rows.y.to_numpy()) 
        
        self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
    
    
    # @QtCore.Slot(float, float)
    # def on_window_changed(self, start_t, end_t):
    #     # called when the window is updated
    #     if self.enable_debug_print:
    #         print(f'TimeSynchronizedOccupancyPlotter.on_window_changed(start_t: {start_t}, end_t: {end_t})')
    #     if self.enable_debug_print:
    #         profiler = pg.debug.Profiler(disabled=True, delayed=True)
            
    #     self.update(end_t)
    #     self._update_plots()
        
    #     if self.enable_debug_print:
    #         profiler('Finished calling _update_plots()')
            
            
    @property
    def curr_recent_trajectory(self):
        """The animal's most recent trajectory preceding self.active_time_dependent_placefields.last_t"""
        # Fixed time ago backward:
        earliest_trajectory_start_time = self.last_t - self.params.recent_position_trajectory_max_seconds_ago # gets the earliest start time for the current trajectory to display
        return self.active_time_dependent_placefields.filtered_pos_df.position.time_sliced(earliest_trajectory_start_time, self.last_t)[['t','x','y']] # Get all rows within the most recent time

            
# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()