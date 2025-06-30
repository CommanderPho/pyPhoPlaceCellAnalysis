import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

# from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlotterBase import TimeSynchronizedPlotterBase
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent

from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.AnimalTrajectoryPlottingMixin import AnimalTrajectoryPlottingMixin


class TimeSynchronizedOccupancyPlotter(AnimalTrajectoryPlottingMixin, TimeSynchronizedPlotterBase):
    """ Plots the time-dependent occupancy produced by a PfND_TimeDependent instance.
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedOccupancyPlotter import TimeSynchronizedOccupancyPlotter
        included_epochs = None
        computation_config = active_session_computation_configs[0]
        # PfND version:
        t_list = []
        ratemaps_list = []
        active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        curr_sync_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_time_dependent_placefields2D)
        curr_sync_occupancy_plotter.show()

    """
    # Application/Window Configuration Options:
    applicationName = 'TimeSynchronizedOccupancyPlotterApp'
    windowName = 'TimeSynchronizedOccupancyPlotterWindow'
    
    enable_debug_print = False
    
    
    @property
    def occupancy_mode_to_render(self):
        """The occupancy_mode_to_render property."""
        return self.params.occupany_mode_to_render
    @occupancy_mode_to_render.setter
    def occupancy_mode_to_render(self, value):
        self.params.occupany_mode_to_render = value
        # on update, be sure to call self._update_plots()
        self._update_plots()
    
    
    def __init__(self, active_time_dependent_placefields2D, drop_below_threshold: float=0.0000001, occupancy_mode_to_render='seconds_occupancy', application_name=None, window_name=None, parent=None):
        """_summary_
        
        ## allows toggling between the various computed occupancies: such as raw counts,  normalized location, and seconds_occupancy
            occupancy_mode_to_render: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        
        """
        super().__init__(application_name=application_name, window_name=(window_name or TimeSynchronizedOccupancyPlotter.windowName), parent=parent) # Call the inherited classes __init__ method
    
        self.active_time_dependent_placefields = active_time_dependent_placefields2D
        
        self.setup()
        self.params.occupany_mode_to_render = occupancy_mode_to_render
        self.params.drop_below_threshold = drop_below_threshold
        
        self.buildUI()
        self._update_plots()
        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName)
        # self.params.shared_axis_order = 'row-major'
        self.params.shared_axis_order = 'col-major'
        # self.params.shared_axis_order = None
        
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.params.cmap = pg.colormap.get('viridis','matplotlib') # prepare a linear color map
        self.params.image_margins = 0.0
        self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(self.active_time_dependent_placefields.xbin, self.active_time_dependent_placefields.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
        
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

        self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=f'Occupancy -  t = {self.active_time_dependent_placefields.last_t}') # , name=f'Occupancy'
        self.ui.root_plot.setObjectName('Occupancy')
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
        
        self.ui.root_plot.setMouseEnabled(x=False, y=False)
        self.ui.root_plot.setMenuEnabled(enableMenu=False)
        
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
        
        # self.occupancy_mode_to_render: allowed values: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        if self.occupancy_mode_to_render == 'seconds_occupancy':
            image = self.active_time_dependent_placefields.curr_seconds_occupancy.copy()
            image_title = 'curr_seconds_occupancy map'
        elif self.occupancy_mode_to_render == 'num_pos_samples_occupancy':
            image = self.active_time_dependent_placefields.curr_num_pos_samples_occupancy_map.copy()
            image_title = 'curr_num_pos_samples_occupancy map'
        elif self.occupancy_mode_to_render == 'num_pos_samples_smoothed_occupancy':
            image = self.active_time_dependent_placefields.curr_num_pos_samples_smoothed_occupancy_map.copy()
            image_title = 'curr_num_pos_samples_occupancy map (smoothed)'
        elif self.occupancy_mode_to_render == 'normalized_occupancy':
            image = self.active_time_dependent_placefields.curr_normalized_occupancy.copy()
            image_title = 'curr_normalized_occupancy map'
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
            
        self.AnimalTrajectoryPlottingMixin_update_plots()
        
        # self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
        self.setWindowTitle(f'TimeSynchronizedOccupancyPlotter - {image_title} t = {curr_t}')
    

# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()