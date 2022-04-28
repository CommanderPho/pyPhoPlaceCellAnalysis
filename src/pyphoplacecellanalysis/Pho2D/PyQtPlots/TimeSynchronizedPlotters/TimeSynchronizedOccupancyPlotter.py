import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

class TimeSynchronizedOccupancyPlotter(QtWidgets.QWidget):
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
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        
        if application_name is not None:
            self.applicationName = application_name
        else:
            self.applicationName = TimeSynchronizedOccupancyPlotter.applicationName
        
        self.windowName = TimeSynchronizedOccupancyPlotter.windowName
        self.enable_debug_print = TimeSynchronizedOccupancyPlotter.enable_debug_print
        self.setup()
        self.active_time_dependent_placefields = active_time_dependent_placefields2D
        self.params.drop_below_threshold = drop_below_threshold
        
        self.buildUI()
        self.resize(800,800)
        self.setWindowTitle(self.windowName)
        # self.show()
        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName)
        ## Build the colormap to be used:
        # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
        
    def buildUI(self):
        """ for QGridLayout
            addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        """
        self.ui = PhoUIContainer()
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setContentsMargins(0, 0, 0, 0)
        self.ui.layout.setVerticalSpacing(0)
        self.ui.layout.setHorizontalSpacing(0)
        # self.setStyleSheet("background : #1B1B1B; color : #727272")
        #### Build Graphics Objects #####
        self._buildGraphics()        
        self.setLayout(self.ui.layout)
        
    def _buildGraphics(self):
        # Build a single image view to display the image:
        self.ui.imv = pg.ImageView()
        self.ui.layout.addWidget(self.ui.imv, 0, 0) # add the GLViewWidget to the layout at 0, 0
        # Set the color map:
        self.ui.imv.setColorMap(self.params.cmap)
    
    
    def update(self, t):
        # Compute the updated placefields/occupancy for the time t:
        with np.errstate(divide='ignore', invalid='ignore'):
            self.active_time_dependent_placefields.update(t)
        # # Update the plots:
        # self._update_plots()

    def _update_plots(self):
        """
        
        """
        if self.enable_debug_print:
            print(f'TimeSynchronizedOccupancyPlotter._update_plots()')
            
        # Update the existing one:
        
        # Update the plots:
        curr_t = self.active_time_dependent_placefields.last_t
        curr_ratemap = self.active_time_dependent_placefields.ratemap
        
        # image = curr_ratemap.occupancy
        # image = self.active_time_dependent_placefields.curr_normalized_occupancy
        # image_title = 'curr_normalized_occupancy'
        
        image = self.active_time_dependent_placefields.curr_seconds_occupancy.copy()
        image_title = 'curr_seconds_occupancy'
        
        if self.params.drop_below_threshold is not None:
            # image[np.where(occupancy < self.params.drop_below_threshold)] = np.nan # null out the occupancy
            image[np.where(image < self.params.drop_below_threshold)] = np.nan # null out the occupancy
        
        self.ui.imv.setImage(image, xvals=self.active_time_dependent_placefields.xbin)
        self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
        
    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'TimeSynchronizedOccupancyPlotter.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
            
        self.update(end_t)
        self._update_plots()
        
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
            
            
# included_epochs = None
# computation_config = active_session_computation_configs[0]
# active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
#                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
#                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
# curr_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_time_dependent_placefields2D)
# curr_occupancy_plotter.show()