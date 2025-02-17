import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets

from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent

import pyphoplacecellanalysis.External.pyqtgraph as pg
# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

class TimeSynchronizedPlotterBase(QtWidgets.QWidget):
    """ Subclasses generally display time-dependent results produced by a PfND_TimeDependent instance in a manner synchronized with another plotter/renderer.
    
    Usage:
    
        included_epochs = None
        computation_config = active_session_computation_configs[0]
        # PfND version:
        t_list = []
        ratemaps_list = []
        active_time_dependent_placefields2D = PfND_TimeDependent(deepcopy(sess.spikes_df.copy()), deepcopy(sess.position), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        curr_sync_plotter = TimeSynchronizedPlotterBase(active_time_dependent_placefields2D)
        curr_sync_plotter.show()

    """
    # Application/Window Configuration Options:
    applicationName = 'TimeSynchronizedPlotterBaseApp'
    windowName = 'TimeSynchronizedPlotterBaseWindow'
    
    enable_debug_print = False
    
    @property
    def last_t(self):
        """Convinence accessor to active_time_dependent_placefields's last_t property."""
        return self.active_time_dependent_placefields.last_t


    @property
    def active_plot_target(self):
        """The active_plot_target property."""
        raise NotImplementedError(f'subclass must override!')
    

    def __init__(self, application_name=None, window_name=None, parent=None):
        """_summary_
        """
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        
        if application_name is not None:
            self.applicationName = application_name
        else:
            # self.applicationName = TimeSynchronizedPlotterBase.applicationName
            self.applicationName = self.applicationName
            
        if window_name is not None:
            self.windowName = window_name
        else:
            # self.windowName = TimeSynchronizedPlotterBase.windowName
            self.windowName = self.windowName
        
        self.enable_debug_print = TimeSynchronizedPlotterBase.enable_debug_print
        # self.setup()        
        # self.buildUI()
        # self.show()
        
    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        raise NotImplementedError # Inheriting classes must override setup to perform particular setup
        self.app = pg.mkQApp(self.applicationName)
        self.params = VisualizationParameters(self.applicationName)
        
        
    def buildUI(self):
        """ for QGridLayout
            addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        """
        self.ui = PhoUIContainer()
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setObjectName('root_layout')
        self.ui.layout.setContentsMargins(0, 0, 0, 0)
        self.ui.layout.setVerticalSpacing(0)
        self.ui.layout.setHorizontalSpacing(0)
        # self.setStyleSheet("background : #1B1B1B; color : #727272")
        #### Build Graphics Objects #####
        self._buildGraphics()        
        self.setLayout(self.ui.layout)
        self.resize(800,800)
        self.setWindowTitle(self.windowName)
        
    def _buildGraphics(self):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        raise NotImplementedError
        # self.ui.imv = pg.ImageView()
        # self.ui.layout.addWidget(self.ui.imv, 0, 0) # add the GLViewWidget to the layout at 0, 0
        # # Set the color map:
        # self.ui.imv.setColorMap(self.params.cmap)
    
    
    def update(self, t, defer_render=False):
        raise NotImplementedError

    def _update_plots(self):
        """ Implementor must override! """
        raise NotImplementedError
    
        
    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'TimeSynchronizedPlotterBase.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
            
        self.update(end_t, defer_render=False)
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
            
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.Slot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        
        
    