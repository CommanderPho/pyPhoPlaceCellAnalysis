# from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui
# from pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

# from pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget

class MatplotlibTimeSynchronizedWidget(CustomMatplotlibWidget):
    """ Extends CustomMatplotlibWidget with time-synchronization properties 
    
    Example::
    
        mw = MatplotlibTimeSynchronizedWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, disable_toolbar=True, size=(5.0, 4.0), dpi=100, **kwargs):
        super(MatplotlibTimeSynchronizedWidget, self).__init__(disable_toolbar=disable_toolbar, size=size, dpi=dpi, **kwargs)
        
        
    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        curr_ax = self.ax
        curr_ax.set_xlim(start_t, end_t)
        self.draw()
        
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.Slot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        
        
        

class ScrollableMatplotlibTimeSynchronizedWidget(QtWidgets.QMainWindow):
    """
    
    Usage:
        from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget, ScrollableMatplotlibTimeSynchronizedWidget

        _temp_out = curr_active_pipeline.display('_display_plot_decoded_epoch_slices', active_config_name, debug_test_max_num_slices=16)
        params, plots_data, plots, ui = _temp_out
        ui.mw.setMinimumHeight(params.all_plots_height)

        scrollable_mw_window = ScrollableMatplotlibTimeSynchronizedWidget(params.name, ui=ui, window_title=params.window_title, scrollAreaContentsWidget=ui.mw)

        
    """
    def __init__(self, name, ui=None, window_title=None, scrollAreaContentsWidget=None):
        super().__init__()
        self.name = name
        if ui is None:
            ui = PhoUIContainer(name=name)
            ui.connections = PhoUIContainer(name=name)
        
        self.ui = ui
            
        if window_title is None:
            window_title = name
        self.window_title = window_title
        
        self.ui.scrollAreaContentsWidget = scrollAreaContentsWidget
        
        self.initUI()

    def initUI(self):
        
        self.ui.scrollAreaWidget = QtWidgets.QScrollArea() # Scroll Area which contains the widgets, set as the centralWidget
        
        #Scroll Area Properties
        self.ui.scrollAreaWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn) #  Qt.ScrollBarAlwaysOn
        self.ui.scrollAreaWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # Qt.ScrollBarAlwaysOff
        self.ui.scrollAreaWidget.setWidgetResizable(True)
        
        if self.ui.scrollAreaContentsWidget is not None:
            # Set contents widget if we have it:
            self.ui.scrollAreaWidget.setWidget(self.ui.scrollAreaContentsWidget)

        self.setCentralWidget(self.ui.scrollAreaWidget)

        self.setGeometry(600, 100, 1000, 900)
        self.setWindowTitle('Scroll Area Demonstration')
        self.show()

        return
    
    
    def setScrollAreaContents(self, widget: QtWidgets.QWidget):
        """Sets the widget contained in the scrollArea after the fact

        Args:
            widget (_type_): _description_
        """
        self.ui.scrollAreaContentsWidget = widget # Widget that contains the collection of Vertical Box
        self.ui.scrollAreaWidget.setWidget(self.ui.scrollAreaContentsWidget)
        