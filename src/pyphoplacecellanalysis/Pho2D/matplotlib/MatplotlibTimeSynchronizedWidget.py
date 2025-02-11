# from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget

class MatplotlibTimeSynchronizedWidget(CustomMatplotlibWidget):
    """ Extends CustomMatplotlibWidget with time-synchronization properties 
    
    #TODO 2024-01-06 06:13: - [ ] What does the "time-synchronization" refer to?  It seems to be used in stacks of pre-computed epoch posteriors only.
        Oh right, it's  also used somewhere in SpikeRaster2D to add interactive matplotlib-based plots below I think    

    ?? There's also `TimeSynchronizedPlotterBase`, which this class does not inherit from but the majority of the actually time-synchronized plotting subclasses do.
    
    
    Example::

        from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget
        mw = MatplotlibTimeSynchronizedWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True, scrollAreaContents_MinimumHeight=params.all_plots_height)
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance
    

    def __init__(self, disable_toolbar=True, size=(5.0, 4.0), dpi=72, **kwargs):
        super(MatplotlibTimeSynchronizedWidget, self).__init__(disable_toolbar=disable_toolbar, size=size, dpi=dpi, **kwargs)
        
        
    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.Slot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        ## Update all children axes:
        for curr_ax in self.axes:
            curr_ax.set_xlim(start_t, end_t)
        self.draw()
        
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.Slot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        
        
    
        