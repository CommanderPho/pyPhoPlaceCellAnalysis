import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore
from pyphoplacecellanalysis.External.pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget


class MatplotlibTimeSynchronizedWidget(MatplotlibWidget):
    """ Extends MatplotlibWidget with time-synchronization properties 
    
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
        
        