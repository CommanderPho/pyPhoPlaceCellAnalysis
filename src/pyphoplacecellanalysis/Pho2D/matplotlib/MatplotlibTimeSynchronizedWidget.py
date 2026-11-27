# from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
import time
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets, QtGui
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget

class MatplotlibTimeSynchronizedWidget(CustomMatplotlibWidget):
    """ Extends CustomMatplotlibWidget with time-synchronization properties 
    
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
        self._last_draw_perf: float = 0.0
        self._draw_coalesce_timer = QtCore.QTimer(self)
        self._draw_coalesce_timer.setSingleShot(True)
        self._draw_coalesce_timer.timeout.connect(self._flush_coalesced_draw_idle)
        
        
    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @pyqtExceptionPrintingSlot()
    def _flush_coalesced_draw_idle(self):
        self._last_draw_perf = time.perf_counter()
        if getattr(self, 'ui', None) is not None and getattr(self.ui, 'canvas', None) is not None:
            self.ui.canvas.draw_idle()


    @pyqtExceptionPrintingSlot(float, float)
    def on_window_changed(self, start_t, end_t=None):
        # called when the window is updated
        # SignalProxy packs (start_t, end_t) into a single tuple arg — unpack if needed:
        if end_t is None and isinstance(start_t, (tuple, list)) and len(start_t) == 2:
            start_t, end_t = start_t
        if (start_t is None) or (end_t is None) or (not np.isfinite(start_t)) or (not np.isfinite(end_t)):
            return
        if end_t <= start_t:
            end_t = start_t + 1e-3
        ## Update all children axes immediately so xlim queries stay synchronous:
        for curr_ax in self.axes:
            curr_ax.set_xlim(float(start_t), float(end_t))
        ## END for curr_ax in self.axes....

        # Coalesce expensive Agg canvas redraws to <= 30 Hz via draw_idle() so rapid scrolling never stalls the GUI thread:
        now = time.perf_counter()
        elapsed = now - getattr(self, '_last_draw_perf', 0.0)
        if (not self._draw_coalesce_timer.isActive()) and (elapsed >= 0.033):
            self._last_draw_perf = now
            self.ui.canvas.draw_idle()
        elif not self._draw_coalesce_timer.isActive():
            remaining_ms = max(1, int((0.033 - elapsed) * 1000.0))
            self._draw_coalesce_timer.start(remaining_ms)


    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @pyqtExceptionPrintingSlot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        
        
    
        