from copy import deepcopy
import numpy as np
import pandas as pd
from qtpy import QtCore, QtWidgets


class CrosshairsTracingMixin:
    """ Implementors render/track crosshairs used for tracing values
    
    self.params.wants_crosshairs = False
    
    
    sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance
    
    
    Usage:
    
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.Mixins.CrosshairsTracingMixin import CrosshairsTracingMixin
    
    
    """
    sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance
    

    def add_crosshairs(self, plot_item, name, matrix=None, xbins=None, ybins=None, enable_y_trace:bool=False):
        """ adds crosshairs that allow the user to hover a bin and have the label dynamically display the bin (x, y) and value."""
        raise NotImplementedError(f'must override in implementor')
    
    def remove_crosshairs(self, plot_item, name=None):
        """ Removes crosshairs"""
        raise NotImplementedError(f'must override in implementor')
    

    def update_crosshair_trace(self, wants_crosshairs_trace: bool):
        """ updates the crosshair trace peferences """
        raise NotImplementedError(f'must override in implementor')
    