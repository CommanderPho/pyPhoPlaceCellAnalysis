from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

""" AssociatedOutputWidgetNodeMixin.py """


class AssociatedOutputWidgetNodeMixin:
    """Implementor should Node subclass that displays an output view widget
        Provides self.view and automatically handles removing the associated view when the node is removed.
    """
    view: QtGui.QWidget
    
    @property
    def view(self):
        """The view property."""
        return self._view
    @view.setter
    def view(self, value):
        self._view = value
        
    @property
    def on_remove_function(self):
        """The on_remove_function property."""
        return self._on_remove_function
    @on_remove_function.setter
    def on_remove_function(self, value):
        self._on_remove_function = value


    def __init__(self, name):
        self.view = None
        self.on_remove_function = None
        ## Initialize node with only a single input terminal
        Node.__init__(self, name, terminals={'data': {'io':'in'}})
        
    def setView(self, view, on_remove_function=None):  ## setView must be called by the program
        self.view = view
        self.on_remove_function = on_remove_function
        # removes the added widget from the interface when this node is closed.
        self.sigClosed.connect(self.on_remove_view) # sigClosed is called when the Node parent class calls its self.close() function, which occurs when it's being removed from the flowchart
        
    
    def on_remove_view(self, event):
        """ Called when the view is to be removed"""
        # print("AssociatedOutputWidgetNodeMixin.on_remove_view()")
        if self.view is not None:
            if self.on_remove_function is not None:
                self.on_remove_function(self.view) # call on_remove_function with self to remove self from the layout
                
            self.view.deleteLater() # How to dynamically remove the widget
    
    # def process(self, data, display=True):
    #     ## if process is called with display=False, then the flowchart is being operated
    #     ## in batch processing mode, so we should skip displaying to improve performance.
        
    #     if display and self.view is not None:
    #         ## the 'data' argument is the value given to the 'data' terminal
    #         if data is None:
    #             self.view.setImage(np.zeros((1,1))) # give a blank array to clear the view
    #         else:
    #             self.view.setImage(data)

    # def close(self):
    #     """Cleans up after the node--removes terminals, graphicsItem, widget"""
    #     super(AssociatedOutputWidgetNodeMixin, self).close() # call super to clean up
    #     # self.sigClosed.emit(self)