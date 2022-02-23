from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

""" AssociatedOutputWidgetNodeMixin.py """

class AssociatedAppNodeMixin:
    """Implementors own an app, meaning a singleton instance of QApplication. """
    
    @property
    def app(self):
        """The app property."""
        return self._app
    @app.setter
    def app(self, value):
        self._app = value
    
    # def __init__(self, app):
    #     super(AssociatedAppNodeMixin, self).__init__()
    #     self._app = app
        
    def setApp(self, app):
        self.app = app



class AssociatedOutputWidgetNodeMixin:
    """Implementor should be Node subclass that displays an output view widget
        Provides self.view and automatically handles removing the associated view when the node is removed.
        Provides self.setView(view, on_remove_function) to setup the view
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
    def owned_parent_container(self):
        """The owned_parent_container property."""
        return self._owned_parent_container
    @owned_parent_container.setter
    def owned_parent_container(self, value):
        self._owned_parent_container = value
        
    
    @property
    def on_remove_function(self):
        """The on_remove_function property."""
        return self._on_remove_function
    @on_remove_function.setter
    def on_remove_function(self, value):
        self._on_remove_function = value


    @property
    def on_add_function(self):
        """The on_add_function property."""
        return self._on_add_function
    @on_add_function.setter
    def on_add_function(self, value):
        self._on_add_function = value
        
    # def __init__(self, name):
    #     self.view = None
    #     self.on_remove_function = None
    #     ## Initialize node with only a single input terminal
    #     Node.__init__(self, name, terminals={'data': {'io':'in'}})
        
    def setView(self, owned_parent_container=None, view=None, on_add_function=None, on_remove_function=None):  ## setView must be called by the program
        self._owned_parent_container = owned_parent_container
        self._view = view
        self._on_add_function = on_add_function
        self._on_remove_function = on_remove_function

        self.on_create_view(None)

        # removes the added widget from the interface when this node is closed.
        self.sigClosed.connect(self.on_remove_view) # sigClosed is called when the Node parent class calls its self.close() function, which occurs when it's being removed from the flowchart
        
    
    def on_create_view(self, event):
        """ Called to create/build the view using the on_add_function """
        if self.on_add_function is not None:
            self.view, self.owned_parent_container = self.on_add_function() # call on_remove_function with self to remove self from the layout
                
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