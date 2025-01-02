from typing import Callable
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.DraggableGraphicsWidgetMixin import MouseInteractionCriteria, DraggableGraphicsWidgetMixin


class CustomGraphicsLayoutWidget(DraggableGraphicsWidgetMixin, pg.GraphicsLayoutWidget):
    """ can forward events to a child 
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomGraphicsLayoutWidget
    
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_event_handler_child = None  # The child item to forward events to

    def set_target_event_forwarding_child(self, child):
        """Set the child item to which events should be forwarded."""
        self.target_event_handler_child = child


    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #
    
    def mouseDragEvent(self, ev):
        if self.target_event_handler_child:
            # Forward the event to the child
            self.target_event_handler_child.mouseDragEvent(ev)
        else:
            # Default behavior if no target child is set
            super().mouseDragEvent(ev)     

            
    def mouseClickEvent(self, ev):
        if self.target_event_handler_child:
            # Forward the event to the child
            self.target_event_handler_child.mouseClickEvent(ev)
        else:
            # Default behavior if no target child is set
            super().mouseClickEvent(ev)     

    def hoverEvent(self, ev):
        if self.target_event_handler_child:
            # Forward the event to the child
            self.target_event_handler_child.hoverEvent(ev)
        else:
            # Default behavior if no target child is set
            super().hoverEvent(ev)                 
