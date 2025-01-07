from typing import Callable
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

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
        print(f'CustomGraphicsLayoutWidget.mouseDragEvent(ev: {ev}')
        if self.target_event_handler_child:
            # Forward the event to the child
            self.target_event_handler_child.mouseDragEvent(ev)
        else:
            # Default behavior if no target child is set
            super().mouseDragEvent(ev)     

            
    def mouseClickEvent(self, ev):
        print(f'CustomGraphicsLayoutWidget.mouseClickEvent(ev: {ev}')
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



@metadata_attributes(short_name=None, tags=['scroll', 'ui', 'viewbox', 'gui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-07 01:50', related_items=[])
class CustomViewBox(pg.ViewBox):
    """ 
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomViewBox
    """
    sigLeftDrag = QtCore.Signal(object)  # optional custom signal
    
    def __init__(self, *args, **kwds):
        kwds['enableMenu'] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        
    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        # if ev.button() == QtCore.Qt.MouseButton.RightButton:
        #     ## this mode enables right-mouse clicks to reset the plot range
        #     self.autoRange()            
        print(f'.mouseClickEvent(ev: {ev})')
        # Custom logic
        ev.accept()  # or ev.ignore() if you want default handling
        # Optionally call super() if desired:
        # super().mousePressEvent(ev)
        
    
    ## reimplement mouseDragEvent to disable continuous axis zoom
    def mouseDragEvent(self, ev, axis=None):
        print(f'.mouseDragEvent(ev: {ev}, axis={axis})')
        if (ev.button() == QtCore.Qt.MouseButton.RightButton): # (axis is not None) and
            ev.accept()
            # ev.ignore()
        elif (ev.button() == QtCore.Qt.MouseButton.LeftButton):
            # axis is not None and 
            # Emit a signal or directly update the slider here
            new_start_point = self.mapSceneToView(ev.pos()) # PyQt5.QtCore.QPointF
            new_start_t = new_start_point.x()
            print(f'new_start_t: {new_start_t}')
            self.sigLeftDrag.emit(new_start_t)
            ev.accept()
        else:
            # pg.ViewBox.mouseDragEvent(self, ev, axis=axis)            
            # Custom dragging logic
            ev.accept()
            # super().mouseDragEvent(ev, axis=axis)  # only if you want default drag/pan
        

