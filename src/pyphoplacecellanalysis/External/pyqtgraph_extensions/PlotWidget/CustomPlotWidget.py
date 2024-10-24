from PyQt5 import QtWidgets
import pyqtgraph as pg
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

__all__ = ['CustomPlotWidget']

# @function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-21 20:56', related_items=[])
class CustomPlotWidget(pg.PlotWidget):
    """ Tries to improve click handling
    from pyphoplacecellanalysis.External.pyqtgraph_extensions.CustomPlotWidget import CustomPlotWidget
    from pyphoplacecellanalysis.External.pyqtgraph_extensions.PlotWidget.CustomPlotWidget import CustomPlotWidget

    """
    # signals wrapped from PlotItem / ViewBox
    # sigRangeChanged = pyqtSignal(object, object)
    # sigTransformChanged = pyqtSignal(object)

    ## Signals from GraphicsScene
    sigMouseHover = pyqtSignal(object, object) ## emits a list of objects hovered over
    sigMouseMoved = pyqtSignal(object, object) ## emits position of mouse on every move
    sigMouseClicked = pyqtSignal(object, object) ## emitted when mouse is clicked. Check for event.isAccepted() to see whether the event has already been acted on.

    @property
    def item_data(self) -> Dict:
        """Data to identify this item."""
        return self._item_data
    @item_data.setter
    def item_data(self, value):
        self._item_data = value

    def __init__(self, *args, item_data: Optional[Dict]=None, **kwargs):
        super(CustomPlotWidget, self).__init__(*args, **kwargs)
        if item_data is None:
            item_data = {}
        self._item_data = deepcopy(item_data)
        self.init_UI()
        

    def init_UI(self):
        """ sets up UI and connects all needed signals """
        # self.scene() is a pyqtgraph.GraphicsScene.GraphicsScene.GraphicsScene
        self.scene().sigMouseClicked.connect(self.mouse_clicked)   
        self.scene().sigMouseHover.connect(self.mouse_hover)
        self.scene().sigMouseMoved.connect(self.mouse_moved)


    ## Override mousePressEvent
    def mousePressEvent(self, ev):
        """ not sure how this version interacts with the others. """
        super().mousePressEvent(ev)
        # self.sigMouseClicked.emit(self, ev)
        self.scene().sigMouseClicked.emit(ev)
        
    def mouse_clicked(self, mouseClickEvent):
        # mouseClickEvent is a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
        # print('clicked plot 0x{:x}, event: {}'.format(id(self), mouseClickEvent))
        self.sigMouseClicked.emit(self, mouseClickEvent) # pass self
        
    def mouse_moved(self, evt):
        # mouseClickEvent is a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
        # print('clicked plot 0x{:x}, event: {}'.format(id(self), evt))
        self.sigMouseMoved.emit(self, evt)
        
    def mouse_hover(self, evt):
        # mouseClickEvent is a pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent
        # print('clicked plot 0x{:x}, event: {}'.format(id(self), evt))
        self.sigMouseHover.emit(self, evt)


