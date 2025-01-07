from typing import Callable
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
from dataclasses import dataclass


@dataclass
class MouseInteractionCriteria(object):
    """Docstring for MouseInteractionCriteria."""
    drag: Callable
    hover: Callable
    click: Callable
    

class DraggableGraphicsWidgetMixin:
    """ 

    Requires:
        self.custom_mouse_click_criteria_fn
        self.movable
        
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.DraggableGraphicsWidgetMixin import MouseInteractionCriteria, DraggableGraphicsWidgetMixin

    """
    
    # @property
    # def custom_mouse_drag_criteria_fn(self):
    #     """The custom_mouse_drag_criteria_fn property."""
    #     return self._custom_area_mouse_action_criteria.drag
    # @custom_mouse_drag_criteria_fn.setter
    # def custom_mouse_drag_criteria_fn(self, value):
    #     self._custom_area_mouse_action_criteria.drag = value
    # @property
    # def custom_mouse_hover_criteria_fn(self):
    #     """The custom_mouse_hover_criteria_fn property."""
    #     return self._custom_area_mouse_action_criteria.hover
    # @custom_mouse_hover_criteria_fn.setter
    # def custom_mouse_hover_criteria_fn(self, value):
    #     self._custom_area_mouse_action_criteria.hover = value
    # @property
    # def custom_mouse_click_criteria_fn(self):
    #     """The custom_mouse_click_criteria_fn property."""
    #     return self._custom_area_mouse_action_criteria.click
    # @custom_mouse_click_criteria_fn.setter
    # def custom_mouse_click_criteria_fn(self, value):
    #     self._custom_area_mouse_action_criteria.click = value
        

    def DraggableGraphicsWidgetMixin_initUI(self):
        pass
        # self._custom_bound_data = custom_bound_data
        
        # ## Setup the mouse action critiera for the background rectangle (excluding the two end-position lines, which are set below):
        # if regionAreaMouseInteractionCriteria is None:
        #     # Original/Default Conditions
        #     regionAreaMouseInteractionCriteria = MouseInteractionCriteria(drag=lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton),
        #                                                                 hover=lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)),
        #                                                                 click=lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton)
        #     )
            
        #     # Actually override drag:
        #     def _override_accept_either_mouse_button_drags(an_evt):
        #         can_accept = an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        #         can_accept = can_accept and an_evt.acceptDrags(QtCore.Qt.MouseButton.MiddleButton)
        #         return can_accept
        #     regionAreaMouseInteractionCriteria.hover = _override_accept_either_mouse_button_drags
            
        #     regionAreaMouseInteractionCriteria.drag = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton) or (an_evt.button() == QtCore.Qt.MouseButton.MiddleButton)
            
            
        # self._custom_area_mouse_action_criteria = regionAreaMouseInteractionCriteria
        


    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #
    
    def mouseDragEvent(self, ev):
        drag_criteria_fn = self.custom_mouse_drag_criteria_fn
        if drag_criteria_fn is None:
            drag_criteria_fn = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton) # 
            
        if not self.movable or not drag_criteria_fn(ev):
            return
        ev.accept()
        
        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True
            
        if not self.moving:
            return
            
        self.lines[0].blockSignals(True)  # only want to update once
        for i, l in enumerate(self.lines):
            l.setPos(self.cursorOffsets[i] + ev.pos())
        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()
        
        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)
            
    def mouseClickEvent(self, ev):
        click_criteria_fn = self.custom_mouse_click_criteria_fn
        if click_criteria_fn is None:
            click_criteria_fn = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) # Original/Default Condition
            
        if self.moving and click_criteria_fn(ev):
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        hover_criteria_fn = self.custom_mouse_hover_criteria_fn
        if hover_criteria_fn is None:
            hover_criteria_fn = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton))
        if self.movable and (not ev.isExit()) and hover_criteria_fn(ev):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
            