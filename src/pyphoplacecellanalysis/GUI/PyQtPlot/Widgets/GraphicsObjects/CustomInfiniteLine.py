from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine


class CustomInfiniteLine(InfiniteLine):
    """ A custom pg.InfiniteLine subclass with custom requirements to hold a customizable modifier key or mouse button to access hover/drag functionality.
    
    =============================== ===================================================
    **Signals:**
    sigDragged(self)
    sigPositionChangeFinished(self)
    sigPositionChanged(self)
    sigClicked(self, ev)
    =============================== ===================================================
    """
    def __init__(self, pos=None, angle=90, pen=None, movable=False, bounds=None, hoverPen=None, label=None, labelOpts=None, span=(0, 1), markers=None, name=None,
                 custom_mouse_drag_criteria_fn=None, custom_mouse_hover_criteria_fn=None, custom_mouse_click_criteria_fn=None):
        InfiniteLine.__init__(self, pos=pos, angle=angle, pen=pen, movable=movable, bounds=bounds, hoverPen=hoverPen, label=label, labelOpts=labelOpts, span=span, markers=markers, name=name)
        self._custom_mouse_drag_criteria_fn = custom_mouse_drag_criteria_fn
        self._custom_mouse_hover_criteria_fn = custom_mouse_hover_criteria_fn
        self._custom_mouse_click_criteria_fn = custom_mouse_click_criteria_fn
        
        
    @property
    def custom_mouse_drag_criteria_fn(self):
        """The custom_mouse_drag_criteria_fn property."""
        return self._custom_mouse_drag_criteria_fn
    @custom_mouse_drag_criteria_fn.setter
    def custom_mouse_drag_criteria_fn(self, value):
        self._custom_mouse_drag_criteria_fn = value
    @property
    def custom_mouse_hover_criteria_fn(self):
        """The custom_mouse_hover_criteria_fn property."""
        return self._custom_mouse_hover_criteria_fn
    @custom_mouse_hover_criteria_fn.setter
    def custom_mouse_hover_criteria_fn(self, value):
        self._custom_mouse_hover_criteria_fn = value
    @property
    def custom_mouse_click_criteria_fn(self):
        """The custom_mouse_click_criteria_fn property."""
        return self._custom_mouse_click_criteria_fn
    @custom_mouse_click_criteria_fn.setter
    def custom_mouse_click_criteria_fn(self, value):
        self._custom_mouse_click_criteria_fn = value
        
        
    def mouseDragEvent(self, ev):
        drag_criteria_fn = self.custom_mouse_drag_criteria_fn
        if drag_criteria_fn is None:
            drag_criteria_fn = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton) # Original/Default Condition
        # drag_criteria_fn = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) or (an_evt.button() == QtCore.Qt.MouseButton.MiddleButton)
        # if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
        if self.movable and drag_criteria_fn(ev):
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return

            self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.moving = False
                self.sigPositionChangeFinished.emit(self)

    def mouseClickEvent(self, ev):
        self.sigClicked.emit(self, ev)
        
        click_criteria_fn = self.custom_mouse_click_criteria_fn
        if click_criteria_fn is None:
            click_criteria_fn = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) # Original/Default Condition
        if self.moving and click_criteria_fn(ev):
        # if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.setPos(self.startPosition)
            self.moving = False
            self.sigDragged.emit(self)
            self.sigPositionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        hover_criteria_fn = self.custom_mouse_hover_criteria_fn
        if hover_criteria_fn is None:
            # hover_criteria_fn = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton) or an_evt.acceptDrags(QtCore.Qt.MouseButton.MiddleButton))
            # hover_criteria_fn = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton) and (an_evt.modifiers() == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)))
            # hover_criteria_fn = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton) and ((an_evt.modifiers() == QtCore.Qt.ControlModifier) or (an_evt.modifiers() == QtCore.Qt.AltModifier) or (an_evt.modifiers() == QtCore.Qt.ShiftModifier)))
            hover_criteria_fn = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)) # Original/Default Condition
            # ev.modifiers(): PyQt5.QtCore.Qt.KeyboardModifiers
            
        # if (not ev.isExit()) and self.movable and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
        if (not ev.isExit()) and self.movable and hover_criteria_fn(ev):
            # print(f'hoverEvent: ev.modifiers(): {ev.modifiers()}')
            # curr_modifiers = ev.modifiers() # PyQt5.QtCore.Qt.KeyboardModifiers
            # if curr_modifiers == QtCore.Qt.ControlModifier:
            #     print(f'\t ControlModifier!')
            # if curr_modifiers == QtCore.Qt.NoModifier:
            #     print(f'\t NoModifier!')
            # elif ((curr_modifiers == QtCore.Qt.ControlModifier) or (curr_modifiers == QtCore.Qt.AltModifier) or (curr_modifiers == QtCore.Qt.ShiftModifier)):
            #     print(f'\t Any Known Modifier (Ctrl or Shift or Alt)!')
            # elif curr_modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier):
            #     # The | operator is supposed to be boolean or, but it doesn't work.
            #     print(f'\t All Simultaneous Known Modifier (Ctrl AND Shift AND Alt)!')
            # elif curr_modifiers == QtCore.Qt.ControlModifier:
            #     print(f'\t ControlModifier!')
            # elif curr_modifiers == QtCore.Qt.AltModifier:
            #     print(f'\t AltModifier!')
            # elif curr_modifiers == QtCore.Qt.ShiftModifier:
            #     print(f'\t ShiftModifier!')
            # else:
            #     print(f'\t Unrecognized Modifier!')
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
    