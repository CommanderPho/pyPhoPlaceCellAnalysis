from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.LinearRegionItem import LinearRegionItem
# from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomInfiniteLine import CustomInfiniteLine



class CustomLinearRegionItem(LinearRegionItem):
    
    def __init__(self, values=(0, 1), orientation='vertical', brush=None, pen=None, hoverBrush=None, hoverPen=None, movable=True, bounds=None, span=(0, 1), swapMode='sort', clipItem=None):
        """Create a new LinearRegionItem.
        
        ==============  =====================================================================
        **Arguments:**
        values          A list of the positions of the lines in the region. These are not
                        limits; limits can be set by specifying bounds.
        orientation     Options are 'vertical' or 'horizontal'
                        The default is 'vertical', indicating that the region is bounded
                        by vertical lines.
        brush           Defines the brush that fills the region. Can be any arguments that
                        are valid for :func:`mkBrush <pyqtgraph.mkBrush>`. Default is
                        transparent blue.
        pen             The pen to use when drawing the lines that bound the region.
        hoverBrush      The brush to use when the mouse is hovering over the region.
        hoverPen        The pen to use when the mouse is hovering over the region.
        movable         If True, the region and individual lines are movable by the user; if
                        False, they are static.
        bounds          Optional [min, max] bounding values for the region
        span            Optional [min, max] giving the range over the view to draw
                        the region. For example, with a vertical line, use
                        ``span=(0.5, 1)`` to draw only on the top half of the
                        view.
        swapMode        Sets the behavior of the region when the lines are moved such that
                        their order reverses:

                          * "block" means the user cannot drag one line past the other
                          * "push" causes both lines to be moved if one would cross the other
                          * "sort" means that lines may trade places, but the output of
                            getRegion always gives the line positions in ascending order.
                          * None means that no attempt is made to handle swapped line
                            positions.

                        The default is "sort".
        clipItem        An item whose bounds will be used to limit the region bounds.
                        This is useful when a LinearRegionItem is added on top of an
                        :class:`~pyqtgraph.ImageItem` or
                        :class:`~pyqtgraph.PlotDataItem` and the visual region should
                        not extend beyond its range. This overrides ``bounds``.
        ==============  =====================================================================
        """
        
        # Call parent __init__ function:
        LinearRegionItem.__init__(self, values=values, orientation=orientation, brush=brush, pen=pen, hoverBrush=hoverBrush, hoverPen=hoverPen, movable=movable, bounds=bounds, span=span, swapMode=swapMode, clipItem=clipItem)
        
        ## Parent Implementation:
        # GraphicsObject.__init__(self)
        # self.orientation = orientation
        # self.blockLineSignal = False
        # self.moving = False
        # self.mouseHovering = False
        # self.span = span
        # self.swapMode = swapMode
        # self.clipItem = clipItem

        # self._boundingRectCache = None
        # self._clipItemBoundsCache = None
        
        # note LinearRegionItem.Horizontal and LinearRegionItem.Vertical
        # are kept for backward compatibility.
        lineKwds = dict(
            movable=movable,
            bounds=bounds,
            span=span,
            pen=pen,
            hoverPen=hoverPen,
        )
        
        ## Add the custom mouse event criteria as arguments to the CustomInfiniteLine's
        lineKwds['custom_mouse_drag_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton) # Original/Default Condition
        # lineKwds['custom_mouse_drag_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) or (an_evt.button() == QtCore.Qt.MouseButton.MiddleButton)
        
        lineKwds['custom_mouse_hover_criteria_fn'] = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)) # Original/Default Condition
        # lineKwds['custom_mouse_hover_criteria_fn'] = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton) and ((an_evt.modifiers() == QtCore.Qt.ControlModifier) or (an_evt.modifiers() == QtCore.Qt.AltModifier) or (an_evt.modifiers() == QtCore.Qt.ShiftModifier)))
        lineKwds['custom_mouse_click_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) # Original/Default Condition
        
        # custom_mouse_drag_criteria_fn=None, custom_mouse_hover_criteria_fn=None, custom_mouse_click_criteria_fn=None
        self._setup_rebuild_with_custom_lines(values, lineKwds)

    
    
    def _setup_rebuild_with_custom_lines(self, values, lineKwds):
        """ rebuilds using CustomInfiniteLine items for self.lines instead of the simple InfiniteLine used in the base class"""
        ## Remove old lines:
        for l in self.lines:
            l.setParentItem(None)
            l.sigPositionChanged.disconnect()
            l.sigPositionChangeFinished.disconnect()
            # l.sigClicked.disconnect()
        # self.lines[0].sigPositionChanged.connect(self._line0Moved)
        # self.lines[1].sigPositionChanged.connect(self._line1Moved)
        self.lines = [] # remove all
        

        if self.orientation in ('horizontal', LinearRegionItem.Horizontal):
            self.lines = [
                # rotate lines to 180 to preserve expected line orientation 
                # with respect to region. This ensures that placing a '<|' 
                # marker on lines[0] causes it to point left in vertical mode
                # and down in horizontal mode. 
                CustomInfiniteLine(QtCore.QPointF(0, values[0]), angle=0, **lineKwds), 
                CustomInfiniteLine(QtCore.QPointF(0, values[1]), angle=0, **lineKwds)]
            tr = QtGui.QTransform.fromScale(1, -1)
            self.lines[0].setTransform(tr, True)
            self.lines[1].setTransform(tr, True)
        elif self.orientation in ('vertical', LinearRegionItem.Vertical):
            self.lines = [
                CustomInfiniteLine(QtCore.QPointF(values[0], 0), angle=90, **lineKwds), 
                CustomInfiniteLine(QtCore.QPointF(values[1], 0), angle=90, **lineKwds)]
        else:
            raise Exception("Orientation must be 'vertical' or 'horizontal'.")
        
        for l in self.lines:
            l.setParentItem(self)
            l.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.lines[0].sigPositionChanged.connect(self._line0Moved)
        self.lines[1].sigPositionChanged.connect(self._line1Moved)
        # setMovable when done, which updates the lines
        self.setMovable(lineKwds.get('movable', True))

        
    def mouseDragEvent(self, ev):
        if not self.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
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
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        if self.movable and (not ev.isExit()) and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
            