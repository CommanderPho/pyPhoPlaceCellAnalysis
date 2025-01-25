from typing import Callable
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.LinearRegionItem import LinearRegionItem
# from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsObject import GraphicsObject
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomInfiniteLine import CustomInfiniteLine
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.DraggableGraphicsWidgetMixin import MouseInteractionCriteria, DraggableGraphicsWidgetMixin



class CustomLinearRegionItem(DraggableGraphicsWidgetMixin, LinearRegionItem):
    """ A custom pg.LinearRegionItem` subclass that allows the user to easily translate/scroll the window without resizing it.
    
    NOTE: also uses CustomInfiniteLine subclass
    
    
    The widget is made of the middle area sandwiched between two infinite lines.
    
    The middle area's interaction config (which mouse keys can interact, etc) is defined by:
    regionAreaMouseInteractionCriteria

    The two lines is defined by:    
    endLinesMouseInteractionCriteria
    
    By default both left and middle clicks can drag the area, but **only right-clicks can resize the region** (by dragging one of the two lines).
    
    
    TODO:
        - [ ] I'd like to add a "minimum-width" property that prevents the user from resizing the window to a sliver so small that they're no longer able to grab it. Instead even if they were resizing once the window reaches its minumum width it stops resizing and starts sliding/translating instead.
            - I suppose this would alter the `swapMode` property too: maybe "push" already nearly does what I want this feature to do?
 

    
    ======================= ====================================================
    **Signals**
    sigRegionChangeFinished Emitted when the user stops dragging the ROI (or
                            one of its handles) or if the ROI is changed
                            programatically.
    sigRegionChangeStarted  Emitted when the user starts dragging the ROI (or
                            one of its handles).
    sigRegionChanged        Emitted any time the position of the ROI changes,
                            including while it is being dragged by the user.
    sigHoverEvent           Emitted when the mouse hovers over the ROI.
    sigClicked              Emitted when the user clicks on the ROI.
                            Note that clicking is disabled by default to prevent
                            stealing clicks from objects behind the ROI. To 
                            enable clicking, call 
                            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton). 
                            See QtWidgets.QGraphicsItem documentation for more 
                            details.
    sigRemoveRequested      Emitted when the user selects 'remove' from the 
                            ROI's context menu (if available).
    ======================= ====================================================
    
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChangeStarted = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    sigHoverEvent = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigRemoveRequested = QtCore.Signal(object)
    
            
    """
    # sigHoverEvent = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigRemoveRequested = QtCore.Signal(object)

    def __init__(self, values=(0, 1), orientation='vertical', brush=None, pen=None, hoverBrush=None, hoverPen=None, movable=True, bounds=None, span=(0, 1), swapMode='sort', clipItem=None,
                 regionAreaMouseInteractionCriteria: MouseInteractionCriteria=None, endLinesMouseInteractionCriteria: MouseInteractionCriteria=None, custom_bound_data=None, removable=True):
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
        self.menu = None ## to hold a reference to the context menu
        self.removable = removable
        self._custom_bound_data = custom_bound_data
        
        ## Setup the mouse action critiera for the background rectangle (excluding the two end-position lines, which are set below):
        if regionAreaMouseInteractionCriteria is None:
            # Original/Default Conditions
            regionAreaMouseInteractionCriteria = MouseInteractionCriteria(drag=lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton),
                                                                        hover=lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)),
                                                                        click=lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) ## allow right-clicking
            )
            
            # Actually override drag:
            def _override_accept_either_mouse_button_drags(an_evt):
                can_accept = an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
                can_accept = can_accept and an_evt.acceptDrags(QtCore.Qt.MouseButton.MiddleButton)
                return can_accept
            regionAreaMouseInteractionCriteria.hover = _override_accept_either_mouse_button_drags
            
            regionAreaMouseInteractionCriteria.drag = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton) or (an_evt.button() == QtCore.Qt.MouseButton.MiddleButton)
            
            
        self._custom_area_mouse_action_criteria = regionAreaMouseInteractionCriteria
        
        
        # ## Add the custom mouse event criteria as arguments to the CustomInfiniteLine's
        # lineKwds['custom_mouse_drag_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton) # Original/Default Condition
        # # lineKwds['custom_mouse_drag_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) or (an_evt.button() == QtCore.Qt.MouseButton.MiddleButton)
        
        # lineKwds['custom_mouse_hover_criteria_fn'] = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)) # Original/Default Condition
        # # lineKwds['custom_mouse_hover_criteria_fn'] = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton) and ((an_evt.modifiers() == QtCore.Qt.ControlModifier) or (an_evt.modifiers() == QtCore.Qt.AltModifier) or (an_evt.modifiers() == QtCore.Qt.ShiftModifier)))
        # lineKwds['custom_mouse_click_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) # Original/Default Condition
        
        if endLinesMouseInteractionCriteria is None:
            # Original/Default Conditions
            endLinesMouseInteractionCriteria = MouseInteractionCriteria(drag=lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.LeftButton),
                                                                        hover=lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton)),
                                                                        click=lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton)
            )
            ## Only allow right-button to adjust lines/handles directly:
            endLinesMouseInteractionCriteria.hover = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.RightButton))
            endLinesMouseInteractionCriteria.drag = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton)

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
        lineKwds = dict(movable=movable, bounds=bounds, span=span, pen=pen, hoverPen=hoverPen)
        
        ## Add the custom mouse event criteria as arguments to the CustomInfiniteLine's
        lineKwds['custom_mouse_drag_criteria_fn'] = endLinesMouseInteractionCriteria.drag
        # lineKwds['custom_mouse_drag_criteria_fn'] = lambda an_evt: (an_evt.button() == QtCore.Qt.MouseButton.RightButton) or (an_evt.button() == QtCore.Qt.MouseButton.MiddleButton)
        
        lineKwds['custom_mouse_hover_criteria_fn'] = endLinesMouseInteractionCriteria.hover
        # lineKwds['custom_mouse_hover_criteria_fn'] = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton) and ((an_evt.modifiers() == QtCore.Qt.ControlModifier) or (an_evt.modifiers() == QtCore.Qt.AltModifier) or (an_evt.modifiers() == QtCore.Qt.ShiftModifier)))
        lineKwds['custom_mouse_click_criteria_fn'] = endLinesMouseInteractionCriteria.click
        
        # custom_mouse_drag_criteria_fn=None, custom_mouse_hover_criteria_fn=None, custom_mouse_click_criteria_fn=None
        self._setup_rebuild_with_custom_lines(values, lineKwds)

    @property
    def custom_mouse_drag_criteria_fn(self):
        """The custom_mouse_drag_criteria_fn property."""
        return self._custom_area_mouse_action_criteria.drag
    @custom_mouse_drag_criteria_fn.setter
    def custom_mouse_drag_criteria_fn(self, value):
        self._custom_area_mouse_action_criteria.drag = value
    @property
    def custom_mouse_hover_criteria_fn(self):
        """The custom_mouse_hover_criteria_fn property."""
        return self._custom_area_mouse_action_criteria.hover
    @custom_mouse_hover_criteria_fn.setter
    def custom_mouse_hover_criteria_fn(self, value):
        self._custom_area_mouse_action_criteria.hover = value
    @property
    def custom_mouse_click_criteria_fn(self):
        """The custom_mouse_click_criteria_fn property."""
        return self._custom_area_mouse_action_criteria.click
    @custom_mouse_click_criteria_fn.setter
    def custom_mouse_click_criteria_fn(self, value):
        self._custom_area_mouse_action_criteria.click = value
        

    @property
    def custom_bound_data(self):
        """ custom_bound_data is an object of any type containing user data that can be used to identify the purpose of the region, such as an epoch_id: int or a more complex structure."""
        return self._custom_bound_data
    @custom_bound_data.setter
    def custom_bound_data(self, value):
        self._custom_bound_data = value
        
    

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
        # if ev.button() == QtCore.Qt.MouseButton.RightButton and self.contextMenuEnabled():
        if click_criteria_fn(ev) and self.contextMenuEnabled():
            self.raiseContextMenu(ev)
            ev.accept()
        elif self.acceptedMouseButtons() & ev.button():
            ev.accept()
            self.sigClicked.emit(self, ev)
        else:
            ev.ignore()
            

    def hoverEvent(self, ev):
        hover_criteria_fn = self.custom_mouse_hover_criteria_fn
        if hover_criteria_fn is None:
            hover_criteria_fn = lambda an_evt: (an_evt.acceptDrags(QtCore.Qt.MouseButton.LeftButton))
        if self.movable and (not ev.isExit()) and hover_criteria_fn(ev):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)
            


    # ==================================================================================================================== #
    # Context Menus                                                                                                        #
    # ==================================================================================================================== #
    def contextMenuEnabled(self):
        return self.removable


    def raiseContextMenu(self, ev):
        if not self.contextMenuEnabled():
            return
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle("Epoch")
            remAct = QtGui.QAction("Remove Epoch", self.menu)
            remAct.triggered.connect(self.removeClicked)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
        # ROI menu may be requested when showing the handle context menu, so
        # return the menu but disable it if the ROI isn't removable
        self.menu.setEnabled(self.contextMenuEnabled())
        return self.menu

    def removeClicked(self):
        ## Send remove event only after we have exited the menu event handler
        QtCore.QTimer.singleShot(0, self._emitRemoveRequest)

    def _emitRemoveRequest(self):
        self.sigRemoveRequested.emit(self)

