"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""
import copy
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets


## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect() 
## (see QGraphicsItem documentation)
class CustomIntervalRectsItem(pg.GraphicsObject):
    """ Created to render the 2D Intervals as rectangles in a pyqtgraph, but does it in a more interactive and less graphically efficient way that presevers interactions with the child epochs
    
        Based on pyqtgraph's CandlestickItem example
       
    Rectangle Item Specification: 
        Renders rectangles, with each specified by a tuple of the form:
            (start_t, series_vertical_offset, duration_t, series_height, pen, brush)

        Note that this is analagous to the position arguments of `QRectF`:
            (left, top, width, height) and (pen, brush)
            
            
    TODO: BUG: Right click currently invokes the custom example context menu that allows you to select between blue/green etc. This is triggered even when you right click on an area that's between the actual interval rect items (when you click in the blank-space between rects).
        Want this to only be triggered when on an interval. And pass through to its parent otherwise.     
        
            
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomIntervalRectsItem import CustomIntervalRectsItem, main
        active_interval_rects_item = CustomIntervalRectsItem(data)
        
        ## Add the active_interval_rects_item to the main_plot_widget: 
        main_plot_widget = spike_raster_window.spike_raster_plt_2d.plots.main_plot_widget # PlotItem
        main_plot_widget.addItem(active_interval_rects_item)

        ## Remove the active_interval_rects_item:
        main_plot_widget.removeItem(active_interval_rects_item)

    """
    # pressed = False
    clickable = True
    hoverEnter = QtCore.pyqtSignal()
    hoverExit = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal(int)
    

    def __init__(self, data):
        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        # note that the use of super() is often avoided because Qt does not allow to inherit from multiple QObject subclasses.
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush
        self.setAcceptHoverEvents(True)
        self.buttons = {}

        # Build the UI:
        self.build_all_interval_rectangle_children(self.data)

    def paint(self, p, *args):
        # required for QGraphicsObject subclasses
        pass # NOTE: no longer does anything here, as it owns children that are responsible for rendering

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on or else we will get artifacts and possibly crashing.
        return self.childrenBoundingRect() # NOTE: now the bounding rect is determined entirely by its children


    # ==================================================================================================================== #
    # Primary Data Function                                                                                                #
    # ==================================================================================================================== #
    def _perform_add_interval_rect_child(self, id, start_t, series_vertical_offset, duration_t, series_height, pen=None, brush=None):
        """ builds a new single interval item (rectangle) and attaches it as a child

            Doesn't return anything, as the child is stored in its internal buttons array

        """
        # if a button already exists with the same id, remove it
        if id in self.buttons:
            oldItem = self.buttons.pop(id)
            if self.scene():
                self.scene().removeItem(oldItem)
            # oldItem.setParent(None) # AttributeError: 'QGraphicsPathItem' object has no attribute 'setParent'
            oldItem.setParentItem(None)

        # compute the extents of the item
        itemExtentRect = QtCore.QRectF(start_t, series_vertical_offset, duration_t, series_height) # QRectF: (left, top, width, height)

        # create the circle section path
        curr_path = QtGui.QPainterPath()
        curr_path.addRect(itemExtentRect)
        # close the path back to the starting position; theoretically unnecessary,
        # but better safe than sorry
        curr_path.closeSubpath()

        # create a child item for the item
        item = QtWidgets.QGraphicsPathItem(curr_path, self)
        item.setPen(pen if pen else (QtGui.QPen(QtCore.Qt.transparent)))
        item.setBrush(brush if brush else QtGui.QColor(180, 140, 70))   # filling of the rectangles by a passed color
        self.buttons[id] = item


    def build_all_interval_rectangle_children(self, buttonData):
        """ the primary function called to set the data. Called at initialization, but even if called later should successfully rebuild the items and reuse them as it can

        Usage:
            self.build_all_interval_rectangle_children(self.data)
        """
        for idx, (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in enumerate(buttonData):
            ## Calls self.addIntervalRectangleChild to actually make or update the rectangle item:
            self._perform_add_interval_rect_child(idx, start_t, series_vertical_offset, duration_t, series_height, pen=pen, brush=brush)


    # Copy Constructors: _________________________________________________________________________________________________ #
    def __copy__(self):
        independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
        return CustomIntervalRectsItem(independent_data_copy)
    
    def __deepcopy__(self, memo):
        independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
        return CustomIntervalRectsItem(independent_data_copy)
        # return CustomIntervalRectsItem(copy.deepcopy(self.data, memo))


    # # ==================================================================================================================== #
    # # Events Copied from https://github.com/CommanderPho/pyqt-xcode/blob/master/menurect.py                                #
    # # ==================================================================================================================== #

    # def hoverEnterEvent(self, event):
    #     if self.clickable:
    #         self.hoverEnter.emit()


    # def hoverLeaveEvent(self, event):
    #     if self.clickable:
    #         self.hoverExit.emit()


    # def mousePressEvent(self, event):
    #     if self.clickable:
    #         pressed = True


    # def mouseReleaseEvent(self, event):
    #     if self.clickable:
    #         pressed = False
    #         self.clicked.emit()



    # ==================================================================================================================== #
    # RadialMenu-copied interactivity functions                                                                            #
    # ==================================================================================================================== #
    def itemAtPos(self, pos):
        for button in self.buttons.values():
            if button.shape().contains(pos):
                return button

    def checkHover(self, pos):
        hoverButton = self.itemAtPos(pos)
        for button in self.buttons.values():
            # set a visible border only for the hovered item
            button.setPen(QtCore.Qt.red if button == hoverButton else QtCore.Qt.transparent)

    def hoverEnterEvent(self, event):
        self.checkHover(event.pos())

    def hoverMoveEvent(self, event):
        self.checkHover(event.pos())

    def hoverLeaveEvent(self, event):
        for button in self.buttons.values():
            button.setPen(QtCore.Qt.transparent) # restore the non-hovered border

    def mousePressEvent(self, ev):
        clickButton = self.itemAtPos(ev.pos())
        if clickButton:
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                ## On left click, do the normal click-reponse:
                # find the button that was clicked so we can emit a clicked signal with its id
                for id, btn in self.buttons.items():
                    if btn == clickButton:
                        self.clicked.emit(id)

            elif ev.button() == QtCore.Qt.MouseButton.RightButton:
                ## On right-click, raise the context menu:

                # if self.mouseShape().contains(ev.pos()):
                #     ev.accept()
                #     self.sigClicked.emit(self, ev)

                # TODO: like the LeftButton case, we want the context menu to be w.r.t. the clicked item, so we'll need to find and store the clicked button and save it for when the context menu returns                
                if self.raiseContextMenu(ev):
                    ev.accept() # note that I think this means it won't pass the right click along to its parent view, might messup widget-wide menus




    # # ==================================================================================================================== #
    # # Context Menu and Interaction Handling                                                                                #
    # # ==================================================================================================================== #
    # def mouseShape(self):
    #     """
    #     Return a QPainterPath representing the clickable shape of the curve

    #     """
    #     if self._mouseShape is None:
    #         view = self.getViewBox()
    #         if view is None:
    #             return QtGui.QPainterPath()
    #         stroker = QtGui.QPainterPathStroker()
    #         path = self.getPath()
    #         path = self.mapToItem(view, path)
    #         stroker.setWidth(self.opts['mouseWidth'])
    #         mousePath = stroker.createStroke(path)
    #         self._mouseShape = self.mapFromItem(view, mousePath)
    #     return self._mouseShape
    
    

    # # On right-click, raise the context menu
    # def mouseClickEvent(self, ev):
    #     print(f'CustomIntervalRectsItem.mouseClickEvent(ev: {ev})')
    #     if ev.button() == QtCore.Qt.MouseButton.RightButton:
    #         # if self.mouseShape().contains(ev.pos()):
    #         #     ev.accept()
    #         #     self.sigClicked.emit(self, ev)
                
            
    #         if self.raiseContextMenu(ev):
    #             ev.accept() # note that I think this means it won't pass the right click along to its parent view, might messup widget-wide menus

    def raiseContextMenu(self, ev):
        """ works to spawn the context menu in the appropriate location """
        print(f'CustomIntervalRectsItem.raiseContextMenu(ev: {ev})')
        menu = self.getContextMenus()
        
        # Let the scene add on to the end of our context menu
        # (this is optional)
        # menu = self.scene().addParentContextMenus(self, menu, ev)
        
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))
        return True

    # This method will be called when this item's _children_ want to raise
    # a context menu that includes their parents' menus.
    def getContextMenus(self, event=None):
        """ builds the context menus as needed """
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            # self.menu.setTitle(self.name+ " options..")
            self.menu.setTitle("IntervalRectItem options..")
            
            green = QtGui.QAction("Turn green", self.menu)
            green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green
            
            blue = QtGui.QAction("Turn blue", self.menu)
            blue.triggered.connect(self.setBlue)
            self.menu.addAction(blue)
            self.menu.green = blue
            
            alpha = QtWidgets.QWidgetAction(self.menu)
            alphaSlider = QtWidgets.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider
        return self.menu

    # Define context menu callbacks
    def setGreen(self):
        # self.pen = pg.mkPen('g')
        print(f'.setGreen()...')
        for i, a_tuple in enumerate(self.data):
            # a_tuple : (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
            # list(a_tuple)
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = a_tuple
            override_pen = pg.mkPen('g')
            override_brush = pg.mkBrush('g')
            self.data[i] = (start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush)
        
        # Need to rebuild the children:
        self.build_all_interval_rectangle_children(self.data)
        # inform Qt that this item must be redrawn.
        self.update()

    def setBlue(self):
        # self.pen = pg.mkPen('b')
        # override_pen = pg.mkPen('b')
        print(f'.setBlue()...')
        for i, a_tuple in enumerate(self.data):
            # a_tuple : (start_t, series_vertical_offset, duration_t, series_height, pen, brush)
            # list(a_tuple)
            start_t, series_vertical_offset, duration_t, series_height, pen, brush = a_tuple
            override_pen = pg.mkPen('b')
            override_brush = pg.mkBrush('b')
            self.data[i] = (start_t, series_vertical_offset, duration_t, series_height, override_pen, override_brush)
            
        # Need to rebuild the children:
        self.build_all_interval_rectangle_children(self.data)
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a/255.)
        
        



class RectangleRenderTupleHelpers:
    """ class for use in copying, serializing, etc the list of tuples used by CustomIntervalRectsItem """
    @staticmethod
    def QPen_to_dict(a_pen):
        return {'color': pg.colorStr(a_pen.color()),'width':a_pen.widthF()}

    @staticmethod
    def QBrush_to_dict(a_brush):
        return {'color': pg.colorStr(a_brush.color())} # ,'gradient':a_brush.gradient()

    
    @classmethod
    def get_serialized_data(cls, tuples_data):
        """ converts the list of (float, float, float, float, QPen, QBrush) tuples into a list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) for serialization. """            
        return [(start_t, series_vertical_offset, duration_t, series_height, cls.QPen_to_dict(pen), cls.QBrush_to_dict(brush)) for (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in tuples_data]

    
    @staticmethod
    def get_deserialized_data(seralized_tuples_data):
        """ converts the list of (float, float, float, float, pen_color_hex:str, brush_color_hex:str) tuples back to the original (float, float, float, float, QPen, QBrush) list
        
        Inverse operation of .get_serialized_data(...).
        
        Usage:
            seralized_tuples_data = RectangleRenderTupleHelpers.get_serialized_data(tuples_data)
            tuples_data = RectangleRenderTupleHelpers.get_deserialized_data(seralized_tuples_data)
        """ 
        return [(start_t, series_vertical_offset, duration_t, series_height, pg.mkPen(pen_color_hex), pg.mkBrush(**brush_color_hex)) for (start_t, series_vertical_offset, duration_t, series_height, pen_color_hex, brush_color_hex) in seralized_tuples_data]

    @classmethod
    def copy_data(cls, tuples_data):
        seralized_tuples_data = cls.get_serialized_data(tuples_data).copy()
        return cls.get_deserialized_data(seralized_tuples_data)
            


# ==================================================================================================================== #
# MAIN TESTING                                                                                                         #
# ==================================================================================================================== #
def main():
    # data = [  ## fields are (series_offset, start_t, duration_t).
    #     (1., 10, 13),
    #     (2., 13, 17, 9, 20, 'w'),
    #     (3., 17, 14, 11, 23, 'w'),
    #     (4., 14, 15, 5, 19, 'w'),
    #     (5., 15, 9, 8, 22, 'w'),
    #     (6., 9, 15, 8, 16, 'w'),
    # ]
    
    
    # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
    #     (1., 10, 13),
    #     (2., 13, 17, 9, 20, 'w'),
    #     (3., 17, 14, 11, 23, 'w'),
    #     (4., 14, 15, 5, 19, 'w'),
    #     (5., 15, 9, 8, 22, 'w'),
    #     (6., 9, 15, 8, 16, 'w'),
    # ]
    
    
    series_start_offsets = [1, 5, 7]
    
    # Have series_offsets which are centers and series_start_offsets which are bottom edges:
    curr_border_color = pg.mkColor('r')
    curr_border_color.setAlphaF(0.8)

    curr_fill_color = pg.mkColor('w')
    curr_fill_color.setAlphaF(0.2)

    # build pen/brush from color
    curr_series_pen = pg.mkPen(curr_border_color)
    curr_series_brush = pg.mkBrush(curr_fill_color)
    # data = [  ## fields are (start_t, series_vertical_offset, duration_t, series_height, pen, brush).
    #     (40.0, 0.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
    #     (41.0, 1.0, 2.0, 1.0, curr_series_pen, curr_series_brush),
    #     (44.0, series_start_offsets[0], 4.0, 1.0, curr_series_pen, curr_series_brush),
    #     (45.0, series_start_offsets[-1], 4.0, 1.0, curr_series_pen, curr_series_brush),
    # ]
    data = []
    step_x_offset = 0.5
    for i in np.arange(len(series_start_offsets)):
        curr_x_pos = (40.0+(step_x_offset*float(i)))
        data.append((curr_x_pos, series_start_offsets[i], 0.5, 1.0, curr_series_pen, curr_series_brush))
        
    
    item = CustomIntervalRectsItem(data)

    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: customGraphicsItem')
    
if __name__ == '__main__':
    
    # (start_t, duration_t, start_alt_axis, alt_axis_size, pen_color, brush_color)
    main()
    pg.exec()
    