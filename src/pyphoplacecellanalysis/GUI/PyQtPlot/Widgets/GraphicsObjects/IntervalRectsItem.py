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
class IntervalRectsItem(pg.GraphicsObject):
    """ Created to render the 2D Intervals as rectangles in a pyqtgraph 
    
        Based on pyqtgraph's CandlestickItem example
       
    Rectangle Item Specification: 
        Renders rectangles, with each specified by a tuple of the form:
            (start_t, series_vertical_offset, duration_t, series_height, pen, brush)

        Note that this is analagous to the position arguments of `QRectF`:
            (left, top, width, height) and (pen, brush)
            
            
    TODO: BUG: Right click currently invokes the custom example context menu that allows you to select between blue/green etc. This is triggered even when you right click on an area that's between the actual interval rect items (when you click in the blank-space between rects).
        Want this to only be triggered when on an interval. And pass through to its parent otherwise.     
        
            
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, main
        active_interval_rects_item = IntervalRectsItem(data)
        
        ## Add the active_interval_rects_item to the main_plot_widget: 
        main_plot_widget = spike_raster_window.spike_raster_plt_2d.plots.main_plot_widget # PlotItem
        main_plot_widget.addItem(active_interval_rects_item)

        ## Remove the active_interval_rects_item:
        main_plot_widget.removeItem(active_interval_rects_item)

    """
    pressed = False
    clickable = True
    hoverEnter = QtCore.pyqtSignal()
    hoverExit = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal()
    


    def __init__(self, data):
        # menu creation is deferred because it is expensive and often
        # the user will never see the menu anyway.
        self.menu = None
        # note that the use of super() is often avoided because Qt does not 
        # allow to inherit from multiple QObject subclasses.
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush
        self.generatePicture()
        self.setAcceptHoverEvents(True)
    
    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        
        # White background bars:
        p.setPen(pg.mkPen('w'))
        p.setBrush(pg.mkBrush('r'))
        
        # for (series_offset, start_t, duration_t) in self.data:
            # # QRectF: (left, top, width, height)
            # p.drawRect(QtCore.QRectF(start_t, series_offset-series_height, duration_t, series_height))
            
        for (start_t, series_vertical_offset, duration_t, series_height, pen, brush) in self.data:
            p.setPen(pen)
            p.setBrush(brush) # filling of the rectangles by a passed color:
            # p.drawRect(QtCore.QRectF(start_t, series_vertical_offset-series_height, duration_t, series_height)) # QRectF: (left, top, width, height)
            p.drawRect(QtCore.QRectF(start_t, series_vertical_offset, duration_t, series_height)) # QRectF: (left, top, width, height)

        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())

    ## Copy Constructors:
    def __copy__(self):
        independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
        return IntervalRectsItem(independent_data_copy)
    
    def __deepcopy__(self, memo):
        independent_data_copy = RectangleRenderTupleHelpers.copy_data(self.data)
        return IntervalRectsItem(independent_data_copy)
        # return IntervalRectsItem(copy.deepcopy(self.data, memo))


    # ==================================================================================================================== #
    # Events Copied from https://github.com/CommanderPho/pyqt-xcode/blob/master/menurect.py                                #
    # ==================================================================================================================== #

    def hoverEnterEvent(self, event):
        if self.clickable:
            self.hoverEnter.emit()


    def hoverLeaveEvent(self, event):
        if self.clickable:
            self.hoverExit.emit()


    def mousePressEvent(self, event):
        if self.clickable:
            pressed = True


    def mouseReleaseEvent(self, event):
        if self.clickable:
            pressed = False
            self.clicked.emit()


    # ==================================================================================================================== #
    # Context Menu and Interaction Handling                                                                                #
    # ==================================================================================================================== #
    def mouseShape(self):
        """
        Return a QPainterPath representing the clickable shape of the curve

        """
        if self._mouseShape is None:
            view = self.getViewBox()
            if view is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            path = self.getPath()
            path = self.mapToItem(view, path)
            stroker.setWidth(self.opts['mouseWidth'])
            mousePath = stroker.createStroke(path)
            self._mouseShape = self.mapFromItem(view, mousePath)
        return self._mouseShape
    
    


    # On right-click, raise the context menu
    def mouseClickEvent(self, ev):
        print(f'IntervalRectsItem.mouseClickEvent(ev: {ev})')
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            # if self.mouseShape().contains(ev.pos()):
            #     ev.accept()
            #     self.sigClicked.emit(self, ev)
                
            
            if self.raiseContextMenu(ev):
                ev.accept() # note that I think this means it won't pass the right click along to its parent view, might messup widget-wide menus

    def raiseContextMenu(self, ev):
        """ works to spawn the context menu in the appropriate location """
        print(f'IntervalRectsItem.raiseContextMenu(ev: {ev})')
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
        
        # Need to regenerate picture
        self.generatePicture()
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
            
        # Need to regenerate picture
        self.generatePicture()
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a/255.)
        
        



class RectangleRenderTupleHelpers:
    """ class for use in copying, serializing, etc the list of tuples used by IntervalRectsItem """
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
        
    
    item = IntervalRectsItem(data)

    item.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: IntervalRectsItem')
    
if __name__ == '__main__':
    
    # (start_t, duration_t, start_alt_axis, alt_axis_size, pen_color, brush_color)
    main()
    pg.exec()
    