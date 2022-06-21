"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""
import copy
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui


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
            
        
            
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, main
        active_interval_rects_item = IntervalRectsItem(data)
        
        ## Add the active_interval_rects_item to the main_plot_widget: 
        main_plot_widget = spike_raster_window.spike_raster_plt_2d.plots.main_plot_widget # PlotItem
        main_plot_widget.addItem(active_interval_rects_item)

        ## Remove the active_interval_rects_item:
        main_plot_widget.removeItem(active_interval_rects_item)

    """
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: start_t, series_vertical_offset, duration_t, series_height, pen, brush
        self.generatePicture()
    
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
            
        
def main():
    data = [  ## fields are (series_offset, start_t, duration_t).
        (1., 10, 13),
        (2., 13, 17, 9, 20, 'w'),
        (3., 17, 14, 11, 23, 'w'),
        (4., 14, 15, 5, 19, 'w'),
        (5., 15, 9, 8, 22, 'w'),
        (6., 9, 15, 8, 16, 'w'),
    ]
    item = IntervalRectsItem(data)
    plt = pg.plot()
    plt.addItem(item)
    plt.setWindowTitle('pyqtgraph example: customGraphicsItem')
    
if __name__ == '__main__':
    
    # (start_t, duration_t, start_alt_axis, alt_axis_size, pen_color, brush_color)
    main()
    pg.exec()
    