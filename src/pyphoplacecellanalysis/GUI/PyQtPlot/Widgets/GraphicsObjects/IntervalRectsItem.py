"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui


## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect() 
## (see QGraphicsItem documentation)
class IntervalRectsItem(pg.GraphicsObject):
    """ Created to render the 2D Intervals as rectangles in a pyqtgraph 
    
        Based on pyqtgraph's CandlestickItem example
    """
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: time, open, close, min, max
        self.generatePicture()
    
    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        
        # White background bars:
        p.setPen(pg.mkPen('w'))
        p.setBrush(pg.mkBrush('r'))
        # Get height of each series by subtracking two adjacent series positions (they're all the same size, so this works
        # series_height = (self.data[1][0] - self.data[0][0]) / 3.
        # series_height = float(self.data[1][0] - self.data[0][0])
        series_height = 1.0
        for (series_offset, start_t, duration_t) in self.data:
            # p.setBrush(pg.mkBrush('r')) # TODO: enable filling of the rectangles by a passed color:
            # QRectF: (left, top, width, height)
            p.drawRect(QtCore.QRectF(start_t, series_offset-series_height, duration_t, series_height))
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())


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
    # data = [  ## fields are (time, open, close, min, max).
    # 	(1., 10, 13, 5, 15),
    # 	(2., 13, 17, 9, 20),
    # 	(3., 17, 14, 11, 23),
    # 	(4., 14, 15, 5, 19),
    # 	(5., 15, 9, 8, 22),
    # 	(6., 9, 15, 8, 16),
    # ]
 
    # (start_t, duration_t, start_alt_axis, alt_axis_size, pen_color, brush_color)
    main()
    pg.exec()
    