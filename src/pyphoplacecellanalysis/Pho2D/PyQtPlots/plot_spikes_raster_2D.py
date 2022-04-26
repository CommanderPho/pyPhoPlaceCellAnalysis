import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets


## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect() 
## (see QGraphicsItem documentation)




  
class SpikesRasterItem(pg.GraphicsObject):
    def __init__(self, config, data = None):
        pg.GraphicsObject.__init__(self)
        self._config = config
        self._data = None
        if data is None:
            self.setData([[] for a_config in self._config])
        else:
            self.setData(data)
    
    def setData(self, data):
        ## data must have fields: time, open, close, min, max
        self._data = data
        assert len(self._data) == len(self._config), f"len(self._data) must equal len(self._config) but len(self._data): {len(self._data)} and len(self._config): {len(self._config)}"
        self.generatePicture()
        
    
    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        # p.setPen(pg.mkPen('w'))
        for i, cell_id, curr_pen, lower_y, upper_y in self._config:
            p.setPen(curr_pen)            
            for t in self._data[i]:
                p.drawLine(QtCore.QPointF(t, lower_y), QtCore.QPointF(t, upper_y))

        # w = (self.data[1][0] - self.data[0][0]) / 3.
        # for (t, open, close, min, max) in self.data:
        #     p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
        #     if open > close:
        #         p.setBrush(pg.mkBrush('r'))
        #     else:
        #         p.setBrush(pg.mkBrush('g'))
        #     p.drawRect(QtCore.QRectF(t-w, open, w*2, close-open))
        p.end()
    
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
    
    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())



  

# class SpikesRasterItem(pg.GraphicsObject):
#     def __init__(self, data):
#         pg.GraphicsObject.__init__(self)
#         self._data = []
#         self.setData(data)
    
#     def setData(self, data):
# 		## data must have fields: time, open, close, min, max
#         self._data = data
#         self.generatePicture()
        
    
#     def generatePicture(self):
#         ## pre-computing a QPicture object allows paint() to run much more quickly, 
#         ## rather than re-drawing the shapes every time.
#         self.picture = QtGui.QPicture()
#         p = QtGui.QPainter(self.picture)
#         p.setPen(pg.mkPen('w'))
        
        
        
        
#         for (t, v) in self._data:
#             if v != 0:
#                 p.drawLine(QtCore.QPointF(t, 0), QtCore.QPointF(t, v))
                
                
        
#         w = (self.data[1][0] - self.data[0][0]) / 3.
#         for (t, open, close, min, max) in self.data:
#             p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
#             if open > close:
#                 p.setBrush(pg.mkBrush('r'))
#             else:
#                 p.setBrush(pg.mkBrush('g'))
#             p.drawRect(QtCore.QRectF(t-w, open, w*2, close-open))
#         p.end()
    
#     def paint(self, p, *args):
#         p.drawPicture(0, 0, self.picture)
    
#     def boundingRect(self):
#         ## boundingRect _must_ indicate the entire area that will be drawn on
#         ## or else we will get artifacts and possibly crashing.
#         ## (in this case, QPicture does all the work of computing the bouning rect for us)
#         return QtCore.QRectF(self.picture.boundingRect())
