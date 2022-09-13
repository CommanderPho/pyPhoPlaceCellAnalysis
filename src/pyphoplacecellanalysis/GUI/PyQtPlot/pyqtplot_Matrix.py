import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, mkQApp, QtGui


""" https://www.geeksforgeeks.org/pyqtgraph-getting-histogram-object-for-image-view/?ref=rp


"""

class MatrixRenderingWindow(QtWidgets.QMainWindow):
    """ example application main window 
    
    Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.
    
    """
    def __init__(self, *args, matrix=None, columns=None, **kwargs):
        super(MatrixRenderingWindow, self).__init__(*args, **kwargs)
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('pyqtgraph example: Correlation matrix display')
        self.resize(600,500)
        


        # default_matrix = np.array([
        #     [ 1.        ,  0.5184571 , -0.70188642],
        #     [ 0.5184571 ,  1.        , -0.86094096],
        #     [-0.70188642, -0.86094096,  1.        ]
        # ])
  
        # matrix = kwargs.get('matrix', default_matrix)
        # columns = kwargs.get('columns', ["A", "B", "C"]) 


        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major
        
        correlogram = pg.ImageItem()
        # create transform to center the corner element on the origin, for any assigned image:
        tr = QtGui.QTransform().translate(-0.5, -0.5) 
        correlogram.setTransform(tr)
        correlogram.setImage(matrix)

        plotItem = gr_wid.addPlot()      # add PlotItem to the main GraphicsLayoutWidget
        plotItem.invertY(True)           # orient y axis to run top-to-bottom
        plotItem.setDefaultPadding(0.0)  # plot without padding data range
        plotItem.addItem(correlogram)    # display correlogram
        
        # show full frame, label tick marks at top and left sides, with some extra space for labels:
        plotItem.showAxes( True, showValues=(True, True, False, False), size=20 )

        # define major tick marks and labels:
        if columns is not None:
            ticks = [(idx, label) for idx, label in enumerate(columns)]
            for side in ('left','top','right','bottom'):
                plotItem.getAxis(side).setTicks( (ticks, []) ) # add list of major ticks; no minor ticks
            plotItem.getAxis('bottom').setHeight(10) # include some additional space at bottom of figure

        colorMap = pg.colormap.get("CET-D1")     # choose perceptually uniform, diverging color map
        # generate an adjustabled color bar, initially spanning -1 to 1:
        bar = pg.ColorBarItem( values=(-1,1), cmap=colorMap) 
        # link color bar and color map to correlogram, and show it in plotItem:
        bar.setImageItem(correlogram, insert_in=plotItem)    

        self.show()
        
        


## Start Qt event loop
if __name__ == '__main__':
    pg.mkQApp("Correlation matrix display")
    main_window = MatrixRenderingWindow()
    main_window.show()
    # win, app = MatrixRenderingWindow()
    pg.exec()


