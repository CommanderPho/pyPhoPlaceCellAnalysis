import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, mkQApp, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage


class BinnedImageRenderingWindow(QtWidgets.QMainWindow):
    """ Renders a Matrix of binned data in the window.
        Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.    
        
        History:
            Based off of pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix.MatrixRenderingWindow
    """
    
    def __init__(self, *args, matrix=None, xbins=None, ybins=None, **kwargs):
        super(BinnedImageRenderingWindow, self).__init__(*args, **kwargs)
        # green - orange - red
        Gradients['gor'] = {'ticks': [(0.0, (74, 158, 71)), (0.5, (255, 230, 0)), (1, (191, 79, 76))], 'mode': 'rgb'}
        
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('BinnedImageRenderingWindow')
        self.resize(600,500)
        plotItem = gr_wid.addPlot(title="Avg Velocity per Pos (X, Y)", row=0, col=0)      # add PlotItem to the main GraphicsLayoutWidget
        # plotItem.invertY(True)           # orient y axis to run top-to-bottom
        plotItem.setDefaultPadding(0.0)  # plot without padding data range
        
        # Full Histogram:
        lut = pg.HistogramLUTItem(orientation="horizontal")
        plotItem.setMouseEnabled(x=False, y=False)
        
        gr_wid.nextRow()
        gr_wid.addItem(lut)

        # load the gradient
        lut.gradient.loadPreset('gor')

        ## NonUniformImage:
        image = NonUniformImage(xbins, ybins, matrix)
        
        ## Normal ImageItem():
        # pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major
        # image = pg.ImageItem()
        # create transform to center the corner element on the origin, for any assigned image:
        # tr = QtGui.QTransform().translate(-0.5, -0.5) 
        # image.setTransform(tr)
        # image.setImage(matrix.T)        
        # image.setLookupTable(lut)
        image.setLookupTable(lut, autoLevel=True)
        image.setZValue(-1)
        plotItem.addItem(image)

        h = image.getHistogram()
        lut.plot.setData(*h)

        # show full frame, label tick marks at top and left sides, with some extra space for labels:
        plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        # # define major tick marks and labels:
        # if xbins is not None:
        #     xticks = [(idx, label) for idx, label in enumerate(xbins)]
        #     for side in ('top','bottom'):
        #         plotItem.getAxis(side).setTicks((xticks, [])) # add list of major ticks; no minor ticks        
        # if ybins is not None:
        #     yticks = [(idx, label) for idx, label in enumerate(ybins)]
        #     for side in ('left','right'):
        #         plotItem.getAxis(side).setTicks((yticks, [])) # add list of major ticks; no minor ticks
        # plotItem.showGrid(x = True, y = True, alpha = 0.65)
                
        # # Color Map:
        # colorMap = pg.colormap.get("viridis")     # choose perceptually uniform, diverging color map
        # # generate an adjustabled color bar, initially spanning -1 to 1:
        # bar = pg.ColorBarItem(colorMap=colorMap, label='Avg Velocity')
        # # bar = pg.ColorBarItem(values=(-1,1), cmap=colorMap) 
        # # link color bar and color map to correlogram, and show it in plotItem:
        # bar.setImageItem(image, insert_in=plotItem)    
        
        self.show()

