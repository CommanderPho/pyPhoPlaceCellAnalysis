import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, mkQApp, QtGui
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage


from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


def add_bin_ticks(plot_item, xbins=None, ybins=None):
    """ adds the ticks/grid for xbins and ybins to the plot_item """
    # show full frame, label tick marks at top and left sides, with some extra space for labels:
    plot_item.showAxes(True, showValues=(True, True, False, False), size=10)
    # define major tick marks and labels:
    if xbins is not None:
        xticks = [(idx, label) for idx, label in enumerate(xbins)]
        for side in ('top','bottom'):
            plot_item.getAxis(side).setTicks((xticks, [])) # add list of major ticks; no minor ticks        
    if ybins is not None:
        yticks = [(idx, label) for idx, label in enumerate(ybins)]
        for side in ('left','right'):
            plot_item.getAxis(side).setTicks((yticks, [])) # add list of major ticks; no minor ticks
    plot_item.showGrid(x = True, y = True, alpha = 0.65)
    return plot_item


def build_binned_imageItem(plot_item, params, xbins=None, ybins=None, matrix=None, data_label='Avg Velocity'):
        local_plots_data = RenderPlotsData(name=data_label)
        local_plots = RenderPlots(name=data_label)
        
        # plotItem.invertY(True)           # orient y axis to run top-to-bottom
        # Normal ImageItem():
        local_plots.imageItem = pg.ImageItem(matrix.T)
        plot_item.addItem(local_plots.imageItem)

        # Color Map:
        if hasattr(params, 'colorMap'):
            colorMap = params.colorMap
        else:
            colorMap = pg.colormap.get("viridis")
        # generate an adjustabled color bar
        local_plots.colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=colorMap, label=data_label)
        # link color bar and color map to correlogram, and show it in plotItem:
        local_plots.colorBarItem.setImageItem(local_plots.imageItem, insert_in=plot_item)
        
        local_plots_data.matrix_min = np.nanmin(matrix)
        local_plots_data.matrix_max = np.nanmax(matrix)
        # Set the colorbar to the range:
        local_plots.colorBarItem.setLevels(low=local_plots_data.matrix_min, high=local_plots_data.matrix_max)
        
        # self.params = VisualizationParameters(name='BasicBinnedImageRenderingWindow')
        # self.plots_data = RenderPlotsData(name='BasicBinnedImageRenderingWindow')
        # self.plots = RenderPlots(name='BasicBinnedImageRenderingWindow')
        # self.ui = PhoUIContainer(name='BasicBinnedImageRenderingWindow')
        return local_plots, local_plots_data
        
        

# class BinnedImageRenderingWindow(QtWidgets.QMainWindow):
#     """ Renders a Matrix of binned data in the window.
#         NOTE: uses pg.NonUniformImage and includes an interactive histogram.
#         Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.    
        
#         History:
#             Based off of pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix.MatrixRenderingWindow
#     """
    
#     def __init__(self, matrix=None, xbins=None, ybins=None, defer_show=False, **kwargs):
#         super(BinnedImageRenderingWindow, self).__init__(**kwargs)
#         # green - orange - red
#         Gradients['gor'] = {'ticks': [(0.0, (74, 158, 71)), (0.5, (255, 230, 0)), (1, (191, 79, 76))], 'mode': 'rgb'}
        
#         gr_wid = pg.GraphicsLayoutWidget(show=True)
#         self.setCentralWidget(gr_wid)
#         self.setWindowTitle('BinnedImageRenderingWindow')
#         self.resize(600,500)
#         plotItem = gr_wid.addPlot(title="Avg Velocity per Pos (X, Y)", row=0, col=0)      # add PlotItem to the main GraphicsLayoutWidget
#         # plotItem.invertY(True)           # orient y axis to run top-to-bottom
#         plotItem.setDefaultPadding(0.0)  # plot without padding data range
#         plotItem.setMouseEnabled(x=False, y=False)
        
#         # Full Histogram:
#         lut = pg.HistogramLUTItem(orientation="horizontal")
#         gr_wid.nextRow()
#         gr_wid.addItem(lut)

#         # load the gradient
#         lut.gradient.loadPreset('gor')

#         ## NonUniformImage:
#         image = NonUniformImage(xbins, ybins, matrix)
#         image.setLookupTable(lut, autoLevel=True)
#         image.setZValue(-1)
#         plotItem.addItem(image)

#         h = image.getHistogram()
#         lut.plot.setData(*h)

#         # show full frame, label tick marks at top and left sides, with some extra space for labels:
#         plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        
#         if not defer_show:
#             self.show()



class BasicBinnedImageRenderingWindow(QtWidgets.QMainWindow):
    """ Renders a Matrix of binned data in the window.NonUniformImage and includes no histogram.
        NOTE: uses basic pg.ImageItem instead of pg.
        Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.    
        
        History:
            Based off of pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix.MatrixRenderingWindow
    """
    
    def __init__(self, matrix=None, xbins=None, ybins=None, defer_show=False, title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity', **kwargs):
        super(BasicBinnedImageRenderingWindow, self).__init__(**kwargs)
        self.params = VisualizationParameters(name='BasicBinnedImageRenderingWindow')
        self.plots_data = RenderPlotsData(name='BasicBinnedImageRenderingWindow')
        self.plots = RenderPlots(name='BasicBinnedImageRenderingWindow')
        self.ui = PhoUIContainer(name='BasicBinnedImageRenderingWindow')
        
        self.params.colorMap = pg.colormap.get("viridis")
        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major

        ## Create:        
        self.ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(self.ui.graphics_layout)
        self.setWindowTitle(title)
        self.resize(600,500)
        
        self.plots.mainPlotItem = self.ui.graphics_layout.addPlot(title=title, row=0, col=0)      # add PlotItem to the main GraphicsLayoutWidget
        # plotItem.invertY(True)           # orient y axis to run top-to-bottom
        self.plots.mainPlotItem.setDefaultPadding(0.0)  # plot without padding data range
        self.plots.mainPlotItem.setMouseEnabled(x=False, y=False)
        self.plots.mainPlotItem = add_bin_ticks(plot_item=self.plots.mainPlotItem, xbins=xbins, ybins=ybins)

        # self.plots.mainImageItem, self.params.colorMap, self.plots.colorBarItem, local_plots_data = build_binned_imageItem(self.plots.mainPlotItem, xbins=xbins, ybins=ybins, matrix=matrix, data_label=variable_label)
        
        
        
        local_plots, local_plots_data = build_binned_imageItem(self.plots.mainPlotItem, self.params, xbins=xbins, ybins=ybins, matrix=matrix, data_label=variable_label)
        self.plots_data[local_plots_data.name] = local_plots_data
        self.plots[local_plots.name] = local_plots
        
        # # Normal ImageItem():
        # self.plots.mainImageItem = pg.ImageItem(matrix.T)
        # # create transform to center the corner element on the origin, for any assigned image:
        # # tr = QtGui.QTransform().translate(-0.5, -0.5) 
        # # image.setTransform(tr)
        # # image.setImage(matrix.T)
        # self.plots.mainPlotItem.addItem(self.plots.mainImageItem)

        # # Color Map:
        # self.params.colorMap = pg.colormap.get("viridis")
        # # generate an adjustabled color bar
        # self.plots.colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=self.params.colorMap, label='Avg Velocity')
        # # bar = pg.ColorBarItem(values=(-1,1), cmap=colorMap) 
        # # link color bar and color map to correlogram, and show it in plotItem:
        # self.plots.colorBarItem.setImageItem(self.plots.mainImageItem, insert_in=self.plots.mainPlotItem)
        
        # self.plots_data.matrix = matrix.copy()
        # self.plots_data.matrix_min = np.nanmin(self.plots_data.matrix)
        # self.plots_data.matrix_max = np.nanmax(self.plots_data.matrix)
        # # Set the colorbar to the range:
        # self.plots.colorBarItem.setLevels(low=self.plots_data.matrix_min, high=self.plots_data.matrix_max)
        
        if not defer_show:
            self.show()

    def add_data(self, row=1, col=0, matrix=None, xbins=None, ybins=None, title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity'):
        newPlotItem = self.ui.graphics_layout.addPlot(title=title, row=row, col=col)      # add PlotItem to the main GraphicsLayoutWidget
        newPlotItem.setDefaultPadding(0.0)  # plot without padding data range
        newPlotItem.setMouseEnabled(x=False, y=False)
        newPlotItem = add_bin_ticks(plot_item=newPlotItem, xbins=xbins, ybins=ybins)
        
        self.plots_data[title] = matrix.copy()
        self.plots[title] = newPlotItem
        
        local_plots, local_plots_data = build_binned_imageItem(newPlotItem, self.params, xbins=xbins, ybins=ybins, matrix=matrix, data_label=variable_label)
        self.plots_data[local_plots_data.name] = local_plots_data
        self.plots[local_plots.name] = local_plots
        # newImageItem, newcolorMap, newColorBarItem, local_plots_data = build_binned_imageItem(newPlotItem, xbins=xbins, ybins=ybins, matrix=matrix, data_label=variable_label)
        # new_plots_data[local_plots_data.name] = local_plots_data