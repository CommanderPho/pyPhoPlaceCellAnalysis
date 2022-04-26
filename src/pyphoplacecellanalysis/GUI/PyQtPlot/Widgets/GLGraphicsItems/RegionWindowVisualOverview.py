import dataclasses
from typing import Optional
import sys

import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@dataclasses.dataclass
class AddRegionWidget(pg.GraphicsLayoutWidget):
    """ A widget that allows one to drag an adjustable and resizable 2D window around to indicate the current region of the plot you want to look at.
    
    Based off of pyqtgraph's LinearRegionItem()
    
    Attributes #
    ----------
    parent: Optional[QtWidgets.QWidget] default=None
Parent screen
    main_plotter: pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem
Main graph
    zoom_plotter: pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem
A graph that zooms the main graph in the region
    region: pyqtgraph.graphicsItems.LinearRegionItem.LinearRegionItem
        zoom_region that specifies the x-axis region of the plotter
    """
    parent: Optional[QtWidgets.QWidget] = None

    def __post_init__(self) -> None:
        """Superclass loading and plot,region added"""
        super(AddRegionWidget, self).__init__(parent=self.parent)

        self.add_plot()
        self.add_region()
        self.connect_slot()

    def add_plot(self) -> None:
        """add plot"""
        self.main_plotter = self.addPlot(row=0, col=0)
        self.main_plotter.showGrid(x=True, y=True, alpha=0.8)
        main_curve1 = self.main_plotter.plot(pen=pg.mkPen('#f00'))
        main_curve2 = self.main_plotter.plot(pen=pg.mkPen('#0f0'))
        main_curve3 = self.main_plotter.plot(pen=pg.mkPen('#00f'))
        main_curve1.setData(SAMPLE_DATA1)
        main_curve2.setData(SAMPLE_DATA2)
        main_curve3.setData(SAMPLE_DATA3)

        self.zoom_plotter = self.addPlot(row=0, col=1)
        #Adjust the y-axis according to the value
        self.zoom_plotter.setAutoVisible(y=True)
        self.zoom_plotter.showGrid(x=True, y=True, alpha=0.8)
        zoom_curve1 = self.zoom_plotter.plot(pen=pg.mkPen('#f00'))
        zoom_curve2 = self.zoom_plotter.plot(pen=pg.mkPen('#0f0'))
        zoom_curve3 = self.zoom_plotter.plot(pen=pg.mkPen('#00f'))
        zoom_curve1.setData(SAMPLE_DATA1)
        zoom_curve2.setData(SAMPLE_DATA2)
        zoom_curve3.setData(SAMPLE_DATA3)

        self.zoom_plotter.setXRange(0.0, len(SAMPLE_DATA1) / 8, padding=0)

        self.ci.layout.setColumnStretchFactor(0, 8)
        self.ci.layout.setColumnStretchFactor(1, 5)

    def add_region(self) -> None:
        """Add region"""
        self.region = pg.LinearRegionItem()
        #Region height setting. There are multiple regions&If they overlap, the one with the higher Z can be operated.(Since there is only one this time, set it to 10 appropriately)
        self.region.setZValue(10)
        self.main_plotter.addItem(self.region, ignoreBounds=True)
        self.update_region()

    def connect_slot(self) -> None:
        """slot connection"""
        self.region.sigRegionChanged.connect(self.update_zoom_plotter)
        self.zoom_plotter.sigRangeChanged.connect(self.update_region)

    @QtCore.pyqtSlot()
    def update_zoom_plotter(self) -> None:
        """self when the region moves.zoom_Change plotter area"""
        self.region.setZValue(10)
        min_x, max_x = self.region.getRegion()
        self.zoom_plotter.setXRange(min_x, max_x, padding=0)

    @QtCore.pyqtSlot()
    def update_region(self) -> None:
        """self.zoom_Change the region of the region when the plotter moves viewRange returns the display range of the graph. The type is
	        [[Xmin, Xmax], [Ymin, Ymax]]
        """
        rgn = self.zoom_plotter.viewRange()[0]
        self.region.setRegion(rgn)
