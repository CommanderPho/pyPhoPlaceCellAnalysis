import functools
from functools import partial
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import numpy as np
import pandas as pd

from pyphocorehelpers.print_helpers import DocumentationFilePrinter, print_keys_if_possible # used in `DebugPrinterStat`

from silx.gui.plot.stats.stats import StatBase
from silx.gui.utils import concurrent

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.plot import PlotWindow, ImageView
from silx.gui.plot.Profile import ProfileToolBar

from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.tools.roi import RoiModeSelectorAction
from silx.gui.plot.items.roi import RectangleROI, BandROI, LineROI
from silx.gui.plot.items import LineMixIn, SymbolMixIn, FillMixIn
from silx.gui.plot.actions import control as control_actions

from silx.gui.plot.ROIStatsWidget import ROIStatsWidget
from silx.gui.plot.StatsWidget import UpdateModeWidget
from silx.gui.plot import Plot2D

# from pyphoplacecellanalysis.GUI.Silx.silx_helpers import _RoiStatsDisplayExWindow

class AutoHideToolBar(qt.QToolBar):
    """A toolbar which hide itself if no actions are visible"""

    def actionEvent(self, event):
        if event.type() == qt.QEvent.ActionChanged:
            self._updateVisibility()
        return qt.QToolBar.actionEvent(self, event)

    def _updateVisibility(self):
        visible = False
        for action in self.actions():
            if action.isVisible():
                visible = True
                break
        self.setVisible(visible)

class _RoiStatsWidget(qt.QMainWindow):
    """
    A widget used to display a table of stats for the ROIs
    Associates ROIStatsWidget and UpdateModeWidget
    """
    def __init__(self, parent=None, plot=None, mode=None):
        assert plot is not None
        qt.QMainWindow.__init__(self, parent)
        self._roiStatsWindow = ROIStatsWidget(plot=plot)
        self.setCentralWidget(self._roiStatsWindow)

        # update mode docker
        self._updateModeControl = UpdateModeWidget(parent=self)
        self._docker = qt.QDockWidget(parent=self)
        self._docker.setWidget(self._updateModeControl)
        self.addDockWidget(qt.Qt.TopDockWidgetArea, self._docker)
        self.setWindowFlags(qt.Qt.Widget)

        # connect signal / slot
        self._updateModeControl.sigUpdateModeChanged.connect(self._roiStatsWindow._setUpdateMode)
        callback = functools.partial(self._roiStatsWindow._updateAllStats, is_request=True)
        self._updateModeControl.sigUpdateRequested.connect(callback)

        # expose API
        self.registerROI = self._roiStatsWindow.registerROI
        self.setStats = self._roiStatsWindow.setStats
        self.addItem = self._roiStatsWindow.addItem
        self.removeItem = self._roiStatsWindow.removeItem
        self.setUpdateMode = self._updateModeControl.setUpdateMode

        # setup
        self._updateModeControl.setUpdateMode('auto')


class _RoiStatsDisplayExWindow(qt.QMainWindow):
    """
    Simple window to group the different statistics actors
    """
    def __init__(self, parent=None, mode=None):
        qt.QMainWindow.__init__(self, parent)
        self.plot = Plot2D()
        self.plot.getDefaultColormap().setName('viridis')
        # self.plot.setKeepDataAspectRatio(True)

        self.setCentralWidget(self.plot)

        # 1D roi management
        self._curveRoiWidget = self.plot.getCurvesRoiDockWidget().widget()
        # hide last columns which are of no use now
        # for index in (5, 6, 7, 8):
        #     self._curveRoiWidget.roiTable.setColumnHidden(index, True)

        # 2D - 3D roi manager
        self._regionManager = RegionOfInterestManager(parent=self.plot)

        # Create the table widget displaying
        self._2DRoiWidget = RegionOfInterestTableWidget()
        self._2DRoiWidget.setRegionOfInterestManager(self._regionManager)

        # tabWidget for displaying the rois
        self._roisTabWidget = qt.QTabWidget(parent=self)
        if hasattr(self._roisTabWidget, 'setTabBarAutoHide'):
            self._roisTabWidget.setTabBarAutoHide(True)

        # widget for displaying stats results and update mode
        self._statsWidget = _RoiStatsWidget(parent=self, plot=self.plot)

        # create Dock widgets
        self._roisTabWidgetDockWidget = qt.QDockWidget(parent=self)
        self._roisTabWidgetDockWidget.setWidget(self._roisTabWidget)
        self.addDockWidget(qt.Qt.TopDockWidgetArea, self._roisTabWidgetDockWidget)

        # create Dock widgets
        self._roiStatsWindowDockWidget = qt.QDockWidget(parent=self)
        self._roiStatsWindowDockWidget.setWidget(self._statsWidget)
        # move the docker contain in the parent widget
        # self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._statsWidget._docker) # worthless. Just asks how to refresh
        self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._roiStatsWindowDockWidget)

        # expose API
        self.setUpdateMode = self._statsWidget.setUpdateMode


    def setRois(self, rois1D=None, rois2D=None):
        rois1D = rois1D or ()
        rois2D = rois2D or ()
        self._curveRoiWidget.setRois(rois1D)
        for roi1D in rois1D:
            self._statsWidget.registerROI(roi1D)

        for roi2D in rois2D:
            self._regionManager.addRoi(roi2D)
            self._statsWidget.registerROI(roi2D)

        # update manage tab visibility
        if len(rois2D) > 0:
            self._roisTabWidget.addTab(self._2DRoiWidget, '2D roi(s)')
        if len(rois1D) > 0:
            self._roisTabWidget.addTab(self._curveRoiWidget, '1D roi(s)')

    def setStats(self, stats):
        self._statsWidget.setStats(stats=stats)

    def addItem(self, item, roi):
        self._statsWidget.addItem(roi=roi, plotItem=item)
        

# ==================================================================================================================== #
# `StatBase` Subclasses                                                                                                #
# ==================================================================================================================== #

custom_value_formatting_fn = partial(DocumentationFilePrinter.string_rep_if_short_enough, max_length=280, max_num_lines=1)
new_custom_item_formatter = partial(DocumentationFilePrinter._default_rich_text_formatter, value_formatting_fn=custom_value_formatting_fn)


class DebugPrinterStat(StatBase):
    """ Prints the context passed to `self.calculate(...)` to easily inspect incoming values
    Simple calculation of the line integral
    
    Usage:
        from pyphoplacecellanalysis.GUI.Silx.silx_helpers import DebugPrinterStat
        
    
    context: silx.gui.plot.stats.stats._CurveContext
	│   ├── _onlimits: NoneType
	│   ├── _from_: NoneType
	│   ├── _to_: NoneType
	│   ├── kind: str
	│   ├── min: int
	│   ├── max: int
	│   ├── data: tuple - (2, 6)
	│   ├── roi: NoneType
	│   ├── onlimits: bool
	│   ├── values: numpy.ma.core.MaskedArray - (6,)
	│   ├── axes: tuple - (1, 6)
	│   ├── xData: numpy.ndarray - (6,)
	│   ├── yData: numpy.ndarray - (6,)
    
    
    - context: silx.gui.plot.stats._ImageContext = <silx.gui.plot.stats.stats._ImageContext object at 0x000001DD028B3F70>
	- _mask_x_min: NoneType = None
	- _mask_x_max: NoneType = None
	- _mask_y_min: NoneType = None
	- _mask_y_max: NoneType = None
	- kind: str = image
	- min: float = 0.0
	- max: float = 0.17010370226218707
	- data: ndarray = [[0 6.26086e-05 0 0 0 1.36084e-05]<br> [0 5.09288e-05 0 0 0 1.34711e-05]<br> [0 3.49613e-05 0 0 0 1.00388e-05]<br> [0 2.07753e-05 0 0 0 4.49698e-06]<br> [0 1.19496e-05 0 0 0 1.13178e-06]<br> [0 8.87921e-06 0 0 0 1.80649e-07]<br> [0 9.77556e-06 0 0 0 2.75724e-08]<br> [0 1.10077... - (57, 6)
	- roi: NoneType = None
	- onlimits: bool = False
	- values: numpy.ma.MaskedArray = [[0.0 6.260861473791977e-05 0.0 0.0 0.0 1.3608427040290595e-05]<br> [0.0 5.092883650578533e-05 0.0 0.0 0.0 1.3471062013121828e-05]<br> [0.0 3.496131374127707e-05 0.0 0.0 0.0 1.0038814918889437e-05]<br> [0.0 2.0775284663506163e-05 0.0 0.0 0.0 4.4969817506273435e-06]<br> [0.0 1.... - (57, 6)
	- axes: tuple = (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]), array([0, 1, 2, 3, 4, 5])) - (2,)
	- origin: tuple = (0.0, 0.0) - (2,)
	- scale: tuple = (1.0, 1.0) - (2,)
    
    """
    def __init__(self):
        StatBase.__init__(self, name='debug_printer', compatibleKinds=('curve', "image",)) # compatibleKinds= ("curve", "image", "scatter", "histogram")

    def calculate(self, context):
        print_keys_if_possible('context', context, max_depth=4, custom_item_formatter=new_custom_item_formatter)
        return 'none'
    


class Integral(StatBase):
    """
    Simple calculation of the line integral
    """
    def __init__(self):
        StatBase.__init__(self, name='integral', compatibleKinds=('curve',))

    def calculate(self, context):
        xData, yData = context.data
        return np.trapz(x=xData, y=yData)


class DispersionImageStat(StatBase):
    """ Computes the dispersion in each time-bin

    - context: silx.gui.plot.stats._ImageContext = <silx.gui.plot.stats.stats._ImageContext object at 0x000001DD028B3F70>
	- _mask_x_min: NoneType = None
	- _mask_x_max: NoneType = None
	- _mask_y_min: NoneType = None
	- _mask_y_max: NoneType = None
	- kind: str = image
	- min: float = 0.0
	- max: float = 0.17010370226218707
	- data: ndarray = [[0 6.26086e-05 0 0 0 1.36084e-05]<br> [0 5.09288e-05 0 0 0 1.34711e-05]<br> [0 3.49613e-05 0 0 0 1.00388e-05]<br> [0 2.07753e-05 0 0 0 4.49698e-06]<br> [0 1.19496e-05 0 0 0 1.13178e-06]<br> [0 8.87921e-06 0 0 0 1.80649e-07]<br> [0 9.77556e-06 0 0 0 2.75724e-08]<br> [0 1.10077... - (57, 6)
	- roi: NoneType = None
	- onlimits: bool = False
	- values: numpy.ma.MaskedArray = [[0.0 6.260861473791977e-05 0.0 0.0 0.0 1.3608427040290595e-05]<br> [0.0 5.092883650578533e-05 0.0 0.0 0.0 1.3471062013121828e-05]<br> [0.0 3.496131374127707e-05 0.0 0.0 0.0 1.0038814918889437e-05]<br> [0.0 2.0775284663506163e-05 0.0 0.0 0.0 4.4969817506273435e-06]<br> [0.0 1.... - (57, 6)
	- axes: tuple = (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]), array([0, 1, 2, 3, 4, 5])) - (2,)
	- origin: tuple = (0.0, 0.0) - (2,)
	- scale: tuple = (1.0, 1.0) - (2,)
    
    """
    def __init__(self):
        StatBase.__init__(self, name='dispersion', compatibleKinds=("image",)) # compatibleKinds= ("curve", "image", "scatter", "histogram")

    def calculate(self, context):        
        arrData = context.data # (57, 6)
        # Assuming arr is your 2D numpy array
        spatial_dispersion = np.var(arrData, axis=0)
        return spatial_dispersion
    

class COM(StatBase):
    """
    Compute data center of mass
    """
    def __init__(self):
        StatBase.__init__(self, name='COM', description="Center of mass")

    def calculate(self, context):
        if context.kind in ('curve', 'histogram'):
            xData, yData = context.data
            deno = np.sum(yData).astype(np.float32)
            if deno == 0.0:
                return 0.0
            else:
                return np.sum(xData * yData).astype(np.float32) / deno
        elif context.kind == 'scatter':
            xData, yData, values = context.data
            values = values.astype(np.float64)
            deno = np.sum(values)
            if deno == 0.0:
                return float('inf'), float('inf')
            else:
                comX = np.sum(xData * values) / deno
                comY = np.sum(yData * values) / deno
                return comX, comY
            
