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
            


# ==================================================================================================================== #
# Image Painting/Modification Tools built from the Mask Tools                                                          #
# ==================================================================================================================== #
"""

NOTE: the parameters for the pencil are fairly entangled in the Silx library defns: `silx.src.silx.gui.plot.PlotInteraction.PlotInteraction._setInteractiveMode`

"""

import numpy as np
from silx.gui import qt
from silx.gui.plot import Plot1D
from silx.gui.plot import Plot2D
from silx.gui.plot._BaseMaskToolsWidget import BaseMask, BaseMaskToolsWidget, BaseMaskToolsDockWidget
from silx.gui.plot.MaskToolsWidget import ImageMask, MaskToolsWidget, MaskToolsDockWidget

from silx._utils import NP_OPTIONAL_COPY
from silx.gui import qt, icons
from silx.gui.widgets.FloatEdit import FloatEdit
from silx.gui.colors import Colormap
from silx.gui.colors import rgba
from silx.gui.plot.actions.mode import PanModeAction


class ContinuousImageMask(ImageMask):
    """A 2D mask field with update operations.
    Coords follows (row, column) convention and are in mask array coords.
    This is meant for internal use by :class:`ContinuousMaskToolsWidget`.
    
    """
    # sigChanged = qt.Signal()
    # """Signal emitted when the mask has changed"""

    # sigStateChanged = qt.Signal()
    # """Signal emitted for each mask commit/undo/redo operation"""

    # sigUndoable = qt.Signal(bool)
    # """Signal emitted when undo becomes possible/impossible"""

    # sigRedoable = qt.Signal(bool)
    # """Signal emitted when redo becomes possible/impossible"""
    
    def __init__(self, image=None):
        """
        :param image: :class:`silx.gui.plot.items.ImageBase` instance
        """
        ImageMask.__init__(self, image)
        self.reset(shape=(0, 0))  # Init the mask with a 2D shape

class ContinuousMaskToolsWidget(MaskToolsWidget):
    """Widget with tools for drawing mask on an image in a PlotWidget."""
    _maxLevelNumber = 255
    def __init__(self, parent=None, plot=None):
        # super(ContinuousMaskToolsWidget, self).__init__(parent=parent, plot=plot, mask=ContinuousImageMask())
        super(ContinuousMaskToolsWidget, self).__init__(parent=parent, plot=plot) # , mask=ContinuousImageMask()
        self._mask = ContinuousImageMask()

    # ==================================================================================================================== #
    # Overrides                                                                                                            #
    # ==================================================================================================================== #
    def _initWidgets(self):
        """OVERRIDE Create widgets"""
        layout = qt.QBoxLayout(qt.QBoxLayout.LeftToRight)
        layout.addWidget(self._initMaskGroupBox())
        layout.addWidget(self._initDrawGroupBox())
        layout.addWidget(self._initThresholdGroupBox())
        layout.addWidget(self._initOtherToolsGroupBox())
        layout.addStretch(1)
        self.setLayout(layout)
        
    # Callback Overrides _________________________________________________________________________________________________ #        
    def _pencilForceChanged(self, force: float):
        """ 
        self.pencilContinuousSpinBox.valueChanged.connect(self._pencilForceChanged)
        self.pencilContinuousSlider.valueChanged.connect(self._pencilForceChanged)
        """
        old = self.pencilContinuousSpinBox.blockSignals(True)
        try:
            self.pencilContinuousSpinBox.setValue(force)
        finally:
            self.pencilContinuousSpinBox.blockSignals(old)

        old = self.pencilContinuousSlider.blockSignals(True)
        try:
            self.pencilContinuousSlider.setValue(force)
        finally:
            self.pencilContinuousSlider.blockSignals(old)
        self._updateInteractiveMode()
        

    def _getPencilForce(self) -> float:
        """Returns the force of the pencil to use in data coordinates`

        :rtype: float
        """
        return self.pencilContinuousSpinBox.value()

    def _activePencilMode(self):
        """OVERRIDE: Handle pencil action mode triggering. Overrides to provide force. 
        """
        self._releaseDrawingMode()
        self._drawingMode = "pencil"
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        width = self._getPencilWidth()
        force = self._getPencilForce() # #TODO 2024-08-01 03:36: - [ ] make use of this
        
        self.plot.setInteractiveMode(
            "draw", shape="pencil", source=self, color=color, width=width, # , force=force
        )
        self._updateDrawingModeWidgets()
        

    # Drawing Overrides: _________________________________________________________________________________________________ #
    def _initDrawGroupBox(self):
        """OVERRIDE: Init drawing tools widgets"""
        layout = qt.QVBoxLayout()

        self.browseAction = PanModeAction(self.plot, self.plot)
        self.addAction(self.browseAction)

        # Draw tools
        self.rectAction = qt.QAction(
            icons.getQIcon("shape-rectangle"), "Rectangle selection", self
        )
        self.rectAction.setToolTip(
            "Rectangle selection tool: (Un)Mask a rectangular region <b>R</b>"
        )
        self.rectAction.setShortcut(qt.QKeySequence(qt.Qt.Key_R))
        self.rectAction.setCheckable(True)
        self.rectAction.triggered.connect(self._activeRectMode)
        self.addAction(self.rectAction)

        self.ellipseAction = qt.QAction(
            icons.getQIcon("shape-ellipse"), "Circle selection", self
        )
        self.ellipseAction.setToolTip(
            "Rectangle selection tool: (Un)Mask a circle region <b>R</b>"
        )
        self.ellipseAction.setShortcut(qt.QKeySequence(qt.Qt.Key_R))
        self.ellipseAction.setCheckable(True)
        self.ellipseAction.triggered.connect(self._activeEllipseMode)
        self.addAction(self.ellipseAction)

        self.polygonAction = qt.QAction(
            icons.getQIcon("shape-polygon"), "Polygon selection", self
        )
        self.polygonAction.setShortcut(qt.QKeySequence(qt.Qt.Key_S))
        self.polygonAction.setToolTip(
            "Polygon selection tool: (Un)Mask a polygonal region <b>S</b><br>"
            "Left-click to place new polygon corners<br>"
            "Left-click on first corner to close the polygon"
        )
        self.polygonAction.setCheckable(True)
        self.polygonAction.triggered.connect(self._activePolygonMode)
        self.addAction(self.polygonAction)

        self.pencilAction = qt.QAction(
            icons.getQIcon("draw-pencil"), "Pencil tool", self
        )
        self.pencilAction.setShortcut(qt.QKeySequence(qt.Qt.Key_P))
        self.pencilAction.setToolTip("Pencil tool: (Un)Mask using a pencil <b>P</b>")
        self.pencilAction.setCheckable(True)
        self.pencilAction.triggered.connect(self._activePencilMode)
        self.addAction(self.pencilAction)

        self.drawActionGroup = qt.QActionGroup(self)
        self.drawActionGroup.setExclusive(True)
        self.drawActionGroup.addAction(self.rectAction)
        self.drawActionGroup.addAction(self.ellipseAction)
        self.drawActionGroup.addAction(self.polygonAction)
        self.drawActionGroup.addAction(self.pencilAction)

        actions = (
            self.browseAction,
            self.rectAction,
            self.ellipseAction,
            self.polygonAction,
            self.pencilAction,
        )
        drawButtons = []
        for action in actions:
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            drawButtons.append(btn)
        container = self._hboxWidget(*drawButtons)
        layout.addWidget(container)

        # Mask/Unmask radio buttons
        maskRadioBtn = qt.QRadioButton("Mask")
        maskRadioBtn.setToolTip(
            "Drawing masks with current level. Press <b>Ctrl</b> to unmask"
        )
        maskRadioBtn.setChecked(True)

        unmaskRadioBtn = qt.QRadioButton("Unmask")
        unmaskRadioBtn.setToolTip(
            "Drawing unmasks with current level. Press <b>Ctrl</b> to mask"
        )

        self.maskStateGroup = qt.QButtonGroup()
        self.maskStateGroup.addButton(maskRadioBtn, 1)
        self.maskStateGroup.addButton(unmaskRadioBtn, 0)

        self.maskStateWidget = self._hboxWidget(maskRadioBtn, unmaskRadioBtn)
        layout.addWidget(self.maskStateWidget)

        self.maskStateWidget.setHidden(True)

        # Pencil settings
        self.pencilSetting = self._createPencilSettings(None)
        self.pencilSetting.setVisible(False)
        layout.addWidget(self.pencilSetting)

        layout.addStretch(1)

        drawGroup = qt.QGroupBox("~Draw tools")
        drawGroup.setLayout(layout)
        return drawGroup

    def _createPencilSettings(self, parent=None):
        """ OVERRIDE: to define pencil settings widget """
        pencilSetting = qt.QWidget(parent)

        self.pencilSpinBox = qt.QSpinBox(parent=pencilSetting)
        self.pencilSpinBox.setRange(1, 1024)
        pencilToolTip = """Set pencil drawing tool size in pixels of the image on which to make the mask."""
        self.pencilSpinBox.setToolTip(pencilToolTip)

        self.pencilSlider = qt.QSlider(qt.Qt.Horizontal, parent=pencilSetting)
        self.pencilSlider.setRange(1, 50)
        self.pencilSlider.setToolTip(pencilToolTip)

        pencilLabel = qt.QLabel("Pencil size:", parent=pencilSetting)

        layout = qt.QGridLayout()
        layout.addWidget(pencilLabel, 0, 0)
        layout.addWidget(self.pencilSpinBox, 0, 1)
        layout.addWidget(self.pencilSlider, 1, 1)
        
        ## continuous value spinbox:
        self.pencilContinuousSpinBox = qt.QDoubleSpinBox(parent=pencilSetting)
        self.pencilContinuousSpinBox.setRange(0.1, 10.0)
        pencilContinuousToolTip = """Set pencil drawing tool 'force' which determines how much weight each draw operation changes."""
        self.pencilContinuousSpinBox.setToolTip(pencilContinuousToolTip)

        self.pencilContinuousSlider = qt.QSlider(qt.Qt.Horizontal, parent=pencilSetting)
        self.pencilContinuousSlider.setRange(1, 10)
        self.pencilContinuousSlider.setToolTip(pencilContinuousToolTip)

        pencilForceLabel = qt.QLabel("Pencil force:", parent=pencilSetting)
        layout.addWidget(pencilForceLabel, 1, 0)
        layout.addWidget(self.pencilContinuousSpinBox, 1, 1)
        layout.addWidget(self.pencilContinuousSlider, 2, 1)
        
        pencilSetting.setLayout(layout)

        self.pencilSpinBox.valueChanged.connect(self._pencilWidthChanged)
        self.pencilSlider.valueChanged.connect(self._pencilWidthChanged)

        self.pencilContinuousSpinBox.valueChanged.connect(self._pencilForceChanged)
        self.pencilContinuousSlider.valueChanged.connect(self._pencilForceChanged)

        return pencilSetting

    def _initThresholdGroupBox(self):
        """Init thresholding widgets"""

        self.belowThresholdAction = qt.QAction(
            icons.getQIcon("plot-roi-below"), "Mask below threshold", self
        )
        self.belowThresholdAction.setToolTip(
            "Mask image where values are below given threshold"
        )
        self.belowThresholdAction.setCheckable(True)
        self.belowThresholdAction.setChecked(True)

        self.betweenThresholdAction = qt.QAction(
            icons.getQIcon("plot-roi-between"), "Mask within range", self
        )
        self.betweenThresholdAction.setToolTip(
            "Mask image where values are within given range"
        )
        self.betweenThresholdAction.setCheckable(True)

        self.aboveThresholdAction = qt.QAction(
            icons.getQIcon("plot-roi-above"), "Mask above threshold", self
        )
        self.aboveThresholdAction.setToolTip(
            "Mask image where values are above given threshold"
        )
        self.aboveThresholdAction.setCheckable(True)

        self.thresholdActionGroup = qt.QActionGroup(self)
        self.thresholdActionGroup.setExclusive(True)
        self.thresholdActionGroup.addAction(self.belowThresholdAction)
        self.thresholdActionGroup.addAction(self.betweenThresholdAction)
        self.thresholdActionGroup.addAction(self.aboveThresholdAction)
        self.thresholdActionGroup.triggered.connect(self._thresholdActionGroupTriggered)

        self.loadColormapRangeAction = qt.QAction(
            icons.getQIcon("view-refresh"), "Set min-max from colormap", self
        )
        self.loadColormapRangeAction.setToolTip(
            "Set min and max values from current colormap range"
        )
        self.loadColormapRangeAction.setCheckable(False)
        self.loadColormapRangeAction.triggered.connect(
            self._loadRangeFromColormapTriggered
        )

        widgets = []
        for action in self.thresholdActionGroup.actions():
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            widgets.append(btn)

        spacer = qt.QWidget(parent=self)
        spacer.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        widgets.append(spacer)

        loadColormapRangeBtn = qt.QToolButton()
        loadColormapRangeBtn.setDefaultAction(self.loadColormapRangeAction)
        widgets.append(loadColormapRangeBtn)

        toolBar = self._hboxWidget(*widgets, stretch=False)

        config = qt.QGridLayout()
        config.setContentsMargins(0, 0, 0, 0)

        self.minLineLabel = qt.QLabel("Min:", self)
        self.minLineEdit = FloatEdit(self, value=0)
        config.addWidget(self.minLineLabel, 0, 0)
        config.addWidget(self.minLineEdit, 0, 1)

        self.maxLineLabel = qt.QLabel("Max:", self)
        self.maxLineEdit = FloatEdit(self, value=0)
        config.addWidget(self.maxLineLabel, 1, 0)
        config.addWidget(self.maxLineEdit, 1, 1)

        self.applyMaskBtn = qt.QPushButton("Apply mask")
        self.applyMaskBtn.clicked.connect(self._maskBtnClicked)

        layout = qt.QVBoxLayout()
        layout.addWidget(toolBar)
        layout.addLayout(config)
        layout.addWidget(self.applyMaskBtn)
        layout.addStretch(1)

        self.thresholdGroup = qt.QGroupBox("Threshold")
        self.thresholdGroup.setLayout(layout)

        # Init widget state
        self._thresholdActionGroupTriggered(self.belowThresholdAction)
        return self.thresholdGroup

        # track widget visibility and plot active image changes

    def _initOtherToolsGroupBox(self):
        layout = qt.QVBoxLayout()

        self.maskNanBtn = qt.QPushButton("Mask not finite values")
        self.maskNanBtn.setToolTip("Mask Not a Number and infinite values")
        self.maskNanBtn.clicked.connect(self._maskNotFiniteBtnClicked)
        layout.addWidget(self.maskNanBtn)
        layout.addStretch(1)

        self.otherToolGroup = qt.QGroupBox("Other tools")
        self.otherToolGroup.setLayout(layout)
        return self.otherToolGroup




        
class ContinuousMaskToolsDockWidget(MaskToolsDockWidget):
    """:class:`ContinuousMaskToolsWidget` embedded in a QDockWidget.

    For integration in a :class:`PlotWindow`.

    :param parent: See :class:`QDockWidget`
    :param plot: The PlotWidget this widget is operating on
    :paran str name: The title of this widget
    """

    def __init__(self, parent=None, plot=None, name="Mask"):
        widget = ContinuousMaskToolsWidget(plot=plot)
        super(ContinuousMaskToolsDockWidget, self).__init__(parent=parent, widget=widget, name=name)


def getMaskToolsDockWidget(self):
    """DockWidget with image mask panel (lazy-loaded)."""
    if self._maskToolsDockWidget is None:
        self._maskToolsDockWidget = ContinuousMaskToolsDockWidget(plot=self, name="Mask")
        self._maskToolsDockWidget.hide()
        self._maskToolsDockWidget.toggleViewAction().triggered.connect(self._handleDockWidgetViewActionTriggered)
        self._maskToolsDockWidget.visibilityChanged.connect(self._handleFirstDockWidgetShow)
        return self._maskToolsDockWidget

