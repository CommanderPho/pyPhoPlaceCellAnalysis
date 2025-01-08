# Spike3DRasterRightSidebarWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\ZoomAndNavigationSidebarControls\Spike3DRasterRightSidebarWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp, uic

## IMPORTS:
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.External.pyqtgraph.widgets.LayoutWidget import LayoutWidget


## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'Spike3DRasterRightSidebarWidget.ui')

# LayoutWidget

class Spike3DRasterRightSidebarWidget(QtWidgets.QWidget):
    """ A simple container to hold interactive widgets
    
    
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        self.setVisible(False) # collapses and hides the sidebar
        # self.ui.layout_widget # a LayoutWidget

        # self.ui.btnToggleCollapseExpand # a LayoutWidget
        self.ui.layout_widget.setMinimumWidth(200.0)

        # self.setVisible(True) # shows the sidebar



class SpikeRasterRightSidebarOwningMixin:
    """ renders the UI 

        Currently used in Spike3DRasterWindowWidget to implement the right sidebar
    """
    
    @property
    def right_sidebar_widget(self) -> Spike3DRasterRightSidebarWidget:
        return self.ui.rightSideContainerWidget
    
    @property
    def right_sidebar_contents_container(self) -> LayoutWidget:
        return self.right_sidebar_widget.ui.layout_widget

    
    def toggle_right_sidebar(self):
        is_visible = self.right_sidebar_widget.isVisible()
        self.right_sidebar_widget.setVisible(not is_visible) # collapses and hides the sidebar
        # self.right_sidebar_widget.setVisible(True) # shows the sidebar

    @pyqtExceptionPrintingSlot(bool)
    def set_right_sidebar_visibility(self, is_visible:bool):
        self.right_sidebar_widget.setVisible(is_visible) 


    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_connectSignals(self, right_side_bar_controls):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        right_side_bar_connections = []
        # right_side_bar_connections.append(right_side_bar_controls.animation_time_step_changed.connect(self.on_animation_timestep_valueChanged))
        # right_side_bar_connections.append(right_side_bar_controls.temporal_zoom_factor_changed.connect(self.on_temporal_zoom_factor_valueChanged))
        # right_side_bar_connections.append(right_side_bar_controls.render_window_duration_changed.connect(self.on_render_window_duration_valueChanged))
        return right_side_bar_connections
        



    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # CALLED:
         # Set Initial values:
        # right_side_bar_controls = self.ui.rightSidebarWidget
        
        # right_side_bar_controls.ui.verticalSliderZoom.blockSignals(True)
        # right_side_bar_controls.ui.spinAnimationTimeStep.blockSignals(True)
        # right_side_bar_controls.ui.spinTemporalZoomFactor.blockSignals(True)
        # right_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(True)
        
        # right_side_bar_controls.ui.verticalSliderZoom.setValue(round(self.temporal_zoom_factor))
        # right_side_bar_controls.ui.spinAnimationTimeStep.setValue(self.animation_time_step)
        # right_side_bar_controls.ui.spinTemporalZoomFactor.setValue(round(self.temporal_zoom_factor))
        # right_side_bar_controls.ui.spinRenderWindowDuration.setValue(self.render_window_duration)

        # right_side_bar_controls.ui.verticalSliderZoom.blockSignals(False)
        # right_side_bar_controls.ui.spinAnimationTimeStep.blockSignals(False)
        # right_side_bar_controls.ui.spinTemporalZoomFactor.blockSignals(False)
        # right_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(False)
        pass
        
        
    @pyqtExceptionPrintingSlot()
    def SpikeRasterRightSidebarOwningMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @pyqtExceptionPrintingSlot(float, float)
    def SpikeRasterRightSidebarOwningMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # TODO: NOT CALLED
        pass


## Start Qt event loop
if __name__ == '__main__':
    app = pg.mkQApp()
    widget = Spike3DRasterRightSidebarWidget()
    widget.show()
    sys.exit(app.exec_())
