import sys
import os
import numpy as np

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

# from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarBase import Ui_leftSideToolbarWidget # Generated file from .ui
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
# from qtpy import QtCore, QtWidgets
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\ZoomAndNavigationSidebarControls\Spike3DRasterLeftSidebarControlBarBase.ui automatically by PhoPyQtClassGenerator VSCode Extension



## IMPORTS:
# 

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'Spike3DRasterLeftSidebarControlBarWidget.ui')


# def trap_exc_during_debug(*args):
#     # when app raises uncaught exception, print info
#     print(args)


# # install exception hook: without this, uncaught exception would cause application to exit
# sys.excepthook = trap_exc_during_debug


class Spike3DRasterLeftSidebarControlBar(QWidget):
    """ A controls bar with buttons loaded from a Qt .ui file. 
    
    self.ui.btnToggleCrosshairTrace
    self.ui.lblCrosshairTraceStaticLabel
    self.ui.lblCrosshairTraceValue
    
    self.ui.verticalScrollBar
    
    spinAnimationTimeStep
    spinRenderWindowDuration
    spinTemporalZoomFactor
    
    self.ui.btnToggleCollapseExpand
    
    """
    
    animation_time_step_changed = pyqtSignal(float) # returns bool indicating whether is_playing
    temporal_zoom_factor_changed = pyqtSignal(float)
    render_window_duration_changed = pyqtSignal(float)
        
    crosshair_trace_toggled = pyqtSignal()

    # @property
    # def lblCrosshairTraceValue(self):
    #     """The lblCrosshairTraceValue property."""
    #     return self.ui.lblCrosshairTraceValue


    @property
    def crosshair_trace_time(self) -> float:
        """The crosshair_trace_time property."""
        # return self.ui.lblCrosshairTraceValue.getText()
        return None
    @crosshair_trace_time.setter
    def crosshair_trace_time(self, value: float):
        if value is not None:
            self.ui.lblCrosshairTraceValue.setText(f"{value}")
            self.ui.lblCrosshairTraceStaticLabel.setVisible(True)
            self.ui.lblCrosshairTraceValue.setVisible(True)
        else:
            ## Hide it
            self.ui.lblCrosshairTraceStaticLabel.setVisible(False)
            self.ui.lblCrosshairTraceValue.setVisible(False)


    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        
        self.initUI() ## when did this get disabled?
        self.show() # Show the GUI


    def initUI(self):
        
        self.ui.btnToggleCollapseExpand.setVisible(False)
        
        # Disable the scroll bar
        self.ui.verticalScrollBar.hide()
        self.ui.verticalScrollBar.setVisible(False)

        # Setup the vertical zoom slider
        # self.ui.verticalSliderZoom
        self.ui.verticalSliderZoom.setVisible(False)
        
        # Connect Signals:
        # self.ui.verticalSliderZoom.valueChanged.connect(self.temporal_zoom_slider_valueChanged)
        self.ui.spinAnimationTimeStep.sigValueChanged.connect(self.animation_time_step_valueChanged)
        self.ui.spinTemporalZoomFactor.sigValueChanged.connect(self.temporal_zoom_factor_valueChanged)
        self.ui.spinRenderWindowDuration.sigValueChanged.connect(self.render_window_duration_valueChanged)
        self.ui.btnToggleCrosshairTrace.clicked.connect(self.crosshair_trace_button_Toggled)
        # self.ui.btnToggleCrosshairTrace.clicked.

        
        

        self.ui.btnToggleCrosshairTrace.setVisible(True)
        self.ui.lblCrosshairTraceStaticLabel.setVisible(False)
        self.ui.lblCrosshairTraceValue.setVisible(False)
             

        

    @pyqtExceptionPrintingSlot(object)
    def animation_time_step_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        # old_value = self.animation_time_step
        self.animation_time_step_changed.emit(sb.value())

        # self.animation_time_step = 
        # changedLabel.setText("Final value: %s" % str(sb.value()))
    
    @pyqtExceptionPrintingSlot(object)
    def temporal_zoom_factor_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        # old_value = self.temporal_zoom_factor
        updated_val = sb.value()
        
        # self.ui.verticalSliderZoom.blockSignals(True)
        self.temporal_zoom_factor_changed.emit(updated_val)
        # update slider
        # slider_int_val = (updated_val * 1000.0)+0.0
        # self.ui.verticalSliderZoom.setValue(slider_int_val)
        # self.ui.verticalSliderZoom.blockSignals(False) # done
                
    @pyqtExceptionPrintingSlot(object)
    def render_window_duration_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        # old_value = self.render_window_duration
        self.render_window_duration_changed.emit(sb.value())
        # self.render_window_duration = 
        
    @pyqtExceptionPrintingSlot(int)
    def temporal_zoom_slider_valueChanged(self, int_slider_val):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        print(f'temporal_zoom_slider_valueChanged({int_slider_val})')
        float_slider_val = (float(int_slider_val)-0.0)/1000.0
        print(f'\t float_slider_val: {float_slider_val}')
        # TODO: emit the temporal changed signal:    
        # self.temporal_zoom_factor_changed.emit(float_slider_val)
                        

    # @pyqtExceptionPrintingSlot()
    def crosshair_trace_button_Toggled(self):
        print(f'Spike3DRasterLeftSidebarControlBar.crosshair_trace_button_Toggled(): self.ui.btnToggleCrosshairTrace.isChecked(): {self.ui.btnToggleCrosshairTrace.isChecked()}')
        wants_crosshair_trace_visible: bool = self.ui.btnToggleCrosshairTrace.isChecked()
        self.ui.lblCrosshairTraceStaticLabel.setVisible(wants_crosshair_trace_visible)
        self.ui.lblCrosshairTraceValue.setVisible(wants_crosshair_trace_visible)
        # self.crosshair_trace_toggled.emit(wants_crosshair_trace_visible)
        self.crosshair_trace_toggled.emit()

    # def __str__(self):
    #      return
     
     

class SpikeRasterLeftSidebarControlsMixin:
    """ renders the UI controls for the Spike3DRasterWindowWidget class 
        Follows Conventions outlined in ModelViewMixin Conventions.md
        
        Implementors must have:
    
            @pyqtExceptionPrintingSlot(float)
            def on_animation_timestep_valueChanged(self, updated_val)
            
            @pyqtExceptionPrintingSlot(float)
            def on_temporal_zoom_factor_valueChanged(self, updated_val)
        
            @pyqtExceptionPrintingSlot(float)
            def on_render_window_duration_valueChanged(self, updated_val)
        
        Currently used in Spike3DRasterWindowWidget to implement the left sidebar
    """
    @property
    def left_side_bar_controls(self) -> Spike3DRasterLeftSidebarControlBar:
        """The left_side_bar_controls property."""
        return self.ui.leftSideToolbarWidget

        

    @pyqtExceptionPrintingSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @pyqtExceptionPrintingSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @pyqtExceptionPrintingSlot()
    def SpikeRasterLeftSidebarControlsMixin_connectSignals(self, left_side_bar_controls):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        left_side_bar_connections = []
        left_side_bar_connections.append(left_side_bar_controls.animation_time_step_changed.connect(self.on_animation_timestep_valueChanged))
        left_side_bar_connections.append(left_side_bar_controls.temporal_zoom_factor_changed.connect(self.on_temporal_zoom_factor_valueChanged))
        left_side_bar_connections.append(left_side_bar_controls.render_window_duration_changed.connect(self.on_render_window_duration_valueChanged))
        left_side_bar_connections.append(left_side_bar_controls.crosshair_trace_toggled.connect(self.on_crosshair_trace_toggled)) # #TODO 2025-02-10 16:50: - [ ] Add handler for enable/disable crosshairs trace
        return left_side_bar_connections
        
            
    @pyqtExceptionPrintingSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # CALLED:
         # Set Initial values:
        left_side_bar_controls = self.ui.leftSideToolbarWidget
        
        left_side_bar_controls.ui.verticalSliderZoom.blockSignals(True)
        left_side_bar_controls.ui.spinAnimationTimeStep.blockSignals(True)
        left_side_bar_controls.ui.spinTemporalZoomFactor.blockSignals(True)
        left_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(True)
        
        left_side_bar_controls.ui.verticalSliderZoom.setValue(round(self.temporal_zoom_factor))
        left_side_bar_controls.ui.spinAnimationTimeStep.setValue(self.animation_time_step)
        left_side_bar_controls.ui.spinTemporalZoomFactor.setValue(round(self.temporal_zoom_factor))
        left_side_bar_controls.ui.spinRenderWindowDuration.setValue(self.render_window_duration)

        left_side_bar_controls.ui.verticalSliderZoom.blockSignals(False)
        left_side_bar_controls.ui.spinAnimationTimeStep.blockSignals(False)
        left_side_bar_controls.ui.spinTemporalZoomFactor.blockSignals(False)
        left_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(False)
        
        
    @pyqtExceptionPrintingSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @pyqtExceptionPrintingSlot(float, float)
    def SpikeRasterLeftSidebarControlsMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # Called in the Implementor's update_window(...) function
        print(f'SpikeRasterLeftSidebarControlsMixin_on_window_update(new_start: {new_start}, new_end: {new_end}')
        
        left_side_bar_controls = self.ui.leftSideToolbarWidget

        if (new_start is not None) and (new_end is not None):
            ## Block signals:
            left_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(True)

            # Force completion of any ongoing edits to prevent conflicts
            if left_side_bar_controls.ui.spinRenderWindowDuration.hasFocus():
                # First interpret any text in the editor
                left_side_bar_controls.ui.spinRenderWindowDuration.interpretText()
                # Then remove focus to cancel any ongoing edit
                left_side_bar_controls.ui.spinRenderWindowDuration.clearFocus()

            ## Update values:
            new_duration: float = new_end - new_start
            
            # Check if value is actually different to avoid unnecessary updates
            current_value = left_side_bar_controls.ui.spinRenderWindowDuration.value()
            if abs(current_value - new_duration) > 1e-6:  # Compare with small epsilon for float comparison
                left_side_bar_controls.ui.spinRenderWindowDuration.setValue(new_duration)

            ## Unblock when done:
            left_side_bar_controls.ui.spinRenderWindowDuration.blockSignals(False)
            
        
    
## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = Spike3DRasterLeftSidebarControlBar()
    widget.show()
    sys.exit(app.exec_())

