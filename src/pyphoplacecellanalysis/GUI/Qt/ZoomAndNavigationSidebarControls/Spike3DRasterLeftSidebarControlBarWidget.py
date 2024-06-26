import sys
import numpy as np

from qtpy import QtCore, QtWidgets
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarBase import Ui_leftSideToolbarWidget # Generated file from .ui


def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug


class Spike3DRasterLeftSidebarControlBar(QtWidgets.QWidget):
    """ A controls bar with buttons loaded from a Qt .ui file. """
    
    animation_time_step_changed = QtCore.Signal(float) # returns bool indicating whether is_playing
    temporal_zoom_factor_changed = QtCore.Signal(float)
    render_window_duration_changed = QtCore.Signal(float)
        
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_leftSideToolbarWidget()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        # Disable the scroll bar
        self.ui.verticalScrollBar.hide()
        # Setup the vertical zoom slider
        # self.ui.verticalSliderZoom

        # Connect Signals:
        # self.ui.verticalSliderZoom.valueChanged.connect(self.temporal_zoom_slider_valueChanged)
        self.ui.spinAnimationTimeStep.sigValueChanged.connect(self.animation_time_step_valueChanged)
        self.ui.spinTemporalZoomFactor.sigValueChanged.connect(self.temporal_zoom_factor_valueChanged)
        self.ui.spinRenderWindowDuration.sigValueChanged.connect(self.render_window_duration_valueChanged)
             
 
    @QtCore.Slot(object)
    def animation_time_step_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        # old_value = self.animation_time_step
        self.animation_time_step_changed.emit(sb.value())

        # self.animation_time_step = 
        # changedLabel.setText("Final value: %s" % str(sb.value()))
    
    @QtCore.Slot(object)
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
                
    @QtCore.Slot(object)
    def render_window_duration_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        # old_value = self.render_window_duration
        self.render_window_duration_changed.emit(sb.value())
        # self.render_window_duration = 
        
    @QtCore.Slot(int)
    def temporal_zoom_slider_valueChanged(self, int_slider_val):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        print(f'temporal_zoom_slider_valueChanged({int_slider_val})')
        float_slider_val = (float(int_slider_val)-0.0)/1000.0
        print(f'\t float_slider_val: {float_slider_val}')
        # TODO: emit the temporal changed signal:    
        # self.temporal_zoom_factor_changed.emit(float_slider_val)
                        
    def __str__(self):
         return
     
     

class SpikeRasterLeftSidebarControlsMixin:
    """ renders the UI controls for the Spike3DRaster_Vedo class 
        Follows Conventions outlined in ModelViewMixin Conventions.md
        
        Implementors must have:
    
            @QtCore.Slot(float)
            def on_animation_timestep_valueChanged(self, updated_val)
            
            @QtCore.Slot(float)
            def on_temporal_zoom_factor_valueChanged(self, updated_val)
        
            @QtCore.Slot(float)
            def on_render_window_duration_valueChanged(self, updated_val)
        
        Currently used in Spike3DRasterWindowWidget to implement the left sidebar
    """
    
    @QtCore.Slot()
    def SpikeRasterLeftSidebarControlsMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.Slot()
    def SpikeRasterLeftSidebarControlsMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @QtCore.Slot()
    def SpikeRasterLeftSidebarControlsMixin_connectSignals(self, left_side_bar_controls):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        left_side_bar_connections = []
        left_side_bar_connections.append(left_side_bar_controls.animation_time_step_changed.connect(self.on_animation_timestep_valueChanged))
        left_side_bar_connections.append(left_side_bar_controls.temporal_zoom_factor_changed.connect(self.on_temporal_zoom_factor_valueChanged))
        left_side_bar_connections.append(left_side_bar_controls.render_window_duration_changed.connect(self.on_render_window_duration_valueChanged))
        return left_side_bar_connections
        
            
    @QtCore.Slot()
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
        
        
    @QtCore.Slot()
    def SpikeRasterLeftSidebarControlsMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @QtCore.Slot(float, float)
    def SpikeRasterLeftSidebarControlsMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # TODO: NOT CALLED
        pass
    
    

        
    
     
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testWidget = Spike3DRasterLeftSidebarControlBar()
    # testWidget.show()
    sys.exit(app.exec_())
