import numpy as np

from qtpy import QtCore, QtWidgets
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarBase import Ui_leftSideToolbarWidget # Generated file from .ui



class Spike3DRasterLeftSidebarControlBar(QtWidgets.QWidget):
    """ A controls bar with buttons loaded from a Qt .ui file. """
    
    # TODO: add signals here:
    
    
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

        # Set Initial values:
        self.ui.verticalSliderZoom.setValue(self.temporal_zoom_factor)
        
        
        self.ui.spinAnimationTimeStep.setValue(self.animation_time_step)
        self.ui.spinTemporalZoomFactor.setValue(self.temporal_zoom_factor)
        self.ui.spinRenderWindowDuration.setValue(self.render_window_duration)
        
        # Connect Signals:
        self.ui.verticalSliderZoom.valueChanged.connect(self.temporal_zoom_slider_valueChanged)
        self.ui.spinAnimationTimeStep.sigValueChanged.connect(self.animation_time_step_valueChanged)
        self.ui.spinTemporalZoomFactor.sigValueChanged.connect(self.temporal_zoom_factor_valueChanged)
        self.ui.spinRenderWindowDuration.sigValueChanged.connect(self.render_window_duration_valueChanged)
             
    def animation_time_step_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        old_value = self.animation_time_step
        self.animation_time_step = sb.value()
        # changedLabel.setText("Final value: %s" % str(sb.value()))
    
    def temporal_zoom_factor_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        old_value = self.temporal_zoom_factor
        self.ui.verticalSliderZoom.blockSignals(True)
        self.temporal_zoom_factor = sb.value()
        # update slider
        slider_int_val = (self.temporal_zoom_factor * 1000.0)+0.0
        self.ui.verticalSliderZoom.setValue(slider_int_val)
        self.ui.verticalSliderZoom.blockSignals(False) # done
                
    def render_window_duration_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        old_value = self.render_window_duration
        self.render_window_duration = sb.value()
        
    def temporal_zoom_slider_valueChanged(self, int_slider_val):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        print(f'temporal_zoom_slider_valueChanged({int_slider_val})')
        float_slider_val = (float(int_slider_val)-0.0)/1000.0
        print(f'\t float_slider_val: {float_slider_val}')
        # old_value = self.render_window_duration
        # self.render_window_duration = sb.value()
        

    def disable_render_window_controls(self):
        # Wrapped in try block so it won't throw an error if these controls were never added (self.setup_render_window_controls() was never called)
        try:
            self.ui.spinAnimationTimeStep.blockSignals(True)
            self.ui.spinTemporalZoomFactor.blockSignals(True)
            self.ui.spinRenderWindowDuration.blockSignals(True)
            
            
            # self.ui.spinAnimationTimeStep.setEnabled(False)
            # self.ui.spinTemporalZoomFactor.setValue(10.0)
            self.ui.spinTemporalZoomFactor.setValue(1.0)
            self.temporal_zoom_factor = 1.0
            
            self.ui.spinRenderWindowDuration.setValue(self.render_window_duration) # set to render window duration

            self.ui.spinTemporalZoomFactor.setReadOnly(True)
            self.ui.spinRenderWindowDuration.setReadOnly(True)

            # self.ui.spinAnimationTimeStep.setEnabled(False)
            # self.ui.spinTemporalZoomFactor.setEnabled(False)
            self.ui.spinRenderWindowDuration.setEnabled(False)
            
            self.ui.spinAnimationTimeStep.blockSignals(False)
            self.ui.spinTemporalZoomFactor.blockSignals(False)
            self.ui.spinRenderWindowDuration.blockSignals(False)

        except Exception as e:
            print(f'disable_render_window_controls(): called but do not seem to have RenderWindowControls enabled: err: {e}')
            # raise e
            return
                        
    def __str__(self):
         return
     
     

class SpikeRasterLeftSidebarControlsMixin:
    """ renders the UI controls for the Spike3DRaster_Vedo class 
        Follows Conventions outlined in ModelViewMixin Conventions.md
        
        Currently Unused after being removed from Spike3DRaster_Vedo
    """
    
    @QtCore.pyqtSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.pyqtSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @QtCore.pyqtSlot()
    def SpikeRasterLeftSidebarControlsMixin_connectSignals(self, left_side_bar_controls):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        left_side_bar_connections = []
        left_side_bar_connections.append(left_side_bar_controls.play_pause_toggled.connect(self.play_pause))
        left_side_bar_connections.append(left_side_bar_controls.jump_left.connect(self.on_jump_left))
        left_side_bar_connections.append(left_side_bar_controls.jump_right.connect(self.on_jump_right))
        left_side_bar_connections.append(left_side_bar_controls.reverse_toggled.connect(self.on_reverse_held))
        return left_side_bar_connections
        
            
    @QtCore.pyqtSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # CALLED:
        
        # controls_frame = QtWidgets.QFrame()
        # controls_layout = QtWidgets.QHBoxLayout() # H-box layout
        
        # # controls_layout = QtWidgets.QGridLayout()
        # # controls_layout.setContentsMargins(0, 0, 0, 0)
        # # controls_layout.setVerticalSpacing(0)
        # # controls_layout.setHorizontalSpacing(0)
        # # controls_layout.setStyleSheet("background : #1B1B1B; color : #727272")
        
        # # Set-up the rest of the Qt window
        # button = QtWidgets.QPushButton("My Button makes the cone red")
        # button.setToolTip('This is an example button')
        # button.clicked.connect(self.onClick)
        # controls_layout.addWidget(button)
        
        # button2 = QtWidgets.QPushButton("<")
        # button2.setToolTip('<')
        # # button2.clicked.connect(self.onClick)
        # controls_layout.addWidget(button2)
        
        # button3 = QtWidgets.QPushButton(">")
        # button3.setToolTip('>')
        # controls_layout.addWidget(button3)
        
        # # Set Final Layouts:
        # controls_frame.setLayout(controls_layout)
        
        controls_frame = Spike3DRasterBottomPlaybackControlBar() # Initialize new controls class from the Spike3DRasterBottomPlaybackControlBar class.
        controls_layout = controls_frame.layout() # Get the layout
        
        self.SpikeRasterLeftSidebarControlsMixin_connectSignals(controls_frame)

        
        return controls_frame, controls_layout


    @QtCore.pyqtSlot()
    def SpikeRasterLeftSidebarControlsMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @QtCore.pyqtSlot(float, float)
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
