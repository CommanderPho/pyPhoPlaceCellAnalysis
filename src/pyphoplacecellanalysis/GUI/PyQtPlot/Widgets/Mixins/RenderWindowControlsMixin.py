import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np


class RenderWindowControlsMixin:
    """ 
    # self.ui.spinBoxCurrentFrame.blockSignals(True)
    """    
    def setup_render_window_controls(self):
        """ Build the right controls bar:
        
        creates self.ui.right_controls_panel
        
        """
        ####################################################
        ####  Controls Bar Right #######        
        self.ui.right_controls_labels = []
        self.ui.right_controls_panel = QtWidgets.QWidget()
        self.ui.right_controls_panel.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding) # Expands to fill the vertical height, but occupy only the preferred width
        self.ui.right_controls_panel.setMaximumWidth(100.0)
        # Try to make the bottom widget bar transparent:
        self.ui.right_controls_panel.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.ui.right_controls_panel.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.right_controls_panel.setStyleSheet("background:transparent;")
        
        # Playback Slider Bottom Bar:
        self.ui.layout_right_bar = QtWidgets.QVBoxLayout()
        self.ui.layout_right_bar.setContentsMargins(10, 4, 4, 4)
        self.ui.right_controls_panel.setLayout(self.ui.layout_right_bar)
        # self.ui.layout_right_bar.addSpacing(50)

        # Playback Slider:
        # self.ui.slider_right = QtWidgets.QSlider(QtCore.Qt.Vertical)
        # # self.ui.slider_right.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # self.ui.slider_right.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        # # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        # self.ui.slider_right.setRange(0, 100)
        # self.ui.slider_right.setSingleStep(1)
        # # self.ui.slider_right.setSingleStep(2)
        # self.ui.slider_right.setValue(0)
        # # self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # # sliderMoved vs valueChanged? vs sliderChange?
        # self.ui.layout_right_bar.addWidget(self.ui.slider_right)
        
        # Animation_time_step:
        label = QtWidgets.QLabel('animation_time_step')
        label.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.ui.right_controls_labels.append(label)
        self.ui.layout_right_bar.addWidget(label)
        self.ui.spinAnimationTimeStep = pg.SpinBox(value=self.animation_time_step, suffix='Sec', siPrefix=True, dec=True, step=0.01, minStep=0.01)
        self.ui.spinAnimationTimeStep.sigValueChanged.connect(self.animation_time_step_valueChanged)
        self.ui.layout_right_bar.addWidget(self.ui.spinAnimationTimeStep)

        # temporal_zoom_factor:
        label = QtWidgets.QLabel('temporal_zoom_factor')
        label.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.ui.right_controls_labels.append(label)
        self.ui.layout_right_bar.addWidget(label)
        self.ui.spinTemporalZoomFactor = pg.SpinBox(value=self.temporal_zoom_factor, suffix='x', siPrefix=False, dec=True, step=0.1, minStep=0.1)
        self.ui.spinTemporalZoomFactor.sigValueChanged.connect(self.temporal_zoom_factor_valueChanged)
        self.ui.layout_right_bar.addWidget(self.ui.spinTemporalZoomFactor)
        
        # render_window_duration:
        label = QtWidgets.QLabel('render_window_duration')
        label.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.ui.right_controls_labels.append(label)
        self.ui.layout_right_bar.addWidget(label)
        self.ui.spinRenderWindowDuration = pg.SpinBox(value=self.render_window_duration, suffix='Sec', siPrefix=True, dec=True, step=0.5, minStep=0.1)
        self.ui.spinRenderWindowDuration.sigValueChanged.connect(self.render_window_duration_valueChanged)
        self.ui.layout_right_bar.addWidget(self.ui.spinRenderWindowDuration)

        self.ui.layout_right_bar.addSpacing(50)
        
    def disable_render_window_controls(self):
        self.ui.spinAnimationTimeStep.blockSignals(True)
        self.ui.spinTemporalZoomFactor.blockSignals(True)
        self.ui.spinRenderWindowDuration.blockSignals(True)
        
        
        # self.ui.spinAnimationTimeStep.setEnabled(False)
        self.ui.spinTemporalZoomFactor.setValue(10.0)
        self.ui.spinRenderWindowDuration.setValue(self.render_window_duration) # set to render window duration

        self.ui.spinTemporalZoomFactor.setReadOnly(True)
        self.ui.spinRenderWindowDuration.setReadOnly(True)

        # self.ui.spinAnimationTimeStep.setEnabled(False)
        # self.ui.spinTemporalZoomFactor.setEnabled(False)
        self.ui.spinRenderWindowDuration.setEnabled(False)
        
        self.ui.spinAnimationTimeStep.blockSignals(False)
        self.ui.spinTemporalZoomFactor.blockSignals(False)
        self.ui.spinRenderWindowDuration.blockSignals(False)
        
        
    def animation_time_step_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        old_value = self.animation_time_step
        self.animation_time_step = sb.value()
        # changedLabel.setText("Final value: %s" % str(sb.value()))
    
    def temporal_zoom_factor_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        old_value = self.temporal_zoom_factor
        self.temporal_zoom_factor = sb.value()
        
    def render_window_duration_valueChanged(self, sb):
        # print(f'sb: {sb}, sb.value(): {str(sb.value())}')
        old_value = self.render_window_duration
        self.render_window_duration = sb.value()
        
        