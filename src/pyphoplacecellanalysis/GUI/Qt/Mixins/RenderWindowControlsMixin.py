import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
import qtawesome as qta
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider




    

class RenderPlaybackControlsMixin:
    """ 

    """
    
    WantsPlaybackControls = False
    
    def setup_render_playback_controls(self):
        """ Build the bottom playback controls bar:
        
        creates self.ui.panel_bottom_bar
        
        """
        
        self.slidebar_val = 0
        
        ####################################################
        ####  Controls Bar Bottom #######
        ####    Slide Bar Bottom #######
        self.ui.panel_bottom_bar = QtWidgets.QWidget()
        self.ui.panel_bottom_bar.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.ui.panel_bottom_bar.setMaximumHeight(50.0) # maximum height
        
        # Try to make the bottom widget bar transparent:
        self.ui.panel_bottom_bar.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.ui.panel_bottom_bar.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.panel_bottom_bar.setStyleSheet("background:transparent;")
        
        # Playback Slider Bottom Bar:
        self.ui.layout_slide_bar = QtWidgets.QHBoxLayout()
        self.ui.layout_slide_bar.setContentsMargins(6, 3, 4, 4)
        self.ui.panel_bottom_bar.setLayout(self.ui.layout_slide_bar)

        # New Button: Play/Pause
        self.ui.button_play_pause = ToggleButton()
        self.ui.button_play_pause.setMinimumHeight(25)
        self.ui.button_play_pause.setMinimumWidth(30)
        self.ui.play_pause_model = ToggleButtonModel(None, self)
        self.ui.play_pause_model.setStateMap(
            {
                True: {
                    "text": "",
                    "icon": qta.icon("fa.play", scale_factor=0.7, color='white', color_active='orange')
                },
                False: {
                    "text": "",
                    "icon": qta.icon("fa.pause", scale_factor=0.7, color='white', color_active='orange')
                }
            }
        )
        self.ui.button_play_pause.setModel(self.ui.play_pause_model)
        self.ui.button_play_pause.clicked.connect(self.play_pause)
        self.ui.layout_slide_bar.addWidget(self.ui.button_play_pause)
        
        
        # Playback Slider:
        # # self.ui.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # # self.ui.slider = HighlightedJumpSlider(QtCore.Qt.Horizontal)
        # self.ui.slider = HighlightedJumpSlider()
        # self.ui.slider.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        # self.ui.slider.setRange(0, 100)
        # self.ui.slider.setSingleStep(1)
        # self.ui.slider.setValue(0)
        # # self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # # sliderMoved vs valueChanged? vs sliderChange?
        # self.ui.layout_slide_bar.addWidget(self.ui.slider)
        
            
        # Button: Reverse:
        self.ui.btnReverse = QtWidgets.QPushButton("Reverse")
        self.ui.btnReverse.setCheckable(True) # set checkable to make it a toggle button
        self.ui.btnReverse.setMinimumHeight(25)
        self.ui.btnReverse.setMinimumWidth(30)
        self.ui.btnReverse.setStyleSheet("background-color : lightgrey") # setting default color of button to light-grey
        self.ui.btnReverse.clicked.connect(self.on_reverse_held)
        self.ui.layout_slide_bar.addWidget(self.ui.btnReverse)
        
        # Button: Jump Left:
        self.ui.btnLeft = QtWidgets.QPushButton("<-")
        self.ui.btnLeft.setMinimumHeight(25)
        self.ui.btnLeft.setMinimumWidth(30)
        self.ui.btnLeft.clicked.connect(self.on_jump_left)
        self.ui.layout_slide_bar.addWidget(self.ui.btnLeft)
        
        # Button: Jump Right:
        self.ui.btnRight = QtWidgets.QPushButton("->")
        self.ui.btnRight.setMinimumHeight(25)
        self.ui.btnRight.setMinimumWidth(30)
        self.ui.btnRight.clicked.connect(self.on_jump_right)
        self.ui.layout_slide_bar.addWidget(self.ui.btnRight)
        
        # Add the bottom bar:
        self.ui.layout.addWidget(self.ui.panel_bottom_bar, 1, 0, 1, 2) # Spans both columns (lays under the right_controls panel)
        

    
    # Called when the play/pause button is clicked:
    def play_pause(self):
        # THIS IS NOT TRUE: the play_pause_model uses inverted logic, so we negate the current state value to determine if is_playing
        is_playing = self.ui.play_pause_model.getState()
        # print(f'is_playing: {is_playing}')
        
        if (not is_playing) or self.slidebar_val == 1:
            if self.slidebar_val == 1:
                # self.ui.slider.setValue(0)
                pass
            # self.play_pause_model.setState(not is_playing)
            self.animationThread.start()

        else:
            # self.play_pause_model.setState(not is_playing)
            self.animationThread.terminate()
            
        
        # self.ui.play_pause_model.blockSignals(True)
        # self.ui.play_pause_model.setState(not is_playing)
        # self.ui.play_pause_model.blockSignals(False)
        
        # if self.ui.btn_slide_run.tag == "paused" or self.slidebar_val == 1:
        #     if self.slidebar_val == 1:
        #         self.ui.slider.setValue(0)
            
        #     self.ui.btn_slide_run.setText("||")
        #     self.ui.btn_slide_run.tag = "running"
        #     self.animationThread.start()

        # elif self.ui.btn_slide_run.tag == "running":
        #     self.ui.btn_slide_run.setText(">")
        #     self.ui.btn_slide_run.tag = "paused"
        #     self.animationThread.terminate()


    

    def on_jump_left(self):
        # Skip back some frames
        self.shift_animation_frame_val(-5)
        
    def on_jump_right(self):
        # Skip forward some frames
        self.shift_animation_frame_val(5)
        

    def on_reverse_held(self):
        # Change the direction of playback by changing the sign of the updating.
        # if button is checked    
        self.is_playback_reversed = self.ui.btnReverse.isChecked()
        if self.ui.btnReverse.isChecked():
            # setting background color to light-blue
            self.ui.btnReverse.setStyleSheet("background-color : lightblue")
  
        # if it is unchecked
        else:
            # set background color back to light-grey
            self.ui.btnReverse.setStyleSheet("background-color : lightgrey")
    
        
        
class RenderWindowControlsMixin:
    """ Provides GUI controls related to the active render window, such as the window duration, the zoom multiplier, etc. Relates to the left toolbar in the SpikesRasterWindow UI.
    
    Implemented by SpikeRasterBase
    # self.ui.spinBoxCurrentFrame.blockSignals(True)
    """
    
    WantsRenderWindowControls = False
    
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
        # verticalSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        # self.ui.layout_right_bar.addWidget(verticalSpacer)
        
        # Add the right controls bar:
        self.ui.layout.addWidget(self.ui.right_controls_panel, 0, 1, 2, 1) # Span both rows
        
        
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
        
        