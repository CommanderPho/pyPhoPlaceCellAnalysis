from copy import deepcopy
import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import qtawesome as qta

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider

# import qdarkstyle

from pyphoplacecellanalysis.General.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem


""" 
FPS     Milliseconds Per Frame
20      50.00 ms
25      40.00 ms
30      33.34 ms
60      16.67 ms
"""

""" For threading info see:
    https://stackoverflow.com/questions/41526832/pyqt5-qthread-signal-not-working-gui-freeze

    For PyOpenGL Requirements, see here: https://stackoverflow.com/questions/57971352/pip-install-pyopengl-accelerate-doesnt-work-on-windows-10-python-3-7 and below.
    I found unofficial Windows builds here:
    https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl

    I downloaded PyOpenGL-3.1.3b2-cp37-cp37m-win_amd64.whl and PyOpenGL_accelerate-3.1.3b2-cp37-cp37m-win_amd64.whl. Next, I navigate to my Downloads folder in a Windows terminal and start the installation:

"""


""" Windowed Spiking Datasource Features

Transforming the events into either 2D or 3D representations for visualization should NOT be part of this class' function.
Separate 2D and 3D event visualization functions should be made to transform events from this class into appropriate point/datastructure representations for the visualization framework being used.

# Local window properties
Get (window_start, window_end) times

# Global data properties
Get (earliest_datapoint_time, latest_datapoint_time) # globally, for the entire timeseries



"""

def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug


class UpdateRunner(QtCore.QThread):
    update_signal = QtCore.pyqtSignal()

    def __init__(self, update_frequency=0.04):
        self.update_frequency = update_frequency
        QtCore.QThread.__init__(self)

    def run(self):
        while(True):
            self.update_signal.emit()
            # probably do a different form of rate limiting instead (like use SignalProxy)? Actually this might be okay because it's on a different thread.
            time.sleep(self.update_frequency) # 40.0 ms = 25 FPS
            # time.sleep(.32) # 320ms
            # time.sleep(0.05) # probably do a different form of rate limiting instead (like use SignalProxy)? Actually this might be okay because it's on a different thread.
            
                

class SpikeRasterBase(NeuronIdentityAccessingMixin, SpikeRenderingBaseMixin, SpikesWindowOwningMixin, SpikesDataframeOwningMixin, QtWidgets.QWidget):
    """ Displays a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Spike3DRaster import Spike3DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike3DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    temporal_mapping_changed = QtCore.pyqtSignal() # signal emitted when the mapping from the temporal window to the spatial layout is changed
    close_signal = QtCore.pyqtSignal() # Called when the window is closing. 
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
    
    @property
    def unit_ids(self):
        """The unit_ids from the whole df (not just the current window)"""
        return np.unique(self.spikes_window.df['unit_id'].to_numpy())
    
    @property
    def n_cells(self):
        """The number_units property."""
        return len(self.unit_ids)
    @property
    def n_half_cells(self):
        """ """
        return np.ceil(float(self.n_cells)/2.0)
    @property
    def n_full_cell_grid(self):
        """ """
        return 2.0 * self.n_half_cells # could be one more than n


    @property
    def temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_zoom_factor * self.render_window_duration
    @property
    def half_temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_axis_length / 2.0
    
    @property
    def animation_time_step(self):
        """ How much to step forward in time at each frame of animation. """
        return self.params.animation_time_step
    @animation_time_step.setter
    def animation_time_step(self, value):
        self.params.animation_time_step = value

    # from NeuronIdentityAccessingMixin
    @property
    def neuron_ids(self):
        """ an alias for self.cell_ids required for NeuronIdentityAccessingMixin """
        return self.cell_ids

    @property
    def cell_ids(self):
        """ e.g. the list of valid cell_ids (unique aclu values) """
        # return self.unit_ids
        return np.unique(self.spikes_window.df['aclu'].to_numpy()) 
    
    
    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""
        
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL']
        lines_dict[af.AlignTop | af.AlignRight] = ['TR', 
                                                   f"n_cells : {self.n_cells}",
                                                   f'render_window_duration: {self.render_window_duration}',
                                                   f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'playback_rate_multiplier: {self.playback_rate_multiplier}']
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']
        return lines_dict
    
    
    ######  Get/Set Properties ######:
    @property
    def temporal_zoom_factor(self):
        """The time dilation factor that maps spikes in the current window to y-positions along the time axis multiplicatively.
            Increasing this factor will result in a more spatially expanded time axis while leaving the visible window unchanged.
        """
        return self.params.temporal_zoom_factor
    @temporal_zoom_factor.setter
    def temporal_zoom_factor(self, value):
        self.params.temporal_zoom_factor = value
        self.temporal_mapping_changed.emit()
        

    ## STATE PROPERTIES
    @property
    def is_playback_reversed(self):
        """The is_playback_reversed property."""
        return self.params.is_playback_reversed
    @is_playback_reversed.setter
    def is_playback_reversed(self, value):
        self.params.is_playback_reversed = value
        
    @property
    def animation_playback_direction_multiplier(self):
        """The animation_reverse_multiplier property."""
        if self.params.is_playback_reversed:
            return -1.0
        else:
            return 1.0


    @property
    def playback_update_frequency(self):
        """The rate at which the separate animation thread attempts to update the interface. ReadOnly."""
        return self._playback_update_frequency

    @property
    def playback_rate_multiplier(self):
        """ 1x playback (real-time) occurs when self.playback_update_frequency == self.animation_time_step. 
            if self.animation_time_step = 2.0 * self.playback_update_frequency => for each update the window will step double the time_step forward in time than it would be default, meaning a 2.0x playback_rate_multiplier.
        """
        return (self.animation_time_step / self.playback_update_frequency)
    @playback_rate_multiplier.setter
    def playback_rate_multiplier(self, value):
        """ since self.playback_update_frequency is fixed, only self.animation_time_step can be adjusted to set the playback_rate_multiplier. """
        desired_playback_rate_multiplier = value
        self.animation_time_step = self.playback_update_frequency * desired_playback_rate_multiplier


    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, **kwargs):
        super(SpikeRasterBase, self).__init__(*args, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        self.params = VisualizationParameters('')
        
        self.slidebar_val = 0
        self._spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        
        # Config
        self._playback_update_frequency = SpikeRasterBase.PlaybackUpdateFrequency
        self.is_speed_burst_mode_active = False
        self.speedBurstPlaybackRate = SpikeRasterBase.SpeedBurstPlaybackRate
        
        self.params.is_playback_reversed = False
        self.params.side_bin_margins = 0.0 # space to sides of the first and last cell on the y-axis
        
        self.params.center_mode = 'zero_centered'
        # self.params.bin_position_mode = ''bin_center'
        self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
        self.params.animation_time_step = 0.04 
        
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        
        if neuron_colors is None:
            # neuron_colors = [pg.mkColor((i, self.n_cells*1.3)) for i, cell_id in enumerate(self.unit_ids)]
            neuron_colors = []
            for i, cell_id in enumerate(self.unit_ids):
                curr_color = pg.mkColor((i, self.n_cells*1.3))
                curr_color.setAlphaF(0.5)
                neuron_colors.append(curr_color)
    
        self.params.neuron_qcolors = deepcopy(neuron_colors)

        # allocate new neuron_colors array:
        self.params.neuron_colors = np.zeros((4, self.n_cells))
        for i, curr_qcolor in enumerate(self.params.neuron_qcolors):
            curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            self.params.neuron_colors[:, i] = curr_color[:]
            # self.params.neuron_colors[:, i] = curr_color[:]
            
        # self.params.neuron_colors = [self.params.neuron_qcolors[i].getRgbF() for i, cell_id in enumerate(self.unit_ids)] 
        # self.params.neuron_colors = deepcopy(neuron_colors)
        self.params.neuron_colors_hex = None
        
        # spike_raster_plt.params.neuron_colors[0].getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        
        # get hex colors:
        #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000' 
        # self.params.neuron_colors_hex = [to_hex(self.params.neuron_colors[:,i], keep_alpha=False) for i, cell_id in enumerate(self.unit_ids)]
        self.params.neuron_colors_hex = [self.params.neuron_qcolors[i].name(QtGui.QColor.HexRgb) for i, cell_id in enumerate(self.unit_ids)] 
        
        # included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        
        # self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        # self.spikes_df['cell_idx'] = self.spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
        
        self.setup()
        
        # build the UI components:
        self.buildUI()
        
        # Setup Signals:
        # self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)



    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        raise NotImplementedError # Inheriting classes must override setup to perform particular setup
    
        self.app = pg.mkQApp("SpikeRasterBase")
        
        # Configure pyqtgraph config:
        try:
            import OpenGL
            pg.setConfigOption('useOpenGL', True)
            pg.setConfigOption('enableExperimental', True)
        except Exception as e:
            print(f"Enabling OpenGL failed with {e}. Will result in slow rendering. Try installing PyOpenGL.")
            
        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', "#1B1B1B")
        pg.setConfigOption('foreground', "#727272")
        
        
    def _buildGraphics(self):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        raise NotImplementedError
    
    
    def buildUI(self):
        """ for QGridLayout
            addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        """
        self.ui = PhoUIContainer()
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setContentsMargins(0, 0, 0, 0)
        self.ui.layout.setVerticalSpacing(0)
        self.ui.layout.setHorizontalSpacing(0)
        self.setStyleSheet("background : #1B1B1B; color : #727272")
        
        #### Build Graphics Objects #####
        self._buildGraphics()
        
        ####################################################
        ####  Controls Bar Bottom #######
        ####    Slide Bar Bottom #######
        self.ui.panel_slide_bar = QtWidgets.QWidget()
        self.ui.panel_slide_bar.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.ui.panel_slide_bar.setMaximumHeight(50.0) # maximum height
        
        # Try to make the bottom widget bar transparent:
        self.ui.panel_slide_bar.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.ui.panel_slide_bar.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.panel_slide_bar.setStyleSheet("background:transparent;")
        
        # Playback Slider Bottom Bar:
        self.ui.layout_slide_bar = QtWidgets.QHBoxLayout()
        self.ui.layout_slide_bar.setContentsMargins(6, 3, 4, 4)
        self.ui.panel_slide_bar.setLayout(self.ui.layout_slide_bar)

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
        # self.ui.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.ui.slider = HighlightedJumpSlider(QtCore.Qt.Horizontal)
        self.ui.slider = HighlightedJumpSlider()
        self.ui.slider.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        self.ui.slider.setRange(0, 100)
        self.ui.slider.setSingleStep(1)
        self.ui.slider.setValue(0)
        # self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # sliderMoved vs valueChanged? vs sliderChange?
        self.ui.layout_slide_bar.addWidget(self.ui.slider)
        
            
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
        self.ui.slider_right = QtWidgets.QSlider(QtCore.Qt.Vertical)
        # self.ui.slider_right.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.ui.slider_right.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        self.ui.slider_right.setRange(0, 100)
        self.ui.slider_right.setSingleStep(1)
        # self.ui.slider_right.setSingleStep(2)
        self.ui.slider_right.setValue(0)
        # self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # sliderMoved vs valueChanged? vs sliderChange?
        self.ui.layout_right_bar.addWidget(self.ui.slider_right)
        
        
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
        
        # addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        
        # Add the bottom bar:
        self.ui.layout.addWidget(self.ui.panel_slide_bar, 1, 0, 1, 2) # Spans both columns (lays under the right_controls panel)
        
        # Add the right controls bar:
        self.ui.layout.addWidget(self.ui.right_controls_panel, 0, 1, 2, 1) # Span both rows
         
        # Set the root (self) layout properties
        self.setLayout(self.ui.layout)
        self.resize(1920, 900)
        self.setWindowTitle('SpikeRasterBase')
        # Connect window update signals
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        self.spikes_window.window_changed_signal.connect(self.on_window_changed)

        # Slider update thread:        
        self.animationThread = UpdateRunner(update_frequency=self.playback_update_frequency)
        # self.sliderThread.update_signal.connect(self.increase_slider_val)
        self.animationThread.update_signal.connect(self.increase_animation_frame_val)
        
        # self.show()
      
      
      
    ###################################
    #### EVENT HANDLERS
    ##################################
    # @QtCore.pyqtSlot()
    # def on_adjust_temporal_spatial_mapping(self):
    #     """ called when the spatio-temporal mapping property is changed.
        
    #     Should change whenever any of the following change:
    #         self.temporal_zoom_factor
    #         self.render_window_duration
            
    #     """
    #     # Adjust the three axes planes:
    #     self.ui.gx.resetTransform()
    #     self.ui.gx.rotate(90, 0, 1, 0)
    #     self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
    #     self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
    #     # self.ui.x_txtitem.resetTransform()
    #     # self.ui.x_txtitem.translate(-self.half_temporal_axis_length, self.n_half_cells, 0.0)
    #     self.ui.x_txtitem.setData(pos=(-self.half_temporal_axis_length, self.n_half_cells, 0.0))
        
    #     self.ui.gy.resetTransform()
    #     self.ui.gy.rotate(90, 1, 0, 0)
    #     self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
    #     self.ui.gy.setSize(self.temporal_axis_length, 20)
    #     # self.ui.y_txtitem.resetTransform()
    #     # self.ui.y_txtitem.translate(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0)
    #     self.ui.y_txtitem.setData(pos=(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0))
        
    #     self.ui.gz.resetTransform()
    #     self.ui.gz.translate(0, 0, self.z_floor) # Shift down by 10 units in the z-dir
    #     self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
    #     # self.ui.z_txtitem.resetTransform()
    #     # self.ui.z_txtitem.translate(-self.half_temporal_axis_length, -self.n_half_cells, (self.z_floor + -0.5))
    #     self.ui.z_txtitem.setData(pos=(-self.half_temporal_axis_length, -self.n_half_cells, (self.z_floor + -0.5)))
        
    #     self._update_neuron_id_graphics()


    # Input Handelers:        
    def keyPressEvent(self, e):
        """ called automatically when a keyboard key is pressed and this widget has focus. 
        TODO: doesn't actually work right now.
        """
        print(f'keyPressEvent(e.key(): {e.key()})')
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif e.key() == QtCore.Qt.Key_Backspace:
            print('TODO')
        elif e.key() == QtCore.Qt.Key_Left:
            self.shift_animation_frame_val(-1) # jump back one frame
            
        elif e.key() == QtCore.Qt.Key_Right:
            self.shift_animation_frame_val(1) # jump forward one frame
            
        elif e.key() == QtCore.Qt.Key_Space:
            self.play_pause()
        elif e.key() == QtCore.Qt.Key_P:
            self.toggle_speed_burst()
            
        else:
            pass
            
            
    # def key_handler(self, event):
    #     print("MainVideoPlayerWindow key handler: {0}".format(str(event.key())))
    #     if event.key() == QtCore.Qt.Key_Escape and self.is_full_screen:
    #         self.toggle_full_screen()
    #     if event.key() == QtCore.Qt.Key_F:
    #         self.toggle_full_screen()
    #     if event.key() == QtCore.Qt.Key_Space:
    #         self.play_pause()
    #     if event.key() == QtCore.Qt.Key_P:
    #         self.toggle_speed_burst()


    def wheel_handler(self, event):
        print(f'wheel_handler(event.angleDelta().y(): {event.angleDelta().y()})')
        # self.modify_volume(1 if event.angleDelta().y() > 0 else -1)
        # self.set_media_position(1 if event.angleDelta().y() > 0 else -1)



    def on_spikes_df_changed(self):
        """ changes:
            self.unit_ids
            self.n_full_cell_grid
        """
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_spikes_df_changed()')
        

    def on_window_duration_changed(self):
        """ changes self.half_render_window_duration """
        print(f'SpikeRasterBase.on_window_duration_changed()')


        
    def on_window_changed(self):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_window_changed()')
        profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        profiler('Finished calling _update_plots()')
        
            
    def _update_plots(self):
        """ Implementor must override! """
        raise NotImplementedError
        

    
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
    
    # Called when the play/pause button is clicked:
    def play_pause(self):
        # THIS IS NOT TRUE: the play_pause_model uses inverted logic, so we negate the current state value to determine if is_playing
        is_playing = self.ui.play_pause_model.getState()
        # print(f'is_playing: {is_playing}')
        
        if (not is_playing) or self.slidebar_val == 1:
            if self.slidebar_val == 1:
                self.ui.slider.setValue(0)
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
    
    
    
    def shift_animation_frame_val(self, shift_frames: int):
        next_start_timestamp = self.spikes_window.active_window_start_time + (self.animation_playback_direction_multiplier * self.animation_time_step * float(shift_frames))
        self.spikes_window.update_window_start(next_start_timestamp)
        # TODO: doesn't update the slider or interact with the slider in any way.
        
    # Called from SliderRunner's thread when it emits the update_signal:    
    def increase_animation_frame_val(self):
        self.shift_animation_frame_val(1)
        
        
        
    # Slider Functions:
    # def _compute_window_transform(self, relative_offset):
    #     """ computes the transform from 0.0-1.0 as the slider would provide to the offset given the current information. """
    #     earliest_t, latest_t = self.spikes_window.total_df_start_end_times
    #     total_spikes_df_duration = latest_t - earliest_t # get the duration of the entire spikes df
    #     render_window_offset = (total_spikes_df_duration * relative_offset) + earliest_t
    #     return render_window_offset
    
    # def increase_slider_val(self):
    #     slider_val = self.ui.slider.value() # integer value between 0-100
    #     if self.enable_debug_print:
    #         print(f'SpikeRasterBase.increase_slider_val(): slider_val: {slider_val}')
    #     if slider_val < 100:
    #         self.ui.slider.setValue(slider_val + 1)
    #     else:
    #         print("thread ended..")
    #         self.ui.btn_slide_run.setText(">")
    #         self.ui.btn_slide_run.tag = "paused"
    #         self.sliderThread.terminate()

    # def slider_val_changed(self, val):
    #     self.slidebar_val = val / 100
    #     # Gets the transform from relative (0.0 - 1.0) to absolute timestamp offset
    #     curr_t = self._compute_window_transform(self.slidebar_val)
        
    #     if self.enable_debug_print:
    #         print(f'SpikeRasterBase.slider_val_changed(): self.slidebar_val: {self.slidebar_val}, curr_t: {curr_t}')
    #         print(f'BEFORE: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    #      # set the start time which will trigger the update cascade and result in on_window_changed(...) being called
    #     self.spikes_window.update_window_start(curr_t)
    #     if self.enable_debug_print:
    #         print(f'AFTER: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    
    

        
        
        
    # #### from pyqtgraph_animated3Dplot_pairedLines's animation style ###:
    # def start(self):
    #     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #         QtGui.QApplication.instance().exec_()
      
            
    # # def set_plotdata(self, name, points, color, width):
    # #     # self.traces in the original
    # #     self.ui.gl_line_plots[name].setData(pos=points, color=color, width=width, mode='lines', antialias=True)
        
    # def update(self):
    #     """ called on timer timeout """
    #     self._update_plots()
    #     self.shift_animation_frame_val(1)
        
    # def animation(self):
    #     timer = QtCore.QTimer()
    #     timer.timeout.connect(self.update)
    #     # timer.start(20)
    #     timer.start(50)
    #     self.start()
        
    # def computeTransform(self, x, y, t = None):
    #     if t == None:
    #         v1_x = (1 * (1 - self.slidebar_val)) + (self.v1_x * self.slidebar_val)
    #         v1_y = (0 * (1 - self.slidebar_val)) + (self.v1_y * self.slidebar_val)

    #         v2_y = (1 * (1 - self.slidebar_val)) + (self.v2_y * self.slidebar_val)
    #         v2_x = (0 * (1 - self.slidebar_val)) + (self.v2_x * self.slidebar_val)
    #     else:
    #         v1_x = self.v1_x
    #         v1_y = self.v1_y
    #         v2_x = self.v2_x
    #         v2_y = self.v2_y
    #     return ((v1_x * x) + (v2_x * y), (v1_y * x) + (v2_y * y))






    # Speed Burst Features:
    def toggle_speed_burst(self):
        curr_is_speed_burst_enabled = self.is_speed_burst_mode_active
        updated_speed_burst_enabled = (not curr_is_speed_burst_enabled)
        if (updated_speed_burst_enabled):
            self.engage_speed_burst()
        else:
            self.disengage_speed_burst()

    # Engages a temporary speed burst 
    def engage_speed_burst(self):
        print("Speed burst enabled!")
        self.is_speed_burst_mode_active = True
        # Set the playback speed temporarily to the burst speed
        self.media_player.set_rate(self.speedBurstPlaybackRate)

        self.ui.toolButton_SpeedBurstEnabled.setEnabled(True)
        self.ui.doubleSpinBoxPlaybackSpeed.setEnabled(False)
        self.ui.button_slow_down.setEnabled(False)
        self.ui.button_speed_up.setEnabled(False)
        
    def disengage_speed_burst(self):
        print("Speed burst disabled!")
        self.is_speed_burst_mode_active = False
        # restore the user specified playback speed
        self.media_player.set_rate(self.ui.doubleSpinBoxPlaybackSpeed.value)

        self.ui.toolButton_SpeedBurstEnabled.setEnabled(False)
        self.ui.doubleSpinBoxPlaybackSpeed.setEnabled(True)
        self.ui.button_slow_down.setEnabled(True)
        self.ui.button_speed_up.setEnabled(True)







# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = SpikeRasterBase()
#     v.animation()
