from copy import deepcopy
import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
# import qdarkstyle

from pyphoplacecellanalysis.General.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem



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


class SliderRunner(QtCore.QThread):
    update_signal = QtCore.pyqtSignal()

    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        while(True):
            self.update_signal.emit()
            # time.sleep(.32) # 320ms
            time.sleep(0.05) # probably do a different form of rate limiting instead (like use SignalProxy)? Actually this might be okay because it's on a different thread.
            
                

class Spike3DRaster(NeuronIdentityAccessingMixin, SpikeRenderingBaseMixin, SpikesWindowOwningMixin, SpikesDataframeOwningMixin, QtWidgets.QWidget):
    """ Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Spike3DRaster import Spike3DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike3DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    temporal_mapping_changed = QtCore.pyqtSignal() # signal emitted when the mapping from the temporal window to the spatial layout is changed
    
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
    def overlay_text_lines(self):
        """The lines of text to be displayed in the overlay."""
        lines = []
        lines.append(f'active_time_window: {self.spikes_window.active_time_window}')
        lines.append(f"n_cells : {self.n_cells}")
        lines.append(f'active num spikes: {self.active_windowed_df.shape[0]}')
        lines.append(f'render_window_duration: {self.render_window_duration}')
        lines.append(f'animation_time_step: {self.animation_time_step}')
        lines.append(f'temporal_axis_length: {self.temporal_axis_length}')
        lines.append(f'temporal_zoom_factor: {self.temporal_zoom_factor}')
        return lines
    
    
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
        


    @property
    def axes_walls_z_height(self):
        """The axes_walls_z_height property."""
        return self._axes_walls_z_height
    
    
    @property
    def z_floor(self):
        """The offset of the floor in the z-axis."""
        return -10
    
    
    

    # @property
    # def cell_id_axis_length(self):
    #     """The cell_id_axis_length property."""
    #     return self._cell_id_axis_length





    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, **kwargs):
        super(Spike3DRaster, self).__init__(*args, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        self.params = VisualizationParameters('')
        
        self.slidebar_val = 0
        self._spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        self.params.spike_start_z = -10.0
        # self.spike_end_z = 0.1
        self.params.spike_end_z = -6.0
        self.params.side_bin_margins = 0.0 # space to sides of the first and last cell on the y-axis
        
        self.params.center_mode = 'zero_centered'
        
        # self.params.bin_position_mode = ''bin_center'
        self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
        
        # return 0.05 # each animation timestep is a fixed 50ms
        # return 0.03 # faster then 30fps
        self.params.animation_time_step = 0.03 
        
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
        
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        self.app = pg.mkQApp("Spike3DRaster")
        
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
        
        # build the UI components:
        self.buildUI()
        
        # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        


    def on_jump_left(self):
        # Skip back some frames
        self.shift_animation_frame_val(-5)
        
    def on_jump_right(self):
        # Skip forward some frames
        self.shift_animation_frame_val(5)
        
    def on_reverse_held(self):
        # Change the direction of playback by changing the sign of the updating.
        self.shift_animation_frame_val(5)
        
    
    
        
        
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
        
        ##### Main Raster Plot Content Top ##########
        self.ui.main_gl_widget = gl.GLViewWidget()
        # self.ui.main_gl_widget.show()
        self.ui.main_gl_widget.resize(1000,600)
        # self.ui.main_gl_widget.setWindowTitle('pyqtgraph: 3D Raster Spikes Plotting')
        self.ui.main_gl_widget.setCameraPosition(distance=40)
        self.ui.layout.addWidget(self.ui.main_gl_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        #### Build Graphics Objects #####
        self._buildGraphics(self.ui.main_gl_widget) # pass the GLViewWidget
        
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

        # Playback Slider:
        self.ui.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ui.slider.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        self.ui.slider.setRange(0, 100)
        self.ui.slider.setSingleStep(1)
        # self.ui.slider.setSingleStep(2)
        self.ui.slider.setValue(0)
        self.ui.slider.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.ui.slider.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.slider.setStyleSheet("background:transparent;")        
        # self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # sliderMoved vs valueChanged? vs sliderChange?
        self.ui.layout_slide_bar.addWidget(self.ui.slider)

        # Button: Play/Pause
        self.ui.btn_slide_run = QtWidgets.QPushButton(">")
        self.ui.btn_slide_run.setMinimumHeight(25)
        self.ui.btn_slide_run.setMinimumWidth(30)
        self.ui.btn_slide_run.tag = "paused"
        self.ui.btn_slide_run.clicked.connect(self.btn_slide_run_clicked)
        self.ui.layout_slide_bar.addWidget(self.ui.btn_slide_run)
        
        # Button: Reverse:
        self.ui.btnReverse = QtWidgets.QPushButton("Reverse")
        self.ui.btnReverse.setMinimumHeight(25)
        self.ui.btnReverse.setMinimumWidth(30)
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
        self.ui.right_controls_panel = QtWidgets.QWidget()
        self.ui.right_controls_panel.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding) # Expands to fill the vertical height, but occupy only the preferred width
        self.ui.right_controls_panel.setMaximumWidth(50.0)
        # Try to make the bottom widget bar transparent:
        self.ui.right_controls_panel.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.ui.right_controls_panel.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.right_controls_panel.setStyleSheet("background:transparent;")
        
        # Playback Slider Bottom Bar:
        self.ui.layout_right_bar = QtWidgets.QVBoxLayout()
        self.ui.layout_right_bar.setContentsMargins(6, 3, 4, 4)
        self.ui.right_controls_panel.setLayout(self.ui.layout_right_bar)

        # Playback Slider:
        self.ui.slider_right = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.ui.slider_right.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        self.ui.slider_right.setRange(0, 100)
        self.ui.slider_right.setSingleStep(1)
        # self.ui.slider_right.setSingleStep(2)
        self.ui.slider_right.setValue(0)
        self.ui.slider_right.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        self.ui.slider_right.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.slider_right.setStyleSheet("background:transparent;")        
        # self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # sliderMoved vs valueChanged? vs sliderChange?
        self.ui.layout_right_bar.addWidget(self.ui.slider_right)

        
        
        # addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        
        # Add the bottom bar:
        self.ui.layout.addWidget(self.ui.panel_slide_bar, 1, 0, 1, 2) # Spans both columns (lays under the right_controls panel)
        
        # Add the right controls bar:
        self.ui.layout.addWidget(self.ui.right_controls_panel, 0, 1, 2, 1) # Span both rows
         
        
        
        # Set the root (self) layout properties
        self.setLayout(self.ui.layout)
        self.resize(1920, 900)
        self.setWindowTitle('Spike3DRaster')
        # Connect window update signals
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        self.spikes_window.window_changed_signal.connect(self.on_window_changed)

        # Slider update thread:        
        self.sliderThread = SliderRunner()
        # self.sliderThread.update_signal.connect(self.increase_slider_val)
        self.sliderThread.update_signal.connect(self.increase_animation_frame_val)
        
        self.show()
      
    def _buildGraphics(self, w):
        # Add debugging widget:
        
        # Adds a helper widget that displays the x/y/z vector at the origin:
        if self.enable_debug_widgets:
            self.ui.ref_axes_indicator = GLDebugAxisItem()
            self.ui.ref_axes_indicator.setSize(x=15.0, y=10.0, z=5.0)
            w.addItem(self.ui.ref_axes_indicator)

        # The 2D viewport overlay that contains text:
        self.ui.viewport_overlay = GLViewportOverlayPainterItem()
        w.addItem(self.ui.viewport_overlay)
        # Update the additional display lines information on the overlay:
        self.ui.viewport_overlay.additional_overlay_text_lines = self.overlay_text_lines
                
        # Add axes planes:
        # X-plane:
        x_color = (255, 155, 155, 76.5)
        self.ui.gx = gl.GLGridItem(color=x_color) # 'x' plane, red
        self.ui.gx.rotate(90, 0, 1, 0)
        self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
        self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
        self.ui.gx.setSpacing(10.0, 1) 
        w.addItem(self.ui.gx)
        self.ui.x_txtitem = gl.GLTextItem(pos=(-self.half_temporal_axis_length, self.n_half_cells, 0.0), text='x', color=x_color) # The axis label text 
        w.addItem(self.ui.x_txtitem)

        # Y-plane:
        y_color = (155, 255, 155, 76.5)
        self.ui.gy = gl.GLGridItem(color=y_color) # 'y' plane, green
        self.ui.gy.rotate(90, 1, 0, 0)
        # gy.translate(0, -10, 0)
        self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
        self.ui.gy.setSize(self.temporal_axis_length, 20)
        self.ui.gy.setSpacing(1, 10.0) # unit along the y axis itself, only one subdivision along the z-axis
        w.addItem(self.ui.gy)
        self.ui.y_txtitem = gl.GLTextItem(pos=(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0), text='y', color=y_color) # The axis label text 
        w.addItem(self.ui.y_txtitem)
        
        # XY-plane (with normal in z-dir):
        z_color = (155, 155, 255, 76.5)
        self.ui.gz = gl.GLGridItem(color=z_color) # 'z' plane, blue
        self.ui.gz.translate(0, 0, self.z_floor) # Shift down by 10 units in the z-dir
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        self.ui.gz.setSpacing(20.0, 1)
        # gz.setSize(n_full_cell_grid, n_full_cell_grid)
        w.addItem(self.ui.gz)
        self.ui.z_txtitem = gl.GLTextItem(pos=(-self.half_temporal_axis_length, -self.n_half_cells, (self.z_floor + 0.5)), text='z', color=z_color)  # The axis label text 
        w.addItem(self.ui.z_txtitem)
        
        
        
        self.ui.gl_test_points = []
        md = gl.MeshData.sphere(rows=10, cols=20)
        m1 = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=False, drawEdges=True, edgeColor=(1,1,1,1))
        m1.translate(5, 0, 0)
        m1.setGLOptions('additive')
        w.addItem(m1)
        self.ui.gl_test_points.append(m1)
        
        
        # Custom 3D raster plot:
        
        # TODO: EFFICIENCY: For a potentially considerable speedup, could compute the "curr_x" values for all cells at once and add as a column to the dataframe since it only depends on the current window parameters (changes when window changes).
            ## OH, but the window changes every frame update (as that's what it means to animate the spikes as a function of time). Maybe not a big speedup.
        
        self.ui.gl_line_plots = [] # create an empty array for each GLLinePlotItem, of which there will be one for each unit.
        
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
        self._build_neuron_id_graphics(w, y)
        
        # Plot each unit one at a time:
        for i, cell_id in enumerate(self.unit_ids):
            curr_color = pg.mkColor((i, self.n_cells*1.3))
            curr_color.setAlphaF(0.5)
            # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
            
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id].copy() # is .copy() needed here since nothing is updated???
            # curr_unit_id = curr_cell_df['unit_id'].to_numpy() # this will map to the y position
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map 
            yi = y[i] # get the correct y-position for all spikes of this cell
            # print(f'cell_id: {cell_id}, yi: {yi}')
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
            curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
            
            # Z-positions:
            # z = curr_spike_t[np.arange(100)] # get the first 20 spikes for each
            spike_bottom_zs = np.full_like(curr_x, self.params.spike_start_z)
            spike_top_zs = np.full_like(curr_x, self.params.spike_end_z)
            curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
        
            # sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            # sp1.translate(5,5,0)
            # w.addItem(sp1)
            
            # Build lines:
            pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # pts = np.column_stack([x, np.full_like(x, yi), z]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor((cell_id,n*1.3)), width=(cell_id+1)/10., antialias=True)
            plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=1.0, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement

            # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
            # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
            
            w.addItem(plt)
            self.ui.gl_line_plots.append(plt)


    def _build_neuron_id_graphics(self, w, y_pos):
        # builds the text items to indicate the neuron ID for each neuron in the df.
        all_cell_ids = self.cell_ids
        self.ui.glCellIdTextItems = []
        for i, cell_id in enumerate(all_cell_ids):
            curr_color = pg.mkColor((i, self.n_cells*1.3))
            curr_color.setAlphaF(0.5)
            # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
            curr_id_txtitem = gl.GLTextItem(pos=(-self.half_temporal_axis_length, y_pos[i], (self.z_floor - 0.5)), text=f'{cell_id}', color=curr_color)
            w.addItem(curr_id_txtitem) # add to the current widget
            # add to the cell_ids array
            self.ui.glCellIdTextItems.append(curr_id_txtitem)
                    


    # def _build_axes_arrow_graphics(self, w):
        
    #     md = gl.MeshData.cylinder(rows=10, cols=20, radius=[1., 2.0], length=5.)
        
        
        
    ###################################
    #### EVENT HANDLERS
    ##################################
    
    @QtCore.pyqtSlot()
    def on_adjust_temporal_spatial_mapping(self):
        """ called when the spatio-temporal mapping property is changed.
        
        Should change whenever any of the following change:
            self.temporal_zoom_factor
            self.render_window_duration
            
        """
        # Adjust the three axes planes:
        self.ui.gx.resetTransform()
        self.ui.gx.rotate(90, 0, 1, 0)
        self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
        self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
        self.ui.x_txtitem.resetTransform()
        self.ui.x_txtitem.translate(-self.half_temporal_axis_length, self.n_half_cells, 0.0)
        
        self.ui.gy.resetTransform()
        self.ui.gy.rotate(90, 1, 0, 0)
        self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
        self.ui.gy.setSize(self.temporal_axis_length, 20)
        self.ui.y_txtitem.resetTransform()
        self.ui.y_txtitem.translate(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0)
        
        self.ui.gz.resetTransform()
        self.ui.gz.translate(0, 0, self.z_floor) # Shift down by 10 units in the z-dir
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        self.ui.z_txtitem.resetTransform()
        self.ui.z_txtitem.translate(-self.half_temporal_axis_length, -self.n_half_cells, (self.z_floor + -0.5))

        
        
    def keyPressEvent(self, e):
        """ called automatically when a keyboard key is pressed and this widget has focus. 
        TODO: doesn't actually work right now.
        """
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif e.key() == QtCore.Qt.Key_Backspace:
            print('TODO')
        elif e.key() == QtCore.Qt.Key_Left:
            self.shift_animation_frame_val(-1) # jump back one frame
            
        elif e.key() == QtCore.Qt.Key_Right:
            self.shift_animation_frame_val(1) # jump forward one frame
        else:
            pass
            

    def on_spikes_df_changed(self):
        """ changes:
            self.unit_ids
            self.n_full_cell_grid
        """
        debug_print=True
        if debug_print:
            print(f'Spike3DRaster.on_spikes_df_changed()')
        # TODO: these '.translate(...)' instructions might not be right if they're relative to the original transform. May need to translate back to by the inverse of the old value, and then do the fresh transform with the new value. Or compute the difference between the old and new.
        self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
        self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        self.rebuild_main_gl_line_plots_if_needed()
        

    def on_window_duration_changed(self):
        """ changes self.half_render_window_duration """
        print(f'Spike3DRaster.on_window_duration_changed()')
        self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
        self.ui.gy.setSize(self.temporal_axis_length, 20)
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        # update grids. on_window_changed should be triggered separately        
        
    def on_window_changed(self):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRaster.on_window_changed()')
        profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        profiler('Finished calling _update_plots()')
        
            
    def _update_plots(self):
        """ performance went:
        FROM:
            > Entering Spike3DRaster.on_window_changed
            Finished calling _update_plots(): 1179.6892 ms
            < Exiting Spike3DRaster.on_window_changed, total time: 1179.7600 ms

        TO:
            > Entering Spike3DRaster.on_window_changed
            Finished calling _update_plots(): 203.8840 ms
            < Exiting Spike3DRaster.on_window_changed, total time: 203.9544 ms

        Just by removing the lines that initialized the color. Conclusion is that pg.mkColor((cell_id, self.n_cells*1.3)) must be VERY slow.
    
        """
        if self.enable_debug_print:
            print(f'Spike3DRaster._update_plots()')
        assert (len(self.ui.gl_line_plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.gl_line_plots): {len(self.ui.gl_line_plots)} and self.n_cells: {self.n_cells}!"
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
        # Plot each unit one at a time:
        for i, cell_id in enumerate(self.unit_ids):    
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id]
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map
            # efficiently get curr_spike_t by filtering for unit and column at the same time
            # curr_spike_t = self.active_windowed_df.loc[self.active_windowed_df.spikes.time_variable_name, (self.active_windowed_df['unit_id']==cell_id)].values # .to_numpy()
            
            curr_unit_n_spikes = len(curr_spike_t)
            
            yi = y[i] # get the correct y-position for all spikes of this cell
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_render_window_duration, +self.half_render_window_duration))
            curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
            # curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
            curr_paired_x = curr_x.repeat(2)
            
            # Z-positions:
            # spike_bottom_zs = np.full_like(curr_x, self.params.spike_start_z)
            # spike_top_zs = np.full_like(curr_x, self.params.spike_end_z)
            # curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
            curr_paired_spike_zs = np.squeeze(np.tile(np.array([self.params.spike_start_z, self.params.spike_end_z]), curr_unit_n_spikes)) # repeat pair of z values once for each spike
        
            # Build lines:
            pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
            self.ui.gl_line_plots[i].setData(pos=pts, mode='lines') # update the current data
            
            # self.ui.main_gl_widget.addItem(plt)
            # self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
            
    
        # Update the additional display lines information on the overlay:
        self.ui.viewport_overlay.additional_overlay_text_lines = self.overlay_text_lines
        
        
            
    def rebuild_main_gl_line_plots_if_needed(self, debug_print=True):
        """ adds or removes GLLinePlotItems to self.ui.gl_line_plots based on the current number of cells. """
        n_extant_plts = len(self.ui.gl_line_plots)
        if (n_extant_plts < self.n_cells):
            # need to create new plots for the difference
            if debug_print:
                print(f'!! Spike3DRaster.rebuild_main_gl_line_plots_if_needed(): building additional plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
            for new_unit_i in np.arange(n_extant_plts-1, self.n_cells, 1):
                cell_id = self.unit_ids[new_unit_i]
                # curr_color = pg.mkColor((cell_id, self.n_cells*1.3))
                # curr_color.setAlphaF(0.5)
                curr_color = self.params.neuron_qcolors[cell_id] # get the pre-build color
                plt = gl.GLLinePlotItem(pos=[], color=curr_color, width=1.0, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
                # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
                # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
                self.ui.main_gl_widget.addItem(plt)
                self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
                
        elif (n_extant_plts > self.n_cells):
            # excess plots, need to remove (or at least hide) them:              
            if debug_print:
                print(f'!! Spike3DRaster.rebuild_main_gl_line_plots_if_needed(): removing excess plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
            for extra_unit_i in np.arange(n_extant_plts, self.n_cells, 1):
                plt = self.ui.gl_line_plots[extra_unit_i] # get the unit to be removed 
                self.ui.main_gl_widget.removeItem(plt)
            # remove from the array
            del self.ui.gl_line_plots[n_extant_plts:] # from n_extant_plts up to the end of the list
        else:
            return # the correct number of items are already in the list
        
        assert (len(self.ui.gl_line_plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.gl_line_plots): {len(self.ui.gl_line_plots)} and self.n_cells: {self.n_cells}!"

    def _compute_window_transform(self, relative_offset):
        """ computes the transform from 0.0-1.0 as the slider would provide to the offset given the current information. """
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        total_spikes_df_duration = latest_t - earliest_t # get the duration of the entire spikes df
        render_window_offset = (total_spikes_df_duration * relative_offset) + earliest_t
        return render_window_offset
    
    
    
    
    
    def btn_slide_run_clicked(self):
        if self.ui.btn_slide_run.tag == "paused" or self.slidebar_val == 1:
            if self.slidebar_val == 1:
                self.ui.slider.setValue(0)
            
            self.ui.btn_slide_run.setText("||")
            self.ui.btn_slide_run.tag = "running"
            self.sliderThread.start()

        elif self.ui.btn_slide_run.tag == "running":
            self.ui.btn_slide_run.setText(">")
            self.ui.btn_slide_run.tag = "paused"
            self.sliderThread.terminate()

    def increase_slider_val(self):
        slider_val = self.ui.slider.value() # integer value between 0-100
        if self.enable_debug_print:
            print(f'Spike3DRaster.increase_slider_val(): slider_val: {slider_val}')
        if slider_val < 100:
            self.ui.slider.setValue(slider_val + 1)
        else:
            print("thread ended..")
            self.ui.btn_slide_run.setText(">")
            self.ui.btn_slide_run.tag = "paused"
            self.sliderThread.terminate()

    def slider_val_changed(self, val):
        self.slidebar_val = val / 100
        # Gets the transform from relative (0.0 - 1.0) to absolute timestamp offset
        curr_t = self._compute_window_transform(self.slidebar_val)
        
        if self.enable_debug_print:
            print(f'Spike3DRaster.slider_val_changed(): self.slidebar_val: {self.slidebar_val}, curr_t: {curr_t}')
            print(f'BEFORE: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
         # set the start time which will trigger the update cascade and result in on_window_changed(...) being called
        self.spikes_window.update_window_start(curr_t)
        if self.enable_debug_print:
            print(f'AFTER: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    
        
    def increase_animation_frame_val(self):
        self.shift_animation_frame_val(1)
        
    def shift_animation_frame_val(self, shift_frames: int):
        next_start_timestamp = self.spikes_window.active_window_start_time + (self.animation_time_step * float(shift_frames))
        self.spikes_window.update_window_start(next_start_timestamp)
        # TODO: doesn't update the slider or interact with the slider in any way.
        
        
    #### from pyqtgraph_animated3Dplot_pairedLines's animation style ###:
    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
            
    def set_plotdata(self, name, points, color, width):
        # self.traces in the original
        self.ui.gl_line_plots[name].setData(pos=points, color=color, width=width, mode='lines', antialias=True)
        
    def update(self):
        self._update_plots()
        self.shift_animation_frame_val(1)
        
    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        # timer.start(20)
        timer.start(50)
        self.start()
        
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

# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike3DRaster()
#     v.animation()
