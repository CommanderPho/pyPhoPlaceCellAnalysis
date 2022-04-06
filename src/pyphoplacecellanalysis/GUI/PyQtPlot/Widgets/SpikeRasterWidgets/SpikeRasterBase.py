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

from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


# import qdarkstyle

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial, DataSeriesToSpatialTransformingMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLGraphicsItems.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLGraphicsItems.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderWindowControlsMixin import RenderWindowControlsMixin, RenderPlaybackControlsMixin

from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController


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



# Note that in addition to the above-mentioned mapping, there's an additional mapping that must be performed due to 'temporal_zoom_factor', a visualization property belonging to the RasterPlot class.


"""

def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug



class RenderPlots(PrettyPrintable, SimplePrintable, metaclass=OrderedMeta):
    def __init__(self, name) -> None:
        # super(RenderEpochs, self).__init__(**kwargs)
        self.name = name
        # self.__dict__ = (self.__dict__ | kwargs)
        
    # def __init__(self, name, **kwargs) -> None:
    #     # super(VisualizationParameters, self).__init__(**kwargs)
    #     self.name = name
    #     # self.__dict__ = (self.__dict__ | kwargs)
    
    
class UnitSortableMixin:
    """ Implementor allows changing the sort order of the units (neurons) displayed by setting indicies directly via the self.unit_sort_order property.
    
    Requires:
        self._unit_sort_order
        self.n_cells
    """
    unit_sort_order_changed_signal = QtCore.pyqtSignal(object) # Called when the window is closing. 
    
    @property
    def unit_sort_order(self):
        """The unit_sort_order property.
            Requires self._unit_sort_order to be a ndarray of indicies with the same length as self.unit_ids
        """
        return self._unit_sort_order
    @unit_sort_order.setter
    def unit_sort_order(self, value):
        assert len(value) == self.n_cells, f"len(self._unit_sort_order): {len(self._unit_sort_order)} must equal self.n_cells: {self.n_cells} but it does not!"
        self._unit_sort_order = value
        # Emit the sort order changed signal:
        self.unit_sort_order_changed_signal.emit(self._unit_sort_order)
        
    

class SpikeRasterBase(UnitSortableMixin, DataSeriesToSpatialTransformingMixin, NeuronIdentityAccessingMixin, SpikeRenderingBaseMixin, SpikesWindowOwningMixin, SpikesDataframeOwningMixin, TimeWindowPlaybackPropertiesMixin, RenderPlaybackControlsMixin, RenderWindowControlsMixin, QtWidgets.QWidget):
    """ Displays a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
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
                                                    f'playback_rate_multiplier: {self.playback_rate_multiplier}'
                                                    ]
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']
        return lines_dict

    @property
    def temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_zoom_factor * self.render_window_duration
    @property
    def half_temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_axis_length / 2.0
        
    # TimeWindowPlaybackPropertiesMixin requirement:
    @property
    def animation_active_time_window(self):
        """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate."""
        return self._spikes_window
    
    
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
        

    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, neuron_sort_order=None, **kwargs):
        super(SpikeRasterBase, self).__init__(*args, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        self.params = VisualizationParameters('')
        
        self.slidebar_val = 0
        self._spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        
        # Config
        self.params.wantsRenderWindowControls = self.WantsRenderWindowControls
        self.params.wantsPlaybackControls = self.WantsPlaybackControls
        
        # self.params.wantsRenderWindowControls = SpikeRasterBase.WantsRenderWindowControls
        # self.params.wantsPlaybackControls = SpikeRasterBase.WantsPlaybackControls
        
        
        
        
        # self._playback_update_frequency = SpikeRasterBase.PlaybackUpdateFrequency
        # self.speedBurstPlaybackRate = SpikeRasterBase.SpeedBurstPlaybackRate
        # self.params.is_speed_burst_mode_active = False
        # self.params.is_playback_reversed = False
        # self.params.animation_time_step = 0.04 
        
        
        
        self.params.side_bin_margins = 0.0 # space to sides of the first and last cell on the y-axis
        
        self.params.center_mode = 'zero_centered'
        # self.params.bin_position_mode = ''bin_center'
        self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        # self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
        self.params.temporal_zoom_factor = 40.0 / float(self.spikes_window.timeWindow.window_duration) 
        
    
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        
        
        # Neurons and sort-orders:
        if neuron_sort_order is None:
            neuron_sort_order = np.arange(len(self.unit_ids)) # default sort order is sorted by unit_ids
        self._unit_sort_order = neuron_sort_order
        assert len(self._unit_sort_order) == len(self.unit_ids), f"len(self._unit_sort_order): {len(self._unit_sort_order)} must equal len(self.unit_ids): {len(self.unit_ids)} but it does not!"
        
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
        
        
        # make root container for plots
        self.plots = RenderPlots('')
        
        self.playback_controller = TimeWindowPlaybackController()
        # Setup the animation playback object for the time window:
        # self.playback_controller = TimeWindowPlaybackController()
        # self.playback_controller.setup(self._spikes_window)
        self.playback_controller.setup(self) # pass self to have properties set
        
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
        
        if self.params.wantsPlaybackControls:
            # Build the bottom playback controls bar:
            self.setup_render_playback_controls()

        if self.params.wantsRenderWindowControls:
            # Build the right controls bar:
            self.setup_render_window_controls() # creates self.ui.right_controls_panel

                
        # addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
         
        # Set the root (self) layout properties
        self.setLayout(self.ui.layout)
        self.resize(1920, 900)
        self.setWindowTitle('SpikeRasterBase')
        # Connect window update signals
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        # self.spikes_window.window_changed_signal.connect(self.on_window_changed)
        self.spikes_window.window_updated_signal.connect(self.on_window_changed)

        # self.show()
      
      
      
    ###################################
    #### EVENT HANDLERS
    ##################################

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


    @QtCore.pyqtSlot()
    def on_spikes_df_changed(self):
        """ changes:
            self.unit_ids
            self.n_full_cell_grid
        """
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_spikes_df_changed()')
        
    @QtCore.pyqtSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        print(f'SpikeRasterBase.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')


    @QtCore.pyqtSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        profiler('Finished calling _update_plots()')
        
            
    def _update_plots(self):
        """ Implementor must override! """
        raise NotImplementedError
        