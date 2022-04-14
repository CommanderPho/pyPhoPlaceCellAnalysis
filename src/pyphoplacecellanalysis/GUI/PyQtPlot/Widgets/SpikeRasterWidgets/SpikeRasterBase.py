from copy import deepcopy
import time
import sys
from typing import OrderedDict
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import qtawesome as qta
# import qdarkstyle

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters, RenderPlots
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial, DataSeriesToSpatialTransformingMixin

from pyphoplacecellanalysis.GUI.Qt.Mixins.RenderWindowControlsMixin import RenderWindowControlsMixin, RenderPlaybackControlsMixin

from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers


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


""" Initialization/Setup Call Order:

In a subclasses __init__(...) function, it immediately calls its superclass (SpikeRasterBase)'s __init__(...) function. The superclass's __init__ calls the setup functions in the order listed below. The majority of the customization in the subclass should be done by overriding these methods, not doing special stuff elsewhere.


__init__(...):
    self.setup()
        # In this function any special self.params values that this class needs should be set to defaults or passed-in values.
    
    # build the UI components:
    self.buildUI()
        # In buildUI() you're free to use anything setup in the self.setup() function, which has now finished executing completely.
        self._buildGraphics()



self._update_plots()


"""



def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug



    
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
    
    # Application/Window Configuration Options:
    applicationName = 'SpikeRasterBase'
    windowName = 'SpikeRasterBase'
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
    
    @property
    def unit_ids(self):
        """The unit_ids from the whole df (not just the current window)"""
        return np.unique(self.spikes_window.df['unit_id'].to_numpy())
    
    @property
    def ordered_unit_ids(self):
        """ Requires the self.unit_sort_order property implemented in UnitSortableMixin """
        return self.unit_ids[self.unit_sort_order]
    
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
    def ordered_neuron_ids(self):
        """ Requires the self.unit_sort_order property implemented in UnitSortableMixin """
        return self.neuron_ids[self.unit_sort_order]
    
    @property
    def ordered_cell_ids(self):
        """ Requires the self.unit_sort_order property implemented in UnitSortableMixin """
        return self.cell_ids[self.unit_sort_order]
    
   
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
        

    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, **kwargs):
        super(SpikeRasterBase, self).__init__(**kwargs)
        # Initialize member variables:
        
        # Helper container variables
        self.params = params
        self._spikes_window = spikes_window
        
        # Config
        self.params.wantsRenderWindowControls = self.WantsRenderWindowControls # from RenderWindowControlsMixin
        self.params.wantsPlaybackControls = self.WantsPlaybackControls # from RenderPlaybackControlsMixin
        

        self.params.side_bin_margins = 0.0 # space to sides of the first and last cell on the y-axis        
        self.params.center_mode = 'zero_centered'
        # self.params.bin_position_mode = ''bin_center'
        self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        # self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
        self.params.temporal_zoom_factor = 40.0 / float(self.spikes_window.timeWindow.window_duration) 
            
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        self.enable_overwrite_invalid_unit_ids = True
        
        SpikeRasterBase.helper_setup_neuron_colors_and_order(self, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
        
        # make root container for plots
        self.plots = RenderPlots('')
        
        self.playback_controller = playback_controller
        # Setup the animation playback object for the time window:
        # self.playback_controller = TimeWindowPlaybackController()
        # self.playback_controller.setup(self._spikes_window)
        # self.playback_controller.setup(self) # pass self to have properties set
        
        self.setup()
                
        # build the UI components:
        self.buildUI()
        
        # Setup Signals:
        # self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        
        ## TODO: BUG: MAJOR: Since the application instance is being assigned to self.app (for any of the widgets that create it) I think that aboutToQuit is called any time any of the widgets are going to close. Although I guess that doesn't explain the errors.
        
        # Connect app quit to onClose event:
        # self.app.aboutToQuit.connect(self.onClose) # Connect the onClose event
        

    @classmethod
    def init_from_unified_spike_raster_app(cls, unified_app, **kwargs):
        """ Helps to create an depdendent instance of the app/window from a master UnifiedSpikeRasterApp 
        
        unified_app: UnifiedSpikeRasterApp
        """
        return cls(params=unified_app.params, spikes_window=unified_app.spikes_window, playback_controller=unified_app.playback_controller, **kwargs)
        
    @classmethod
    def init_from_independent_data(cls, spikes_df, window_duration=15.0, window_start_time=0.0, neuron_colors=None, neuron_sort_order=None, **kwargs):
        """ Helps to create an independent master instance of the app/window. """
        # Helper container variables
        params = VisualizationParameters('')
        spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        playback_controller = TimeWindowPlaybackController()
        # Setup the animation playback object for the time window:
        # self.playback_controller = TimeWindowPlaybackController()
        # self.playback_controller.setup(self._spikes_window)
        # playback_controller.setup(self) # pass self to have properties set    
        return cls(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, **kwargs)
    
    
    @classmethod
    def helper_setup_neuron_colors_and_order(cls, raster_plotter, neuron_colors=None, neuron_sort_order=None):
        """ 
        raster_plotter: a raster plotter
        
        Requires Properties:
            .unit_ids, .neuron_ids
            .spikes_df
            .enable_overwrite_invalid_unit_ids

        Requires Functions:
            .find_cell_IDXs_from_cell_ids(...)
            ._setup_neurons_color_data(...)
            
        Sets Properties:
            ._unit_sort_order
            .cell_id_to_unit_id_map
            .unit_id_to_cell_id_map
        
        """
        # Neurons and sort-orders:
        old_neuron_IDXs = raster_plotter.unit_ids.copy() # backup the old unit_ids
        print(f'\t\t raster_plotter.unit_ids: {raster_plotter.unit_ids} (len: {len(raster_plotter.unit_ids)})\n \t\t raster_plotter.cell_ids: {raster_plotter.cell_ids} (len: {len(raster_plotter.cell_ids)})')
        new_neuron_IDXs = raster_plotter.find_cell_IDXs_from_cell_ids(raster_plotter.neuron_ids)
        print(f'\t\t new_neuron_IDXs: {new_neuron_IDXs} (len(new_neuron_IDXs): {len(new_neuron_IDXs)})')
        # build a map between the old and new neuron_IDXs:
        old_to_new_map = OrderedDict(zip(old_neuron_IDXs, new_neuron_IDXs))
        new_to_old_map = OrderedDict(zip(new_neuron_IDXs, old_neuron_IDXs))
        neuron_id_to_new_IDX_map = OrderedDict(zip(raster_plotter.neuron_ids, new_neuron_IDXs)) # provides the new_IDX corresponding to any neuron_id (aclu value)
        
        if raster_plotter.enable_overwrite_invalid_unit_ids:
            print("WARNING: raster_plotter.enable_overwrite_invalid_unit_ids is True, so dataframe 'unit_id' and 'cell_idx' will be overwritten!")
            raster_plotter.overwrite_invalid_unit_ids(raster_plotter.spikes_df, neuron_id_to_new_IDX_map)
        
        # Build important maps between raster_plotter.unit_ids and raster_plotter.cell_ids:
        raster_plotter.cell_id_to_unit_id_map = OrderedDict(zip(raster_plotter.cell_ids, raster_plotter.unit_ids)) # maps cell_ids to unit_ids
        raster_plotter.unit_id_to_cell_id_map = OrderedDict(zip(raster_plotter.unit_ids, raster_plotter.cell_ids)) # maps unit_ids to cell_ids
        
        if neuron_sort_order is None:
            neuron_sort_order = np.arange(len(raster_plotter.unit_ids)) # default sort order is sorted by unit_ids
        raster_plotter._unit_sort_order = neuron_sort_order
        assert len(raster_plotter._unit_sort_order) == len(raster_plotter.unit_ids), f"len(raster_plotter._unit_sort_order): {len(raster_plotter._unit_sort_order)} must equal len(raster_plotter.unit_ids): {len(raster_plotter.unit_ids)} but it does not!"
        
        # Setup Coloring:
        raster_plotter._setup_neurons_color_data(neuron_colors, coloring_mode='color_by_index_order')
        
    
    @classmethod
    def overwrite_invalid_unit_ids(cls, spikes_df, neuron_id_to_new_IDX_map):
        # if self.enable_overwrite_invalid_unit_ids:
        print("WARNING: self.enable_overwrite_invalid_unit_ids is True, so dataframe 'unit_id' and 'cell_idx' will be overwritten!")
        spikes_df['old_unit_id'] = spikes_df['unit_id'].copy()
        # self.spikes_df['unit_id'] = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        # self.spikes_df['unit_id'] = np.array(self.find_cell_IDXs_from_cell_ids(self.spikes_df['aclu'].to_numpy()), dtype=int)
        # included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        # included_cell_INDEXES = np.array(self.find_cell_IDXs_from_cell_ids(self.spikes_df['aclu'].to_numpy()), dtype=int)
        included_cell_INDEXES = np.array([neuron_id_to_new_IDX_map[an_included_cell_ID] for an_included_cell_ID in spikes_df['aclu'].to_numpy()], dtype=int) # get the indexes from the cellIDs
        print('\t computed included_cell_INDEXES.')
        spikes_df['unit_id'] = included_cell_INDEXES.copy()
        print("\t set spikes_df['unit_id']")
        # self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        spikes_df['cell_idx'] = spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
        print("\t set spikes_df['cell_idx']")
        print("\t done updating 'unit_id' and 'cell_idx'.")
        
    
    
    """ Cell Coloring functions:
    """
    def _setup_neurons_color_data(self, neuron_colors_list, coloring_mode='color_by_index_order'):
        """ 
        neuron_colors_list: a list of neuron colors
        
        Sets:
            self.params.neuron_qcolors
            self.params.neuron_qcolors_map
            self.params.neuron_colors: ndarray of shape (4, self.n_cells)
            self.params.neuron_colors_hex
        """
        
        unsorted_unit_ids = self.unit_ids
        
        if neuron_colors_list is None:
            neuron_colors_list = DataSeriesColorHelpers._build_cell_color_map(unsorted_unit_ids, mode=coloring_mode)
            for a_color in neuron_colors_list:
                a_color.setAlphaF(0.5)
                
                
            # neuron_unit_id_to_colors_index_map = OrderedDict(zip(unsorted_unit_ids, neuron_colors_list))
            neuron_colors_map = OrderedDict(zip(unsorted_unit_ids, neuron_colors_list))
            
            # neuron_colors = []
            # for i, cell_id in enumerate(self.unit_ids):
            #     curr_color = pg.mkColor((i, self.n_cells*1.3))
            #     curr_color.setAlphaF(0.5)
            #     neuron_colors.append(curr_color)
    
        self.params.neuron_qcolors = deepcopy(neuron_colors_list)
        self.params.neuron_qcolors_map = deepcopy(neuron_colors_map)

        # allocate new neuron_colors array:
        self.params.neuron_colors = np.zeros((4, self.n_cells))
        for i, curr_qcolor in enumerate(self.params.neuron_qcolors):
            curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            self.params.neuron_colors[:, i] = curr_color[:]
            # self.params.neuron_colors[:, i] = curr_color[:]
        
        self.params.neuron_colors_hex = None
        
        # spike_raster_plt.params.neuron_colors[0].getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        
        # get hex colors:
        #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'
        self.params.neuron_colors_hex = [self.params.neuron_qcolors[i].name(QtGui.QColor.HexRgb) for i, cell_id in enumerate(self.unit_ids)] 
        
       

    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        raise NotImplementedError # Inheriting classes must override setup to perform particular setup
        # self.app = pg.mkQApp("SpikeRasterBase")
        self.app = pg.mkQApp(self.applicationName)
        
        
        

    
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
        
        self.setWindowTitle(self.windowName)
        # self.setWindowTitle('SpikeRasterBase')
        
        # Connect window update signals
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        # self.spikes_window.window_changed_signal.connect(self.on_window_changed)
        self.spikes_window.window_updated_signal.connect(self.on_window_changed)

        
    def _buildGraphics(self):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        raise NotImplementedError
    
      
                  
    def _update_plots(self):
        """ Implementor must override! """
        raise NotImplementedError
    
    
    ###################################
    #### EVENT HANDLERS
    ##################################

    def debug_print_instance_info(self):
        print('debug_print_instance_info():')
        print(f'\t.applicationName: {self.applicationName}\n\t.windowName: {self.windowName}\n')
    
    
    
    def closeEvent(self, event):
        """closeEvent(self, event): pyqt default event, doesn't have to be registered. Called when the widget will close.
        """
        reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            print('Window closed')
        else:
            event.ignore()



    # @QtCore.pyqtSlot()
    def onClose(self):
        ## TODO: this seems to get called excessively, at least for Spike3DRaster. It happens even when accessing invalid properties and stuff. Not sure what's up, something is broken.
        print(f'onClose()')
        
        self.debug_print_instance_info()
        self.close_signal.emit() # emit to indicate that we're closing this window

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
        

        
# hih