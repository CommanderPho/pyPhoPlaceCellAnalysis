from copy import deepcopy
import time
import sys
from typing import Optional, OrderedDict, Union
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import qtawesome as qta
# import qdarkstyle

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters, RenderPlots, RenderPlotsData
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeRenderingBaseMixin, SpikesDataframeOwningMixin

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial, DataSeriesToSpatialTransformingMixin

from pyphoplacecellanalysis.GUI.Qt.Mixins.RenderWindowControlsMixin import RenderWindowControlsMixin, RenderPlaybackControlsMixin

from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers, UnitColoringMode

# Pipeline Logging:
import logging
# from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import pipeline_module_logger
from pyphocorehelpers.print_helpers import build_run_log_task_identifier, build_logger
from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass, LoggingBaseClassLoggerOwningMixin
_GLOBAL_spike_raster_logger = None 

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
    if _GLOBAL_spike_raster_logger is not None:
        _GLOBAL_spike_raster_logger.error(f'in trap_exc_during_debug(*args: {args})\n this was installed as the sys.excepthook in SpikeRasterBase above the main class.')

# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug

    
class UnitSortableMixin:
    """ Implementor allows changing the sort order of the units (neurons) displayed by setting indicies directly via the self.unit_sort_order property.
    
    Requires:
        self._unit_sort_order
        self.n_cells
    """
    unit_sort_order_changed_signal = QtCore.pyqtSignal(object) # Called when the sort order is changed. 
    
    @property
    def unit_sort_order(self):
        """The unit_sort_order property.
            Requires self._unit_sort_order to be a ndarray of indicies with the same length as self.fragile_linear_neuron_IDXs
        """
        return self._unit_sort_order
    @unit_sort_order.setter
    def unit_sort_order(self, value):
        assert len(value) == self.n_cells, f"len(value): {len(value)} must equal self.n_cells: {self.n_cells} but it does not!"
        # assert len(self._unit_sort_order) == self.n_cells, f"len(self._unit_sort_order): {len(self._unit_sort_order)} must equal self.n_cells: {self.n_cells} but it does not!"
        self._unit_sort_order = value
        # Emit the sort order changed signal:
        self.unit_sort_order_changed_signal.emit(self._unit_sort_order)
        

class SpikeRasterBase(LoggingBaseClassLoggerOwningMixin, UnitSortableMixin, DataSeriesToSpatialTransformingMixin, NeuronIdentityAccessingMixin, SpikeRenderingBaseMixin, SpikesWindowOwningMixin, SpikesDataframeOwningMixin, RenderPlaybackControlsMixin, RenderWindowControlsMixin, QtWidgets.QWidget):
    
    """ Displays a raster plot with the spikes occuring along a plane. 
    
    Note: fragile_linear_neuron_IDXs: sequentially increasing sequence starting from 0 and going to n_cells - 1. No elements missing:
    fragile_linear_neuron_IDXs: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
    cell_ids: the actual aclu values:
    cell_ids: [ 2  3  4  5  7  8  9 10 11 12 14 17 18 21 22 23 24 25 26 27 28 29 33 34 38 39 42 44 45 46 47 48 53 55 57 58 61 62 63 64]
    neuron_ids: ALIAS for cell_ids
    
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
    
    enable_window_close_confirmation = False
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
    
    # Required for SpikesWindowOwningMixin:
    @property
    def spikes_window(self):
        """The spikes_window property."""
        return self._spikes_window
    
    
    @property
    def fragile_linear_neuron_IDXs(self):
        """The fragile_linear_neuron_IDXs from the whole df (not just the current window)"""
        return np.unique(self.spikes_window.df['fragile_linear_neuron_IDX'].to_numpy())
    
    @property
    def ordered_fragile_linear_neuron_IDXs(self):
        """ Requires the self.unit_sort_order property implemented in UnitSortableMixin """
        return self.fragile_linear_neuron_IDXs[self.unit_sort_order]
    
    @property
    def n_cells(self):
        """The number_units property."""
        return len(self.fragile_linear_neuron_IDXs)
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
        """ NOTE: the temporal_axis_length actually refers to the length of the active_window, as it's used in the pyqtgraph Spike3DRaster class."""
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
        
    @property
    def logger(self):
        """The logger property."""
        return self._logger
    @logger.setter
    def logger(self, value):
        self._logger = value


    @property
    def LoggingBaseClassLoggerOwningMixin_logger(self) -> Optional[LoggingBaseClass]:
        """`LoggingBaseClassLoggerOwningMixin`-conformance required property."""
        #TODO 2025-01-06 12:01: - [ ] IMPLEMENT
        return None
        # return self._logger
    

        
    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, application_name=None, should_show=True, **kwargs):
        """ 
        
        spikes_window: SpikesDataframeWindow
        
        """
        super(SpikeRasterBase, self).__init__(**kwargs)
        # Initialize member variables:
        _GLOBAL_spike_raster_logger = build_logger('Spike3D.display.SpikeRasterBase', file_logging_dir=None, debug_print=False) # Only now do we build the module logger. This way it isn't made when the SpikeRaster plots aren't even used.
        self._logger = _GLOBAL_spike_raster_logger
        self.logger.info(f'SpikeRasterBase.__init__(...)')
        
        # Helper container variables
        self.params = params
        self._spikes_window = spikes_window
        
        
        if application_name is not None:
            self.applicationName = application_name # set instance application name if it isn't None. Otherwise just use the class value.
        
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
        self.enable_overwrite_invalid_fragile_linear_neuron_IDXs = True
        
        self.enable_show_on_init = should_show
        
        SpikeRenderingBaseMixin.helper_setup_neuron_colors_and_order(self, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
        
        # make root container for plots
        self.plots = RenderPlots(self.applicationName)
        self.plots_data = RenderPlotsData(self.applicationName)
        
        
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
        
        
        # Connect window update signals
        # Only subscribe to the more advanced LiveWindowedData-style window update signals that also provide data
        self.spikes_window.windowed_data_window_duration_changed_signal.connect(self.on_windowed_data_window_duration_changed)
        self.spikes_window.windowed_data_window_updated_signal.connect(self.on_windowed_data_window_changed)
        
        
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
    def init_from_independent_data(cls, spikes_df, window_duration=15.0, window_start_time=0.0, neuron_colors=None, neuron_sort_order=None, enable_independent_playback_controller=False, **kwargs):
        """ Helps to create an independent master instance of the app/window. """
        # Helper container variables
        params = VisualizationParameters('')
        spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        
        if enable_independent_playback_controller:
            playback_controller = TimeWindowPlaybackController()
        else:
            # don't allow playback controller.
            playback_controller = None
        
        # Setup the animation playback object for the time window:
        # self.playback_controller = TimeWindowPlaybackController()
        # self.playback_controller.setup(self._spikes_window)
        # playback_controller.setup(self) # pass self to have properties set    
        return cls(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, **kwargs)
    
    
    
    
    """ Cell Coloring functions:
    """
    def _setup_neurons_color_data(self, neuron_colors_list=None, coloring_mode:UnitColoringMode=UnitColoringMode.COLOR_BY_INDEX_ORDER):
        """ 
        neuron_colors_list: a list of neuron colors
            if None provided will call DataSeriesColorHelpers._build_cell_qcolor_list(...) to build them.
        
        Requires:
            self.fragile_linear_neuron_IDXs
            self.n_cells
        
        Sets:
            self.params.neuron_qcolors
            self.params.neuron_qcolors_map
            self.params.neuron_colors: ndarray of shape (4, self.n_cells)
            self.params.neuron_colors_hex
            

        Known Calls: Seemingly only called from:
            SpikesRenderingBaseMixin.helper_setup_neuron_colors_and_order(...)
        """
        
        unsorted_fragile_linear_neuron_IDXs = self.fragile_linear_neuron_IDXs
        
        if neuron_colors_list is None:
            neuron_qcolors_list = DataSeriesColorHelpers._build_cell_qcolor_list(unsorted_fragile_linear_neuron_IDXs, mode=coloring_mode, provided_cell_colors=None)
            for a_color in neuron_qcolors_list:
                a_color.setAlphaF(0.5)
        else:
            ## TODO: otherwise we have some provided colors that we should convert into the correct format
            neuron_qcolors_list = DataSeriesColorHelpers._build_cell_qcolor_list(unsorted_fragile_linear_neuron_IDXs, mode=coloring_mode, provided_cell_colors=neuron_colors_list.copy()) # builts a list of qcolors
                                
        # neuron_fragile_linear_neuron_IDX_to_colors_index_map = OrderedDict(zip(unsorted_fragile_linear_neuron_IDXs, neuron_colors_list))
        neuron_qcolors_map = OrderedDict(zip(unsorted_fragile_linear_neuron_IDXs, neuron_qcolors_list))
    
        self.params.neuron_qcolors = deepcopy(neuron_qcolors_list)
        self.params.neuron_qcolors_map = deepcopy(neuron_qcolors_map)

        # allocate new neuron_colors array:
        self.params.neuron_colors = np.zeros((4, self.n_cells))
        for i, curr_qcolor in enumerate(self.params.neuron_qcolors):
            curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            self.params.neuron_colors[:, i] = curr_color[:]
            # self.params.neuron_colors[:, i] = curr_color[:]
        
        self.params.neuron_colors_hex = None
        
        # get hex colors:
        #  getting the name of a QColor with .name(QtGui.QColor.HexRgb) results in a string like '#ff0000'
        #  getting the name of a QColor with .name(QtGui.QColor.HexArgb) results in a string like '#80ff0000'
        self.params.neuron_colors_hex = [ColorFormatConverter.qColor_to_hexstring(self.params.neuron_qcolors[i], include_alpha=False) for i, cell_id in enumerate(self.fragile_linear_neuron_IDXs)] 
        
       
       
       
    def update_neurons_color_data(self, updated_neuron_render_configs):
        """updates the colors for each neuron/cell given the updated_neuron_render_configs map

        Args:
            updated_neuron_render_configs (_type_): _description_
            
        Updates:

        """
        # updated_color_dict = {cell_id:cell_config.color for cell_id, cell_config in updated_neuron_render_configs.items()} ## TODO: efficiency: pass only the colors that changed instead of all the colors:
        updated_color_dict = {}
        
        for cell_id, cell_config in updated_neuron_render_configs.items():
            a_fragile_linear_neuron_IDX = self.cell_id_to_fragile_linear_neuron_IDX_map[cell_id]
            curr_qcolor = cell_config.qcolor
            
            # Determine if the color changed: Easiest to compare the hex value string:
            did_color_change = (self.params.neuron_colors_hex[a_fragile_linear_neuron_IDX] != cell_config.color) # the hex color
            
            # Overwrite the old colors:
            self.params.neuron_qcolors_map[a_fragile_linear_neuron_IDX] = curr_qcolor
            self.params.neuron_qcolors[a_fragile_linear_neuron_IDX] = curr_qcolor
            # Overwrite the old secondary/derived colors:
            curr_rgbf_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            self.params.neuron_colors[:, a_fragile_linear_neuron_IDX] = curr_rgbf_color[:]
            self.params.neuron_colors_hex[a_fragile_linear_neuron_IDX] = cell_config.color # the hex color
            
            if did_color_change:
                # If the color changed, add it to the changed array:
                updated_color_dict[cell_id] = cell_config.color
        
        self.on_neuron_colors_changed(updated_color_dict)

    
    @QtCore.pyqtSlot(object)
    def on_neuron_colors_changed(self, neuron_id_color_update_dict):
        """ Called when the neuron colors have finished changing (changed) to update the rendered elements.
        """
        pass
    
    
    
    def setup(self):
        self.logger.info(f'SpikeRasterBase.setup()')
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        raise NotImplementedError # Inheriting classes must override setup to perform particular setup
        # self.app = pg.mkQApp("SpikeRasterBase")
        self.app = pg.mkQApp(self.applicationName)
        
        
        

    
    def buildUI(self):
        """ for QGridLayout
            addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        """
        self.logger.info(f'SpikeRasterBase.buildUI()')
        self.ui = PhoUIContainer()
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setObjectName('root_layout')
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
        # self.spikes_window.timeWindow.window_duration_changed_signal.connect(self.on_window_duration_changed)
        # self.spikes_window.timeWindow.window_changed_signal.connect(self.on_window_changed)
        # self.spikes_window.timeWindow.window_updated_signal.connect(self.on_window_changed)

        
    def _buildGraphics(self):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        self.logger.info(f'SpikeRasterBase._buildGraphics()')
        raise NotImplementedError
    
      
                  
    def _update_plots(self):
        """ Implementor must override! """
        self.logger.info(f'SpikeRasterBase._update_plots()')
        raise NotImplementedError
    
    
    ###################################
    #### EVENT HANDLERS
    ##################################

    def debug_print_instance_info(self):
        print('debug_print_instance_info():')
        print(f'\t.applicationName: {self.applicationName}\n\t.windowName: {self.windowName}\n')
        self.logger.info(f'SpikeRasterBase: \t.applicationName: {self.applicationName}\n\t.windowName: {self.windowName}\n')
    
    
    
    def closeEvent(self, event):
        """closeEvent(self, event): pyqt default event, doesn't have to be registered. Called when the widget will close.
        """
        if self.enable_window_close_confirmation:
            reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        else:
            reply = QtWidgets.QMessageBox.Yes
            
        if reply == QtWidgets.QMessageBox.Yes:
            self.close_signal.emit() # emit to indicate that we're closing this window
            self.onClose() # ensure onClose() is called
            event.accept()
            print('Window closed')
        else:
            event.ignore()
            



    # @QtCore.pyqtSlot()
    def onClose(self):
        ## TODO: this seems to get called excessively, at least for Spike3DRaster. It happens even when accessing invalid properties and stuff. Not sure what's up, something is broken.
        print(f'onClose()')
        self.logger.info(f'onClose()')
        self.debug_print_instance_info()
        

    # Input Handelers:        
    def keyPressEvent(self, e):
        """ called automatically when a keyboard key is pressed and this widget has focus. 
        
        """
        print(f'keyPressEvent(e.key(): {e.key()})')
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif e.key() == QtCore.Qt.Key_Backspace:
            print('TODO')
        elif e.key() == QtCore.Qt.Key_Left:
            # Left Arrow
            self.shift_animation_frame_val(-1) # jump back one frame            
        elif e.key() == QtCore.Qt.Key_Right:
            # Right Arrow
            self.shift_animation_frame_val(1) # jump forward one frame            

        elif e.key() == QtCore.Qt.Key_PageDown:
            # PageDown Key
            print(f'Key_PageDown Pressed')
            self.on_jump_window_right() # jump forward one full window

        elif e.key() == QtCore.Qt.Key_PageUp:
            # PageUp Key
            print(f'Key_PageUp Pressed')
            self.on_jump_window_left() # jump back one window

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
        print(f'SpikeRasterBase.wheel_handler(event.angleDelta().y(): {event.angleDelta().y()})')
        # self.modify_volume(1 if event.angleDelta().y() > 0 else -1)
        # self.set_media_position(1 if event.angleDelta().y() > 0 else -1)





    @QtCore.pyqtSlot()
    def on_spikes_df_changed(self):
        """ changes:
            self.fragile_linear_neuron_IDXs
            self.n_full_cell_grid
        """
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_spikes_df_changed()')
        
    @QtCore.pyqtSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')


    @QtCore.pyqtSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
        
        
    
    @QtCore.pyqtSlot(float, float, float, object)
    def on_windowed_data_window_duration_changed(self, start_t, end_t, duration, updated_data_value):
        """ changes self.half_render_window_duration """
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_windowed_data_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration}, updated_data_value: ...)')

    @QtCore.pyqtSlot(float, float, object)
    def on_windowed_data_window_changed(self, start_t, end_t, updated_data_value):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'SpikeRasterBase.on_windowed_data_window_changed(start_t: {start_t}, end_t: {end_t}, updated_data_value: ...)')
        if self.enable_debug_print:
            profiler = pg.debug.Profiler(disabled=True, delayed=True)
        self._update_plots()
        if self.enable_debug_print:
            profiler('Finished calling _update_plots()')
    

        
# hih