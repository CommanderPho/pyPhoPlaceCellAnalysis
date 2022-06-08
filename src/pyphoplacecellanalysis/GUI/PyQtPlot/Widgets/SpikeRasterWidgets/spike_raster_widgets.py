# .SpikeRasterWidgets

from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo


from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters
from pyphoplacecellanalysis.General.Mixins.TimeWindowPlaybackMixin import TimeWindowPlaybackPropertiesMixin, TimeWindowPlaybackController, TimeWindowPlaybackControllerActionsMixin
from pyphoplacecellanalysis.General.Model.SpikesDataframeWindow import SpikesDataframeWindow, SpikesWindowOwningMixin

# for reandering in a single window:
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget


""" 
Each separate call to Spikes3DRaster, Spikes2DRaster, etc shouldn't nec. create a whole new app. We want the ability for data such as the spikes_window to be shared between these windows.

TimeWindowPlaybackController

"""
class UnifiedSpikeRasterApp(TimeWindowPlaybackControllerActionsMixin, TimeWindowPlaybackPropertiesMixin, QtCore.QObject):
    """ An attempt to make a singleton global app instance to hold the main window and synchronized playback controls and other global properties.
        Currently Unused!
    """
    
    # TimeWindowPlaybackPropertiesMixin requirement:
    @property
    def animation_active_time_window(self):
        """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate."""
        return self._spikes_window
    
    # Get/Set Properties:
    @property
    def spikes_window(self):
        """The spikes_window property."""
        return self._spikes_window
    @spikes_window.setter
    def spikes_window(self, value):
        self._spikes_window = value
    
    def __init__(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None):
        super(UnifiedSpikeRasterApp, self).__init__() # QtCore.QObject.__init__(self)
        
        # Set app name
        self.name = core_app_name
        
        self.params = VisualizationParameters('')
        self._spikes_window = SpikesDataframeWindow(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        self.playback_controller = TimeWindowPlaybackController()
        self.playback_controller.setup(self) # pass self to have properties set
        
        
        

        
def build_spike_3d_raster_with_2d_controls(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=False, application_name=None):
    """ builds a 3D Raster plot for spikes with 2D controls in a separate window
    
    Inputs:
        separate_windows: bool - If True, the 3d plotter and its 2d controls are rendered in separate windows. Otherwise they're rendered in a single Spike3DRasterWindowWidget
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        curr_computations_results = curr_active_pipeline.computation_results[curr_epoch_name]
        # Build the output widget:
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_with_2d_controls(curr_spikes_df)
        
    """
    if separate_windows:
        spike_raster_plt_3d = Spike3DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name)
        # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
        spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
        spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
        spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d.spikes_window.update_window_start_end)
        spike_raster_plt_3d.disable_render_window_controls()
        # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
        spike_raster_plt_3d.setWindowTitle('Main 3D Raster Window')
        WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d, debug_print=False)
        WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d, spike_raster_plt_2d) # Align the two windows
        spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
        spike_raster_window = None
    else:
        spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, type_of_3d_plotter='pyqtgraph')
        spike_raster_plt_2d = spike_raster_window.spike_raster_plt_2d
        spike_raster_plt_3d = spike_raster_window.spike_raster_plt_3d
        spike_3d_to_2d_window_connection = spike_raster_window.spike_3d_to_2d_window_connection
    
    return spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window



def build_spike_3d_raster_vedo_with_2d_controls(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, extant_spike_raster_plt_3d_vedo = None, separate_windows=False, application_name=None):
    """ builds a vedo-based 3D Raster plot for spikes with 2D controls in a separate window

    # NOTE: It appears this only works if the 2D Raster plot (pyqtgraph-based) is created before the Spike3DRaster_Vedo (Vedo-based). This is probably due to the pyqtgraph's instancing of the QtApplication.
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        curr_computations_results = curr_active_pipeline.computation_results[curr_epoch_name]
        # Build the output widget:
        spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_vedo_with_2d_controls(curr_spikes_df)
    
    """
    # Build the 2D Raster Plotter
    spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name,  parent=None) # setting , parent=spike_raster_plt_3d makes a single window
    spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
    # Update the 2D Scroll Region to the initial value:
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
    

    # Build the 3D Vedo Raster plotter
    spike_raster_plt_3d_vedo = Spike3DRaster_Vedo.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name)
    spike_raster_plt_3d_vedo.setWindowTitle('Main 3D (Vedo) Raster Window')
    spike_raster_plt_3d_vedo.disable_render_window_controls()
    # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
    
    # Set the 3D Vedo plots' window to the current values of the 2d plot:
    spike_raster_plt_3d_vedo.spikes_window.update_window_start_end(spike_raster_plt_2d.spikes_window.active_time_window[0], spike_raster_plt_2d.spikes_window.active_time_window[1])

    # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
    spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d_vedo.spikes_window.update_window_start_end)
    
    # Position the Windows As a Stack in the top-left corner:
    WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d_vedo, debug_print=False)
    WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d_vedo, spike_raster_plt_2d) # Align the two windows
    
    # Update the scroll position programmatically with block_signals=False to ensure the 3D plot is synced:
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
    
    # Stand-in for future return value:
    spike_raster_window = None
    return spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window



# fd

