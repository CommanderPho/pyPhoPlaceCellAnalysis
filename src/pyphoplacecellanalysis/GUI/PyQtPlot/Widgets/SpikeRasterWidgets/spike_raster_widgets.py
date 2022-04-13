# .SpikeRasterWidgets

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo


def build_spike_3d_raster_with_2d_controls(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None):
    """ builds a 3D Raster plot for spikes with 2D controls in a separate window
    
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
    spike_raster_plt_3d = Spike3DRaster(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
    # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
    spike_raster_plt_2d = Spike2DRaster(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
    spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
    spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d.spikes_window.update_window_start_end)
    spike_raster_plt_3d.disable_render_window_controls()
    # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
    spike_raster_plt_3d.setWindowTitle('Main 3D Raster Window')
    WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d, debug_print=False)
    WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d, spike_raster_plt_2d) # Align the two windows
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
    return spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection



def build_spike_3d_raster_vedo_with_2d_controls(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, extant_spike_raster_plt_3d_vedo = None):
    """ builds a vedo-based 3D Raster plot for spikes with 2D controls in a separate window
    
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
#     if extant_spike_raster_plt_3d_vedo is not None:
#         spike_raster_plt_3d = extant_spike_raster_plt_3d_vedo
#         curr_spikes_df = spike_raster_plt_3d.spikes_df
        
# #         temp_window_duration = (spike_raster_plt_3d.spikes_window.active_window_end_time - spike_raster_plt_3d.spikes_window.active_window_start_time)
# #         temp_window_start_time = spike_raster_plt_3d.spikes_window.active_window_start_time
# #         temp_neuron_colors = None
# #         temp_neuron_sort_order = None

# #         print(f'window_duration={temp_window_duration}, window_start_time={temp_window_start_time}, neuron_colors=None, neuron_sort_order=None')

#     else:
#         spike_raster_plt_3d = Spike3DRaster_Vedo(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
    
    # Build the 3D Vedo Raster plotter
    spike_raster_plt_3d_vedo = Spike3DRaster_Vedo(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
    # Build the 2D Raster Plotter
    spike_raster_plt_2d = Spike2DRaster(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
    spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
    # Update the 2D Scroll Region to the initial value:
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)

#     # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
#     spike_raster_plt_2d = Spike2DRaster(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
    
#     # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
#     # spike_raster_plt_2d = Spike2DRaster(spike_raster_plt_3d.spikes_df, window_duration=temp_window_duration, window_start_time=temp_window_start_time, neuron_colors=temp_neuron_colors, neuron_sort_order=temp_neuron_sort_order, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
#     spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')

    spike_3d_to_2d_window_connection = None
    
    # Set the 3D Vedo plots' window to the current values of the 2d plot:
    spike_raster_plt_3d_vedo.spikes_window.update_window_start_end(spike_raster_plt_2d.spikes_window.active_time_window[0], spike_raster_plt_2d.spikes_window.active_time_window[1])

    spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d_vedo.spikes_window.update_window_start_end)


    # spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d.spikes_window.update_window_start_end)
    # spike_raster_plt_3d.disable_render_window_controls()
    # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
    # spike_raster_plt_3d.setWindowTitle('Main 3D (Vedo) Raster Window')
    
    WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d_vedo, debug_print=False)
    WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d_vedo, spike_raster_plt_2d) # Align the two windows
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
    
    return spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection

# Comes in with a spike_raster_plt_3d_vedo:

# spike_raster_plt_3d_vedo.spikes_df

# spike_raster_plt_3d_vedo.neuron_colors
# spike_raster_plt_3d_vedo.neuron_sort_order







# fd

