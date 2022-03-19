import numpy as np
import pandas as pd

from pyphoplacecellanalysis.General.Mixins.AllFunctionEnumeratingMixin import AllFunctionEnumeratingMixin

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Spike3DRaster import Spike3DRaster



class SpikeRastersDisplayFunctions(AllFunctionEnumeratingMixin):
    """ Functions related to visualizing 2D and 3D Spike raster plots. """
    
    def _display_spike_rasters_pyqtplot_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 2D raster plot
        """ 
        spike_raster_plt_2d = Spike2DRaster(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None)
        return spike_raster_plt_2d

    def _display_spike_rasters_pyqtplot_3D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot with independent/standalone controls built-in
        """ 
        spike_raster_plt_3d = Spike3DRaster(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None)
        return spike_raster_plt_3d


    def _display_spike_rasters_pyqtplot_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        spike_raster_plt_3d = Spike3DRaster(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None)
        spike_raster_plt_2d = Spike2DRaster(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None)
        spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d.spikes_window.update_window_start_end)
        spike_raster_plt_3d.disable_render_window_controls()
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_plt_3d, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection}




