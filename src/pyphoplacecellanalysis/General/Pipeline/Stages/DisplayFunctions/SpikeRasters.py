import numpy as np
import pandas as pd

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls


## TODO: update these to use the correct format! This format has been invalidated!

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
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None)
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_plt_3d, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection}




