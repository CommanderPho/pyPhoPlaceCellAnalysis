import numpy as np
import pandas as pd
from functools import partial

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

from pyphoplacecellanalysis.GUI.Qt.GlobalApplicationMenus.LocalMenus_AddRenderable import LocalMenus_AddRenderable # for custom context menus

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls, build_spike_3d_raster_vedo_with_2d_controls
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.ConnectionControlsMenuMixin import ConnectionControlsMenuMixin
from pyphoplacecellanalysis.GUI.Qt.Mixins.Menus.CreateNewConnectedWidgetMenuMixin import CreateNewConnectedWidgetMenuMixin


## TODO: update these to use the correct format! This format has been invalidated!

class SpikeRastersDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing 2D and 3D Spike raster plots. """
    
    # external_independent_widget_fcns = ['_display_spike_rasters_pyqtplot_2D', '_display_spike_rasters_pyqtplot_3D', '_display_spike_rasters_vedo_3D', '_display_spike_rasters_pyqtplot_3D_with_2D_controls', '_display_spike_rasters_vedo_3D_with_2D_controls', '_display_spike_rasters_window']
    
    @staticmethod
    def _display_spike_rasters_pyqtplot_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 2D raster plot
        """ 
        spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess)
        return spike_raster_plt_2d

    @staticmethod
    def _display_spike_rasters_pyqtplot_3D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot with independent/standalone controls built-in
        """ 
        spike_raster_plt_3d = Spike3DRaster.init_from_independent_data(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
        return spike_raster_plt_3d
    
    @staticmethod
    def _display_spike_rasters_vedo_3D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot with independent/standalone controls built-in
        """ 
        spike_raster_plt_3d_vedo = Spike3DRaster_Vedo.init_from_independent_data(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
        return spike_raster_plt_3d_vedo

    ## 2D Controlled 3D Raster Plots:
    @staticmethod
    def _display_spike_rasters_pyqtplot_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot (via pyqtgraph) with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        use_separate_windows = kwargs.get('separate_windows', False)
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window = build_spike_3d_raster_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=use_separate_windows)
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess)  # Adds the custom context menus for SpikeRaster2D
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_plt_3d, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection, 'spike_raster_window': spike_raster_window}

    @staticmethod
    def _display_spike_rasters_vedo_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot (via Vedo) with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        use_separate_windows = kwargs.get('separate_windows', False)
        spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window = build_spike_3d_raster_vedo_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=use_separate_windows)
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess) # Adds the custom context menus for SpikeRaster2D
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d_vedo':spike_raster_plt_3d_vedo, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection, 'spike_raster_window': spike_raster_window}

    @staticmethod
    def _display_spike_rasters_window(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Displays a Spike3DRasterWindowWidget with a configurable set of raster widgets and controls in it.
        """
        use_separate_windows = kwargs.get('separate_windows', False)
        type_of_3d_plotter = kwargs.get('type_of_3d_plotter', 'pyqtgraph')
        active_plotting_config = active_config.plotting_config
        active_config_name = kwargs.get('active_config_name', 'Unknown')
        # active_config_name = active_config.
        
        # spike_raster_window = Spike3DRasterWindowWidget(computation_result.sess.spikes_df)
        # spike_raster_window = Spike3DRasterWindowWidget(computation_result.sess.spikes_df, neuron_colors=provided_neuron_id_to_color_map, neuron_sort_order=None, type_of_3d_plotter=type_of_3d_plotter, application_name=f'Spike Raster Window - {active_config_name}')
        spike_raster_window = Spike3DRasterWindowWidget(computation_result.sess.spikes_df, type_of_3d_plotter=type_of_3d_plotter, application_name=f'Spike Raster Window - {active_config_name}')
        
        # Set Window Title Options:
        spike_raster_window.setWindowFilePath(str(computation_result.sess.filePrefix.resolve()))
        spike_raster_window.setWindowTitle(f'Spike Raster Window - {active_config_name} - {str(computation_result.sess.filePrefix.resolve())}')
        
        ## Adds the custom renderable menu to the top-level menu of the plots in Spike2DRaster
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_window.spike_raster_plt_2d, computation_result.sess)  # Adds the custom context menus for SpikeRaster2D        
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        ## Note that curr_main_menu_window is usually not the same as spike_raster_window, instead curr_main_menu_window wraps it and produces the final output window
        curr_main_menu_window, menuConnections, connections_actions_dict = ConnectionControlsMenuMixin.try_add_connections_menu(spike_raster_window)
        spike_raster_window.main_menu_window = curr_main_menu_window # to retain the changes
        
        if owning_pipeline_reference is not None:
            display_output = owning_pipeline_reference.display_output
            curr_main_menu_window, menuCreateNewConnectedWidget, createNewConnected_actions_dict = CreateNewConnectedWidgetMenuMixin.try_add_create_new_connected_widget_menu(spike_raster_window, owning_pipeline_reference, active_config, display_output)
            spike_raster_window.main_menu_window = curr_main_menu_window # to retain the changes
            
        else:
            print(f'WARNING: _display_spike_rasters_window(...) has no owning_pipeline_reference in its parameters, so it cannot add the CreateNewConnectedWidgetMenuMixin menus.')
            menuCreateNewConnectedWidget = None
            createNewConnected_actions_dict = None
            
        return {'spike_raster_plt_2d':spike_raster_window.spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_window.spike_raster_plt_3d, 'spike_raster_window': spike_raster_window}


