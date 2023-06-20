from copy import deepcopy
import numpy as np
import pandas as pd
from functools import partial
from attrs import define, Factory
from indexed import IndexedOrderedDict

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots, RenderPlotsData
from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color # required for the different emphasis states in ._build_cell_configs()

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers # for build_neurons_color_data
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui # for QColor build_neurons_color_data
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable # for custom context menus

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls, build_spike_3d_raster_vedo_with_2d_controls
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.ConnectionControlsMenuMixin import ConnectionControlsMenuMixin
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.CreateNewConnectedWidgetMenuMixin import CreateNewConnectedWidgetMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.DebugMenuProviderMixin import DebugMenuProviderMixin
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.CreateLinkedWidget_MenuProvider import CreateLinkedWidget_MenuProvider
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.DockedWidgets_MenuProvider import DockedWidgets_MenuProvider

from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper


## TODO: update these to use the correct format! This format has been invalidated!

class SpikeRastersDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing 2D and 3D Spike raster plots. """
    
    # external_independent_widget_fcns = ['_display_spike_rasters_pyqtplot_2D', '_display_spike_rasters_pyqtplot_3D', '_display_spike_rasters_vedo_3D', '_display_spike_rasters_pyqtplot_3D_with_2D_controls', '_display_spike_rasters_vedo_3D_with_2D_controls', '_display_spike_rasters_window']
    
    @function_attributes(short_name='spike_rasters_pyqtplot_2D', tags=['display','interactive', 'raster', '2D', 'pyqtplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_pyqtplot_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 2D raster plot
        """ 
        ## Modern version just calls the function to build the spike_rasters_window (_display_spike_rasters_window) with type_of_3d_plotter=None
        kwargs['type_of_3d_plotter'] = None # make sure that 'type_of_3d_plotter' of kwargs is None either way so no 3D plotter is rendered, overriding the user's argument if needed:
        return SpikeRastersDisplayFunctions._display_spike_rasters_window(computation_result, active_config, enable_saving_to_disk=enable_saving_to_disk, **kwargs)

    @function_attributes(short_name='spike_rasters_pyqtplot_3D', tags=['display','interactive', 'raster', '3D', 'pyqtplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_pyqtplot_3D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot with independent/standalone controls built-in
        """ 
        spike_raster_plt_3d = Spike3DRaster.init_from_independent_data(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
        return spike_raster_plt_3d
    
    @function_attributes(short_name='spike_rasters_vedo_3D', tags=['display','interactive', 'raster', '3D', 'vedo'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_vedo_3D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot with independent/standalone controls built-in
        """ 
        spike_raster_plt_3d_vedo = Spike3DRaster_Vedo.init_from_independent_data(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None)
        return spike_raster_plt_3d_vedo

    ## 2D Controlled 3D Raster Plots:

    @function_attributes(short_name='spike_rasters_pyqtplot_3D_with_2D_controls', tags=['display','interactive', 'raster', '2D', 'ui', '3D', 'pyqtplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_pyqtplot_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot (via pyqtgraph) with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        use_separate_windows = kwargs.get('separate_windows', False)
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window = build_spike_3d_raster_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=use_separate_windows)
        
        
        active_config_name = kwargs.get('active_config_name', 'Unknown')
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        assert owning_pipeline_reference is not None
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, owning_pipeline_reference, active_config_name)
        # _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess)  # Adds the custom context menus for SpikeRaster2D
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_plt_3d, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection, 'spike_raster_window': spike_raster_window}


    @function_attributes(short_name='spike_rasters_vedo_3D_with_2D_controls', tags=['display','interactive', 'raster', '2D', 'ui', '3D', 'vedo'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_vedo_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot (via Vedo) with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        use_separate_windows = kwargs.get('separate_windows', False)
        spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window = build_spike_3d_raster_vedo_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=use_separate_windows)
        
        active_config_name = kwargs.get('active_config_name', 'Unknown')
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        assert owning_pipeline_reference is not None
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, owning_pipeline_reference, active_config_name)
        # _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess) # Adds the custom context menus for SpikeRaster2D
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d_vedo':spike_raster_plt_3d_vedo, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection, 'spike_raster_window': spike_raster_window}


    @function_attributes(short_name='spike_rasters_window', tags=['display','interactive', 'primary', 'raster', '2D', 'ui', 'pyqtplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_window(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Displays a Spike3DRasterWindowWidget with a configurable set of raster widgets and controls in it.
        """
        use_separate_windows = kwargs.get('separate_windows', False)
        type_of_3d_plotter = kwargs.get('type_of_3d_plotter', 'pyqtgraph')
        active_plotting_config = active_config.plotting_config
        active_config_name = kwargs.get('active_config_name', 'Unknown')
        active_identifying_context = kwargs.get('active_context', None)
        assert active_identifying_context is not None
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        
        ## Finally, add the display function to the active context
        active_display_fn_identifying_ctx = active_identifying_context.adding_context('display_fn', display_fn_name='display_spike_rasters_window')
        active_display_fn_identifying_ctx_string = active_display_fn_identifying_ctx.get_description(separator='|') # Get final discription string:

        ## It's passed a specific computation_result which has a .sess attribute that's used to determine which spikes are displayed or not.
        spike_raster_window = Spike3DRasterWindowWidget(computation_result.sess.spikes_df, type_of_3d_plotter=type_of_3d_plotter, application_name=f'Spike Raster Window - {active_display_fn_identifying_ctx_string}')
        
        # Set Window Title Options:
        spike_raster_window.setWindowFilePath(str(computation_result.sess.filePrefix.resolve()))
        spike_raster_window.setWindowTitle(f'Spike Raster Window - {active_config_name} - {str(computation_result.sess.filePrefix.resolve())}')
        
        ## Build the additional menus:
        output_references = _build_additional_window_menus(spike_raster_window, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx)

        return {'spike_raster_plt_2d':spike_raster_window.spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_window.spike_raster_plt_3d, 'spike_raster_window': spike_raster_window}


# ==================================================================================================================== #
# 2023-03-31 PyQtGraph Based Standalone Raster Plot                                                                    #
# ==================================================================================================================== #
# ==================================================================================================================== #
# NEW 2023-03-31 - Uses Scatter Plot Based raster like SpikeRaster2D so they can be colored, resized, etc.             #
# ==================================================================================================================== #


""" Raster Plot:

    plot_raster_plot(...) is the only user-facing function. The rest of this crap is just a hack.
    2023-03-31: All this crap was brought in to replace the functionality used in SpikeRaster2D 
"""

# Windowing helpers for spikes_df:
def _add_spikes_df_visualization_columns(manager, spikes_df):
    if 'visualization_raster_y_location' not in spikes_df.columns:
        all_y = [manager.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
        spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes. BUG? NO, RESOLVED: actually, this should be updated when anything that would change .y_fragile_linear_neuron_IDX_map would change, right? Meaning: .y, ... oh, I see. y doesn't change because params.center_mode, params.bin_position_mode, and params.side_bin_margins aren't expected to change. 

    if 'visualization_raster_emphasis_state' not in spikes_df.columns:
        spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default
    return spikes_df

def _build_neurons_color_data(params, fragile_linear_neuron_IDXs, neuron_colors_list=None, coloring_mode='color_by_index_order'):
    """ Cell Coloring function

    neuron_colors_list: a list of neuron colors
        if None provided will call DataSeriesColorHelpers._build_cell_color_map(...) to build them.
    
    Requires:
        fragile_linear_neuron_IDXs
        
    Sets:
        params.neuron_qcolors
        params.neuron_qcolors_map
        params.neuron_colors: ndarray of shape (4, self.n_cells)
        params.neuron_colors_hex

    Known Calls: Seemingly only called from:
        SpikesRenderingBaseMixin.helper_setup_neuron_colors_and_order(...)

    History: Factored out of SpikeRasterBase on 2023-03-31

    Usage:

        params = build_neurons_color_data(params, fragile_linear_neuron_IDXs)
        params

    """	
    unsorted_fragile_linear_neuron_IDXs = fragile_linear_neuron_IDXs
    n_cells = len(unsorted_fragile_linear_neuron_IDXs)

    if neuron_colors_list is None:
        neuron_qcolors_list = DataSeriesColorHelpers._build_cell_color_map(unsorted_fragile_linear_neuron_IDXs, mode=coloring_mode, provided_cell_colors=None)
        for a_color in neuron_qcolors_list:
            a_color.setAlphaF(0.5)
    else:
        neuron_qcolors_list = DataSeriesColorHelpers._build_cell_color_map(unsorted_fragile_linear_neuron_IDXs, mode=coloring_mode, provided_cell_colors=neuron_colors_list.copy()) # builts a list of qcolors
                            
    neuron_qcolors_map = dict(zip(unsorted_fragile_linear_neuron_IDXs, neuron_qcolors_list))

    params.neuron_qcolors = deepcopy(neuron_qcolors_list)
    params.neuron_qcolors_map = deepcopy(neuron_qcolors_map)

    # allocate new neuron_colors array:
    params.neuron_colors = np.zeros((4, n_cells))
    for i, curr_qcolor in enumerate(params.neuron_qcolors):
        curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        params.neuron_colors[:, i] = curr_color[:]
    
    params.neuron_colors_hex = None
    
    # get hex colors:
    params.neuron_colors_hex = [params.neuron_qcolors[i].name(QtGui.QColor.HexRgb) for i, cell_id in enumerate(fragile_linear_neuron_IDXs)]
    return params

@define 
class RasterPlotParams:
    """ factored out of Spike2DRaster to do standalone pyqtgraph plotting of the 2D raster plot. """
    center_mode: str = 'starting_at_zero' # or 'zero_centered'
    bin_position_mode: str = 'bin_center' # or 'left_edges'
    side_bin_margins: float = 0.0

    # Colors:
    neuron_qcolors: list = None
    neuron_colors: np.ndarray = None # of shape (4, self.n_cells)
    neuron_colors_hex: np.ndarray = None #
    neuron_qcolors_map: dict = Factory(dict)

    # Configs:
    config_items: IndexedOrderedDict = Factory(IndexedOrderedDict)

    def build_neurons_color_data(self, fragile_linear_neuron_IDXs, neuron_colors_list=None, coloring_mode='color_by_index_order'):
        """ Cell Coloring function

        neuron_colors_list: a list of neuron colors
            if None provided will call DataSeriesColorHelpers._build_cell_color_map(...) to build them.
            
        Sets:
            params.neuron_qcolors
            params.neuron_qcolors_map
            params.neuron_colors: ndarray of shape (4, self.n_cells)
            params.neuron_colors_hex

        History: Factored out of SpikeRasterBase on 2023-03-31

        """
        self = _build_neurons_color_data(self, fragile_linear_neuron_IDXs, neuron_colors_list=neuron_colors_list, coloring_mode=coloring_mode)

@define
class UnitSortOrderManager(NeuronIdentityAccessingMixin):
    """ factored out of Spike2DRaster to do standalone pyqtgraph plotting of the 2D raster plot. """
    neuron_ids: np.ndarray
    fragile_linear_neuron_IDXs: np.ndarray
    n_cells: int # = len(shared_aclus)
    unit_sort_order: np.ndarray # = np.arange(n_cells) # in-line sort order
    _series_identity_y_values: np.ndarray = None
    _series_identity_lower_y_values: np.ndarray = None
    _series_identity_upper_y_values: np.ndarray = None
    y_fragile_linear_neuron_IDX_map: dict = Factory(dict)
    params: RasterPlotParams = Factory(RasterPlotParams)

    @property
    def series_identity_y_values(self):
        """The series_identity_y_values property."""
        return self._series_identity_y_values

    def update_series_identity_y_values(self, debug_print=False):
        """ updates the fixed self._series_identity_y_values using the DataSeriesToSpatial.build_series_identity_axis(...) function.
        
        Should be called whenever:
            self.n_cells, 
            params.center_mode,
            params.bin_position_mode
            params.side_bin_margins
            self.unit_sort_order
        values change.
        """
        self._series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        self._series_identity_lower_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        self._series_identity_upper_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells

        ## SORT: TODO: This sort condition seems to work and change the sort-order of the cells when self.unit_sort_order is updated... but the colors get all wonky and I'm uncertain if the configs are working correctly. Furthmore, it isn't clear that the spiking is any better aligned.
        # This might be overkill, idk
        # self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs), self._series_identity_y_values)) # Using `self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs)` instead of just `self.fragile_linear_neuron_IDXs` should yield sorted results
        if not np.alltrue(self.unit_sort_order == self.fragile_linear_neuron_IDXs):
            if debug_print:
                print(f'update_series_identity_y_values(): building sorted version...')
            # Copy the `self.series_identity_y_values` and sort them according to `self.unit_sort_order`
            _sorted_map_values = self.series_identity_y_values[self.unit_sort_order].copy() # sort the y-values
            # Builds the sorted version by sorting the map values before building:
            self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, _sorted_map_values)) # Old way

        else:
            if debug_print:
                print(f'update_series_identity_y_values(): (self.unit_sort_order == self.fragile_linear_neuron_IDXs) (default sort).')
            self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, self._series_identity_y_values)) # Old way 


    ## Required for DataSeriesToSpatialTransformingMixin
    def fragile_linear_neuron_IDX_to_spatial(self, fragile_linear_neuron_IDXs):
        """ transforms the fragile_linear_neuron_IDXs in fragile_linear_neuron_IDXs to a spatial offset (such as the y-positions for a 3D raster plot) """
        if self.series_identity_y_values is None:
            self.update_series_identity_y_values()
        fragile_linear_neuron_IDX_series_indicies = self.unit_sort_order[fragile_linear_neuron_IDXs] # get the appropriate series index for each fragile_linear_neuron_IDX given their sort order
        return self.series_identity_y_values[fragile_linear_neuron_IDX_series_indicies]

@define
class RasterScatterPlotManager:
    unit_sort_manager: UnitSortOrderManager
    config_fragile_linear_neuron_IDX_map: dict = None

    @property
    def params(self):
        """Passthrough to params."""
        return self.unit_sort_manager.params
    @params.setter
    def params(self, value):
        self.unit_sort_manager.params = value

    @function_attributes(short_name='_build_cell_configs', tags=['config','private'], input_requires=['self.params.neuron_qcolors_map'], output_provides=['self.params.config_items', 'self.config_fragile_linear_neuron_IDX_map'], uses=['self.find_cell_ids_from_neuron_IDXs', 'build_adjusted_color'], used_by=[], creation_date='2023-03-31 18:46')
    def _build_cell_configs(self):
        """ Adds the neuron/cell configurations that are used to color and format the scatterplot spikes and such. 
        Requires:
            self._series_identity_lower_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='left_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
            self._series_identity_upper_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='right_edges', side_bin_margins = self.params.side_bin_margins) / self.n_cells
        
        NOTE: on self.y vs (self._series_identity_lower_y_values, self._series_identity_upper_y_values): two ndarrays of the same length as self.y but they each express the start/end edges of each series as a ratio of the total.
            this means for example: 
                y:       [0.5, 1.5, 2.5, ..., 65.5, 66.5, 67.5]
                lower_y: [0.0, 0.0147059, 0.0294118, ..., 0.955882, 0.970588, 0.985294]
                upper_y: [0.0147059, 0.0294118, 0.0441176, ..., 0.970588, 0.985294, 1.0]

        Adds:
            self.params.config_items: IndexedOrderedDict
            self.config_fragile_linear_neuron_IDX_map: dict<self.fragile_linear_neuron_IDXs, self.params.config_items>
        
        Known Calls:
            From self._buildGraphics()
            From self.on_neuron_colors_changed(...) and self.on_unit_sort_order_changed(...)
        """
        
        # SpikeEmphasisState
        state_alpha = {SpikeEmphasisState.Hidden: 0.01,
                        SpikeEmphasisState.Deemphasized: 0.1,
                        SpikeEmphasisState.Default: 0.5,
                        SpikeEmphasisState.Emphasized: 1.0,
        }
        
        # state_color_adjust_fcns: functions that take the base color and call build_adjusted_color to get the adjusted color for each state
        state_color_adjust_fcns = {SpikeEmphasisState.Hidden: lambda x: build_adjusted_color(x),
                        SpikeEmphasisState.Deemphasized: lambda x: build_adjusted_color(x, saturation_scale=0.35, value_scale=0.8),
                        SpikeEmphasisState.Default: lambda x: build_adjusted_color(x),
                        SpikeEmphasisState.Emphasized: lambda x: build_adjusted_color(x, value_scale=1.25),
        }
        
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        self.params.config_items = IndexedOrderedDict()
        curr_neuron_ids_list = self.unit_sort_manager.find_cell_ids_from_neuron_IDXs(self.unit_sort_manager.fragile_linear_neuron_IDXs)
        
        # builds one config for each neuron color:
        for i, fragile_linear_neuron_IDX in enumerate(self.unit_sort_manager.fragile_linear_neuron_IDXs):
            curr_neuron_id = curr_neuron_ids_list[i] # aclu value
            
            curr_state_pen_dict = dict()
            for an_emphasis_state, alpha_value in state_alpha.items():
                curr_color = self.params.neuron_qcolors_map[fragile_linear_neuron_IDX]
                curr_color.setAlphaF(alpha_value)
                curr_color = state_color_adjust_fcns[an_emphasis_state](curr_color)
                curr_pen = pg.mkPen(curr_color)
                curr_state_pen_dict[an_emphasis_state] = curr_pen
            
            # curr_config_item = (i, fragile_linear_neuron_IDX, curr_state_pen_dict, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i]) # config item is just a tuple here

            # TEST: Seems like these other values are unused, and only curr_config_item[2] (containing the curr_state_pen_dict) is ever accessed in the subsequent functions.
            curr_config_item = (None, None, curr_state_pen_dict, None, None) # config item is just a tuple here
            self.params.config_items[curr_neuron_id] = curr_config_item # add the current config item to the config items 


        #!! SORT: TODO: CRITICAL: this is where I think we do the sorting! We leave everything else in the natural order, and then sort the `self.params.config_items.values()` in this map (assuming they're what are used:
        ## ORIGINAL Unsorted version:
        self.config_fragile_linear_neuron_IDX_map = dict(zip(self.unit_sort_manager.fragile_linear_neuron_IDXs, self.params.config_items.values()))
        
        # ## Attempted sorted version -- NOTE -- DOES NOT WORK:
        # self.config_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, np.array(list(self.params.config_items.values()))[self.unit_sort_order])) # sort using the `unit_sort_order`


# Note that these raster plots could implement some variant of HideShowSpikeRenderingMixin, SpikeRenderingMixin, etc but these classes frankly suck. 

def _plot_empty_raster_plot_frame(scatter_app_name='pho_test', defer_show=False):
    """ simple helper to initialize the mkQApp, spawn the window, and build the plots and plots_data. """
    ## Perform the plotting:
    app = pg.mkQApp(scatter_app_name)
    win = pg.GraphicsLayoutWidget(show=(not defer_show), title=scatter_app_name)
    win.resize(1000,600)
    win.setWindowTitle(f'pyqtgraph: Raster Spikes: {scatter_app_name}')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    plots = RenderPlots(scatter_app_name)
    plots_data = RenderPlotsData(scatter_app_name)

    return app, win, plots, plots_data


@function_attributes(short_name='plot_raster_plot', tags=['pyqtgraph','raster','2D'], input_requires=[], output_provides=[], uses=['_plot_empty_raster_plot_frame'], used_by=[], creation_date='2023-03-31 20:53')
def plot_raster_plot(spikes_df, shared_aclus, scatter_app_name='pho_test'):
    """ This uses pyqtgraph's scatter function like SpikeRaster2D to render a raster plot with colored ticks by default

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_raster_plot

        app, win, plots, plots_data = plot_raster_plot(_temp_active_spikes_df, shared_aclus)

    """
    neuron_ids = deepcopy(shared_aclus)
    n_cells = len(shared_aclus)
    fragile_linear_neuron_IDXs = np.arange(n_cells)
    unit_sort_order = np.arange(n_cells) # in-line sort order
    params = RasterPlotParams()
    params.build_neurons_color_data(fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs)
    manager = UnitSortOrderManager(neuron_ids=neuron_ids, fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs, n_cells=n_cells, unit_sort_order=unit_sort_order, params=params)
    manager.update_series_identity_y_values()
    raster_plot_manager = RasterScatterPlotManager(unit_sort_manager=manager)
    raster_plot_manager._build_cell_configs()

    # Update the dataframe
    spikes_df = _add_spikes_df_visualization_columns(manager, spikes_df)

    # make root container for plots
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=False)

    # each entry in `config_fragile_linear_neuron_IDX_map` has the form:
    # 	(i, fragile_linear_neuron_IDX, curr_pen, _series_identity_lower_y_values[i], _series_identity_upper_y_values[i])

    ## Build the spots for the raster plot:
    plots_data.all_spots = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(spikes_df, raster_plot_manager.config_fragile_linear_neuron_IDX_map)

    # # Actually setup the plot:
    plots.root_plot = win.addPlot() # this seems to be the equivalent to an 'axes'

    # p1 = win.addPlot(title="SpikesDataframe", x=x, y=y, connect='pairs')
    # p1.setLabel('bottom', 'Timestamp', units='[sec]') # set the x-axis label

    # Common Tick Label
    vtick = QtGui.QPainterPath()

    # Defailt Tick Mark:
    # # vtick.moveTo(0, -0.5)
    # # vtick.lineTo(0, 0.5)
    # vtick.moveTo(0, -0.5)
    # vtick.lineTo(0, 0.5)

    # Thicker Tick Label:
    tick_width = 0.1
    half_tick_width = 0.5 * tick_width
    vtick.moveTo(-half_tick_width, -0.5)
    vtick.addRect(-half_tick_width, -0.5, tick_width, 1.0) # x, y, width, height

    plots.scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=10, pen={'color': 'w', 'width': 1})
    plots.scatter_plot.setObjectName('scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    plots.scatter_plot.opts['useCache'] = True
    plots.scatter_plot.addPoints(plots_data.all_spots) # , hoverable=True
    plots.root_plot.addItem(plots.scatter_plot)

    plots.scatter_plot.addPoints(plots_data.all_spots)

    return app, win, plots, plots_data




@function_attributes(short_name=None, tags=['pyqtgraph','raster','2D'], input_requires=[], output_provides=[], uses=['_plot_empty_raster_plot_frame'], used_by=[], creation_date='2023-06-16 20:45', related_items=['plot_raster_plot'])
def plot_multiple_raster_plot(filter_epochs_df: pd.DataFrame, filter_epoch_spikes_df: pd.DataFrame, included_neuron_ids=None, unit_sort_order=None, epoch_id_key_name='temp_epoch_id', scatter_app_name="Pho Stacked Replays"):
    """ This renders a stack of raster plots

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot

        
        app, win, plots, plots_data = plot_multiple_raster_plot(filter_epochs_df, filter_epoch_spikes_df, included_neuron_ids=shared_aclus, epoch_id_key_name='replay_epoch_id', scatter_app_name="Pho Stacked Replays")

    """
    # ## Create the raster plot for the replay:
    # app, win, plots, plots_data = plot_raster_plot(_active_epoch_spikes_df, shared_aclus, scatter_app_name=f"Raster Epoch[{epoch_idx}]")
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=False)

    plots.layout = win.addLayout()
    plots.ax = {}
    plots_data.all_spots_dict = {}

    if included_neuron_ids is not None:
        neuron_ids = deepcopy(included_neuron_ids) # use the provided neuron_ids
    else:
        neuron_ids = np.sort(filter_epoch_spikes_df.aclu.unique()) # get all the aclus from the entire spikes_df frame
        # aclu_to_idx = {aclus[i]:i for i in range(len(aclus))}        

    n_cells = len(neuron_ids)
    fragile_linear_neuron_IDXs = np.arange(n_cells)
    if unit_sort_order is None:
        unit_sort_order = np.arange(n_cells) # in-line sort order
    else:
        assert len(unit_sort_order) == n_cells
    
    params = RasterPlotParams()
    params.build_neurons_color_data(fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs)
    manager = UnitSortOrderManager(neuron_ids=neuron_ids, fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs, n_cells=n_cells, unit_sort_order=unit_sort_order, params=params)
    manager.update_series_identity_y_values()
    raster_plot_manager = RasterScatterPlotManager(unit_sort_manager=manager)
    raster_plot_manager._build_cell_configs()
    # Update the dataframe
    filter_epoch_spikes_df = _add_spikes_df_visualization_columns(manager, filter_epoch_spikes_df)
    
    # Common Tick Label
    vtick = QtGui.QPainterPath()

    # Thicker Tick Label:
    # tick_width = 0.1
    tick_width = 1.0
    half_tick_width = 0.5 * tick_width
    vtick.moveTo(-half_tick_width, -0.5)
    vtick.addRect(-half_tick_width, -0.5, tick_width, 1.0) # x, y, width, height

    scatter_plot_kwargs = dict(pxMode=True, symbol=vtick, size=2, pen={'color': 'w', 'width': 1})

    ## Build the individual epoch raster plot rows:
    for an_epoch in filter_epochs_df.itertuples():
        # print(f'an_epoch: {an_epoch}')
        # if an_epoch.Index < 10:
        _active_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df[epoch_id_key_name] == an_epoch.Index]
        _active_plot_title: str = f"Epoch[{an_epoch.label}]"
        
        ## Create the raster plot for the replay:
        # if win is None:
        # 	# Initialize
        # 	app, win, plots, plots_data = plot_raster_plot(_active_epoch_spikes_df, shared_aclus, scatter_app_name=f"Raster Epoch[{an_epoch.label}]")
        # else:
        # add a new row
        new_ax = plots.layout.addPlot(row=int(an_epoch.Index), col=0)
        spots = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(_active_epoch_spikes_df, raster_plot_manager.config_fragile_linear_neuron_IDX_map)
        plots_data.all_spots_dict[an_epoch.Index] = spots

        scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', **scatter_plot_kwargs)
        scatter_plot.setObjectName(f'scatter_plot_{_active_plot_title}') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
        scatter_plot.opts['useCache'] = True
        scatter_plot.addPoints(plots_data.all_spots_dict[an_epoch.Index]) # , hoverable=True
        new_ax.addItem(scatter_plot)
        new_ax.setXRange(an_epoch.start, an_epoch.stop)
        new_ax.setYRange(0, n_cells-1)
        # Disable Interactivity
        new_ax.setMouseEnabled(x=False, y=False)
        new_ax.setMenuEnabled(False)
        plots.ax[an_epoch.Index] = new_ax

    return app, win, plots, plots_data


    
@function_attributes(short_name=None, tags=['spikes_df', 'raster', 'helper',' filter'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-19 15:25', related_items=['plot_multiple_raster_plot'])
def _prepare_spikes_df_from_filter_epochs(spikes_df: pd.DataFrame, filter_epochs, included_neuron_ids=None, epoch_id_key_name='temp_epoch_id', no_interval_fill_value=-1, debug_print=False) -> pd.DataFrame:
    """ Prepares the spikes_df to be plotted for a given set of filter_epochs and included_neuron_ids by restricting to these periods/aclus.
    
    Usage:
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs

        long_ratemap = long_pf1D.ratemap
        short_ratemap = short_pf1D.ratemap
        curr_any_context_neurons = _find_any_context_neurons(*[k.neuron_ids for k in [long_ratemap, short_ratemap]])

        filter_epoch_spikes_df = _prepare_spikes_df_from_filter_epochs(long_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=curr_any_context_neurons, epoch_id_key_name='replay_epoch_id', debug_print=False) # replay_epoch_id
        filter_epoch_spikes_df

    """
    from neuropy.utils.mixins.time_slicing import add_epochs_id_identity


    if isinstance(filter_epochs, pd.DataFrame):
        filter_epochs_df = filter_epochs
    else:
        filter_epochs_df = filter_epochs.to_dataframe()
        
    if debug_print:
        print(f'filter_epochs: {filter_epochs.epochs.n_epochs}')
        

    ## Get the spikes during these epochs to attempt to decode from:
    filter_epoch_spikes_df = deepcopy(spikes_df)
    filter_epochs_df = deepcopy(filter_epochs_df) # copy just to make sure no modifications happen.
        
    if included_neuron_ids is not None:
        filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['aclu'].isin(included_neuron_ids)] ## restrict to only the shared aclus for both short and long
        
    filter_epoch_spikes_df, _temp_neuron_id_to_new_IDX_map = filter_epoch_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # I think this must be done prior to restricting to the current epoch, but after restricting to the shared_aclus

    ## Add the epoch ids to each spike so we can easily filter on them:
    filter_epoch_spikes_df = add_epochs_id_identity(filter_epoch_spikes_df, filter_epochs_df, epoch_id_key_name=epoch_id_key_name, epoch_label_column_name=None, no_interval_fill_value=no_interval_fill_value)
    if debug_print:
        print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')
    filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df[epoch_id_key_name] != no_interval_fill_value] # Drop all non-included spikes
    if debug_print:
        print(f'np.shape(filter_epoch_spikes_df): {np.shape(filter_epoch_spikes_df)}')

    # returns `filter_epoch_spikes_df`
    return filter_epoch_spikes_df


# ==================================================================================================================== #
# Menu Builders                                                                                                        #
# ==================================================================================================================== #
def _build_additional_spikeRaster2D_menus(spike_raster_plt_2d, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx):
    active_config_name = active_display_fn_identifying_ctx.filter_name # recover active_config_name from the context

    ## Adds the custom renderable menu to the top-level menu of the plots in Spike2DRaster
    # _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_window.spike_raster_plt_2d, computation_result.sess)  # Adds the custom context menus for SpikeRaster2D
    
    _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, owning_pipeline_reference, active_config_name)  # Adds the custom context menus for SpikeRaster2D
    output_references = [_active_2d_plot_renderable_menus]
    return output_references


def _build_additional_window_menus(spike_raster_window, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx):
        assert owning_pipeline_reference is not None
        active_config_name = active_display_fn_identifying_ctx.filter_name # recover active_config_name from the context

        ## SpikeRaster2D Specific Items:
        output_references = _build_additional_spikeRaster2D_menus(spike_raster_window.spike_raster_plt_2d, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx)

        ## Note that curr_main_menu_window is usually not the same as spike_raster_window, instead curr_main_menu_window wraps it and produces the final output window
        curr_main_menu_window, menuConnections, connections_actions_dict = ConnectionControlsMenuMixin.try_add_connections_menu(spike_raster_window)
        spike_raster_window.main_menu_window = curr_main_menu_window # to retain the changes
        
        if owning_pipeline_reference is not None:
            if active_display_fn_identifying_ctx not in owning_pipeline_reference.display_output:
                owning_pipeline_reference.display_output[active_display_fn_identifying_ctx] = PhoUIContainer() # create a new context
            
            display_output = owning_pipeline_reference.display_output[active_display_fn_identifying_ctx]
            # print(f'display_output: {display_output}')
            curr_main_menu_window, menuCreateNewConnectedWidget, createNewConnected_actions_dict = CreateNewConnectedWidgetMenuHelper.try_add_create_new_connected_widget_menu(spike_raster_window, owning_pipeline_reference, active_config_name, display_output) 
            spike_raster_window.main_menu_window = curr_main_menu_window # to retain the changes
            
        else:
            print(f'WARNING: _display_spike_rasters_window(...) has no owning_pipeline_reference in its parameters, so it cannot add the CreateNewConnectedWidgetMenuMixin menus.')
            menuCreateNewConnectedWidget = None
            createNewConnected_actions_dict = None
            
        # Debug Menu
        _debug_menu_provider = DebugMenuProviderMixin(render_widget=spike_raster_window)
        spike_raster_window.main_menu_window.ui.menus.global_window_menus.debug.menu_provider_obj = _debug_menu_provider
        
        # Docked Menu
        _docked_menu_provider = DockedWidgets_MenuProvider(render_widget=spike_raster_window)
        _docked_menu_provider.DockedWidgets_MenuProvider_on_buildUI(spike_raster_window=spike_raster_window, owning_pipeline_reference=owning_pipeline_reference, context=active_display_fn_identifying_ctx, active_config_name=active_config_name, display_output=owning_pipeline_reference.display_output[active_display_fn_identifying_ctx])
        spike_raster_window.main_menu_window.ui.menus.global_window_menus.docked_widgets.menu_provider_obj = _docked_menu_provider
        
        # Create Linked Widget Menu
        ## Adds the custom renderable menu to the top-level menu of the plots in Spike2DRaster
        active_pf_2D_dt = computation_result.computed_data.get('pf2D_dt', None)
        if active_pf_2D_dt is not None:
            active_pf_2D_dt.reset()
            active_pf_2D_dt.update(t=45.0, start_relative_t=True)

            # _createLinkedWidget_menus = LocalMenus_AddRenderable.add_Create_Paired_Widget_menu(spike_raster_window, active_pf_2D_dt)  # Adds the custom context menus for SpikeRaster2D
            _createLinkedWidget_menu_provider = CreateLinkedWidget_MenuProvider(render_widget=spike_raster_window)
            if owning_pipeline_reference is not None:
                if active_display_fn_identifying_ctx not in owning_pipeline_reference.display_output:
                    owning_pipeline_reference.display_output[active_display_fn_identifying_ctx] = PhoUIContainer() # create a new context
            
                display_output = owning_pipeline_reference.display_output[active_display_fn_identifying_ctx]
                _createLinkedWidget_menu_provider.CreateLinkedWidget_MenuProvider_on_buildUI(spike_raster_window=spike_raster_window, active_pf_2D_dt=active_pf_2D_dt, context=active_display_fn_identifying_ctx, display_output=display_output)
            else:
                print(f'WARNING: owning_pipeline_reference is NONE in  _display_spike_rasters_window!')   
        else:
            print(f'active_pf_2D_dt is None! Skipping Create Paired Widget Menu...')
            _createLinkedWidget_menu_provider = None
    
        spike_raster_window.main_menu_window.ui.menus.global_window_menus.create_linked_widget.menu_provider_obj = _createLinkedWidget_menu_provider

        output_references = output_references.extend([curr_main_menu_window, menuConnections, connections_actions_dict,
            curr_main_menu_window, menuCreateNewConnectedWidget, createNewConnected_actions_dict,
            _debug_menu_provider,
            _docked_menu_provider,
            _createLinkedWidget_menu_provider
        ])
        return output_references