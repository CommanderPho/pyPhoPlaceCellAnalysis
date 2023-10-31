from typing import Any, Optional, Union
from copy import deepcopy
import numpy as np
import pandas as pd
from functools import partial
from attrs import define, Factory
from indexed import IndexedOrderedDict

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin
from neuropy.utils.result_context import overwriting_display_context, providing_context


from pyphocorehelpers.indexing_helpers import partition # needed by `_find_example_epochs` to partition the dataframe by aclus
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots, RenderPlotsData
from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color # required for the different emphasis states in ._build_cell_configs()

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers, UnitColoringMode # for build_neurons_color_data
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
        
        Uses:
            computation_result.sess.spikes_df
            
        
        """
        use_separate_windows = kwargs.pop('separate_windows', False)
        type_of_3d_plotter = kwargs.pop('type_of_3d_plotter', 'pyqtgraph')
        # active_plotting_config = active_config.plotting_config # active_config is unused
        active_config_name = kwargs.pop('active_config_name', 'Unknown')
        active_identifying_context = kwargs.pop('active_context', None)
        assert active_identifying_context is not None
        owning_pipeline_reference = kwargs.pop('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        neuron_colors=kwargs.pop('neuron_colors', None)
        neuron_sort_order=kwargs.pop('neuron_sort_order', None)
        
        included_neuron_ids = kwargs.pop('included_neuron_ids', None)
        spikes_df = computation_result.sess.spikes_df
        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids

        # TODO: slice neuron_sort_order, neuron_colors as well now

        spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids).copy()
        
        ## Finally, add the display function to the active context
        active_display_fn_identifying_ctx = active_identifying_context.adding_context('display_fn', display_fn_name='display_spike_rasters_window')
        active_display_fn_identifying_ctx_string = active_display_fn_identifying_ctx.get_description(separator='|') # Get final discription string:

        ## It's passed a specific computation_result which has a .sess attribute that's used to determine which spikes are displayed or not.
        spike_raster_window = Spike3DRasterWindowWidget(spikes_df, type_of_3d_plotter=type_of_3d_plotter, application_name=f'Spike Raster Window - {active_display_fn_identifying_ctx_string}', neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order)
        # Set Window Title Options:
        a_file_prefix = str(computation_result.sess.filePrefix.resolve())
        spike_raster_window.setWindowFilePath(a_file_prefix)
        spike_raster_window.setWindowTitle(f'Spike Raster Window - {active_config_name} - {a_file_prefix}')
        
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

    def build_neurons_color_data(self, fragile_linear_neuron_IDXs, neuron_colors_list=None, coloring_mode:UnitColoringMode=UnitColoringMode.COLOR_BY_INDEX_ORDER) -> None:
        """ Cell Coloring function

        Inputs:
            neuron_colors_list: a list of neuron colors
                if None provided will call DataSeriesColorHelpers._build_cell_qcolor_list(...) to build them.
            
            mode:
                'preserve_fragile_linear_neuron_IDXs': color is assigned based off of fragile_linear_neuron_IDX value, meaning after re-sorting the fragile_linear_neuron_IDXs the colors will appear visually different along y but will correspond to the same units as before the sort.
                'color_by_index_order': color is assigned based of the raw index order of the passed-in unit ids. This means after re-sorting the units the colors will appear visually the same along y, but will not correspond to the same units.
        
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
            neuron_qcolors_list = DataSeriesColorHelpers._build_cell_qcolor_list(unsorted_fragile_linear_neuron_IDXs, mode=coloring_mode, provided_cell_colors=None)
            for a_color in neuron_qcolors_list:
                a_color.setAlphaF(0.5)
        else:
            neuron_qcolors_list = DataSeriesColorHelpers._build_cell_qcolor_list(unsorted_fragile_linear_neuron_IDXs, mode=coloring_mode, provided_cell_colors=neuron_colors_list.copy()) # builts a list of qcolors
                                
        neuron_qcolors_map = dict(zip(unsorted_fragile_linear_neuron_IDXs, neuron_qcolors_list))

        self.neuron_qcolors = deepcopy(neuron_qcolors_list)
        self.neuron_qcolors_map = deepcopy(neuron_qcolors_map)

        # allocate new neuron_colors array:
        self.neuron_colors = np.zeros((4, n_cells))
        for i, curr_qcolor in enumerate(self.neuron_qcolors):
            curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
            self.neuron_colors[:, i] = curr_color[:]
        
        self.neuron_colors_hex = None
        
        # get hex colors:
        self.neuron_colors_hex = [self.neuron_qcolors[i].name(QtGui.QColor.HexRgb) for i, cell_id in enumerate(fragile_linear_neuron_IDXs)]
        return self






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


    def update_spikes_df_visualization_columns(self, spikes_df: pd.DataFrame, overwrite_existing:bool=True):
        """ updates spike_df's columns: ['visualization_raster_y_location', 'visualization_raster_emphasis_state'] """
        if overwrite_existing or ('visualization_raster_y_location' not in spikes_df.columns):
            all_y = [self.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
            spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes. BUG? NO, RESOLVED: actually, this should be updated when anything that would change .y_fragile_linear_neuron_IDX_map would change, right? Meaning: .y, ... oh, I see. y doesn't change because params.center_mode, params.bin_position_mode, and params.side_bin_margins aren't expected to change. 

        if overwrite_existing or ('visualization_raster_emphasis_state' not in spikes_df.columns):
            # TODO: This might be the one we don't want to overwrite unless it's missing, as we probably don't want to always reset it to default emphasis if a column with customized values already exists.
            spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default
        return spikes_df


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
                        SpikeEmphasisState.Default: 0.95, # SpikeEmphasisState.Default: 0.5,
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

def _plot_empty_raster_plot_frame(scatter_app_name='pho_test', defer_show=False, active_context=None) -> tuple[Any, pg.GraphicsLayoutWidget, RenderPlots, RenderPlotsData]:
    """ simple helper to initialize the mkQApp, spawn the window, and build the plots and plots_data. """
    ## Perform the plotting:
    app = pg.mkQApp(scatter_app_name)
    win = pg.GraphicsLayoutWidget(show=(not defer_show), title=scatter_app_name)
    win.resize(1000,600)
    # window_title_prefix = 'pyqtgraph: Raster Spikes: '
    window_title_prefix = '' # no prefix before the provided title
    win.setWindowTitle(f'{window_title_prefix}{scatter_app_name}')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    plots = RenderPlots(scatter_app_name)
    plots_data = RenderPlotsData(scatter_app_name)
    if active_context is not None:
        plots_data.active_context = active_context

    return app, win, plots, plots_data



def _plot_multi_grid_raster_plot_frame(scatter_app_name='pho_test', defer_show=False, active_context=None) -> tuple[Any, pg.GraphicsLayoutWidget, RenderPlots, RenderPlotsData]:
    """ simple helper to initialize the mkQApp, spawn the window, and build the plots and plots_data. """
    ## Perform the plotting:
    app = pg.mkQApp(scatter_app_name)
    win = pg.GraphicsLayoutWidget(show=(not defer_show), title=scatter_app_name)
    win.resize(1000,600)
    # window_title_prefix = 'pyqtgraph: Raster Spikes: '
    window_title_prefix = '' # no prefix before the provided title
    win.setWindowTitle(f'{window_title_prefix}{scatter_app_name}')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    plots = RenderPlots(scatter_app_name)
    plots_data = RenderPlotsData(scatter_app_name)
    if active_context is not None:
        plots.context = active_context
        plots_data.active_context = active_context

    plots.layout = win.addLayout()
    plots.ax = {}
    plots.scatter_plots = {} # index is the _active_plot_identifier
    plots.grid = {} # index is the _active_plot_identifier

    plots_data.all_spots_dict = {}
    plots_data.all_scatterplot_tooltips_kwargs_dict = {}

    plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list)


    return app, win, plots, plots_data




def _build_default_tick(tick_width: float = 0.1, tick_height: float = 1.0) -> QtGui.QPainterPath:
    """ 

    vtick = _build_default_tick(tick_width=0.1)
    override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItem', pxMode=True, symbol=vtick, size=10.0, pen={'color': 'w', 'width': 1}, brush=pg.mkBrush(color='w'), hoverable=False)
    
    vtick = _build_default_tick(tick_width=0.01, tick_height=1.0)
    override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItem', pxMode=False, symbol=vtick, size=1, pen={'color': 'w', 'width': 1}, brush=pg.mkBrush(color='w'), hoverable=False)

    """
    half_tick_height: float = 0.5 * float(tick_height)

    # Common Tick Label
    vtick = QtGui.QPainterPath()

    
    if ((tick_width is None) or (tick_width == 0.0)):
        # Defailt Tick Mark:
        vtick.moveTo(0, -half_tick_height)
        vtick.lineTo(0, half_tick_height)
    else:
        # Thicker (Rect) Tick Label:
        half_tick_width = 0.5 * float(tick_width)
        
        vtick.moveTo(-half_tick_width, -half_tick_height)
        vtick.addRect(-half_tick_width, -half_tick_height, tick_width, half_tick_height) # x, y, width, height
    return vtick

def build_scatter_plot_kwargs(scatter_plot_kwargs=None):
    """build the default scatter plot kwargs, and merge them with the provided kwargs"""
    # Common Tick Label 
    vtick = _build_default_tick(tick_width=1.0)
    default_scatter_plot_kwargs = dict(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=2, pen={'color': 'w', 'width': 1}, hoverable=True)

    if scatter_plot_kwargs is None:
        merged_kwargs = default_scatter_plot_kwargs
    else:
        # merge the two
        # scatter_plot_kwargs = default_scatter_plot_kwargs | scatter_plot_kwargs
        # Merge the default kwargs with the user-provided kwargs
        merged_kwargs = {**default_scatter_plot_kwargs, **scatter_plot_kwargs}

    print(f'merged_kwargs: {merged_kwargs}')
    return merged_kwargs




def _build_units_y_grid(plot_item) -> pg.GridItem:
    """create a GridItem and add it to the plot
    
    Usage:
        grid = _build_units_y_grid(plot_item)
    
    """
    grid = pg.GridItem()
    plot_item.addItem(grid)
    # set the properties of the grid
    grid.setTickSpacing([], [1, 5, 10 ]) # on the y-axis (units) set lines every 1, 5, and 10 units. Looks great on linux.
    grid.setPen(pg.mkPen('#888888', width=1))
    grid.setZValue(-100)
    return grid


def _build_scatter_plotting_managers(plots_data, spikes_df, included_neuron_ids=None, unit_sort_order=None, unit_colors_list=None):
    """ 
    Does not modify `spikes_df` and it's unused when included_neuron_ids is provided

    Usage:
        plots_data = _build_scatter_plotting_managers(plots_data, included_neuron_ids=included_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list)
    """
    if included_neuron_ids is not None:
        neuron_ids = deepcopy(included_neuron_ids) # use the provided neuron_ids
    else:
        neuron_ids = np.sort(spikes_df.aclu.unique()) # get all the aclus from the entire spikes_df frame
    plots_data.n_cells = len(neuron_ids)
    fragile_linear_neuron_IDXs = np.arange(plots_data.n_cells)
    if unit_sort_order is None:
        unit_sort_order = np.arange(plots_data.n_cells) # in-line sort order
    else:
        assert len(unit_sort_order) == plots_data.n_cells

    params = RasterPlotParams()
    # params.build_neurons_color_data(fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs) # normal coloring of neurons
    params.build_neurons_color_data(fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs, neuron_colors_list=unit_colors_list)
    
    manager = UnitSortOrderManager(neuron_ids=neuron_ids, fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs, n_cells=plots_data.n_cells, unit_sort_order=unit_sort_order, params=params)
    manager.update_series_identity_y_values()
    raster_plot_manager = RasterScatterPlotManager(unit_sort_manager=manager)
    raster_plot_manager._build_cell_configs()
    ## Add the managers to the plot_data
    plots_data.params = params
    plots_data.unit_sort_manager = manager
    plots_data.raster_plot_manager = raster_plot_manager
    return plots_data


def _build_scatterplot(new_ax) -> pg.GridItem:
    """create a GridItem and add it to the plot

    Usage:
        plots.scatter_plots[an_epoch.Index] = _build_scatterplot(new_ax)

    """
    scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', **scatter_plot_kwargs)
    scatter_plot.setObjectName(f'scatter_plot_{_active_plot_title}') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    scatter_plot.opts['useCache'] = False
    scatter_plot.addPoints(plots_data.all_spots_dict[an_epoch.Index]) # , hoverable=True

    ## Add to the axis and set up the axis:
    new_ax.addItem(scatter_plot)

    new_ax.setXRange(an_epoch.start, an_epoch.stop)
    new_ax.setYRange(0, n_cells-1)
    # new_ax.showAxes(True, showValues=(True, True, True, False)) # showValues=(left: True, bottom: True, right: False, top: False) # , size=10       
    new_ax.hideButtons() # Hides the auto-scale button
    new_ax.setDefaultPadding(0.0)  # plot without padding data range
    # Format Labels:
    # left_label: str = f'Epoch[{an_epoch.label}]: {an_epoch.start:.2f}' # Full label
    # left_label: str = f'Epoch[{an_epoch.label}]' # Epoch[idx] style label
    left_label: str = f'[{an_epoch.label}]' # very short (index only) label
    new_ax.getAxis('left').setLabel(left_label)
    # new_ax.getAxis('bottom').setLabel('t')
    # new_ax.getAxis('right').setLabel(f'Epoch[{an_epoch.label}]: {an_epoch.stop:.2f}')

    # new_ax.getAxis('bottom').setTickSpacing(1.0) # 5.0, 1.0 .setTickSpacing(x=[None], y=[1.0])
    # new_ax.showGrid(x=False, y=True, alpha=1.0)
    new_ax.getAxis('bottom').setStyle(showValues=False)

    # Disable Interactivity
    new_ax.setMouseEnabled(x=False, y=False)
    new_ax.setMenuEnabled(False)
    return scatter_plot


def _subfn_build_and_add_scatterplot_row(plots_data, plots, _active_plot_identifier: str, row:int, col:int=0, left_label: Optional[str]=None, scatter_plot_kwargs=None):
    """ Adds a single scatter plot row to the plots_data/plots with identifier '_active_plot_identifier':
    usage:
    
    scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs)
    scatter_plot, new_ax, y_grid = _subfn_build_and_add_scatterplot_row(plots_data, plots, _active_plot_identifier=_active_plot_identifier, row=int(_active_plot_identifier), col=0, left_label=left_label)
    
    """
    if scatter_plot_kwargs is None:
        scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs)


    new_ax = plots.layout.addPlot(row=row, col=col)
    # plots_data.all_spots_dict[_active_plot_identifier], plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(_active_epoch_spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

    scatter_plot = pg.ScatterPlotItem(**scatter_plot_kwargs)
    scatter_plot.setObjectName(f'scatter_plot_{_active_plot_identifier}') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    scatter_plot.opts['useCache'] = False
    # scatter_plot.addPoints(plots_data.all_spots_dict[_active_plot_identifier], **(plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {})) # , hoverable=True
    new_ax.addItem(scatter_plot)
    plots.scatter_plots[_active_plot_identifier] = scatter_plot
    # new_ax.setXRange(an_epoch.start, an_epoch.stop)
    new_ax.setYRange(0, plots_data.n_cells-1)
    # new_ax.showAxes(True, showValues=(True, True, True, False)) # showValues=(left: True, bottom: True, right: False, top: False) # , size=10       
    new_ax.hideButtons() # Hides the auto-scale button
    new_ax.setDefaultPadding(0.0)  # plot without padding data range
    # Format Labels:
    if left_label is not None:
        new_ax.getAxis('left').setLabel(left_label)
    new_ax.getAxis('bottom').setStyle(showValues=False)

    # Disable Interactivity
    new_ax.setMouseEnabled(x=False, y=False)
    new_ax.setMenuEnabled(False)

    # build the y-axis grid to separate the units
    plots.grid[_active_plot_identifier] = _build_units_y_grid(new_ax)
    plots.ax[_active_plot_identifier] = new_ax
    
    return plots.scatter_plots[_active_plot_identifier], plots.ax[_active_plot_identifier], plots.grid[_active_plot_identifier]


@function_attributes(short_name=None, tags=['plotting','raster', 'sort'], input_requires=[], output_provides=[], uses=['_subfn_build_and_add_scatterplot_row'], used_by=[], creation_date='2023-10-30 22:23', related_items=[])
def _plot_multi_sort_raster_browser(spikes_df: pd.DataFrame, included_neuron_ids, unit_sort_orders_dict=None, unit_colors_list_dict=None, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None):
    """ 

    Basic Plotting:    
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _plot_multi_sort_raster_browser

        included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        unit_sort_orders_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (even_long, odd_long, even_short, odd_short)))
        unit_colors_list_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (unit_colors_list, unit_colors_list, unit_colors_list, unit_colors_list)))

        app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = _plot_multi_sort_raster_browser(spikes_df, included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)


    Updating Raster Epoch:

        active_epoch_idx: int = 11
        curr_epoch_spikes = spikes_df[(spikes_df.new_lap_IDX == active_epoch_idx)]
        curr_epoch_df = active_epochs_df[(active_epochs_df.lap_id == (active_epoch_idx+1))]
        curr_epoch = list(curr_epoch_df.itertuples())[0]

        on_update_active_epoch(curr_epoch)



    Updating Raster Display:

        vtick = _build_default_tick(tick_width=0.0, tick_height=0.9)
        override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItemSimpleSpike', pxMode=False, symbol=vtick, size=1, hoverable=False) # , pen=None, brush=None
        on_update_active_scatterplot_kwargs(override_scatter_plot_kwargs)


    """
    # scatter_plot_kwargs = None

    # rebuild_spikes_df_anyway = True # if True, `_prepare_spikes_df_from_filter_epochs` is called to rebuild spikes_df given the filter_epochs_df even if it contains the desired column already.
    # if rebuild_spikes_df_anyway:
    # 	spikes_df = spikes_df.copy() # don't modify the original dataframe
    # 	filter_epochs_df = filter_epochs_df.copy()

    # if rebuild_spikes_df_anyway or (epoch_id_key_name not in spikes_df.columns):
    # 	# missing epoch_id column in the spikes_df, need to rebuild
    # 	spikes_df = _prepare_spikes_df_from_filter_epochs(spikes_df, filter_epochs=filter_epochs_df, included_neuron_ids=included_neuron_ids, epoch_id_key_name=epoch_id_key_name, debug_print=False) # replay_epoch_id

    # ## Create the raster plot for the replay:
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)

    plots.layout = win.addLayout()
    plots.ax = {}
    plots.scatter_plots = {} # index is the _active_plot_identifier
    plots.grid = {} # index is the _active_plot_identifier

    plots_data.all_spots_dict = {}
    plots_data.all_scatterplot_tooltips_kwargs_dict = {}

    # Build the base data that will be copied for each epoch:
    plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=list(unit_sort_orders_dict.values())[0], unit_colors_list=list(unit_colors_list_dict.values())[0])
    # Update the dataframe
    spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(spikes_df)


    # Common Tick Label 
    # override_scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs)
    # vtick_box = _build_default_tick(tick_width=0.01, tick_height=1.0)
    # # override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItem', pxMode=False, symbol=vtick, size=1, pen={'color': 'w', 'width': 1}, brush=pg.mkBrush(color='w'), hoverable=False)
    # override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItem', pxMode=False, symbol=vtick_box, size=1, hoverable=False) # , pen=None, brush=None

    vtick_simple_line = _build_default_tick(tick_width=0.0, tick_height=0.9)
    override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItemSimpleSpike', pxMode=False, symbol=vtick_simple_line, size=1, hoverable=False) # , pen=None, brush=None
    # print(f'override_scatter_plot_kwargs: {override_scatter_plot_kwargs}')

    
    # list(plots.scatter_plots.keys()) # ['long_even', 'long_odd', 'short_even', 'short_odd']

    i = 0
    plots_data_dict = {} # new dict to hold plot data
    plots_spikes_df_dict = {}

    for _active_plot_identifier, active_unit_sort_order in unit_sort_orders_dict.items():
        # new_plots_data = deepcopy(plots_data)
        new_plots_data = plots_data

        if unit_colors_list_dict is not None:
            unit_colors_list = unit_colors_list_dict.get(_active_plot_identifier, None)
        else:
            unit_colors_list = None

        
        new_plots_data = _build_scatter_plotting_managers(new_plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=active_unit_sort_order, unit_colors_list=unit_colors_list)
        plots_data_dict[_active_plot_identifier] = new_plots_data
        
        # Update the dataframe
        plots_spikes_df_dict[_active_plot_identifier] = deepcopy(spikes_df)
        plots_spikes_df_dict[_active_plot_identifier] = plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(plots_spikes_df_dict[_active_plot_identifier])
        ## Build the spots for the raster plot:
        # plots_data.all_spots, plots_data.all_scatterplot_tooltips_kwargs = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)
        
        plots_data.all_spots_dict[_active_plot_identifier], plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(plots_spikes_df_dict[_active_plot_identifier], plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

        _subfn_build_and_add_scatterplot_row(plots_data_dict[_active_plot_identifier], plots, _active_plot_identifier=_active_plot_identifier, row=(i), col=0, left_label=_active_plot_identifier, scatter_plot_kwargs=override_scatter_plot_kwargs)
        i = i+1

        ## Get the scatterplot and update the points:
        a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
        a_scatter_plot.addPoints(plots_data.all_spots_dict[_active_plot_identifier], **(plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {})) # , hoverable=True
        

    def on_update_active_scatterplot_kwargs(override_scatter_plot_kwargs):
        for _active_plot_identifier in ['long_even', 'long_odd', 'short_even', 'short_odd']:
            new_ax = plots.ax[_active_plot_identifier]
            a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
            a_scatter_plot.setData(plots_data.all_spots_dict[_active_plot_identifier], **(plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {}), **override_scatter_plot_kwargs)

    def on_update_active_epoch(an_epoch):
        for _active_plot_identifier in ['long_even', 'long_odd', 'short_even', 'short_odd']:
            new_ax = plots.ax[_active_plot_identifier]
            new_ax.setXRange(an_epoch.start, an_epoch.stop)
            new_ax.getAxis('left').setLabel(f'[{an_epoch.label}]')
            
            # a_scatter_plot = plots.scatter_plots[_active_plot_identifier]

    return app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs



@providing_context(fn_name='plot_raster_plot')
@function_attributes(short_name='plot_raster_plot', tags=['pyqtgraph','raster','2D'], input_requires=[], output_provides=[], uses=['_plot_empty_raster_plot_frame'], used_by=[], creation_date='2023-03-31 20:53')
def plot_raster_plot(spikes_df: pd.DataFrame, included_neuron_ids, unit_sort_order=None, unit_colors_list=None, scatter_plot_kwargs=None, scatter_app_name='pho_test', defer_show=False, active_context=None, **kwargs) -> tuple[Any, pg.GraphicsLayoutWidget, RenderPlots, RenderPlotsData]:
    """ This uses pyqtgraph's scatter function like SpikeRaster2D to render a raster plot with colored ticks by default

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_raster_plot

        app, win, plots, plots_data = plot_raster_plot(_temp_active_spikes_df, shared_aclus)

    """
    
    # make root container for plots
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)
    
    plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list)
    # Update the dataframe
    spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(spikes_df)
    
    ## Build the spots for the raster plot:
    plots_data.all_spots, plots_data.all_scatterplot_tooltips_kwargs = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

    # # Actually setup the plot:
    plots.root_plot = win.addPlot() # this seems to be the equivalent to an 'axes'

    scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs)
    
    plots.scatter_plot = pg.ScatterPlotItem(**scatter_plot_kwargs)
    plots.scatter_plot.setObjectName('scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    plots.scatter_plot.opts['useCache'] = True
    plots.scatter_plot.addPoints(plots_data.all_spots, **(plots_data.all_scatterplot_tooltips_kwargs or {})) # , hoverable=True
    plots.root_plot.addItem(plots.scatter_plot)

    # build the y-axis grid to separate the units
    plots.grid = _build_units_y_grid(plots.root_plot)

    return app, win, plots, plots_data

@providing_context(fn_name='plot_multiple_raster_plot')
@function_attributes(short_name=None, tags=['pyqtgraph','raster','2D'], input_requires=[], output_provides=[], uses=['_prepare_spikes_df_from_filter_epochs', '_plot_empty_raster_plot_frame'], used_by=[], creation_date='2023-06-16 20:45', related_items=['plot_raster_plot'])
def plot_multiple_raster_plot(filter_epochs_df: pd.DataFrame, spikes_df: pd.DataFrame, included_neuron_ids=None, unit_sort_order=None, unit_colors_list=None, scatter_plot_kwargs=None, epoch_id_key_name='temp_epoch_id', scatter_app_name="Pho Stacked Replays", defer_show=False, active_context=None, **kwargs):
    """ This renders a stack of raster plots (one for each Epoch) for the epochs specified in `filter_epochs_df`. The rasters are determined from spikes_df.

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multiple_raster_plot

        
        app, win, plots, plots_data = plot_multiple_raster_plot(filter_epochs_df, spikes_df, included_neuron_ids=shared_aclus, epoch_id_key_name='replay_epoch_id', scatter_app_name="Pho Stacked Replays")

    """

    #TODO 2023-06-20 08:59: - [ ] Can potentially reuse `stacked_epoch_slices_view` (the pyqtgraph version)?

    rebuild_spikes_df_anyway = True # if True, `_prepare_spikes_df_from_filter_epochs` is called to rebuild spikes_df given the filter_epochs_df even if it contains the desired column already.
    if rebuild_spikes_df_anyway:
        spikes_df = spikes_df.copy() # don't modify the original dataframe
        filter_epochs_df = filter_epochs_df.copy()

    if rebuild_spikes_df_anyway or (epoch_id_key_name not in spikes_df.columns):
        # missing epoch_id column in the spikes_df, need to rebuild
        spikes_df = _prepare_spikes_df_from_filter_epochs(spikes_df, filter_epochs=filter_epochs_df, included_neuron_ids=included_neuron_ids, epoch_id_key_name=epoch_id_key_name, debug_print=False) # replay_epoch_id

    # ## Create the raster plot for the replay:
    # app, win, plots, plots_data = plot_raster_plot(_active_epoch_spikes_df, shared_aclus, scatter_app_name=f"Raster Epoch[{epoch_idx}]")
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)
    # setting plot window background color to white
    win.setBackground('w')
    # win.setForeground('k')
    
    plots.layout = win.addLayout()
    plots.ax = {}
    plots.scatter_plots = {} # index is the an_epoch.Index
    plots.grid = {} # index is the an_epoch.Index
    
    plots_data.all_spots_dict = {}
    plots_data.all_scatterplot_tooltips_kwargs_dict = {}
    plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list)

    # Update the dataframe
    spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(spikes_df)
    
    # Common Tick Label 
    scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs)
    # print(f'scatter_plot_kwargs: {scatter_plot_kwargs}')

    ## Build the individual epoch raster plot rows:
    for an_epoch in filter_epochs_df.itertuples():
        # print(f'an_epoch: {an_epoch}')
        # if an_epoch.Index < 10:
        _active_epoch_spikes_df = spikes_df[spikes_df[epoch_id_key_name] == an_epoch.Index]
        _active_plot_title: str = f"Epoch[{an_epoch.label}]" # Epoch[idx]
        # _active_plot_title: str = f"[{an_epoch.label}]" # [idx]
        
        ## Create the raster plot for the replay:
        # if win is None:
        # 	# Initialize
        # 	app, win, plots, plots_data = plot_raster_plot(_active_epoch_spikes_df, shared_aclus, scatter_app_name=f"Raster Epoch[{an_epoch.label}]")
        # else:
        # add a new row
        new_ax = plots.layout.addPlot(row=int(an_epoch.Index), col=0)
        plots_data.all_spots_dict[an_epoch.Index], plots_data.all_scatterplot_tooltips_kwargs_dict[an_epoch.Index] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(_active_epoch_spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

        scatter_plot = pg.ScatterPlotItem(**scatter_plot_kwargs)
        scatter_plot.setObjectName(f'scatter_plot_{_active_plot_title}') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
        scatter_plot.opts['useCache'] = False
        scatter_plot.addPoints(plots_data.all_spots_dict[an_epoch.Index], **(plots_data.all_scatterplot_tooltips_kwargs_dict[an_epoch.Index] or {})) # , hoverable=True
        new_ax.addItem(scatter_plot)
        plots.scatter_plots[an_epoch.Index] = scatter_plot
        new_ax.setXRange(an_epoch.start, an_epoch.stop)
        new_ax.setYRange(0, plots_data.n_cells-1)
        # new_ax.showAxes(True, showValues=(True, True, True, False)) # showValues=(left: True, bottom: True, right: False, top: False) # , size=10       
        new_ax.hideButtons() # Hides the auto-scale button
        new_ax.setDefaultPadding(0.0)  # plot without padding data range
        # Format Labels:
        # left_label: str = f'Epoch[{an_epoch.label}]: {an_epoch.start:.2f}' # Full label
        # left_label: str = f'Epoch[{an_epoch.label}]' # Epoch[idx] style label
        left_label: str = f'[{an_epoch.label}]' # very short (index only) label
        new_ax.getAxis('left').setLabel(left_label)
        # new_ax.getAxis('bottom').setLabel('t')
        # new_ax.getAxis('right').setLabel(f'Epoch[{an_epoch.label}]: {an_epoch.stop:.2f}')

        # new_ax.getAxis('bottom').setTickSpacing(1.0) # 5.0, 1.0 .setTickSpacing(x=[None], y=[1.0])
        # new_ax.showGrid(x=False, y=True, alpha=1.0)
        new_ax.getAxis('bottom').setStyle(showValues=False)

        # Disable Interactivity
        new_ax.setMouseEnabled(x=False, y=False)
        new_ax.setMenuEnabled(False)

        # build the y-axis grid to separate the units
        plots.grid[an_epoch.Index] = _build_units_y_grid(new_ax)

        plots.ax[an_epoch.Index] = new_ax

    return app, win, plots, plots_data


    
@function_attributes(short_name=None, tags=['spikes_df', 'raster', 'helper',' filter'], input_requires=[], output_provides=[], uses=['add_epochs_id_identity'], used_by=['plot_multiple_raster_plot'], creation_date='2023-06-19 15:25', related_items=['plot_multiple_raster_plot'])
def _prepare_spikes_df_from_filter_epochs(spikes_df: pd.DataFrame, filter_epochs, included_neuron_ids=None, epoch_id_key_name='temp_epoch_id', no_interval_fill_value=-1, debug_print=False) -> pd.DataFrame:
    """ Prepares the spikes_df to be plotted for a given set of filter_epochs and included_neuron_ids by restricting to these periods/aclus, 
            - rebuilding the fragile_linear_neuron_IDXs by calling `.rebuild_fragile_linear_neuron_IDXs(...)`
            - adding an additional column to each spike specifying the epoch it belongs to (`epoch_id_key_name`).
    
    Usage:
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _prepare_spikes_df_from_filter_epochs

        long_ratemap = long_pf1D.ratemap
        short_ratemap = short_pf1D.ratemap
        curr_any_context_neurons = _find_any_context_neurons(*[k.neuron_ids for k in [long_ratemap, short_ratemap]])

        spikes_df = _prepare_spikes_df_from_filter_epochs(long_session.spikes_df, filter_epochs=long_replays, included_neuron_ids=curr_any_context_neurons, epoch_id_key_name='replay_epoch_id', debug_print=False) # replay_epoch_id
        spikes_df

    """
    from neuropy.utils.mixins.time_slicing import add_epochs_id_identity


    if isinstance(filter_epochs, pd.DataFrame):
        filter_epochs_df = filter_epochs
    else:
        filter_epochs_df = filter_epochs.to_dataframe()
        
        
    ## Get the spikes during these epochs to attempt to decode from:
    filter_epoch_spikes_df = deepcopy(spikes_df)
    filter_epochs_df = deepcopy(filter_epochs_df) # copy just to make sure no modifications happen.
        
    if included_neuron_ids is not None:
        # filter_epoch_spikes_df = filter_epoch_spikes_df[filter_epoch_spikes_df['aclu'].isin(included_neuron_ids)]
        filter_epoch_spikes_df = filter_epoch_spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids) ## restrict to only the shared aclus for both short and long

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