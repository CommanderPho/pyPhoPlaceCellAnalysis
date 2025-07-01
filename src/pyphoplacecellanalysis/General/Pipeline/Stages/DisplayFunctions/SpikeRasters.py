from typing import Any, Dict, List, Optional, Union, Tuple
from collections import namedtuple
from copy import deepcopy
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
from functools import partial
from attrs import define, field, Factory, asdict
from indexed import IndexedOrderedDict

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin
from neuropy.utils.result_context import overwriting_display_context, providing_context
from neuropy.utils.indexing_helpers import paired_individual_sorting, paired_incremental_sorting, union_of_arrays # `paired_incremental_sort_neurons`

from pyphocorehelpers.indexing_helpers import partition # needed by `_find_example_epochs` to partition the dataframe by aclus
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots, RenderPlotsData
from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color # required for the different emphasis states in ._build_cell_configs()

import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtGui # for QColor
from qtpy.QtGui import QColor, QBrush, QPen
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import ScatterItemData # used in `NewSimpleRaster`
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
    
    @function_attributes(short_name='spike_rasters_pyqtplot_2D', tags=['display','interactive', 'raster', '2D', 'pyqtplot'], input_requires=[], output_provides=[], uses=[], used_by=['Spike3DRasterWindowWidget.find_or_create_if_needed'], creation_date='2023-04-11 03:05')
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

    @function_attributes(short_name='spike_rasters_pyqtplot_3D_with_2D_controls', tags=['display','interactive', 'raster', '2D', 'ui', '3D', 'pyqtplot'], input_requires=[], output_provides=[], uses=['LocalMenus_AddRenderable.add_renderable_context_menu'], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_pyqtplot_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot (via pyqtgraph) with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        use_separate_windows = kwargs.get('separate_windows', False)
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window = build_spike_3d_raster_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=use_separate_windows)
        
        
        active_config_name = kwargs.get('active_config_name', 'Unknown')
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        assert owning_pipeline_reference is not None
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.initialize_renderable_context_menu(spike_raster_plt_2d, owning_pipeline_reference, active_config_name)
        # _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess)  # Adds the custom context menus for SpikeRaster2D
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d':spike_raster_plt_3d, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection, 'spike_raster_window': spike_raster_window}


    @function_attributes(short_name='spike_rasters_vedo_3D_with_2D_controls', tags=['display','interactive', 'raster', '2D', 'ui', '3D', 'vedo'], input_requires=[], output_provides=[], uses=['LocalMenus_AddRenderable.add_renderable_context_menu'], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_vedo_3D_with_2D_controls(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots a standalone 3D raster plot (via Vedo) with a separate 2D raster plot as the window with which you can adjust the viewed window. 
        """ 
        use_separate_windows = kwargs.get('separate_windows', False)
        spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window = build_spike_3d_raster_vedo_with_2d_controls(computation_result.sess.spikes_df, window_duration=1.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=use_separate_windows)
        
        active_config_name = kwargs.get('active_config_name', 'Unknown')
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        assert owning_pipeline_reference is not None
        _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.initialize_renderable_context_menu(spike_raster_plt_2d, owning_pipeline_reference, active_config_name)
        # _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_plt_2d, computation_result.sess) # Adds the custom context menus for SpikeRaster2D
        return {'spike_raster_plt_2d':spike_raster_plt_2d, 'spike_raster_plt_3d_vedo':spike_raster_plt_3d_vedo, 'spike_3d_to_2d_window_connection':spike_3d_to_2d_window_connection, 'spike_raster_window': spike_raster_window}


    @function_attributes(short_name='spike_rasters_window', tags=['display','interactive', 'primary', 'raster', '2D', 'ui', 'pyqtplot'], input_requires=[], output_provides=[], uses=['Spike3DRasterWindowWidget', '_build_additional_window_menus'], used_by=[], creation_date='2023-04-11 03:05')
    @staticmethod
    def _display_spike_rasters_window(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Displays a Spike3DRasterWindowWidget with a configurable set of raster widgets and controls in it.
        
        Uses:
            computation_result.sess.spikes_df
            
        
        """
        use_separate_windows = kwargs.pop('separate_windows', False)
        type_of_3d_plotter = kwargs.pop('type_of_3d_plotter', 'pyqtgraph')
        use_docked_pyqtgraph_plots: bool = kwargs.pop('use_docked_pyqtgraph_plots', False)


        # active_plotting_config = active_config.plotting_config # active_config is unused
        active_config_name = kwargs.pop('active_config_name', 'Unknown')
        active_identifying_context = kwargs.pop('active_context', None)
        assert active_identifying_context is not None
        owning_pipeline_reference = kwargs.pop('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        neuron_colors=kwargs.pop('neuron_colors', None)
        neuron_sort_order=kwargs.pop('neuron_sort_order', None)
        
        included_neuron_ids = kwargs.pop('included_neuron_ids', None)
        spikes_df: pd.DataFrame = computation_result.sess.spikes_df ## pulls from the session here
        if included_neuron_ids is None:
            included_neuron_ids = spikes_df.spikes.neuron_ids

        # TODO: slice neuron_sort_order, neuron_colors as well now

        spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids).copy()
        
        ## Finally, add the display function to the active context
        active_display_fn_identifying_ctx = active_identifying_context.adding_context('display_fn', display_fn_name='display_spike_rasters_window')
        active_display_fn_identifying_ctx_string = active_display_fn_identifying_ctx.get_description(separator='|') # Get final discription string:


        ## It's passed a specific computation_result which has a .sess attribute that's used to determine which spikes are displayed or not.
        spike_raster_window: Spike3DRasterWindowWidget = Spike3DRasterWindowWidget(spikes_df, type_of_3d_plotter=type_of_3d_plotter, application_name=f'Spike Raster Window - {active_display_fn_identifying_ctx_string}', neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order,
                                                                                   params_kwargs=dict(use_docked_pyqtgraph_plots=use_docked_pyqtgraph_plots),
                                                                                   ) ## surprisingly only needs spikes_df !!?!
        # Set Window Title Options:
        a_file_prefix = str(computation_result.sess.filePrefix.resolve())
        spike_raster_window.setWindowFilePath(a_file_prefix)
        spike_raster_window.setWindowTitle(f'Spike Raster Window - {active_config_name} - {a_file_prefix}')
        
        ## Build the additional menus:
        output_references = _build_additional_window_menus(spike_raster_window, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx) ## the menus on the other hand take the entire pipeline, because they might need that valuable DATA

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


NeuronSpikesConfigTuple = namedtuple('NeuronSpikesConfigTuple', ['idx', 'fragile_linear_neuron_IDX', 'curr_state_pen_dict', 'lower_y_value', 'upper_y_value', 'curr_state_brush_dict'])


# ==================================================================================================================== #
# Spike2DRaster-like Managers                                                                                          #
# ==================================================================================================================== #
@define(slots=False)
class RasterPlotParams:
    """ Holds configuration parameters used in determining how to render a raster plot.
    History: factored out of Spike2DRaster to do standalone pyqtgraph plotting of the 2D raster plot.
    """
    center_mode: str = field(default='starting_at_zero') # or 'zero_centered'
    bin_position_mode: str = field(default='bin_center') # or 'left_edges'
    side_bin_margins: float = field(default=0.0)

    # Colors:
    neuron_qcolors: List[QtGui.QColor] = field(default=Factory(list))
    neuron_colors: Optional[np.ndarray] = field(default=None) # of shape (4, self.n_cells)
    neuron_colors_hex: List[str] = field(default=Factory(list)) #
    neuron_qcolors_map: Dict[int, QtGui.QColor] = field(default=Factory(dict)) 

    # Configs:
    config_items: IndexedOrderedDict[int, NeuronSpikesConfigTuple] = field(default=Factory(IndexedOrderedDict))

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

@define(slots=False)
class UnitSortOrderManager(NeuronIdentityAccessingMixin):
    """ factored out of Spike2DRaster to do standalone pyqtgraph plotting of the 2D raster plot.
    
    _neuron_ids, fragile_linear_neuron_IDXs
    
    It looks like modifying `unit_sort_order` only affects:
        .y_fragile_linear_neuron_IDX_map
            
    """
    _neuron_ids: np.ndarray = field() # always kept in the original, unsorted order. np.array([  9,  10,  11,  15,  16,  18,  24,  25,  26,  31,  39,  40,  43,  44,  47,  48,  51,  52,  53,  54,  56,  60,  61,  65,  66,  68,  70,  72,  75,  77,  78,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  98, 101, 102, 104])
    fragile_linear_neuron_IDXs: np.ndarray = field() # np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])
    n_cells: int = field() # = len(shared_aclus)
    unit_sort_order: np.ndarray = field() # = np.arange(n_cells) # in-line sort order, np.array([18, 17, 19,  5, 35, 23, 31,  4, 45, 21, 37, 36, 10,  7, 16,  9,  2, 40, 20, 28, 13, 41, 38, 25, 29, 42,  0, 14, 34, 44, 32, 11, 30, 12, 24,  3, 39,  1,  6, 27,  8, 22, 15, 33, 43, 26])
    _series_identity_y_values: Optional[np.ndarray] = field(default=None) # np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5, 45.5])
    _series_identity_lower_y_values: Optional[np.ndarray] = field(default=None) # np.array([0, 0.0217391, 0.0434783, 0.0652174, 0.0869565, 0.108696, 0.130435, 0.152174, 0.173913, 0.195652, 0.217391, 0.23913, 0.26087, 0.282609, 0.304348, 0.326087, 0.347826, 0.369565, 0.391304, 0.413043, 0.434783, 0.456522, 0.478261, 0.5, 0.521739, 0.543478, 0.565217, 0.586957, 0.608696, 0.630435, 0.652174, 0.673913, 0.695652, 0.717391, 0.73913, 0.76087, 0.782609, 0.804348, 0.826087, 0.847826, 0.869565, 0.891304, 0.913043, 0.934783, 0.956522, 0.978261])
    _series_identity_upper_y_values: Optional[np.ndarray] = field(default=None) # np.array([0.0217391, 0.0434783, 0.0652174, 0.0869565, 0.108696, 0.130435, 0.152174, 0.173913, 0.195652, 0.217391, 0.23913, 0.26087, 0.282609, 0.304348, 0.326087, 0.347826, 0.369565, 0.391304, 0.413043, 0.434783, 0.456522, 0.478261, 0.5, 0.521739, 0.543478, 0.565217, 0.586957, 0.608696, 0.630435, 0.652174, 0.673913, 0.695652, 0.717391, 0.73913, 0.76087, 0.782609, 0.804348, 0.826087, 0.847826, 0.869565, 0.891304, 0.913043, 0.934783, 0.956522, 0.978261, 1])
    y_fragile_linear_neuron_IDX_map: Dict[int, float] = field(default=Factory(dict)) # Dict[fragile_linear_neuron_IDX:y_value]; {0: 18.5, 1: 17.5, 2: 19.5, 3: 5.5, 4: 35.5, 5: 23.5, 6: 31.5, 7: 4.5, 8: 45.5, 9: 21.5, 10: 37.5, 11: 36.5, 12: 10.5, 13: 7.5, 14: 16.5, 15: 9.5, 16: 2.5, 17: 40.5, 18: 20.5, 19: 28.5, 20: 13.5, 21: 41.5, 22: 38.5, 23: 25.5, 24: 29.5, 25: 42.5, 26: 0.5, 27: 14.5, 28: 34.5, 29: 44.5, 30: 32.5, 31: 11.5, 32: 30.5, 33: 12.5, 34: 24.5, 35: 3.5, 36: 39.5, 37: 1.5, 38: 6.5, 39: 27.5, 40: 8.5, 41: 22.5, 42: 15.5, 43: 33.5, 44: 43.5, 45: 26.5}
    params: RasterPlotParams = field(default=Factory(RasterPlotParams))

    @property
    def neuron_ids(self) -> NDArray:
        """ e.g. return np.array(active_epoch_placefields2D.cell_ids) """
        return self._neuron_ids
    
    @property
    def sorted_neuron_ids(self) -> NDArray:
        """ e.g. return the neuron_ids sorted by self.unit_sort_order """
        assert len(self._neuron_ids) == len(self.unit_sort_order)
        return self._neuron_ids[self.unit_sort_order]


    @property
    def series_identity_y_values(self) -> NDArray:
        """The series_identity_y_values property."""
        return self._series_identity_y_values


    # @property
    # def y_from_neuron_ID_map(self) -> Dict[int, float]:
    #     """The series_identity_y_values property."""
    #     sorted_neuron_ids = deepcopy(self.sorted_neuron_ids)
        
    #     return self.y_fragile_linear_neuron_IDX_map


    def __attrs_post_init__(self):
        """ validate and initialize """
        assert len(self.neuron_ids) == self.n_cells
        assert len(self.unit_sort_order) == len(self.unit_sort_order)
        assert len(self.neuron_ids) == len(self.fragile_linear_neuron_IDXs)
        self.update_series_identity_y_values()


    def update_series_identity_y_values(self, debug_print=False):
        """ updates the fixed self._series_identity_y_values using the DataSeriesToSpatial.build_series_identity_axis(...) function.
        
        Updates: self.y_fragile_linear_neuron_IDX_map, (self._series_identity_y_values, self._series_identity_lower_y_values, self._series_identity_upper_y_values)
        
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
            # self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs[self.unit_sort_order].copy(), _sorted_map_values)) # 2023-12-06 - Attempted fix for basically scrambled orders
            # self.y_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs), self._series_identity_y_values)) # Using `self.fragile_linear_neuron_IDX_to_spatial(self.fragile_linear_neuron_IDXs)` instead of just `self.fragile_linear_neuron_IDXs` should yield sorted results

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


    def update_spikes_df_visualization_columns(self, spikes_df: pd.DataFrame, overwrite_existing:bool=True) -> pd.DataFrame:
        """ updates spike_df's columns: ['visualization_raster_y_location', 'visualization_raster_emphasis_state']
        Uses:
            .y_fragile_linear_neuron_IDX_map
            
        """
        if overwrite_existing or ('visualization_raster_y_location' not in spikes_df.columns):
            all_y = [self.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
            spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes. BUG? NO, RESOLVED: actually, this should be updated when anything that would change .y_fragile_linear_neuron_IDX_map would change, right? Meaning: .y, ... oh, I see. y doesn't change because params.center_mode, params.bin_position_mode, and params.side_bin_margins aren't expected to change. 

        if overwrite_existing or ('visualization_raster_emphasis_state' not in spikes_df.columns):
            # TODO: This might be the one we don't want to overwrite unless it's missing, as we probably don't want to always reset it to default emphasis if a column with customized values already exists.
            spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default
        return spikes_df

@define(slots=False)
class RasterScatterPlotManager(NeuronIdentityAccessingMixin):
    """ Consists of `unit_sort_manager: UnitSortOrderManager` and `config_fragile_linear_neuron_IDX_map`
    
    
    Modifying `unit_sort_order` only affects:
        unit_sort_manager.y_fragile_linear_neuron_IDX_map
        
        
    Overall, it looks like:

    
    """
    unit_sort_manager: UnitSortOrderManager = field(default=None)
    config_fragile_linear_neuron_IDX_map: Optional[IndexedOrderedDict[int, NeuronSpikesConfigTuple]] = field(default=None) #  dict<self.fragile_linear_neuron_IDXs, self.params.config_items>

    @property
    def neuron_ids(self) -> NDArray:
        """ passthrough to self.unit_sort_manager. e.g. return np.array(active_epoch_placefields2D.cell_ids) """
        return self.unit_sort_manager.neuron_ids

    @property
    def sorted_neuron_ids(self) -> NDArray:
        """ passthrough to self.unit_sort_manager. e.g. return the neuron_ids sorted by self.unit_sort_order """
        return self.unit_sort_manager.sorted_neuron_ids
    
    @property
    def params(self) -> "RasterPlotParams":
        """Passthrough to params."""
        return self.unit_sort_manager.params
    @params.setter
    def params(self, value: "RasterPlotParams"):
        self.unit_sort_manager.params = value

    @function_attributes(short_name='_build_cell_configs', tags=['config','private'], input_requires=['self.params.neuron_qcolors_map'], output_provides=['self.params.config_items', 'self.config_fragile_linear_neuron_IDX_map'], uses=['self.find_cell_ids_from_neuron_IDXs', 'build_adjusted_color'], used_by=[], creation_date='2023-03-31 18:46')
    def _build_cell_configs(self, should_build_brushes:bool=True):
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
            
            Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(...)
            
        """
        
        # SpikeEmphasisState
        state_alpha = {SpikeEmphasisState.Hidden: 0.01,
                        SpikeEmphasisState.Deemphasized: 0.1,
                        SpikeEmphasisState.Default: 0.95, # SpikeEmphasisState.Default: 0.5,
                        SpikeEmphasisState.Emphasized: 1.0,
        }
        
        # state_color_adjust_fcns: functions that take the base color and call build_adjusted_color to get the adjusted color for each state
        state_color_adjust_fcns = {SpikeEmphasisState.Hidden: lambda x: build_adjusted_color(x, alpha_scale=0.01),
                        SpikeEmphasisState.Deemphasized: lambda x: build_adjusted_color(x, saturation_scale=0.35, value_scale=0.8, alpha_scale=0.1),
                        SpikeEmphasisState.Default: lambda x: build_adjusted_color(x, alpha_scale=0.95),
                        SpikeEmphasisState.Emphasized: lambda x: build_adjusted_color(x, value_scale=1.25, alpha_scale=1.0),
        }
        
        # self._build_neuron_id_graphics(self.ui.main_gl_widget, self.y)
        self.params.config_items = IndexedOrderedDict()
        curr_neuron_ids_list = self.unit_sort_manager.find_cell_ids_from_neuron_IDXs(self.unit_sort_manager.fragile_linear_neuron_IDXs) # this does not seem to necissarily return them in the sorted order
        
        # builds one config for each neuron color:
        for i, fragile_linear_neuron_IDX in enumerate(self.unit_sort_manager.fragile_linear_neuron_IDXs):
            curr_neuron_id = curr_neuron_ids_list[i] # aclu value
            
            curr_state_pen_dict = dict()
            if should_build_brushes:
                curr_state_brush_dict = dict()
            else:
                curr_state_brush_dict = None
                
            for an_emphasis_state, alpha_value in state_alpha.items():
                curr_color = self.params.neuron_qcolors_map[fragile_linear_neuron_IDX]
                curr_color.setAlphaF(alpha_value)
                curr_color = state_color_adjust_fcns[an_emphasis_state](curr_color)
                curr_pen = pg.mkPen(curr_color)
                curr_state_pen_dict[an_emphasis_state] = curr_pen
                if should_build_brushes:
                    curr_brush = pg.mkBrush(curr_color)
                    curr_state_brush_dict[an_emphasis_state] = curr_brush
                    
            # curr_config_item = (i, fragile_linear_neuron_IDX, curr_state_pen_dict, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i]) # config item is just a tuple here

            # TEST: Seems like these other values are unused, and only curr_config_item[2] (containing the curr_state_pen_dict) is ever accessed in the subsequent functions.
            # curr_config_item = (None, None, curr_state_pen_dict, None, None) # config item is just a tuple here
            curr_config_item = NeuronSpikesConfigTuple(None, None, curr_state_pen_dict, None, None, curr_state_brush_dict) # config item is just a tuple here
            self.params.config_items[curr_neuron_id] = curr_config_item # add the current config item to the config items 


        #!! SORT: TODO: CRITICAL: this is where I think we do the sorting! We leave everything else in the natural order, and then sort the `self.params.config_items.values()` in this map (assuming they're what are used:
        ## ORIGINAL Unsorted version:
        self.config_fragile_linear_neuron_IDX_map = dict(zip(self.unit_sort_manager.fragile_linear_neuron_IDXs, self.params.config_items.values()))
        
        # ## Attempted sorted version -- NOTE -- DOES NOT WORK:
        # self.config_fragile_linear_neuron_IDX_map = dict(zip(self.fragile_linear_neuron_IDXs, np.array(list(self.params.config_items.values()))[self.unit_sort_order])) # sort using the `unit_sort_order`


    def __attrs_post_init__(self):
        """ validate and initialize """
        assert self.unit_sort_manager is not None
        self._build_cell_configs()
        assert len(self.config_fragile_linear_neuron_IDX_map) == len(self.params.config_items)
        
    
    def update_sort(self, new_sort_order: NDArray):
        """ conceptually re-sorts the cells """
        assert len(new_sort_order) == len(self.unit_sort_manager.neuron_ids)
        prev_sort = deepcopy(self.unit_sort_manager.unit_sort_order)

        is_new_sort = np.any(new_sort_order != prev_sort)
        if not is_new_sort:
            print(f'WARN: sort did not change.')
            return False
        else:
            # Update the sort
            self.unit_sort_manager.unit_sort_order = deepcopy(new_sort_order)
            self.unit_sort_manager.update_series_identity_y_values()
            self._build_cell_configs() # rebuild self configs
            print(f'new sort complete.')
            return True


    def update_colors(self, new_colors_mapping):
        """ conceptually re-colors the cells """
        raise NotImplementedError
    
    def get_colors(self):
        """ conceptually returns the appropriate colors for the cells """
        raise NotImplementedError



# ==================================================================================================================== #
# Simplified 2023-12-06 Rasters                                                                                        #
# ==================================================================================================================== #


@metadata_attributes(short_name=None, tags=['raster', 'simple', 'working', 'state', 'modal', 'independent'], input_requires=[], output_provides=[], uses=['ScatterItemData'], used_by=[], creation_date='2023-12-06 13:48', related_items=['new_plot_raster_plot'])
@define(slots=False, eq=False)
class NewSimpleRaster:
    """ Simpler one-shot raster plotter that doesn't support the flexiblity of SpikeRaster2D and the managers extracted from it but is much simpler to debug and use. 
    
    All of its fields are Dict with keys of `aclu` (neuron_ID)
    
    To specify a specific sort, pass already sorted neuron_IDs
    
    ```
        if unit_sort_order is None:
            unit_sort_order = np.arange(len(included_neuron_ids))
        active_sorted_neuron_ids = included_neuron_ids[unit_sort_order]
        plots_data.new_sorted_raster = NewSimpleRaster.init_from_neuron_ids(active_sorted_neuron_ids, neuron_colors=unit_colors_list)
    ```
    
    """
    neuron_IDs: NDArray = field(repr=True, metadata={'shape':'n_aclus'})
    neuron_colors: Dict[int, QColor] = field(init=False, repr=False, metadata={'shape':'n_aclus'}) # , default=Factory(dict)
    neuron_y_pos: Dict[int, float] = field(init=False, repr=True, metadata={'shape':'n_aclus'}) # , default=Factory(dict)

    def __attrs_post_init__(self):
        self.neuron_colors = dict()
        self.neuron_y_pos = dict()

    @classmethod
    def init_from_neuron_ids(cls, neuron_IDs, neuron_colors=None):
        _obj = cls(neuron_IDs=neuron_IDs)
        n_cells = len(_obj.neuron_IDs)
        # _obj.neuron_colors = None
        # _obj.neuron_y_pos = None
        
        if neuron_colors is None:	
            neuron_qcolors_list = DataSeriesColorHelpers._build_cell_qcolor_list(np.arange(n_cells), mode=UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS, provided_cell_colors=None)
            _obj.neuron_colors = dict(zip(_obj.neuron_IDs, neuron_qcolors_list))
        else:
            
            if isinstance(neuron_colors, dict):
                assert np.all(np.isin(neuron_IDs, np.array(list(neuron_colors.keys())))), f" if colors dict is provided, all neuron_ids must be present in the neuron_color's keys."
                if len(neuron_colors) > n_cells:
                    print(f'WARN: len(neuron_colors): {len(neuron_colors)} > n_cells: {n_cells}: restricting neuron_colors to the correct aclus, but if colors ever get off this is where it is happening!')
                    _obj.neuron_colors = {k:deepcopy(v) for k,v in neuron_colors.items() if k in neuron_IDs} # only include colors that correspond to active neuron_ids. #TODO 2023-12-11 17:17: - [ ] This might break things!
                else:
                    assert len(neuron_colors) == n_cells
                    _obj.neuron_colors = neuron_colors
            else:
                assert len(neuron_colors) == n_cells
                neuron_qcolors_list = DataSeriesColorHelpers._build_cell_qcolor_list(np.arange(n_cells), mode=UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS, provided_cell_colors=neuron_colors)	
                _obj.neuron_colors = dict(zip(_obj.neuron_IDs, neuron_qcolors_list))
                
        ## build raw y-values:
        _series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(n_cells, center_mode='starting_at_zero', bin_position_mode='bin_center')
        _obj.neuron_y_pos = dict(zip(_obj.neuron_IDs, _series_identity_y_values))
        return _obj


    def update_sort(self, ordered_aclus_list: NDArray):
        """ called to update the sort order of the neuron, adjusting ony `.neuron_y_pos`."""
        raise NotImplementedError


    def update_colors(self, aclu_color_map: Dict[int, QColor]):
        """ updates the neuron's color """
        for aclu, a_color in aclu_color_map.items():
            self.neuron_colors[aclu] = a_color.copy()


    def update_spikes_df_visualization_columns(self, spikes_df: pd.DataFrame, overwrite_existing:bool=True) -> pd.DataFrame:
        """ updates spike_df's columns: ['visualization_raster_y_location', 'visualization_raster_emphasis_state']
        Uses:
            .y_fragile_linear_neuron_IDX_map
            
        Always returns a copy of spikes_df
        
        """
        # Get only the spikes for the shared_aclus:
        a_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(self.neuron_IDs)
        a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

        if overwrite_existing or ('visualization_raster_y_location' not in a_spikes_df.columns):
            # all_y = [self.y_fragile_linear_neuron_IDX_map[a_cell_IDX] for a_cell_IDX in spikes_df['fragile_linear_neuron_IDX'].to_numpy()]
            all_y = [self.neuron_y_pos[aclu] for aclu in a_spikes_df['aclu'].to_numpy()]
            a_spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes. BUG? NO, RESOLVED: actually, this should be updated when anything that would change .y_fragile_linear_neuron_IDX_map would change, right? Meaning: .y, ... oh, I see. y doesn't change because params.center_mode, params.bin_position_mode, and params.side_bin_margins aren't expected to change. 

        if overwrite_existing or ('visualization_raster_emphasis_state' not in a_spikes_df.columns):
            # TODO: This might be the one we don't want to overwrite unless it's missing, as we probably don't want to always reset it to default emphasis if a column with customized values already exists.
            a_spikes_df['visualization_raster_emphasis_state'] = SpikeEmphasisState.Default
        return a_spikes_df


    @function_attributes(short_name=None, tags=['SLOW'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-12 15:21', related_items=[])
    def build_spikes_all_spots_from_df(self, spikes_df: pd.DataFrame, is_spike_included=None, should_return_data_tooltips_kwargs:bool=True, generate_debug_tuples=False, downsampling_rate: int = 1, **kwargs):
        """ builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes 
            Needs to be called whenever:
                spikes_df['visualization_raster_y_location']
                spikes_df['visualization_raster_emphasis_state']
                spikes_df['fragile_linear_neuron_IDX']
            Changes.
        Removed `config_fragile_linear_neuron_IDX_map`
        """
        if (downsampling_rate is not None) and (downsampling_rate > 1):
            active_spikes_df = deepcopy(spikes_df).iloc[::downsampling_rate]  # Take every 10th row
        else:
            active_spikes_df = deepcopy(spikes_df)
        
        # INLINEING `build_spikes_data_values_from_df`: ______________________________________________________________________ #
        # curr_spike_x, curr_spike_y, curr_spike_pens, all_scatterplot_tooltips_kwargs, all_spots, curr_n = cls.build_spikes_data_values_from_df(spikes_df, config_fragile_linear_neuron_IDX_map, is_spike_included=is_spike_included, should_return_data_tooltips_kwargs=should_return_data_tooltips_kwargs, **kwargs)
        # All units at once approach:
        active_time_variable_name = active_spikes_df.spikes.time_variable_name
        # Copy only the relevent columns so filtering is easier:
        filtered_spikes_df = active_spikes_df[[active_time_variable_name, 'visualization_raster_y_location',  'visualization_raster_emphasis_state', 'aclu', 'fragile_linear_neuron_IDX']].copy()
        
        spike_emphasis_states = kwargs.get('spike_emphasis_state', None)
        if spike_emphasis_states is not None:
            assert len(spike_emphasis_states) == np.shape(active_spikes_df)[0], f"if specified, spike_emphasis_states must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(active_spikes_df)[0]} and len(is_included_indicies): {len(spike_emphasis_states)}"
            # Can set it on the dataframe:
            # 'visualization_raster_y_location'
        
        if is_spike_included is not None:
            assert len(is_spike_included) == np.shape(active_spikes_df)[0], f"if specified, is_included_indicies must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(active_spikes_df)[0]} and len(is_included_indicies): {len(is_spike_included)}"
            ## filter them by the is_included_indicies:
            filtered_spikes_df = filtered_spikes_df[is_spike_included]
        
        # Filter the dataframe using that column and value from the list
        curr_spike_t = filtered_spikes_df[active_time_variable_name].to_numpy() # this will map
        curr_spike_y = filtered_spikes_df['visualization_raster_y_location'].to_numpy() # this will map
        
        # Build the "tooltips" for each spike:
        # curr_spike_data_tooltips = [f"{an_aclu}" for an_aclu in spikes_df['aclu'].to_numpy()]
        if should_return_data_tooltips_kwargs:
            # #TODO 2023-12-06 03:35: - [ ] This doesn't look like it can sort the tooltips at all, right? Or does this not matter?
            # all_scatterplot_tooltips_kwargs = cls._build_spike_data_tuples_from_spikes_df(spikes_df, generate_debug_tuples=True) # need the full spikes_df, not the filtered one
            # INLINING: _build_spike_data_tuples_from_spikes_df __________________________________________________________________ #

            if generate_debug_tuples:
                # debug_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'shank', 'cluster', 'aclu', 'qclu', 'x', 'y', 'speed', 'traj', 'lap', 'maze_relative_lap', 'maze_id', 'neuron_type', 'flat_spike_idx', 'x_loaded', 'y_loaded', 'lin_pos', 'fragile_linear_neuron_IDX', 'PBE_id', 'scISI', 'neuron_IDX', 'replay_epoch_id', 'visualization_raster_y_location', 'visualization_raster_emphasis_state']
                debug_datapoint_column_names = [active_spikes_df.spikes.time_variable_name, 'aclu', 'fragile_linear_neuron_IDX', 'visualization_raster_y_location'] # a subset I'm actually interested in for debugging
                active_datapoint_column_names = debug_datapoint_column_names # all values for the purpose of debugging
            else:
                default_datapoint_column_names = [active_spikes_df.spikes.time_variable_name, 'aclu', 'fragile_linear_neuron_IDX']
                active_datapoint_column_names = default_datapoint_column_names
                
            def _tip_fn(x, y, data):
                """ the function required by pg.ScatterPlotItem's `tip` argument to print the tooltip for each spike. """
                from attrs import asdict
                # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in zip(active_datapoint_column_names, data)])
                # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in asdict(data).items()])
                data_string:str = '|'.join([f"{k}: {str(v)}" for k, v in asdict(data).items()])
                print(f'_tip_fn(...): data_string: {data_string}')
                return f"spike: (x={x:.3f}, y={y:.2f})\n{data_string}"

            # spikes_data = spikes_df[active_datapoint_column_names].to_records(index=False).tolist() # list of tuples
            spikes_data = active_spikes_df[active_datapoint_column_names].to_dict('records') # list of dicts
            spikes_data = [ScatterItemData(**v) for v in spikes_data] 
            all_scatterplot_tooltips_kwargs = dict(data=spikes_data, tip=_tip_fn)
            assert len(all_scatterplot_tooltips_kwargs['data']) == np.shape(active_spikes_df)[0], f"if specified, all_scatterplot_tooltips_kwargs must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(active_spikes_df)[0]} and len((all_scatterplot_tooltips_kwargs['data']): {len(all_scatterplot_tooltips_kwargs['data'])}"
        else:
            all_scatterplot_tooltips_kwargs = None
            
        # config_fragile_linear_neuron_IDX_map values are of the form: (i, fragile_linear_neuron_IDX, curr_pen, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i])
        # Emphasis/Deemphasis-Dependent Pens:
        # curr_spike_pens = [config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][2][a_spike_emphasis_state] for a_fragile_linear_neuron_IDX, a_spike_emphasis_state in zip(filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # get the pens for each spike from the configs map
        curr_spike_pens = [pg.mkPen(self.neuron_colors[aclu], width=1) for aclu, a_spike_emphasis_state in zip(filtered_spikes_df['aclu'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # ignores emphasis state ## 2024-11-05 12:36 AttributeError: 'NewSimpleRaster' object has no attribute 'neuron_colors'
        curr_spikes_brushes = [pg.mkBrush(self.neuron_colors[aclu]) for aclu, a_spike_emphasis_state in zip(filtered_spikes_df['aclu'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # ignores emphasis state

        curr_n = len(curr_spike_t) # curr number of spikes
        # builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes
        pos = np.vstack((curr_spike_t, curr_spike_y))
        all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i], 'brush': curr_spikes_brushes[i]} for i in range(curr_n)] # returned spikes {'pos','data','pen'}		
        if should_return_data_tooltips_kwargs:
            return all_spots, all_scatterplot_tooltips_kwargs
        else:
            return all_spots




@function_attributes(short_name=None, tags=['raster', 'simple', 'working', 'stateless'], input_requires=[], output_provides=[], uses=['_plot_empty_raster_plot_frame', 'build_scatter_plot_kwargs', '_build_units_y_grid'], used_by=['_plot_empty_raster_plot_frame', ''], creation_date='2023-12-06 13:49', related_items=['NewSimpleRaster'])
def new_plot_raster_plot(spikes_df: pd.DataFrame, included_neuron_ids, unit_sort_order=None, unit_colors_list=None, scatter_plot_kwargs=None, scatter_app_name='pho_test', defer_show=False, active_context=None, 
                         win=None, plots_data=None, plots=None, add_debug_header_label: bool=True, **kwargs) -> tuple[Any, pg.GraphicsLayoutWidget, RenderPlots, RenderPlotsData]:
    """ This uses `NewSimpleRaster` and pyqtgraph's scatter function to render a simple raster plot. Simpler than the `SpikeRaster2D`-like implementations.

    
    If extant data passed in, updates:
    
        plots_data.spikes_df
        plots_data.all_spots
        plots_data.all_scatterplot_tooltips_kwargs
        
        if add_debug_header_label:
              plots.debug_header_label
        plots.root_plot
        plots.scatter_plot
        plots.grid
        
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot

        app, win, plots, plots_data = new_plot_raster_plot(_temp_active_spikes_df, shared_aclus)

    """
    downsampling_rate: int = kwargs.pop('downsampling_rate', None)
    
    needs_create_new = ((win is None) or (plots_data is None) or (plots is None))
    if needs_create_new:
        # make root container for plots
        app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)
    else:
        app = kwargs.pop('app', None) # no app needed, but passthrough so it doesn't fail
        

    if unit_sort_order is None:
        unit_sort_order = np.arange(len(included_neuron_ids))
    assert len(unit_sort_order) == len(included_neuron_ids)
    active_sorted_neuron_ids = included_neuron_ids[unit_sort_order]
    plots_data.new_sorted_raster = NewSimpleRaster.init_from_neuron_ids(active_sorted_neuron_ids, neuron_colors=unit_colors_list) ## about data, not hte plot itself
    # self.plots.scatter_plot.opts['useCache'] = True
    
    ## Add the source data (spikes_df) to the plot_data
    plots_data.spikes_df = deepcopy(spikes_df)    
    # Update the dataframe
    plots_data.spikes_df = plots_data.new_sorted_raster.update_spikes_df_visualization_columns(spikes_df=plots_data.spikes_df)
    ## Build the spots for the raster plot:
    plots_data.all_spots, plots_data.all_scatterplot_tooltips_kwargs = plots_data.new_sorted_raster.build_spikes_all_spots_from_df(spikes_df=plots_data.spikes_df, should_return_data_tooltips_kwargs=True, generate_debug_tuples=False, downsampling_rate=downsampling_rate)
    # self.plots.scatter_plot.opts['useCache'] = True
    
    # Add header label
    if add_debug_header_label:
        # plots.debug_header_label = pg.LabelItem(justify='right', text='debug_header_label')
        # win.addItem(plots.debug_header_label)
        plots.debug_header_label = win.addLabel("debug_header_label") # , row=1, colspan=4
        win.nextRow()
        # plots.debug_label2 = win.addLabel("Label2") # , col=1, colspan=4
        # win.nextRow()
    
    # # Actually setup the plot:
    if (not plots.has_attr('root_plot')):
        plots.root_plot = win.addPlot(title="Raster") # this seems to be the equivalent to an 'axes'

    if scatter_plot_kwargs is None:
        scatter_plot_kwargs = {} ## make them empty at least


    scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs, tick_width=scatter_plot_kwargs.pop('tick_width', 0.1), tick_height=scatter_plot_kwargs.pop('tick_height', 1.0))
    
    plots.scatter_plot = pg.ScatterPlotItem(**scatter_plot_kwargs)
    plots.scatter_plot.setObjectName('scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    plots.scatter_plot.opts['useCache'] = True
    plots.scatter_plot.addPoints(plots_data.all_spots, **(plots_data.all_scatterplot_tooltips_kwargs or {})) # , hoverable=True
    plots.root_plot.addItem(plots.scatter_plot)

    # build the y-axis grid to separate the units
    plots.grid = _build_units_y_grid(plots.root_plot)
    
    ## Build the y-axis cell labels:
    # Need to get the y-axis positions corresponding to each cell.
    # sorted_neuron_ids = deepcopy(plots_data.new_sorted_raster.neuron_IDs)
    # for aclu in sorted_neuron_ids:
    #     plots_data.new_sorted_raster.neuron_y_pos[aclu]
    a_left_axis = plots.root_plot.getAxis('left') # axisItem
    # a_left_axis.setLabel('test')
    # tick_ydict = {y_pos:f"{int(aclu)}" for y_pos, aclu in zip(a_series_identity_y_values, sorted_neuron_ids)} # {0.5: '68', 1.5: '75', 2.5: '54', 3.5: '10', 4.5: '104', 5.5: '90', 6.5: '44', 7.5: '15', 8.5: '93', 9.5: '79', 10.5: '56', 11.5: '84', 12.5: '78', 13.5: '31', 14.5: '16', 15.5: '40', 16.5: '25', 17.5: '81', 18.5: '70', 19.5: '66', 20.5: '24', 21.5: '98', 22.5: '80', 23.5: '77', 24.5: '60', 25.5: '39', 26.5: '9', 27.5: '82', 28.5: '85', 29.5: '101', 30.5: '87', 31.5: '26', 32.5: '43', 33.5: '65', 34.5: '48', 35.5: '52', 36.5: '92', 37.5: '11', 38.5: '51', 39.5: '72', 40.5: '18', 41.5: '53', 42.5: '47', 43.5: '89', 44.5: '102', 45.5: '61'}
    tick_ydict = {plots_data.new_sorted_raster.neuron_y_pos[aclu]:f"{int(aclu)}" for aclu in plots_data.new_sorted_raster.neuron_IDs}
    a_left_axis.setTicks([tick_ydict.items()])

    return RasterPlotSetupTuple(app, win, plots, plots_data)











# ==================================================================================================================== #
# Main Functions                                                                                                       #
# ==================================================================================================================== #

    

# Note that these raster plots could implement some variant of HideShowSpikeRenderingMixin, SpikeRenderingMixin, etc but these classes frankly suck. 

# Define the namedtuple
RasterPlotSetupTuple = namedtuple('RasterPlotSetupTuple', ['app', 'win', 'plots', 'plots_data']) # tuple[Any, pg.GraphicsLayoutWidget, RenderPlots, RenderPlotsData]


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

    return RasterPlotSetupTuple(app, win, plots, plots_data) # app, win, plots, plots_data



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


def build_scatter_plot_kwargs(scatter_plot_kwargs=None, tick_width: float = 1.0, tick_height: float = 1.0, **kwargs):
    """build the default scatter plot kwargs, and merge them with the provided kwargs
    
    
    build_scatter_plot_kwargs(scatter_plot_kwargs=dict(size=5, hoverable=False), tick_width=0.0, tick_height=1.0)
    
    """
    # Common Tick Label 
    vtick = _build_default_tick(tick_width=tick_width, tick_height=tick_height)
    default_scatter_plot_kwargs = dict(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=2, pen={'color': 'w', 'width': 1}, hoverable=kwargs.pop('hoverable', True), **kwargs)

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
    grid.setTextPen(None) # no text should be generated
    
    grid.setZValue(-100)
    return grid


# class FmtAxisItem(pg.AxisItem):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def tickStrings(self, values, scale, spacing):
#         return [f'{v:.4f}' for v in values]
    


@function_attributes(short_name=None, tags=['scatterplot', 'raster', 'manager'], input_requires=[], output_provides=[], uses=['RasterPlotParams','UnitSortOrderManager','RasterScatterPlotManager'], used_by=['plot_raster_plot', 'plot_multi_sort_raster_browser'], creation_date='2023-12-06 02:24', related_items=[])
def _build_scatter_plotting_managers(plots_data: RenderPlotsData, spikes_df: Optional[pd.DataFrame], included_neuron_ids=None, unit_sort_order=None, unit_colors_list=None) -> RenderPlotsData:
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
    params.build_neurons_color_data(fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs, neuron_colors_list=unit_colors_list, coloring_mode=UnitColoringMode.PRESERVE_FRAGILE_LINEAR_NEURON_IDXS)
    
    manager = UnitSortOrderManager(neuron_ids=neuron_ids, fragile_linear_neuron_IDXs=fragile_linear_neuron_IDXs, n_cells=plots_data.n_cells, unit_sort_order=unit_sort_order, params=params)
    manager.update_series_identity_y_values()
    raster_plot_manager = RasterScatterPlotManager(unit_sort_manager=manager)
    raster_plot_manager._build_cell_configs()
    
    ## Add the managers to the plot_data
    plots_data.params = params
    plots_data.unit_sort_manager = manager
    plots_data.raster_plot_manager = raster_plot_manager
    return plots_data


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


@function_attributes(short_name=None, tags=['plotting','raster', 'sort'], input_requires=[], output_provides=[], uses=['_subfn_build_and_add_scatterplot_row', '_build_scatter_plotting_managers'], used_by=['RankOrderRastersDebugger'], creation_date='2023-10-30 22:23', related_items=[])
def plot_multi_sort_raster_browser(spikes_df: pd.DataFrame, included_neuron_ids, unit_sort_orders_dict=None, unit_colors_list_dict=None, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None):
    """ Plots a neat stack of raster plots.
    
    ISSUES:
    - [ ] TODO 2023-11-27 - does not color (brush) the spikes, so when they are wide enough for this to matter a strange default grey/blue color shows through for all of them. This could be fixed I believe by adding 'brush' property to all_spots dict

    
    
    Basic Plotting:    
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multi_sort_raster_browser

        included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        unit_sort_orders_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (even_long, odd_long, even_short, odd_short)))
        unit_colors_list_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (unit_colors_list, unit_colors_list, unit_colors_list, unit_colors_list)))

        app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = plot_multi_sort_raster_browser(spikes_df, included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)


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

    # ## Create the raster plot for the replay:
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)

    plots.layout = win.addLayout()
    plots.ax = {}
    plots.scatter_plots = {} # index is the _active_plot_identifier
    plots.grid = {} # index is the _active_plot_identifier

    plots_data.all_spots_dict = {}
    plots_data.all_scatterplot_tooltips_kwargs_dict = {}

    vtick_simple_line = _build_default_tick(tick_width=0.0, tick_height=0.9)
    override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItemSimpleSpike', pxMode=False, symbol=vtick_simple_line, size=1, hoverable=False) # , pen=None, brush=None

    # vtick_simple_line = _build_default_tick(tick_width=1.0, tick_height=0.9)
    # override_scatter_plot_kwargs = dict(name='epochSpikeRasterScatterPlotItemSimpleSpike', pxMode=False, symbol=vtick_simple_line, size=1.0, hoverable=False) # , pen=None, brush=None
    # print(f'override_scatter_plot_kwargs: {override_scatter_plot_kwargs}')

    i = 0
    plots_data.plots_data_dict = {} # new dict to hold plot data
    plots_data.plots_spikes_df_dict = {}

    for _active_plot_identifier, active_unit_sort_order in unit_sort_orders_dict.items():

        plots_data.plots_data_dict[_active_plot_identifier] = RenderPlotsData(_active_plot_identifier)
    
        if unit_colors_list_dict is not None:
            unit_colors_list = unit_colors_list_dict.get(_active_plot_identifier, None)
        else:
            unit_colors_list = None

        plots_data.plots_data_dict[_active_plot_identifier] = _build_scatter_plotting_managers(plots_data.plots_data_dict[_active_plot_identifier], spikes_df=spikes_df, included_neuron_ids=deepcopy(included_neuron_ids), unit_sort_order=deepcopy(active_unit_sort_order), unit_colors_list=deepcopy(unit_colors_list))
        
        # Update the dataframe
        plots_data.plots_spikes_df_dict[_active_plot_identifier] = deepcopy(spikes_df)
        plots_data.plots_spikes_df_dict[_active_plot_identifier] = plots_data.plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(plots_data.plots_spikes_df_dict[_active_plot_identifier], overwrite_existing=True)
        ## Build the spots for the raster plot:
        # plots_data.all_spots, plots_data.all_scatterplot_tooltips_kwargs = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)
        plots_data.all_spots_dict[_active_plot_identifier], plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(plots_data.plots_spikes_df_dict[_active_plot_identifier], plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

        _subfn_build_and_add_scatterplot_row(plots_data.plots_data_dict[_active_plot_identifier], plots, _active_plot_identifier=_active_plot_identifier, row=(i), col=0, left_label=_active_plot_identifier, scatter_plot_kwargs=override_scatter_plot_kwargs)
        i = i+1

        ## Get the scatterplot and update the points:
        a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
        a_scatter_plot.addPoints(plots_data.all_spots_dict[_active_plot_identifier], **(plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {})) # , hoverable=True
        

    main_plot_identifiers_list = list(unit_sort_orders_dict.keys()) # ['long_even', 'long_odd', 'short_even', 'short_odd']
    
    def on_update_active_scatterplot_kwargs(override_scatter_plot_kwargs):
        """ captures: main_plot_identifiers_list, plots, plots_data """
        for _active_plot_identifier in main_plot_identifiers_list:
            # for _active_plot_identifier, a_scatter_plot in plots.scatter_plots.items():
            # new_ax = plots.ax[_active_plot_identifier]
            a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
            a_scatter_plot.setData(plots_data.all_spots_dict[_active_plot_identifier], **(plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {}), **override_scatter_plot_kwargs)

    def on_update_active_epoch(an_epoch_idx, an_epoch):
        """ captures: main_plot_identifiers_list, plots """
        for _active_plot_identifier in main_plot_identifiers_list:
            new_ax = plots.ax[_active_plot_identifier]
            new_ax.setXRange(an_epoch.start, an_epoch.stop)
            # new_ax.getAxis('left').setLabel(f'[{an_epoch.label}]')
            
            # a_scatter_plot = plots.scatter_plots[_active_plot_identifier]

    return app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs



@providing_context(fn_name='plot_raster_plot')
@function_attributes(short_name='plot_raster_plot', tags=['pyqtgraph','raster','2D'], input_requires=[], output_provides=[], uses=['_plot_empty_raster_plot_frame', '_build_scatter_plotting_managers'], used_by=[], creation_date='2023-03-31 20:53')
def plot_raster_plot(spikes_df: pd.DataFrame, included_neuron_ids, unit_sort_order=None, unit_colors_list=None, scatter_plot_kwargs=None, scatter_app_name='pho_test', defer_show=False, active_context=None, should_return_data_tooltips_kwargs=True, **kwargs) -> tuple[Any, pg.GraphicsLayoutWidget, RenderPlots, RenderPlotsData]:
    """ This uses pyqtgraph's scatter function like SpikeRaster2D to render a raster plot with colored ticks by default

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_raster_plot

        app, win, plots, plots_data = plot_raster_plot(_temp_active_spikes_df, shared_aclus)

    """
    
    # make root container for plots
    app, win, plots, plots_data = _plot_empty_raster_plot_frame(scatter_app_name=scatter_app_name, defer_show=defer_show, active_context=active_context)
    
    plots_data = _build_scatter_plotting_managers(plots_data, spikes_df=spikes_df, included_neuron_ids=included_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=unit_colors_list)
    
    ## Add the source data (spikes_df) to the plot_data
    plots_data.spikes_df = deepcopy(spikes_df)
    
    # Update the dataframe
    plots_data.spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(plots_data.spikes_df, overwrite_existing=True)
    
    ## Build the spots for the raster plot:
    plots_data.all_spots, plots_data.all_scatterplot_tooltips_kwargs = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(plots_data.spikes_df, plots_data.raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=should_return_data_tooltips_kwargs)

    # Add header label
    plots.debug_header_label = pg.LabelItem(justify='right', text='debug_header_label')
    win.addItem(plots.debug_header_label)
    
    # # Actually setup the plot:
    plots.root_plot = win.addPlot() # this seems to be the equivalent to an 'axes'

    # build_scatter_plot_kwargs(scatter_plot_kwargs=dict(size=5, hoverable=False), tick_width=0.0, tick_height=1.0)
    # build_scatter_plot_kwargs(scatter_plot_kwargs=dict(size=5, hoverable=False, tick_width=0.0, tick_height=1.0))

    scatter_plot_kwargs = build_scatter_plot_kwargs(scatter_plot_kwargs=scatter_plot_kwargs, tick_width=scatter_plot_kwargs.pop('tick_width', 0.1), tick_height=scatter_plot_kwargs.pop('tick_height', 1.0))
    
    plots.scatter_plot = pg.ScatterPlotItem(**scatter_plot_kwargs)
    plots.scatter_plot.setObjectName('scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    plots.scatter_plot.opts['useCache'] = True
    plots.scatter_plot.addPoints(plots_data.all_spots, **(plots_data.all_scatterplot_tooltips_kwargs or {})) # , hoverable=True
    plots.root_plot.addItem(plots.scatter_plot)

    # build the y-axis grid to separate the units
    plots.grid = _build_units_y_grid(plots.root_plot)

    return RasterPlotSetupTuple(app, win, plots, plots_data)

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
    spikes_df = plots_data.unit_sort_manager.update_spikes_df_visualization_columns(spikes_df, overwrite_existing=True)
    
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


@function_attributes(short_name=None, tags=['neuron_ID', 'color'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-28 11:07', related_items=[])
def build_shared_sorted_neuron_color_maps(neuron_IDs_lists, return_255_array: bool=True) -> Tuple[Dict, Dict]:
    """ builds the shared colors for all neuron_IDs in any of the lists. This approach lends itself to globally-unique color mapping, like would be done when wanting to compare between different spike raster plots. 

    Outputs:

    - unit_colors_ndarray_map: Int:NDArray[(4,)] - {5: array([255, 157, 0.278431, 1]), 7: array([252.817, 175.545, 0.202502, 1]), ...}


    unit_qcolors_map, unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    """
    # If you have a set of values that can be larger than the entries in each list:
    any_list_neuron_IDs = np.sort(union_of_arrays(*neuron_IDs_lists)) # neuron_IDs as they appear in any list
    ## build color values from these:
    any_list_n_neurons = len(any_list_neuron_IDs)
    _neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(any_list_n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None, return_255_array=return_255_array)
    unit_colors_ndarray_map: Dict = dict(zip(any_list_neuron_IDs, neuron_colors_ndarray.copy().T)) # Int:NDArray[(4,)] - {5: array([255, 157, 0.278431, 1]), 7: array([252.817, 175.545, 0.202502, 1]), ...}
    unit_qcolors_map: Dict = dict(zip(any_list_neuron_IDs, _neuron_qcolors_list.copy())) # Int:NDArray[(4,)] - {5: array([255, 157, 0.278431, 1]), 7: array([252.817, 175.545, 0.202502, 1]), ...}
    # `unit_colors_map` is main colors output
    return unit_qcolors_map, unit_colors_ndarray_map

# INPUTS: decoders_dict, included_any_context_neuron_ids

@function_attributes(short_name=None, tags=['sort', 'raster', 'sorting', 'important', 'visualization', 'order', 'neuron_ids'], input_requires=[], output_provides=[], uses=['DataSeriesColorHelpers.build_cell_colors', 'paired_incremental_sorting'], used_by=[], creation_date='2023-11-28 10:52', related_items=[])
def paired_incremental_sort_neurons(decoders_dict: Dict, included_any_context_neuron_ids_dict=None, sortable_values_list_dict=None):
    """ Given a set of decoders (or more generally placefields, ratemaps, or anything else with neuron_IDs and a property that can be sorted) return the iterative successive sort.
    
    This means:
    
    [A, B, C, D, E, F]
    [A, B, C, D, E, F]
    Sort Order: [0, 3, 5, 1, 4, 2]
    Sorted: [A, D, F, B, E, C]
    
    
    D0: [A, B, C]
    D1: [A, C, D]
    D2: [A, B, D, E, F]
    D3: [A, B, C, D, E, F]
        
    Sort Order on D0 is:
    [A, B, C]
    [0, 1, 2]
    
    So sorted order on D0 is:
    [A, B, C]

    # For D1 all entries previously ranked are kept in the same order, then the remaining entries are sorted according to their D1-relative sorts:
    
    #TODO 2023-11-30 12:08: - [ ] Example, finish for D2, D3

        
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_incremental_sort_neurons
        
    decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
    # included_any_context_neuron_ids = None

    epoch_active_aclus = np.array([9,  26,  31,  39,  40,  43,  47,  52,  53,  54,  60,  61,  65,  68,  72,  75,  77,  78,  81,  82,  84,  85,  90,  92,  93,  98, 102])
    included_any_context_neuron_ids = deepcopy(epoch_active_aclus)

    sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_incremental_sort_neurons(decoders_dict=track_templates.get_decoders_dict(), included_any_context_neuron_ids=included_any_context_neuron_ids)
    sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
    
    """
    
    # 2023-11-28 - New Sorting using `paired_incremental_sorting`
    original_neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # immutable, required to get proper original indicies after exclusions
    neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
    unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    # `unit_colors_map` is main colors output

    if sortable_values_list_dict is None:
        # build default sortable items list:
        sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} 
    else:
        assert len(sortable_values_list_dict) == len(decoders_dict)
        

    # Here is where we want to filter by specific included_neuron_IDs (before we plot and output):
    if included_any_context_neuron_ids_dict is not None:
        # restrict only to `included_any_context_neuron_ids`
        print(f'restricting only to included_any_context_neuron_ids: {included_any_context_neuron_ids_dict}...')
        is_neuron_IDs_included_lists = [np.isin(neuron_ids, deepcopy(included_any_context_neuron_ids_dict)) for neuron_ids in neuron_IDs_lists]
        neuron_IDs_lists = [neuron_ids[is_neuron_IDs_included] for neuron_ids, is_neuron_IDs_included in zip(neuron_IDs_lists, is_neuron_IDs_included_lists)] # filtered_neuron_IDs_lists
        sortable_values_lists = [deepcopy(a_sortable_values_list) for a_sortable_values_list in sortable_values_list_dict.values()]
        # sortable_values_lists = [deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for a_decoder in decoders_dict.values()] # (46, 56) - (n_neurons, n_pos_bins)
        sortable_values_lists = [a_sortable_vals_list[is_neuron_IDs_included_lists[i]] for i, a_sortable_vals_list in enumerate(sortable_values_lists)] #
        
    else:
        sortable_values_lists = [deepcopy(a_sortable_values_list) for a_sortable_values_list in sortable_values_list_dict.values()]
        # sortable_values_lists = [deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for a_decoder in decoders_dict.values()] # (46, 56) - (n_neurons, n_pos_bins)

    # `sort_helper_original_neuron_id_to_IDX_dicts` needs to be built after filtering (if it occurs)
    sort_helper_original_neuron_id_to_IDX_dicts = [dict(zip(neuron_ids, np.arange(len(neuron_ids)))) for neuron_ids in original_neuron_IDs_lists] # just maps each neuron_id in the list to a fragile_linear_IDX 


    ## DO SORTING: determine sorting:
    sorted_neuron_IDs_lists = paired_incremental_sorting(neuron_IDs_lists, sortable_values_lists)
    # `sort_helper_neuron_id_to_sort_IDX_dicts` dictionaries in the appropriate order (sorted order) with appropriate indexes. Its .values() can be used to index into things originally indexed with aclus.
    sort_helper_neuron_id_to_sort_IDX_dicts = [{aclu:a_sort_helper_neuron_id_to_IDX_map[aclu] for aclu in sorted_neuron_ids} for a_sort_helper_neuron_id_to_IDX_map, sorted_neuron_ids in zip(sort_helper_original_neuron_id_to_IDX_dicts, sorted_neuron_IDs_lists)]
    # sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
    # So unlike other attempts, these colors are sorted along with the aclus for each decoder, and we don't try to keep them separate. Since they're actually in a dict (where conceptually the order doesn't really matter) this should be indistinguishable performance-wise from other implementation.
    sort_helper_neuron_id_to_neuron_colors_dicts = [{aclu:unit_colors_map[aclu] for aclu in sorted_neuron_ids} for sorted_neuron_ids in sorted_neuron_IDs_lists] # [{72: array([11.2724, 145.455, 0.815335, 1]), 84: array([165, 77, 1, 1]), ...}, {72: array([11.2724, 145.455, 0.815335, 1]), 84: array([165, 77, 1, 1]), ...}, ...]
    # `sort_helper_neuron_id_to_sort_IDX_dicts` is main output here:
    return sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts


# Define the namedtuple for additional_data
UnsortedDataTuple = namedtuple(
    'UnsortedDataTuple',
    ['original_neuron_IDs_lists', 'neuron_IDs_lists', 'sortable_values_lists', 'unit_colors_map']
)

PairedSeparatelySortNeuronsResult = namedtuple(
    'PairedSeparatelySortNeuronsResult',
    [
        'sorted_neuron_IDs_lists', 
        'sort_helper_neuron_id_to_neuron_colors_dicts', 
        'sort_helper_neuron_id_to_sort_IDX_dicts', 
        'unsorted_data_tuple'
    ]
)


@function_attributes(short_name=None, tags=['sort', 'raster', 'sorting', 'important', 'visualization', 'order', 'neuron_ids'], input_requires=[], output_provides=[], uses=['DataSeriesColorHelpers.build_cell_colors', 'paired_incremental_sorting'], used_by=[], creation_date='2023-11-29 17:13', related_items=['paired_incremental_sort_neurons'])
def paired_separately_sort_neurons(decoders_dict: Dict, included_any_context_neuron_ids_dict_dict=None, sortable_values_list_dict=None):
    """ Given a set of decoders (or more generally placefields, ratemaps, or anything else with neuron_IDs and a property that can be sorted) return the iterative successive sort.
    
    History: Built from working `paired_incremental_sort_neurons` with an attempt to generalize for raster plotting.
    
    #TODO 2023-11-29 17:36: - [ ] Actually change to use separately sorted items instead of `paired_individual_sorting`
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons
        
    decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
    # included_any_context_neuron_ids = None

    epoch_active_aclus = np.array([9,  26,  31,  39,  40,  43,  47,  52,  53,  54,  60,  61,  65,  68,  72,  75,  77,  78,  81,  82,  84,  85,  90,  92,  93,  98, 102])
    included_any_context_neuron_ids = deepcopy(epoch_active_aclus)

    sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_incremental_sort_neurons(decoders_dict=track_templates.get_decoders_dict(), included_any_context_neuron_ids=included_any_context_neuron_ids)
    sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        
        
    """
    
    # 2023-11-28 - New Sorting using `paired_individual_sorting`
    original_neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # immutable, required to get proper original indicies after exclusions
    neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
    unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    # `unit_colors_map` is main colors output

    if sortable_values_list_dict is None:
        # build default sortable items list:
        sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
    else:
        assert len(sortable_values_list_dict) == len(decoders_dict)
        

    # Here is where we want to filter by specific included_neuron_IDs (before we plot and output):
    if included_any_context_neuron_ids_dict_dict is not None:
        # assert len(included_any_context_neuron_ids_dict_dict) == len(decoders_dict), f"included_any_context_neuron_ids_dict should contain one `included_any_context_neuron_ids` list for each decoder"
        if (len(included_any_context_neuron_ids_dict_dict) != len(decoders_dict)):
            print(f'len(included_any_context_neuron_ids_dict_dict) != len(decoders_dict), assuming this is a single included_any_context_neuron_ids_dict for all decoders like used in `paired_incremental_sort_neurons(...)`. Fixing. ')
            included_any_context_neuron_ids_dict = deepcopy(included_any_context_neuron_ids_dict_dict)
            included_any_context_neuron_ids_dict_dict = {k:deepcopy(included_any_context_neuron_ids_dict) for k, v in decoders_dict.items()}

        assert sortable_values_list_dict is not None
        assert len(sortable_values_list_dict) == len(included_any_context_neuron_ids_dict_dict)
        
        # for a_name, included_any_context_neuron_ids in included_any_context_neuron_ids_dict.items():
        # restrict only to `included_any_context_neuron_ids`
        # print(f'restricting only to included_any_context_neuron_ids: {included_any_context_neuron_ids}...')
        is_neuron_IDs_included_lists = [np.isin(neuron_ids, deepcopy(included_any_context_neuron_ids)) for neuron_ids, included_any_context_neuron_ids in zip(neuron_IDs_lists, included_any_context_neuron_ids_dict_dict.values())]
        neuron_IDs_lists = [neuron_ids[is_neuron_IDs_included] for neuron_ids, is_neuron_IDs_included in zip(neuron_IDs_lists, is_neuron_IDs_included_lists)] # filtered_neuron_IDs_lists
        ## These are the sortable values:
        # sortable_values_lists = [deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for a_decoder in decoders_dict.values()] # (46, 56) - (n_neurons, n_pos_bins)

        sortable_values_lists = [deepcopy(a_sortable_values_list) for a_sortable_values_list in sortable_values_list_dict.values()] # (46, 56) - (n_neurons, n_pos_bins)
        sortable_values_lists = [a_sortable_vals_list[is_neuron_IDs_included_lists[i]] for i, a_sortable_vals_list in enumerate(sortable_values_lists)] #
        
    else:
        
        # sortable_values_lists = [deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for a_decoder in decoders_dict.values()] # (46, 56) - (n_neurons, n_pos_bins)
        sortable_values_lists = [deepcopy(a_sortable_values_list) for a_sortable_values_list in sortable_values_list_dict.values()] # (46, 56) - (n_neurons, n_pos_bins)

    # `sort_helper_original_neuron_id_to_IDX_dicts` needs to be built after filtering (if it occurs)
    sort_helper_original_neuron_id_to_IDX_dicts = [dict(zip(neuron_ids, np.arange(len(neuron_ids)))) for neuron_ids in original_neuron_IDs_lists] # just maps each neuron_id in the list to a fragile_linear_IDX 

    ## DO SORTING: determine sorting:
    sorted_neuron_IDs_lists = paired_individual_sorting(neuron_IDs_lists, sortable_values_lists)

    # `sort_helper_neuron_id_to_sort_IDX_dicts` dictionaries in the appropriate order (sorted order) with appropriate indexes. Its .values() can be used to index into things originally indexed with aclus.
    sort_helper_neuron_id_to_sort_IDX_dicts = [{aclu:a_sort_helper_neuron_id_to_IDX_map[aclu] for aclu in sorted_neuron_ids} for a_sort_helper_neuron_id_to_IDX_map, sorted_neuron_ids in zip(sort_helper_original_neuron_id_to_IDX_dicts, sorted_neuron_IDs_lists)]
    # So unlike other attempts, these colors are sorted along with the aclus for each decoder, and we don't try to keep them separate. Since they're actually in a dict (where conceptually the order doesn't really matter) this should be indistinguishable performance-wise from other implementation.
    sort_helper_neuron_id_to_neuron_colors_dicts = [{aclu:unit_colors_map[aclu] for aclu in sorted_neuron_ids} for sorted_neuron_ids in sorted_neuron_IDs_lists] # [{72: array([11.2724, 145.455, 0.815335, 1]), 84: array([165, 77, 1, 1]), ...}, {72: array([11.2724, 145.455, 0.815335, 1]), 84: array([165, 77, 1, 1]), ...}, ...]
    # `sort_helper_neuron_id_to_sort_IDX_dicts` is main output here:
    return PairedSeparatelySortNeuronsResult(sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, UnsortedDataTuple(original_neuron_IDs_lists, neuron_IDs_lists, sortable_values_lists, unit_colors_map)) #, sorted_pf_tuning_curves



# ==================================================================================================================== #
# Menu Builders                                                                                                        #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['context', 'directional_pf', 'display', 'filter'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-09 20:07', related_items=[])
def _recover_filter_config_name_from_display_context(owning_pipeline_reference, active_display_fn_identifying_ctx) -> str:
    """ recover active_config_name from the context in a way that works for both normal and directional pfs (which include both a .filter_name and .lap_dir value) """
    # active_config_name = active_display_fn_identifying_ctx.filter_name # hardcoded bad way that doesn't work for directional pfs
    # directional-pf compatible way of finding the matching config name:
    reverse_context_name_lookup_map = {v: k for k, v in owning_pipeline_reference.filtered_contexts.items()} # builds an inverse mapping from a given filtered_context Dict[IdentifyingContext, str] to its string name
    recovered_filter_context = active_display_fn_identifying_ctx.get_subset(subset_excludelist=['display_fn_name']) # drop the display_fn_name part to get the original filter context
    active_config_name: str = reverse_context_name_lookup_map[recovered_filter_context]
    return active_config_name


@function_attributes(short_name=None, tags=['menu', 'spike_raster', 'ui'], input_requires=[], output_provides=[], uses=['LocalMenus_AddRenderable'], used_by=['_build_additional_window_menus'], creation_date='2023-11-09 19:32', related_items=[])
def _build_additional_spikeRaster2D_menus(spike_raster_plt_2d, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx):
    from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable

    active_config_name: str = _recover_filter_config_name_from_display_context(owning_pipeline_reference, active_display_fn_identifying_ctx) # recover active_config_name from the context

    ## Adds the custom renderable menu to the top-level menu of the plots in Spike2DRaster
    # _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.add_renderable_context_menu(spike_raster_window.spike_raster_plt_2d, computation_result.sess)  # Adds the custom context menus for SpikeRaster2D
    
    _active_2d_plot_renderable_menus = LocalMenus_AddRenderable.initialize_renderable_context_menu(spike_raster_plt_2d, owning_pipeline_reference, active_config_name)  # Adds the custom context menus for SpikeRaster2D
    
    # spike_raster_plt_2d.menu_action_history_list = []
    
    output_references = [_active_2d_plot_renderable_menus]
    return output_references


@function_attributes(short_name=None, tags=['menu', 'spike_raster', 'gui', 'IMPORTANT'], input_requires=[], output_provides=[], uses=['_build_additional_spikeRaster2D_menus', 'ConnectionControlsMenuMixin', 'CreateNewConnectedWidgetMenuHelper', 'CreateLinkedWidget_MenuProvider', 'DebugMenuProviderMixin', 'DockedWidgets_MenuProvider', ], used_by=['_display_spike_rasters_window'], creation_date='2023-11-09 19:32', related_items=[])
def _build_additional_window_menus(spike_raster_window: Spike3DRasterWindowWidget, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx):
    """ needs the entire pipeline so that data is avilable for any of the optional display menus
        - secondarily so it can call the pipeline's normal .display(...) functions to create new visualizations

    TODO: seems like it should be a Spike3DRasterWindowWidget member property
    
        DockedWidgets_MenuProvider
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _build_additional_window_menus

        # Set Window Title Options:
        a_file_prefix = str(computation_result.sess.filePrefix.resolve())
        spike_raster_window.setWindowFilePath(a_file_prefix)
        spike_raster_window.setWindowTitle(f'Spike Raster Window - {active_config_name} - {a_file_prefix}')

        ## Build the additional menus:
        output_references = _build_additional_window_menus(spike_raster_window, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx) ## the menus on the other hand take the entire pipeline, because they might need that valuable DATA

        
    """
    assert owning_pipeline_reference is not None
    active_config_name: str = _recover_filter_config_name_from_display_context(owning_pipeline_reference, active_display_fn_identifying_ctx) # recover active_config_name from the context

    # if not spike_raster_window.ui.has_attr('_menu_action_history_list'):
    #     spike_raster_window.ui._menu_action_history_list = [] ## a list to show the history

    spike_raster_window.menu_action_history_list = []


    ## SpikeRaster2D Specific Items:
    output_references = _build_additional_spikeRaster2D_menus(spike_raster_window.spike_raster_plt_2d, owning_pipeline_reference, computation_result, active_display_fn_identifying_ctx)

    spike_raster_window.menu_action_history_list = []

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
        print(f'ERROR: _build_additional_window_menus(...) has no owning_pipeline_reference in its parameters, so it cannot add the CreateNewConnectedWidgetMenuMixin menus.')
        menuCreateNewConnectedWidget = None
        createNewConnected_actions_dict = None
        
    # Debug Menu
    _debug_menu_provider = DebugMenuProviderMixin(render_widget=spike_raster_window)
    spike_raster_window.main_menu_window.ui.menus.global_window_menus.debug.menu_provider_obj = _debug_menu_provider

    # Docked Menu
    _docked_menu_provider = DockedWidgets_MenuProvider(render_widget=spike_raster_window)
    _docked_menu_provider.DockedWidgets_MenuProvider_on_buildUI(spike_raster_window=spike_raster_window, owning_pipeline_reference=owning_pipeline_reference, context=active_display_fn_identifying_ctx, active_config_name=active_config_name, display_output=owning_pipeline_reference.display_output[active_display_fn_identifying_ctx], use_time_bin_specific_menus=True)
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
            _createLinkedWidget_menu_provider.CreateLinkedWidget_MenuProvider_on_buildUI(spike_raster_window=spike_raster_window, owning_pipeline_reference=owning_pipeline_reference, active_pf_2D_dt=active_pf_2D_dt, context=active_display_fn_identifying_ctx, active_config_name=active_config_name, display_output=display_output)
        else:
            print(f'WARNING: owning_pipeline_reference is NONE in  _display_spike_rasters_window!')   
            
        spike_raster_window.main_menu_window.ui.menus.global_window_menus.create_linked_widget.menu_provider_obj = _createLinkedWidget_menu_provider # KeyError: 'create_linked_widget' was occuring before, so I moved it into the condition where dts were had and the menus were created 
    else:
        print(f'active_pf_2D_dt is None! Skipping Create Paired Widget Menu...')
        _createLinkedWidget_menu_provider = None

    
    output_references.extend([curr_main_menu_window, menuConnections, connections_actions_dict, # note .extend(...) function works in-place and does not return a result, which is why this function was returning None originally.
        curr_main_menu_window, menuCreateNewConnectedWidget, createNewConnected_actions_dict,
        _debug_menu_provider,
        _docked_menu_provider,
        _createLinkedWidget_menu_provider
    ])
    return output_references