
from copy import deepcopy
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


# Windowing helpers for spikes_df:
from pyphoplacecellanalysis.PhoPositionalData.plotting.visualization_window import VisualizationWindow # Used to build "Windows" into the data points such as the window defining the fixed time period preceeding the current time where spikes had recently fired, etc.
from numpy.lib.stride_tricks import sliding_window_view

# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QHBoxLayout, QSlider, QCheckBox 


# ==================================================================================================================== #
# NEW 2023-03-31 - Uses Scatter Plot Based raster like SpikeRaster2D so they can be colored, resized, etc.             #
# ==================================================================================================================== #

from attrs import define, Factory
from indexed import IndexedOrderedDict
# from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots, RenderPlotsData
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.gui.Qt.color_helpers import build_adjusted_color # required for the different emphasis states in ._build_cell_configs()
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers # for build_neurons_color_data
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui # for QColor build_neurons_color_data
from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin

""" Raster Plot:

    plot_raster_plot(...) is the only user-facing function. The rest of this crap is just a hack.
    2023-03-31: All this crap was brought in to replace the functionality used in SpikeRaster2D 
"""

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


@function_attributes(short_name='plot_raster_plot', tags=['pyqtgraph','raster','2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-31 20:53')
def plot_raster_plot(spikes_df, shared_aclus, scatter_app_name='pho_test'):
    """ This uses pyqtgraph's scatter function like SpikeRaster2D

    Usage:

        from pyphoplacecellanalysis.GUI.PyQtPlot.Examples.pyqtplot_RasterPlot import plot_raster_plot

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
    
    plots = RenderPlots(scatter_app_name)
    plots_data = RenderPlotsData(scatter_app_name)

    # each entry in `config_fragile_linear_neuron_IDX_map` has the form:
    # 	(i, fragile_linear_neuron_IDX, curr_pen, _series_identity_lower_y_values[i], _series_identity_upper_y_values[i])

    ## Build the spots for the raster plot:
    plots_data.all_spots = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(spikes_df, raster_plot_manager.config_fragile_linear_neuron_IDX_map)

    ## Perform the plotting:
    app = pg.mkQApp(scatter_app_name)
    win = pg.GraphicsLayoutWidget(show=True, title=scatter_app_name)
    win.resize(1000,600)
    win.setWindowTitle(f'pyqtgraph: Raster Spikes: {scatter_app_name}')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    # # Actually setup the plot:
    plots.root_plot = win.addPlot()

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


    plots.scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1})
    plots.scatter_plot.setObjectName('scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
    plots.scatter_plot.opts['useCache'] = True
    plots.scatter_plot.addPoints(plots_data.all_spots) # , hoverable=True
    plots.root_plot.addItem(plots.scatter_plot)

    plots.scatter_plot.addPoints(plots_data.all_spots)

    return app, win, plots, plots_data


if __name__ == '__main__':
    plots, win, app = plot_raster_plot()
    pg.exec()