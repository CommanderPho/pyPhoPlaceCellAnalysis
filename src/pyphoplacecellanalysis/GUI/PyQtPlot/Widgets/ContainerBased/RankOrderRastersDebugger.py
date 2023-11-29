import operator
import weakref
from collections import OrderedDict

from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
import numpy as np
import pandas as pd
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
# from pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots

import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum


from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses, ShuffleHelper, Zscorer

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _build_default_tick, build_scatter_plot_kwargs, build_shared_sorted_neuron_color_maps
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import RasterScatterPlotManager, UnitSortOrderManager, _build_default_tick, _build_scatter_plotting_managers, _prepare_spikes_df_from_filter_epochs, _subfn_build_and_add_scatterplot_row
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multi_sort_raster_browser

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin, ScatterItemData
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager



__all__ = ['RankOrderRastersDebugger']



""" 2023-11-03 - Debugger for directional laps shuffles
# Global (all odd/even)
# plots_selected_spikes_df_dict = {}

## EVEN: "RL"
# is_even = (an_epoch.lap_dir == 0)

## ODD: "LR"
# is_odd = (an_epoch.lap_dir == 1)

"""


# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import _debug_plot_directional_template_rasters, build_selected_spikes_df, add_selected_spikes_df_points_to_scatter_plot


@metadata_attributes(short_name=None, tags=['gui'], input_requires=[], output_provides=[], uses=['_debug_plot_directional_template_rasters', 'add_selected_spikes_df_points_to_scatter_plot'], used_by=[], creation_date='2023-11-17 19:59', related_items=[])
@define(slots=False)
class RankOrderRastersDebugger:
    """ RankOrderRastersDebugger displays four rasters showing the same spikes but sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger

    _out = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)
    
    
    Updating Display Epoch:
        The `self.on_update_epoch_IDX(an_epoch_idx=0)` can be used to control which Epoch is displayed, and is synchronized across all four sorts.
    
    """
    global_spikes_df: pd.DataFrame = field(repr=False)
    active_epochs_df: pd.DataFrame = field(repr=False)
    track_templates: TrackTemplates = field(repr=False)
    RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)
    LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)
    
    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False, repr=False)
    ui: PhoUIContainer = field(init=False, repr=False)
    
    active_epoch_IDX: int = field(default=0, repr=True)

    on_idx_changed_callback_function_dict: Dict[str, Callable] = field(default=Factory(dict), repr=False)
    

    @property
    def n_epochs(self) -> int:
        return np.shape(self.active_epochs_df)[0]
    
    @classmethod
    def _build_neuron_y_labels(cls, a_plot_item, a_decoder_color_map):
        """ 2023-11-29 - builds the y-axis text labels for a single one of the four raster plots. """
        [[x1, x2], [y1, y2]] = a_plot_item.getViewBox().viewRange() # get the x-axis range for initial position
        
        _out_text_items = {}
        for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
            # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
            text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
            text.setPos(x2, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
            a_plot_item.addItem(text)

            # ## Mode 2: stillItem
            # text.setFlag(text.GraphicsItemFlag.ItemIgnoresTransformations) # This line is necessary
            # text.setParentItem(a_plot_item) # # Use this instead of `plot.addItem`
            # # This position will be in pixels, not scene coordinates since transforms are ignored. You can use helpers like `mapFromScene()` etc. to translate between pixels and viewbox coordinates
            # text.setPos(300, (10*(cell_i+1)))

            # ## Mode 3: pg.LabelItem - this does not resize when the plot is resized like when the window is resized.
            # text = pg.LabelItem(f"{int(aclu)}", size="12pt", color=pg.mkColor(a_color_vector))
            # text.setParentItem(a_plot_item)
            # text.anchor(itemPos=(1,0), parentPos=(1,0), offset=(-10,(1*(cell_i+1))))

            _out_text_items[aclu] = text
            
        return _out_text_items

    @classmethod
    def _perform_update_cell_y_labels(cls, text_items_dict):
        """ text_items_dict, _out_rank_order_event_raster_debugger, _out_text_items """
        # _active_plot_identifier = 'long_LR'
        # a_plot_item = _out_rank_order_event_raster_debugger.plots.LR_plots.ax[_active_plot_identifier]
        
        for a_plot_item, _out_text_items in text_items_dict.items():
            # a_plot_item = _out_rank_order_event_raster_debugger.plots.LR_plots.scatter_plots[_active_plot_identifier] # AttributeError: 'ScatterPlotItem' object has no attribute 'addItem'
            [[x1, x2], [y1, y2]] = a_plot_item.getViewBox().viewRange() # get the x-axis range
            # midpoint = x1 + ((x1 + x2)/2.0)
            # print(f'x1: {x1}, x2: {x2}, midpoint: {midpoint}')
            for cell_i, (aclu, text) in enumerate(_out_text_items.items()):
                text.setPos(x2, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?


    def _build_cell_y_labels(self):
        """Builds y-axis labels for each of the rasters and stores them in `self.plots.text_items_dict`
        """
        LR_plots_data: RenderPlotsData = self.plots_data.LR_plots_data
        RL_plots_data: RenderPlotsData = self.plots_data.RL_plots_data

        ## Built flat lists across all four rasters so they aren't broken up into LR/RL when indexing:
        _active_plot_identifiers = list(LR_plots_data.plots_data_dict.keys()) + list(RL_plots_data.plots_data_dict.keys()) # ['long_LR', 'short_LR', 'long_RL', 'short_RL']
        _paired_plots_data = [LR_plots_data, LR_plots_data, RL_plots_data, RL_plots_data]
        _paired_plots = [self.plots.LR_plots, self.plots.LR_plots, self.plots.RL_plots, self.plots.RL_plots]

        emphasis_state = SpikeEmphasisState.Default

        self.plots.text_items_dict = {}

        for _active_plot_identifier, plots_data, plots in zip(_active_plot_identifiers, _paired_plots_data, _paired_plots):
            # plots_data: RenderPlotsData = LR_plots_data

            # plots_data.plots_spikes_df_dict[_active_plot_identifier] = plots_data.plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(plots_data.plots_spikes_df_dict[_active_plot_identifier])
            # plots_data.plots_spikes_df_dict[_active_plot_identifier]

            ## Add the neuron_id labels to the rasters:
            raster_plot_manager = plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager
            aclus_list = list(raster_plot_manager.params.config_items.keys())
            a_decoder_color_map = {aclu:raster_plot_manager.params.config_items[aclu].curr_state_pen_dict[emphasis_state].color() for aclu in aclus_list} # Recover color from pen:
            a_plot_item = plots.ax[_active_plot_identifier]	
            self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map)


    def update_cell_y_labels(self):
        """ called whenever the window scrolls or changes to reposition the y-axis labels created with self._build_cell_y_labels """
        self._perform_update_cell_y_labels(self.plots.text_items_dict)


    @classmethod
    def init_rank_order_debugger(cls, global_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame], LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame]):
        """ 
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()
          
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
        from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock, DockDisplayConfig
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig

        _obj = cls(global_spikes_df=global_spikes_df, active_epochs_df=active_epochs_df.copy(), track_templates=track_templates,
             RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict=RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict=LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

        name:str = 'RankOrderRastersDebugger'
        # LR_display_outputs, RL_display_outputs = cls._debug_plot_directional_template_rasters(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates) # `_debug_plot_directional_template_rasters` main plot commmand
        LR_display_outputs, RL_display_outputs = cls._modern_debug_plot_directional_template_rasters(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates) # `_debug_plot_directional_template_rasters` main plot commmand
        LR_app, LR_win, LR_plots, LR_plots_data, LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs = LR_display_outputs
        RL_app, RL_win, RL_plots, RL_plots_data, RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs = RL_display_outputs

        # Embedding in docks:
        # root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(RL_win, LR_win, title='Pho Debug Plot Directional Template Rasters')

        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Debug Plot Directional Template Rasters')

        even_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_even_dock_colors)
        odd_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_odd_dock_colors)

        _out_dock_widgets = {}
        dock_configs = (even_dock_config, odd_dock_config)
        dock_add_locations = (['left'], ['right'])
        
        _out_dock_widgets['RL'] = root_dockAreaWindow.add_display_dock(identifier='RL', widget=RL_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[0], display_config=dock_configs[0])
        _out_dock_widgets['LR'] = root_dockAreaWindow.add_display_dock(identifier='LR', widget=LR_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[1], display_config=dock_configs[1])
        
        ctrls_dock_config = CustomDockDisplayConfig(showCloseButton=False)
        slider = pg.QtWidgets.QSlider(pg.QtCore.Qt.Horizontal)
        slider.setRange(0, (_obj.n_epochs-1))
        slider.setValue(50)
        # layout.addWidget(slider)
        slider.valueChanged.connect(_obj.on_update_epoch_IDX)
        
        ctrl_layout = pg.LayoutWidget()
        ctrl_layout.addWidget(slider, row=0, col=0, colspan=1)
        
        logTextEdit = pg.QtWidgets.QTextEdit()
        logTextEdit.setReadOnly(True)
        logTextEdit.setObjectName("logTextEdit")

        ctrl_layout.addWidget(logTextEdit, row=1, rowspan=3, col=0, colspan=1)

        _out_dock_widgets['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,100), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)

        
        root_dockAreaWindow.resize(500, 600)

        _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, LR_app=LR_app, LR_win=LR_win, LR_plots=LR_plots, RL_app=RL_app, RL_win=RL_win, RL_plots=RL_plots, dock_widgets=_out_dock_widgets, ctrl_widgets={'slider': slider}, text_items_dict=None)
        _obj.plots_data = RenderPlotsData(name=name, LR_plots_data=LR_plots_data, LR_on_update_active_epoch=LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs=LR_on_update_active_scatterplot_kwargs,
                                           RL_plots_data=RL_plots_data, RL_on_update_active_epoch=RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs=RL_on_update_active_scatterplot_kwargs)
        
        _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, slider=slider, logTextEdit=logTextEdit, dock_configs=dock_configs)
        

        try:
            ## rank_order_results.LR_ripple.selected_spikes_df mode:
            if isinstance(LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, pd.DataFrame) and isinstance(RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, pd.DataFrame):
                # already a selected_spikes_df! Use it raw!
                _obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map = deepcopy(RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
                _obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map = deepcopy(LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
            else:
                ## Build the selected spikes df:
                (_obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map), (_obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map) = _obj.build_selected_spikes_df(_obj.track_templates, _obj.active_epochs_df,
                                                                                                                                                                                                                    _obj.RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict,                                                                                                                                                                                                                _obj.LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict)
            ## Add the spikes
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'long_LR')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'short_LR')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'long_RL')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'short_RL')

        except IndexError:
            print(f'WARN: the selected spikes did not work properly, so none will be shown.')
            pass
        

        _obj._build_cell_y_labels() # builds the cell labels

        return _obj



    def on_update_active_epoch(self, an_epoch_idx: int, an_epoch):
        """ captures: LR_on_update_active_epoch, RL_on_update_active_epoch """
        self.plots_data.LR_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
        self.plots_data.RL_on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
        # Update window titles:
        an_epoch_string: str = f'idx: {an_epoch.Index}, t: {an_epoch.start:0.2f}, {an_epoch.stop:0.2f}, lbl: {str(an_epoch.label)}'
        self.plots.dock_widgets['LR'][1].setTitle(f'LR Directional Pf Rasters - epoch_IDX: {int(an_epoch_idx)} - epoch: {an_epoch_string}')
        self.plots.dock_widgets['RL'][1].setTitle(f'RL Directional Pf Rasters - epoch_IDX: {int(an_epoch_idx)} - epoch: {an_epoch_string}')
        
        self.plots.LR_win.setWindowTitle(f'LR Directional Pf Rasters - epoch_IDX: {int(an_epoch_idx)} - epoch: {an_epoch_string}')
        self.plots.RL_win.setWindowTitle(f'RL Directional Pf Rasters - epoch_IDX: {int(an_epoch_idx)} - epoch: {an_epoch_string}')

        self.update_cell_y_labels()
                

    def on_update_epoch_IDX(self, an_epoch_idx: int):
        """ Calls self.on_update_epoch_IDX(...)
        
        captures on_update_active_epoch, active_epochs_df to extract the epoch time range and call `on_update_active_epoch` """
        self.active_epoch_IDX = an_epoch_idx # set the active_epoch_IDX, not the index value
        a_df_idx = self.active_epochs_df.index.to_numpy()[an_epoch_idx]
        print(f'a_df_idx: {a_df_idx}')
        # curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == (a_df_idx+1))] # this +1 here makes zero sense
        curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == a_df_idx)] # this +1 here makes zero sense

        # curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.lap_id == (an_epoch_idx+1))]
        curr_epoch = list(curr_epoch_df.itertuples())[0]
        self.on_update_active_epoch(an_epoch_idx, curr_epoch)

        for a_callback_name, a_callback_fn in self.on_idx_changed_callback_function_dict.items():
            a_callback_fn(self, an_epoch_idx)

    def write_to_log(self, log_messages):
        self.ui.logTextEdit.append(log_messages)
        
        
    def get_ipywidget(self):
        """ Displays a slider that allows the user to select the epoch_IDX instead of having to type it and call it manually

        """
        import ipywidgets as widgets
        # 2023-11-17: Displays a slider that allows the user to select the epoch_IDX instead of having to type it and call it manually
        # https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html#throttling

        self.active_epoch_IDX = 0
        # Define the update function
        def update_function(change):
            global active_epoch_IDX
            new_value = change['new']
            active_epoch_IDX = new_value
            # Add your update logic here
            print(f"Slider value updated to: {new_value}")
            self.on_update_epoch_IDX(new_value) # call the update
            # print(f'n_unique_cells_participating_in_replay[{active_epoch_IDX}]: {n_unique_cells_participating_in_replay[active_epoch_IDX]}')
            selected_epoch_df = self.active_epochs_df[self.active_epochs_df.index == active_epoch_IDX] # should only contain one entry, the selected epoch.
            curr_epoch = list(selected_epoch_df.itertuples())[0] # extract the interval of interest as a namedtuple object
            print(f"curr_epoch: {curr_epoch}")
            
        # Create a slider widget
        slider = widgets.IntSlider(value=0, min=0, max=(self.n_epochs-1), step=1, description='Epoch Index')

        # Link the update function to value changes in the slider
        slider.observe(update_function, names='value')

        # self.ui.slider = slider
        return slider
    
        # Display the slider
        # display(slider)


    @function_attributes(short_name='debug_plot_directional_template_rasters', tags=['directional', 'templates', 'debugger', 'pyqtgraph'], input_requires=[], output_provides=[], uses=['plot_multi_sort_raster_browser'], used_by=[], creation_date='2023-11-02 14:06', related_items=[])
    @classmethod
    def _debug_plot_directional_template_rasters(cls, spikes_df, active_epochs_df, track_templates):
        """ Perform raster plotting by getting our data from track_templates (TrackTemplates)
        There will be four templates, one for each run direction x each maze configuration


        Usage:
            from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import _debug_plot_directional_template_rasters

            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
            global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
            global_laps_epochs_df = global_laps.to_dataframe()
            # app, win, plots, plots_data, (on_update_active_epoch, on_update_active_scatterplot_kwargs) = _debug_plot_directional_template_rasters(global_spikes_df, global_laps_epochs_df, track_templates)

            LR_display_outputs, RL_display_outputs = _debug_plot_directional_template_rasters(global_spikes_df, global_laps_epochs_df, track_templates)

        """
        from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
        from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
        from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
        from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color

        ## spikes_df: get the spikes to plot
        # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        # track_templates.shared_LR_aclus_only_neuron_IDs

        included_neuron_ids = np.sort(np.union1d(track_templates.shared_RL_aclus_only_neuron_IDs, track_templates.shared_LR_aclus_only_neuron_IDs))
        n_neurons = len(included_neuron_ids)

        # Get only the spikes for the shared_aclus:
        spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids)
        # spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
        spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX', epoch_label_column_name=None) # , override_time_variable_name='t_seconds'
        # spikes_df = spikes_df[spikes_df['ripple_id'] != -1]
        spikes_df = spikes_df[(spikes_df['new_lap_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
        spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards


        # CORRECT: Even: RL, Odd: LR
        RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy() # (69, )
        LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy() # (64, )
        RL_long, RL_short = [(a_sort-1) for a_sort in track_templates.decoder_RL_pf_peak_ranks_list] # nope, different sizes: (62,), (69,)
        LR_long, LR_short = [(a_sort-1) for a_sort in track_templates.decoder_LR_pf_peak_ranks_list]

        assert np.shape(RL_long) == np.shape(RL_short), f"{np.shape(RL_long)} != {np.shape(RL_short)}"

        neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None)
        unit_colors_list = neuron_colors_ndarray.copy()

        unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_long, LR_long, RL_short, LR_short)))
        # unit_colors_list_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (unit_colors_list, unit_colors_list, unit_colors_list, unit_colors_list)))

        
        ## Do Even/Odd Separately:
        unit_colors_map = dict(zip(included_neuron_ids, neuron_colors_ndarray.copy().T))
        RL_unit_colors_list = np.array([v for k, v in unit_colors_map.items() if k in RL_neuron_ids]).T # should be (4, len(shared_RL_aclus_only_neuron_IDs))
        LR_unit_colors_list = np.array([v for k, v in unit_colors_map.items() if k in LR_neuron_ids]).T # should be (4, len(shared_RL_aclus_only_neuron_IDs))
        unit_colors_list_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_unit_colors_list, LR_unit_colors_list, RL_unit_colors_list, LR_unit_colors_list)))

        # THE LOGIC MUST BE WRONG HERE. Slicing and dicing each Epoch separately is NOT OKAY. Spikes must be built before-hand. Loser.

        # Even:
        RL_names = ['long_RL', 'short_RL']
        RL_unit_sort_orders_dict = {k:v for k, v in unit_sort_orders_dict.items() if k in RL_names}
        RL_unit_colors_list_dict = {k:v for k, v in unit_colors_list_dict.items() if k in RL_names}
        RL_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(RL_neuron_ids)
        RL_spikes_df, RL_neuron_id_to_new_IDX_map = RL_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        RL_display_outputs = plot_multi_sort_raster_browser(RL_spikes_df, RL_neuron_ids, unit_sort_orders_dict=RL_unit_sort_orders_dict, unit_colors_list_dict=RL_unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters_RL', defer_show=False, active_context=None)
        # RL_app, RL_win, RL_plots, RL_plots_data, RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs = RL_display_outputs

        # Odd:
        LR_names = ['long_LR', 'short_LR']
        LR_unit_sort_orders_dict = {k:v for k, v in unit_sort_orders_dict.items() if k in LR_names}
        LR_unit_colors_list_dict = {k:v for k, v in unit_colors_list_dict.items() if k in LR_names}
        LR_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(LR_neuron_ids)
        LR_spikes_df, LR_neuron_id_to_new_IDX_map = LR_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        LR_display_outputs = plot_multi_sort_raster_browser(LR_spikes_df, LR_neuron_ids, unit_sort_orders_dict=LR_unit_sort_orders_dict, unit_colors_list_dict=LR_unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters_LR', defer_show=False, active_context=None)
        # LR_app, LR_win, LR_plots, LR_plots_data, LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs = LR_display_outputs

        # app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = plot_multi_sort_raster_browser(spikes_df, included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)
        
        return LR_display_outputs, RL_display_outputs



    @classmethod
    def _modern_debug_plot_directional_template_rasters(cls, spikes_df, active_epochs_df, track_templates):
        """ 2023-11-28 **UPDATING** - Perform raster plotting by getting our data from track_templates (TrackTemplates)
        There will be four templates, one for each run direction x each maze configuration


        Usage:
            from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.DirectionalTemplatesRastersDebugger import _debug_plot_directional_template_rasters

            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
            global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
            global_laps_epochs_df = global_laps.to_dataframe()
            # app, win, plots, plots_data, (on_update_active_epoch, on_update_active_scatterplot_kwargs) = _debug_plot_directional_template_rasters(global_spikes_df, global_laps_epochs_df, track_templates)

            LR_display_outputs, RL_display_outputs = _debug_plot_directional_template_rasters(global_spikes_df, global_laps_epochs_df, track_templates)

        """
        from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
        from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
        from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
        from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons

        ## spikes_df: get the spikes to plot
        # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        # track_templates.shared_LR_aclus_only_neuron_IDs


        decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }
        # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`               
        # sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sorted_pf_tuning_curves = paired_incremental_sort_neurons(decoders_dict=decoders_dict, included_any_context_neuron_ids=included_any_context_neuron_ids)

        neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        # _unit_qcolors_map, unit_colors_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
        unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
        # `unit_colors_map` is main colors output

        included_neuron_ids = np.array(list(unit_colors_map.keys()))
        n_neurons = len(included_neuron_ids)

        print(f'included_neuron_ids: {included_neuron_ids}, n_neurons: {n_neurons}')
        
        # included_neuron_ids = np.sort(np.union1d(track_templates.shared_RL_aclus_only_neuron_IDs, track_templates.shared_LR_aclus_only_neuron_IDs))
        # n_neurons = len(included_neuron_ids)

        # Get only the spikes for the shared_aclus:
        spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids)
        # spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
        spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX', epoch_label_column_name=None) # , override_time_variable_name='t_seconds'
        # spikes_df = spikes_df[spikes_df['ripple_id'] != -1]
        spikes_df = spikes_df[(spikes_df['new_lap_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
        spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards


        # CORRECT: Even: RL, Odd: LR
        RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy() # (69, )
        LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy() # (64, )
        RL_long, RL_short = [(a_sort-1) for a_sort in track_templates.decoder_RL_pf_peak_ranks_list] # nope, different sizes: (62,), (69,)
        LR_long, LR_short = [(a_sort-1) for a_sort in track_templates.decoder_LR_pf_peak_ranks_list]

        assert np.shape(RL_long) == np.shape(RL_short), f"{np.shape(RL_long)} != {np.shape(RL_short)}"

        unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_long, LR_long, RL_short, LR_short)))
        # unit_colors_list_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (unit_colors_list, unit_colors_list, unit_colors_list, unit_colors_list)))

        # included_any_context_neuron_ids_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_neuron_ids, LR_neuron_ids, RL_neuron_ids, LR_neuron_ids)))
        # sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_separately_sort_neurons(decoders_dict=decoders_dict,
        #                                                                                                                                                 included_any_context_neuron_ids_dict=included_any_context_neuron_ids_dict,
        #                                                                                                                                                 sortable_values_list_dict=unit_sort_orders_dict)

        # neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None)
        # unit_colors_list = neuron_colors_ndarray.copy()

        
        
        ## Do Even/Odd Separately:
        # unit_colors_map = dict(zip(included_neuron_ids, neuron_colors_ndarray.copy().T))

        ## This is for the NDArray version:
        # RL_unit_colors_list = [pg.mkColor(v) for k, v in unit_colors_map.items() if k in RL_neuron_ids] # should be a list of QColors, this is confirmed to be what is expected for the colors
        # LR_unit_colors_list = [pg.mkColor(v) for k, v in unit_colors_map.items() if k in LR_neuron_ids] # should be a list of QColors

        ## This is for the NDArray version:
        RL_unit_colors_list = np.array([v for k, v in _unit_colors_ndarray_map.items() if k in RL_neuron_ids]).T # should be (4, len(shared_RL_aclus_only_neuron_IDs))
        LR_unit_colors_list = np.array([v for k, v in _unit_colors_ndarray_map.items() if k in LR_neuron_ids]).T # should be (4, len(shared_RL_aclus_only_neuron_IDs))        

        unit_colors_list_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_unit_colors_list, LR_unit_colors_list, RL_unit_colors_list, LR_unit_colors_list))) # the colors dict for all four templates

        # # #TODO 2023-11-29 18:16: - [ ] paired_separately_sort_neurons version:
        # unit_colors_list_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], sort_helper_neuron_id_to_neuron_colors_dicts))
        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], sort_helper_neuron_id_to_sort_IDX_dicts)) # disable this if doesn't work
        
        # THE LOGIC MUST BE WRONG HERE. Slicing and dicing each Epoch separately is NOT OKAY. Spikes must be built before-hand. Loser.

        # Even:
        RL_names = ['long_RL', 'short_RL']
        RL_unit_sort_orders_dict = {k:v for k, v in unit_sort_orders_dict.items() if k in RL_names}
        RL_unit_colors_list_dict = {k:v for k, v in unit_colors_list_dict.items() if k in RL_names}
        RL_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(RL_neuron_ids)
        RL_spikes_df, RL_neuron_id_to_new_IDX_map = RL_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        RL_display_outputs = plot_multi_sort_raster_browser(RL_spikes_df, RL_neuron_ids, unit_sort_orders_dict=RL_unit_sort_orders_dict, unit_colors_list_dict=RL_unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters_RL', defer_show=False, active_context=None)
        # RL_app, RL_win, RL_plots, RL_plots_data, RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs = RL_display_outputs

        # Odd:
        LR_names = ['long_LR', 'short_LR']
        LR_unit_sort_orders_dict = {k:v for k, v in unit_sort_orders_dict.items() if k in LR_names}
        LR_unit_colors_list_dict = {k:v for k, v in unit_colors_list_dict.items() if k in LR_names}
        LR_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(LR_neuron_ids)
        LR_spikes_df, LR_neuron_id_to_new_IDX_map = LR_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        LR_display_outputs = plot_multi_sort_raster_browser(LR_spikes_df, LR_neuron_ids, unit_sort_orders_dict=LR_unit_sort_orders_dict, unit_colors_list_dict=LR_unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters_LR', defer_show=False, active_context=None)
        # LR_app, LR_win, LR_plots, LR_plots_data, LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs = LR_display_outputs

        # app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = plot_multi_sort_raster_browser(spikes_df, included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)
        
        return LR_display_outputs, RL_display_outputs



    @classmethod
    def build_selected_spikes_df(cls, track_templates, active_epochs_df, RL_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict):
        """ "selected" in this sense means those spikes/spots that were used for the rank-order analysis, such as 'first' for the ripples or 'median' for the laps.

            ## Use LR_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, RL_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict to plot the active median spike

        Usage:


            active_epochs_df = global_laps_epochs_df.copy()
            (RL_selected_spike_df, RL_neuron_id_to_new_IDX_map), (LR_selected_spike_df, LR_neuron_id_to_new_IDX_map) = build_selected_spikes_df(track_templates, active_epochs_df, RL_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


        """
        # CORRECT: Even: RL, Odd: LR
        RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy()
        LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy()

        ## WE HAVE TO BUILD OUT selected_spikes_df ahead of time or die trying. Not one epoch at a time.

        # Converts the selected spikes information dict (containing the median/first spikes for each epoch) into a spikes_df capable of being rendered on the raster plot.
        # selected_spike_df_list = []

        RL_selected_spike_df_list = []
        LR_selected_spike_df_list = []

        for an_epoch in active_epochs_df.itertuples():
            active_epoch_idx = an_epoch.Index

            # RL (Even):
            directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict = RL_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict
            shared_directional_aclus_only_neuron_IDs = RL_neuron_ids
            selected_spike_fragile_neuron_IDX = np.squeeze(directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict[active_epoch_idx][:,0]).astype('int')
            selected_spike_aclus = shared_directional_aclus_only_neuron_IDs[selected_spike_fragile_neuron_IDX].astype('int')
            selected_spike_times = np.squeeze(directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict[active_epoch_idx][:,1]) # (n_cells, 2)
            selected_spike_epoch_idx = np.full_like(selected_spike_fragile_neuron_IDX, active_epoch_idx).astype('int')
            
            selected_spike_df = pd.DataFrame({'t': selected_spike_times, 'fragile_linear_neuron_IDX': selected_spike_fragile_neuron_IDX, 'aclu': selected_spike_aclus, 'epoch_IDX': selected_spike_epoch_idx})
            if hasattr(an_epoch, 'lap_dir'):
                selected_spike_lap_dir = np.full_like(selected_spike_fragile_neuron_IDX, an_epoch.lap_dir).astype('int')
                selected_spike_df['lap_dir'] = selected_spike_lap_dir
            RL_selected_spike_df_list.append(selected_spike_df)
            

            # LR (Odd):
            directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict = LR_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict
            shared_directional_aclus_only_neuron_IDs = LR_neuron_ids
            selected_spike_fragile_neuron_IDX = np.squeeze(directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict[active_epoch_idx][:,0]).astype('int')
            selected_spike_aclus = shared_directional_aclus_only_neuron_IDs[selected_spike_fragile_neuron_IDX].astype('int')
            selected_spike_times = np.squeeze(directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict[active_epoch_idx][:,1]) # (n_cells, 2)
            selected_spike_epoch_idx = np.full_like(selected_spike_fragile_neuron_IDX, active_epoch_idx).astype('int')
            selected_spike_df = pd.DataFrame({'t': selected_spike_times, 'fragile_linear_neuron_IDX': selected_spike_fragile_neuron_IDX, 'aclu': selected_spike_aclus, 'epoch_IDX': selected_spike_epoch_idx})
            if hasattr(an_epoch, 'lap_dir'):
                selected_spike_lap_dir = np.full_like(selected_spike_fragile_neuron_IDX, an_epoch.lap_dir).astype('int')
                selected_spike_df['lap_dir'] = selected_spike_lap_dir

            LR_selected_spike_df_list.append(selected_spike_df)


        RL_selected_spike_df = pd.concat(RL_selected_spike_df_list, ignore_index=True)
        # Sort by columns: 't' (ascending), 'aclu' (ascending), 'epoch_IDX' (ascending)
        RL_selected_spike_df = RL_selected_spike_df.sort_values(['t', 'epoch_IDX', 'aclu']).reset_index(drop=True) # someting wong for RLs. WRY?: self.y_fragile_linear_neuron_IDX_map[a_cell_IDX]

        RL_selected_spike_df['t_rel_seconds'] = RL_selected_spike_df['t']
        RL_selected_spike_df['neuron_type'] = False # stupid workaround
        RL_selected_spike_df['flat_spike_idx'] = RL_selected_spike_df.index # stupid workaround

        LR_selected_spike_df = pd.concat(LR_selected_spike_df_list, ignore_index=True)
        # Sort by columns: 't' (ascending), 'aclu' (ascending), 'epoch_IDX' (ascending)
        LR_selected_spike_df = LR_selected_spike_df.sort_values(['t', 'epoch_IDX', 'aclu']).reset_index(drop=True) # someting wong for RLs. WRY?: self.y_fragile_linear_neuron_IDX_map[a_cell_IDX]

        LR_selected_spike_df['t_rel_seconds'] = LR_selected_spike_df['t']
        LR_selected_spike_df['neuron_type'] = False # stupid workaround
        LR_selected_spike_df['flat_spike_idx'] = LR_selected_spike_df.index # stupid workaround
        
        # Need to split into RL/LR versions:
        RL_selected_spike_df, RL_neuron_id_to_new_IDX_map = deepcopy(RL_selected_spike_df).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        LR_selected_spike_df, LR_neuron_id_to_new_IDX_map = deepcopy(LR_selected_spike_df).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        
        return (RL_selected_spike_df, RL_neuron_id_to_new_IDX_map), (LR_selected_spike_df, LR_neuron_id_to_new_IDX_map)

    @classmethod
    def add_selected_spikes_df_points_to_scatter_plot(cls, plots_data, plots, selected_spikes_df, _active_plot_identifier = 'long_RL'):
        """ Called after above `build_selected_spikes_df`

        Usage:
            add_selected_spikes_df_points_to_scatter_plot(plots_data=LR_plots_data, plots=LR_plots, selected_spikes_df=deepcopy(LR_selected_spike_df), _active_plot_identifier = 'long_LR')
            add_selected_spikes_df_points_to_scatter_plot(plots_data=LR_plots_data, plots=LR_plots, selected_spikes_df=deepcopy(LR_selected_spike_df), _active_plot_identifier = 'short_LR')
            add_selected_spikes_df_points_to_scatter_plot(plots_data=RL_plots_data, plots=RL_plots, selected_spikes_df=deepcopy(RL_selected_spike_df), _active_plot_identifier = 'long_RL')
            add_selected_spikes_df_points_to_scatter_plot(plots_data=RL_plots_data, plots=RL_plots, selected_spikes_df=deepcopy(RL_selected_spike_df), _active_plot_identifier = 'short_RL')

        """
        ## Initialize global selected spikes stuff:
        plots_data.all_selected_spots_dict = {}
        plots_data.all_selected_scatterplot_tooltips_kwargs_dict = {}
        plots_data_dict = plots_data.plots_data_dict # derived
        selected_spikes_df = plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(selected_spikes_df)

        ## Build the spots for the raster plot:
        plots_data.all_selected_spots_dict[_active_plot_identifier], plots_data.all_selected_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(selected_spikes_df, plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)
        # Override the pen for the selected spots, the default renders them looking exactly like normal spikes which is no good:
        for a_point in plots_data.all_selected_spots_dict[_active_plot_identifier]:
            a_point['pen'] = pg.mkPen('#ffffff8e', width=3.5)
            a_point['brush'] = pg.mkBrush('#ffffff2f')


        ## Add the median spikes to the plots:
        a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
        a_scatter_plot.addPoints(plots_data.all_selected_spots_dict[_active_plot_identifier])


        

