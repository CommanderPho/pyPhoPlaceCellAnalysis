from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
from nptyping import NDArray
import numpy as np
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import build_shared_sorted_neuron_color_maps
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multi_sort_raster_browser, plot_raster_plot

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState

from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum

from neuropy.utils.indexing_helpers import find_desired_sort_indicies
from pyphoplacecellanalysis.GUI.Qt.Widgets.ScrollBarWithSpinBox.ScrollBarWithSpinBox import ScrollBarWithSpinBox

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, PhoDockAreaContainingWindow
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum, LongShortDisplayConfigManager
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`


from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyqtgraph as pg
import pyqtgraph.exporters
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons, paired_incremental_sort_neurons # _display_directional_template_debugger
from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays


from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color




__all__ = ['TemplateDebugger']



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
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import _debug_plot_directional_template_rasters, build_selected_spikes_df, add_selected_spikes_df_points_to_scatter_plot

@metadata_attributes(short_name=None, tags=['gui', 'incomplete', 'not_used', 'not_implemented'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-11 10:24', related_items=[])
@define(slots=False)
class TemplateDebugger:
    """ TemplateDebugger displays four rasters showing the same spikes but sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger

    _out = TemplateDebugger.init_rank_order_debugger(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


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

    @property
    def active_epoch_tuple(self) -> tuple:
        """ returns a namedtuple describing the single epoch corresponding to `self.active_epoch_IDX`. """
        a_df_idx = self.active_epochs_df.index.to_numpy()[self.active_epoch_IDX]
        curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == a_df_idx)] # this +1 here makes zero sense
        curr_epoch = list(curr_epoch_df.itertuples(name='EpochTuple'))[0]
        return curr_epoch


    def get_active_epoch_spikes_df(self) -> pd.DataFrame:
        active_epoch_tuple = self.active_epoch_tuple
        active_epoch_spikes_df: pd.DataFrame = deepcopy(self.global_spikes_df.spikes.time_sliced(active_epoch_tuple.start, active_epoch_tuple.stop))
        return active_epoch_spikes_df

    def get_epoch_active_aclus(self) -> NDArray:
        """ returns a list of aclus active (having at least one spike) in the current epoch (based on `self.active_epoch`) """
        active_epoch_spikes_df: pd.DataFrame = self.get_active_epoch_spikes_df()
        active_epoch_unique_active_aclus = np.unique(active_epoch_spikes_df['aclu'].to_numpy())
        return active_epoch_unique_active_aclus


    @classmethod
    def init_templates_debugger(cls, global_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame], LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame]):
        """
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig

        _obj = cls(global_spikes_df=global_spikes_df, active_epochs_df=active_epochs_df.copy(), track_templates=track_templates,
             RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict=RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict=LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

        name:str = 'TemplateDebugger'
        
        ## 2023-11-30 - Newest Version using separate rasters:
        _obj.plots_data, _obj.plots = cls._build_internal_raster_plots(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates, debug_print=True)
        #TODO 2023-11-30 15:14: - [ ] Unpacking and putting in docks and such not yet finished. Update functions would need to be done separately.
        rasters_display_outputs = _obj.plots.rasters_display_outputs
        all_apps = {a_decoder_name:a_raster_setup_tuple.app for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_windows = {a_decoder_name:a_raster_setup_tuple.win for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_plots = {a_decoder_name:a_raster_setup_tuple.plots for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_plots_data = {a_decoder_name:a_raster_setup_tuple.plots_data for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}

        main_plot_identifiers_list = list(all_windows.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']

        ## Extract the data items:
        all_separate_data_all_spots = {a_decoder_name:a_raster_setup_tuple.plots_data.all_spots for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_data_all_scatterplot_tooltips_kwargs = {a_decoder_name:a_raster_setup_tuple.plots_data.all_scatterplot_tooltips_kwargs for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_data_new_sorted_rasters = {a_decoder_name:a_raster_setup_tuple.plots_data.new_sorted_raster for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_data_spikes_dfs = {a_decoder_name:a_raster_setup_tuple.plots_data.spikes_df for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}

        # Extract the plot/renderable items
        all_separate_root_plots = {a_decoder_name:a_raster_setup_tuple.plots.root_plot for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_grids = {a_decoder_name:a_raster_setup_tuple.plots.grid for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_scatter_plots = {a_decoder_name:a_raster_setup_tuple.plots.scatter_plot for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        all_separate_debug_header_labels = {a_decoder_name:a_raster_setup_tuple.plots.debug_header_label for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        
        # Embedding in docks:
        # root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(RL_win, LR_win, title='Pho Debug Plot Directional Template Rasters')
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Debug Plot Directional Template Rasters')

        ## Build Dock Widgets:
        def get_utility_dock_colors(orientation, is_dim):
            """ used for CustomDockDisplayConfig for non-specialized utility docks """
            # Common to all:
            if is_dim:
                fg_color = '#aaa' # Grey
            else:
                fg_color = '#fff' # White
                
            # a purplish-royal-blue 
            if is_dim:
                bg_color = '#d8d8d8' 
                border_color = '#717171' 
            else:
                bg_color = '#9d9d9d' 
                border_color = '#3a3a3a' 

            return fg_color, bg_color, border_color


        # decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        _out_dock_widgets = {}
        dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
        # dock_add_locations = (['left'], ['left'], ['right'], ['right'])
        # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))
        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))

        for i, (a_decoder_name, a_win) in enumerate(all_windows.items()):
            if (a_decoder_name == 'short_RL'):
                short_LR_dock = root_dockAreaWindow.find_display_dock('short_LR')
                assert short_LR_dock is not None
                dock_add_locations['short_RL'] = ['bottom', short_LR_dock]
                print(f'using overriden dock location.')
                
            _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name])


       
        # Build callback functions:
        def on_update_active_scatterplot_kwargs(override_scatter_plot_kwargs):
            """ captures: main_plot_identifiers_list, plots, plots_data """
            for _active_plot_identifier in main_plot_identifiers_list:
                # for _active_plot_identifier, a_scatter_plot in plots.scatter_plots.items():
                # new_ax = plots.ax[_active_plot_identifier]
                a_scatter_plot = all_separate_scatter_plots[_active_plot_identifier]
                plots_data = all_separate_plots_data[_active_plot_identifier]
                a_scatter_plot.setData(plots_data.all_spots_dict[_active_plot_identifier], **(plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {}), **override_scatter_plot_kwargs)

        def on_update_active_epoch(an_epoch_idx, an_epoch):
            """ captures: main_plot_identifiers_list, all_separate_root_plots """
            for _active_plot_identifier in main_plot_identifiers_list:
                new_ax = all_separate_root_plots[_active_plot_identifier]
                new_ax.setXRange(an_epoch.start, an_epoch.stop)
                # new_ax.getAxis('left').setLabel(f'[{an_epoch.label}]')
                
                # a_scatter_plot = plots.scatter_plots[_active_plot_identifier]


        ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)

        ctrls_widget = ScrollBarWithSpinBox()
        ctrls_widget.setObjectName("ctrls_widget")
        ctrls_widget.update_range(0, (_obj.n_epochs-1))
        ctrls_widget.setValue(10)

        def valueChanged(new_val:int):
            print(f'valueChanged(new_val: {new_val})')
            _obj.on_update_epoch_IDX(int(new_val))

        ctrls_widget_connection = ctrls_widget.sigValueChanged.connect(valueChanged)
        ctrl_layout = pg.LayoutWidget()
        ctrl_layout.addWidget(ctrls_widget, row=1, rowspan=1)
        ctrl_widgets_dict = dict(ctrls_widget=ctrls_widget, ctrls_widget_connection=ctrls_widget_connection)

        logTextEdit = pg.QtWidgets.QTextEdit()
        logTextEdit.setReadOnly(True)
        logTextEdit.setObjectName("logTextEdit")

        ctrl_layout.addWidget(logTextEdit, row=2, rowspan=3, col=0, colspan=1)

        _out_dock_widgets['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,100), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)

        ## Add two labels in the top row that show the Long/Short column values:
        long_short_info_layout = pg.LayoutWidget()
        long_short_info_layout.setObjectName('layoutLongShortInfo')

        long_info_label = long_short_info_layout.addLabel(text='LONG', row=0, col=0)
        long_info_label.setObjectName('lblLongInfo')
        # long_info_label.setAlignment(pg.QtCore.Qt.AlignCenter)
        long_info_label.setAlignment(pg.QtCore.Qt.AlignLeft)

        short_info_label = long_short_info_layout.addLabel(text='SHORT', row=0, col=1)
        short_info_label.setObjectName('lblShortInfo')
        # short_info_label.setAlignment(pg.QtCore.Qt.AlignCenter)
        short_info_label.setAlignment(pg.QtCore.Qt.AlignRight)
        
        _out_dock_widgets['LongShortColumnsInfo_dock'] = root_dockAreaWindow.add_display_dock(identifier='LongShortColumnsInfo_dock', widget=long_short_info_layout, dockSize=(600,60), dockAddLocationOpts=['top'], display_config=CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False, corner_radius='0px'))
        _out_dock_widgets['LongShortColumnsInfo_dock'][1].hideTitleBar() # hide the dock title bar

        # Add the widgets to the .ui:
        long_short_info_layout = long_short_info_layout
        long_info_label = long_info_label
        short_info_label = short_info_label
        info_labels_widgets_dict = dict(long_short_info_layout=long_short_info_layout, long_info_label=long_info_label, short_info_label=short_info_label)

        root_dockAreaWindow.resize(600, 900)

        ## Build final .plots and .plots_data:
        _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, apps=all_apps, all_windows=all_windows, all_separate_plots=all_separate_plots,
                                  root_plots=all_separate_root_plots, grids=all_separate_grids, scatter_plots=all_separate_scatter_plots, debug_header_labels=all_separate_debug_header_labels,
                                  dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
        _obj.plots_data = RenderPlotsData(name=name, main_plot_identifiers_list=main_plot_identifiers_list,
                                           seperate_all_spots_dict=all_separate_data_all_spots, seperate_all_scatterplot_tooltips_kwargs_dict=all_separate_data_all_scatterplot_tooltips_kwargs, seperate_new_sorted_rasters_dict=all_separate_data_new_sorted_rasters, seperate_spikes_dfs_dict=all_separate_data_spikes_dfs,
                                           on_update_active_epoch=on_update_active_epoch, on_update_active_scatterplot_kwargs=on_update_active_scatterplot_kwargs, **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']})                
        _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs)

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

        except (IndexError, KeyError):
            print(f'WARN: the selected spikes did not work properly, so none will be shown.')
            pass

        _obj._build_cell_y_labels() # builds the cell labels

        ## Cleanup when done:
        for a_decoder_name, a_root_plot in _obj.plots.root_plots.items():
            a_root_plot.setTitle(title=a_decoder_name)
            # a_root_plot.setTitle(title="")
            a_left_axis = a_root_plot.getAxis('left')# axisItem
            a_left_axis.setLabel(a_decoder_name)
            a_left_axis.setStyle(showValues=False)
            a_left_axis.setTicks([])
            # a_root_plot.hideAxis('bottom')
            # a_root_plot.hideAxis('bottom')
            a_root_plot.hideAxis('left')

        # for a_decoder_name, a_scatter_plot_item in _obj.plots.scatter_plots.items():
        #     a_scatter_plot_item.hideAxis('left')

        # Hide the debugging labels
        for a_decoder_name, a_label in _obj.plots.debug_header_labels.items():
            # a_label.setText('NEW')
            a_label.hide() # hide the labels unless we need them.


        return _obj



    # ==================================================================================================================== #
    # Extracted Functions:                                                                                                 #
    # ==================================================================================================================== #
    @classmethod
    def _subfn_rebuild_sort_idxs(cls, decoders_dict: Dict, _out_data: RenderPlotsData, use_incremental_sorting: bool, included_any_context_neuron_ids) -> RenderPlotsData:
        """ captures decoders_dict

        Updates RenderPlotsData

        """

        if use_incremental_sorting:
            # INCRIMENTAL SORTING:
            ref_decoder_name: Optional[str] = list(decoders_dict.keys())[0] # name of the reference coder. Should be 'long_LR'
            sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_incremental_sort_neurons(decoders_dict, included_any_context_neuron_ids)
        else:
            # INDIVIDUAL SORTING:
            ref_decoder_name: Optional[str] = None
            sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids)

        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        # sorted_pf_tuning_curves_dict = {a_decoder_name:a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_decoder_name, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts.keys(), sort_helper_neuron_id_to_sort_IDX_dicts.values())}

            # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
        _out_data.ref_decoder_name = ref_decoder_name
        _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
        _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
        _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts
        _out_data.sorted_pf_tuning_curves = sorted_pf_tuning_curves
        _out_data.unsorted_included_any_context_neuron_ids = deepcopy(included_any_context_neuron_ids)
        return _out_data

    # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`
    @classmethod
    def _subfn_buildUI_directional_template_debugger_data(cls, included_any_context_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, decoders_dict: Dict):
        """ Builds UI """
        _out_data = cls._subfn_rebuild_sort_idxs(decoders_dict, _out_data, use_incremental_sorting=use_incremental_sorting, included_any_context_neuron_ids=included_any_context_neuron_ids)
        # Unpack the updated _out_data:
        sort_helper_neuron_id_to_neuron_colors_dicts = _out_data.sort_helper_neuron_id_to_neuron_colors_dicts
        sorted_pf_tuning_curves = _out_data.sorted_pf_tuning_curves

        ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
        _out_plots.pf1D_heatmaps = {}
        _out_ui.text_items_dict = {}

        for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            if use_incremental_sorting:
                title_str = f'{a_decoder_name}_pf1Ds [sort: {_out_data.ref_decoder_name}]'
            else:
                title_str = f'{a_decoder_name}_pf1Ds'

            curr_curves = sorted_pf_tuning_curves[i]
            _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(curr_curves, title=title_str, show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)

            # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
            curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img

            a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)

            # Coloring the heatmap data for each row of the 1D heatmap:
            curr_data = deepcopy(curr_curves)
            if debug_print:
                print(f'np.shape(curr_data): {np.shape(curr_data)}, np.nanmax(curr_data): {np.nanmax(curr_data)}, np.nanmin(curr_data): {np.nanmin(curr_data)}') # np.shape(curr_data): (34, 62), np.nanmax(curr_data): 0.15320444716258447, np.nanmin(curr_data): 0.0

            _temp_curr_out_colors_heatmap_image = [] # used to accumulate the rows so they can be built into a color image in `out_colors_heatmap_image_matrix`

            _out_ui.text_items_dict[a_decoder_name] = {} # new dict to hold these items.
            for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
                # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
                text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
                text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
                curr_win.addItem(text)
                _out_ui.text_items_dict[a_decoder_name][aclu] = text # add the TextItem to the map

                # modulate heatmap color for this row (`curr_data[i, :]`):
                heatmap_base_color = pg.mkColor(a_color_vector)
                out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
                _temp_curr_out_colors_heatmap_image.append(out_colors_row)

            ## Build the colored heatmap:
            out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            if debug_print:
                print(f"np.shape(out_colors_heatmap_image_matrix): {np.shape(out_colors_heatmap_image_matrix)}") # (34, 62, 4) - (n_cells, n_pos_bins, n_channels_RGBA)

                # Ensure the data is in the correct range [0, 1]
            out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            if enable_cell_colored_heatmap_rows:
                curr_img.updateImage(out_colors_heatmap_image_matrix) # use the color image only if `enable_cell_colored_heatmap_rows==True`
            _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix

        # end `for i, (a_decoder_name, a_decoder)`

        ## Setup the Docks: 
        # decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        _out_ui.dock_widgets = {}
        _out_ui.dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))

        for i, (a_decoder_name, a_heatmap) in enumerate(_out_plots.pf1D_heatmaps.items()):
            if (a_decoder_name == 'short_RL'):
                short_LR_dock = _out_ui.root_dockAreaWindow.find_display_dock('short_LR')
                assert short_LR_dock is not None
                dock_add_locations['short_RL'] = ['bottom', short_LR_dock]
                print(f'using overriden dock location.')

            _out_ui.dock_widgets[a_decoder_name] = _out_ui.root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_heatmap[0], dockSize=(300,200), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=_out_ui.dock_configs[a_decoder_name])
        # end `for i, (a_decoder_name, a_heatmap)`

        return _out_data, _out_plots, _out_ui

    @classmethod
    def _subfn_update_directional_template_debugger_data(cls, included_any_context_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, decoders_dict: Dict):
        """ Just updates the existing UI, doesn't build new elements.

        ## Needs to update:
        _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name]
        curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name]
        _out_ui.dock_widgets[a_decoder_name] # maybe

        """
        _out_data = cls._subfn_rebuild_sort_idxs(decoders_dict, _out_data, use_incremental_sorting=use_incremental_sorting, included_any_context_neuron_ids=included_any_context_neuron_ids)
        # Unpack the updated _out_data:
        sort_helper_neuron_id_to_neuron_colors_dicts = _out_data.sort_helper_neuron_id_to_neuron_colors_dicts
        sorted_pf_tuning_curves = _out_data.sorted_pf_tuning_curves

        ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
        assert _out_plots.pf1D_heatmaps is not None
        for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            if use_incremental_sorting:
                title_str = f'{a_decoder_name}_pf1Ds [sort: {_out_data.ref_decoder_name}]'
            else:
                title_str = f'{a_decoder_name}_pf1Ds'

            curr_curves = sorted_pf_tuning_curves[i]
            # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
            curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img
            a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)

            # Coloring the heatmap data for each row of the 1D heatmap:
            curr_data = deepcopy(curr_curves)
            if debug_print:
                print(f'np.shape(curr_data): {np.shape(curr_data)}, np.nanmax(curr_data): {np.nanmax(curr_data)}, np.nanmin(curr_data): {np.nanmin(curr_data)}') # np.shape(curr_data): (34, 62), np.nanmax(curr_data): 0.15320444716258447, np.nanmin(curr_data): 0.0

            _temp_curr_out_colors_heatmap_image = [] # used to accumulate the rows so they can be built into a color image in `out_colors_heatmap_image_matrix`

            ## Remove all labels and re-add:
            for aclu, a_text_item in _out_ui.text_items_dict[a_decoder_name].items():
                curr_win.removeItem(a_text_item)
                a_text_item.deleteLater()
            _out_ui.text_items_dict[a_decoder_name] = {} # clear the dictionary

            for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):                        
                # text = _out_ui.text_items_dict[a_decoder_name][aclu] # pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
                # Create a new text item:
                text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0))
                text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
                curr_win.addItem(text)
                _out_ui.text_items_dict[a_decoder_name][aclu] = text # add the TextItem to the map

                # modulate heatmap color for this row (`curr_data[i, :]`):
                heatmap_base_color = pg.mkColor(a_color_vector)
                out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
                _temp_curr_out_colors_heatmap_image.append(out_colors_row)
            # end `for cell_i, (aclu, a_color_vector)`

            ## Build the colored heatmap:
            out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            # Ensure the data is in the correct range [0, 1]
            out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            if enable_cell_colored_heatmap_rows:
                curr_img.updateImage(out_colors_heatmap_image_matrix) # use the color image only if `enable_cell_colored_heatmap_rows==True`
            _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix
        # end `for i, (a_decoder_name, a_decoder)`

        return _out_data, _out_plots, _out_ui







    # ==================================================================================================================== #
    # Update Active Epoch Functions                                                                                        #
    # ==================================================================================================================== #

    def on_update_active_epoch(self, an_epoch_idx: int, an_epoch):
        """ captures: LR_on_update_active_epoch, RL_on_update_active_epoch """
        self.plots_data.on_update_active_epoch(an_epoch_idx, an_epoch=an_epoch)
        # Update window titles:
        an_epoch_string: str = f'idx: {an_epoch.Index}, t: {an_epoch.start:0.2f}, {an_epoch.stop:0.2f}, lbl: {str(an_epoch.label)}'
        
        # for i, (a_decoder_name, a_dock_widget) in enumerate(self.plots.dock_widgets.items()):
        for i, (a_decoder_name, a_win) in enumerate(self.plots.all_windows.items()):
            a_dock_widget = self.plots.dock_widgets[a_decoder_name]
            a_dock_widget[1].setTitle(f'{a_decoder_name} - epoch_IDX: {int(an_epoch_idx)} - epoch: {an_epoch_string}')
            a_win.setWindowTitle(f'{a_decoder_name} - epoch_IDX: {int(an_epoch_idx)} - epoch: {an_epoch_string}')
        
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
        
        # Update the widget if needed, but block changes
        # self.ui.ctrls_widget
        # self.ui.ctrls_widget.setValue(an_epoch_idx, False)

        self.on_update_active_epoch(an_epoch_idx, curr_epoch)

        for a_callback_name, a_callback_fn in self.on_idx_changed_callback_function_dict.items():
            a_callback_fn(self, an_epoch_idx)


    ## Update the colors for the individual rasters plotted by multiplot rasters or w/e
    def update_neurons_color_data(self, updated_neuron_render_configs_dict):
        """updates the colors for each neuron/cell given the updated_neuron_render_configs map

        #TODO 2023-11-29 20:13: - [ ] Not yet finished, does not seem to update colors.

        ## Update the colors for the individual rasters plotted by multiplot rasters or w/e
        from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended # for build_cell_display_configs
        # from pyphoplacecellanalysis.External
        ## Get colors from the existing `_out_directional_template_pfs_debugger` to use for the spikes:
        sorted_neuron_IDs_lists = _out_directional_template_pfs_debugger['data']['sorted_neuron_IDs_lists'].copy()
        sort_helper_neuron_id_to_neuron_colors_dicts = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], _out_directional_template_pfs_debugger['data']['sort_helper_neuron_id_to_neuron_colors_dicts'].copy()))

        # separate dict for each one:
        neuron_plotting_configs_dict_dict = {name:{aclu:SingleNeuronPlottingExtended(name=str(aclu), isVisible=False, color=color.name(pg.QtGui.QColor.HexRgb), spikesVisible=False) for aclu, color in v.items()} for name, v in sort_helper_neuron_id_to_neuron_colors_dicts.items()}
        updated_color_dict_dict = _out_rank_order_event_raster_debugger.update_neurons_color_data(updated_neuron_render_configs_dict=neuron_plotting_configs_dict_dict)


        Args:
            updated_neuron_render_configs (_type_): _description_

        Updates:

        """
        ## #TODO 2023-11-29 20:14: - [ ] Has to loop through all four rasters and do this for each of them
        raise NotImplementedError(f"simple raster plots are not supposed to be updated.")
        # LR_plots_data: RenderPlotsData = self.plots_data.LR_plots_data
        # RL_plots_data: RenderPlotsData = self.plots_data.RL_plots_data

        # ## Built flat lists across all four rasters so they aren't broken up into LR/RL when indexing:
        # _active_plot_identifiers = list(LR_plots_data.plots_data_dict.keys()) + list(RL_plots_data.plots_data_dict.keys()) # ['long_LR', 'short_LR', 'long_RL', 'short_RL']
        # _paired_plots_data = [LR_plots_data, LR_plots_data, RL_plots_data, RL_plots_data]
        # _paired_plots = [self.plots.LR_plots, self.plots.LR_plots, self.plots.RL_plots, self.plots.RL_plots]

        # assert len(updated_neuron_render_configs_dict) == len(_active_plot_identifiers)
        # emphasis_state = SpikeEmphasisState.Default
        # # self.plots.text_items_dict = {}

        # updated_color_dict_dict = {}

        # for _active_plot_identifier, plots_data, plots, updated_neuron_render_configs in zip(_active_plot_identifiers, _paired_plots_data, _paired_plots, updated_neuron_render_configs_dict.values()):
        #     # plots_data: RenderPlotsData = LR_plots_data

        #     # plots_data.plots_spikes_df_dict[_active_plot_identifier] = plots_data.plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(plots_data.plots_spikes_df_dict[_active_plot_identifier])
        #     # plots_data.plots_spikes_df_dict[_active_plot_identifier]

        #     ## Add the neuron_id labels to the rasters:
        #     raster_plot_manager = plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager
        #     a_plot_item = plots.ax[_active_plot_identifier]
        #     # self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map)

        #     # updated_color_dict = {cell_id:cell_config.color for cell_id, cell_config in updated_neuron_render_configs.items()} ## TODO: efficiency: pass only the colors that changed instead of all the colors:
        #     updated_color_dict = {}

        #     for cell_id, cell_config in updated_neuron_render_configs.items():
        #         # a_fragile_linear_neuron_IDX = raster_plot_manager.cell_id_to_fragile_linear_neuron_IDX_map[cell_id]
        #         a_fragile_linear_neuron_IDX = raster_plot_manager.unit_sort_manager.find_neuron_IDXs_from_cell_ids(cell_ids=[cell_id])[0]
        #         curr_qcolor = cell_config.qcolor

        #         # Determine if the color changed: Easiest to compare the hex value string:
        #         did_color_change = (raster_plot_manager.params.neuron_colors_hex[a_fragile_linear_neuron_IDX] != cell_config.color) # the hex color

        #         # Overwrite the old colors:
        #         raster_plot_manager.params.neuron_qcolors_map[a_fragile_linear_neuron_IDX] = curr_qcolor
        #         raster_plot_manager.params.neuron_qcolors[a_fragile_linear_neuron_IDX] = curr_qcolor
        #         # Overwrite the old secondary/derived colors:
        #         curr_rgbf_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        #         raster_plot_manager.params.neuron_colors[:, a_fragile_linear_neuron_IDX] = curr_rgbf_color[:]
        #         raster_plot_manager.params.neuron_colors_hex[a_fragile_linear_neuron_IDX] = cell_config.color # the hex color

        #         if did_color_change:
        #             # If the color changed, add it to the changed array:
        #             updated_color_dict[cell_id] = cell_config.color

        #     updated_color_dict_dict[_active_plot_identifier] = updated_color_dict

        #     #TODO 2023-11-29 20:39: - [ ] rebuild the spikes
        #     # raster_plot_manager.unit_sort_manager.update_spikes_df_visualization_columns(self.global_spikes_df)
        #     plots_data.all_spots_dict[_active_plot_identifier], plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(plots_data.plots_spikes_df_dict[_active_plot_identifier], plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

        #     #TODO 2023-11-29 20:40: - [ ] Rebuild the y-axis labels:
        #     # - make sure configs are updated so the colors used to rebuild are correct.
        #     # - todo - remove the old items?
        #     aclus_list = list(raster_plot_manager.params.config_items.keys())
        #     a_decoder_color_map = {aclu:raster_plot_manager.params.config_items[aclu].curr_state_pen_dict[emphasis_state].color() for aclu in aclus_list} # Recover color from pen:

        #     ## Get the y-values for the labels
        #     y_values = raster_plot_manager.unit_sort_manager.fragile_linear_neuron_IDX_to_spatial(raster_plot_manager.unit_sort_manager.find_neuron_IDXs_from_cell_ids(cell_ids=aclus_list))
        #     aclu_y_values_dict = dict(zip(aclus_list, y_values))
        #     self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map, aclu_y_values_dict) # ideally we could just add a color update to the update function

        # return updated_color_dict_dict


    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #

    def write_to_log(self, log_messages):
        """ logs text to the text widget at the bottom """
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


    # ==================================================================================================================== #
    # Cell y-axis Labels                                                                                                   #
    # ==================================================================================================================== #

    @classmethod
    def _build_neuron_y_labels(cls, a_plot_item, a_decoder_color_map, aclu_y_values_dict: Dict):
        """ 2023-11-29 - builds the y-axis text labels for a single one of the four raster plots.
        
        
        Uses:
            a_decoder_color_map, aclu_y_values_dict
        
        """
        [[x1, x2], [y1, y2]] = a_plot_item.getViewBox().viewRange() # get the x-axis range for initial position

        _out_text_items = {}
        for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
            # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
            # anchor=(1,0.5) should be its upper-middle point.
            text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0.5)) # , angle=15
            # text.setPos(x2, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
            text.setPos(x2, aclu_y_values_dict[aclu])
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


    @function_attributes(short_name=None, tags=['cell_y_labels'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-29 20:05', related_items=[])
    def _build_cell_y_labels(self):
        """Builds y-axis labels for each of the rasters and stores them in `self.plots.text_items_dict`
        """
        emphasis_state = SpikeEmphasisState.Default
        if self.plots.text_items_dict is not None:
            #TODO 2023-11-30 08:33: - [ ] Remove the old labels here if they exist.
            print(f'TODO 2023-11-30 08:33: - [ ] Remove the old labels here if they exist.')

        self.plots.text_items_dict = {}

        for _active_plot_identifier, new_sorted_raster in self.plots_data.seperate_new_sorted_rasters_dict.items():
            aclu_y_values_dict = {int(aclu):new_sorted_raster.neuron_y_pos[aclu] for aclu in new_sorted_raster.neuron_IDs}
            a_plot_item = self.plots.root_plots[_active_plot_identifier]
            # f"{int(aclu)}"
            a_decoder_color_map = deepcopy(new_sorted_raster.neuron_colors)
            self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map, aclu_y_values_dict)


    @function_attributes(short_name=None, tags=['cell_y_labels', 'update', 'active_epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-29 20:05', related_items=[])
    def update_cell_y_labels(self):
        """ called whenever the window scrolls or changes to reposition the y-axis labels created with self._build_cell_y_labels """
        # Adjust based on the whether the aclu is active or not:
        curr_active_aclus = self.get_epoch_active_aclus()
        emphasis_state = SpikeEmphasisState.Default

        for _active_plot_identifier, new_sorted_raster in self.plots_data.seperate_new_sorted_rasters_dict.items():
            aclu_y_values_dict = {int(aclu):new_sorted_raster.neuron_y_pos[aclu] for aclu in new_sorted_raster.neuron_IDs}
            a_plot_item = self.plots.root_plots[_active_plot_identifier]
            # f"{int(aclu)}"
            a_decoder_color_map = deepcopy(new_sorted_raster.neuron_colors)
            # get the labels to update:
            _out_text_items = self.plots.text_items_dict[a_plot_item]

            ## Perform the update:
            [[x1, x2], [y1, y2]] = a_plot_item.getViewBox().viewRange() # get the x-axis range
            # print(f'bounds: [[x1:{x1}, x2:{x2}], [y1:{y1}, y2:{y2}]]')
            for cell_i, (aclu, text) in enumerate(_out_text_items.items()):
                # print(f'aclu_y_values_dict[aclu={aclu}]: {aclu_y_values_dict[aclu]}')
                text.setPos(x2, aclu_y_values_dict[aclu])
                is_aclu_active: bool = aclu in curr_active_aclus
                if is_aclu_active:
                    text.setColor(pg.mkColor(a_decoder_color_map[aclu]))
                else:
                    inactive_color = pg.mkColor("#666666")
                    inactive_color.setAlpha(0.5)
                    text.setColor(inactive_color) # dark grey (inactive)
                # text.setVisible(is_aclu_active)



    # ==================================================================================================================== #
    # Core Component Building Classmethods                                                                                 #
    # ==================================================================================================================== #

    





