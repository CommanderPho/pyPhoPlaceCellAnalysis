from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
from nptyping import NDArray
import numpy as np
import pandas as pd
from pathlib import Path
import io
from contextlib import redirect_stdout # used by DocumentationFilePrinter to capture print output

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr

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
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster
from pyphoplacecellanalysis.GUI.Qt.Widgets.ScrollBarWithSpinBox.ScrollBarWithSpinBox import ScrollBarWithSpinBox
from pyphoplacecellanalysis.GUI.Qt.Widgets.LogViewerTextEdit import LogViewer
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import DirectionalRankOrderLikelihoods, RankOrderComputationsContainer, RankOrderResult ## Circular import?

from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel, create_tabbed_table_widget
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot, ExportFiletype

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget

__all__ = ['DockPlanningHelperWindow']


# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #

@define(slots=False, eq=False)
class DockPlanningHelperWindow:
    """ DockPlanningHelperWindow displays four rasters showing the same spikes but sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)
    

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.DockPlanningHelperWindow import DockPlanningHelperWindow
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockPlanningHelperWidget.DockPlanningHelperWidget import DockPlanningHelperWidget

    _out = DockPlanningHelperWindow.init_dock_area_builder(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)


    Updating Display Epoch:
        The `self.on_update_epoch_IDX(an_epoch_idx=0)` can be used to control which Epoch is displayed, and is synchronized across all four sorts.

    """
    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False, repr=False)
    ui: PhoUIContainer = field(init=False, repr=False)
    params: VisualizationParameters = field(init=False, repr=keys_only_repr)

    # Plot Convenience Accessors _________________________________________________________________________________________ #
    # @property
    # def seperate_new_sorted_rasters_dict(self) -> Dict[str, NewSimpleRaster]:
    #     return self.plots_data.seperate_new_sorted_rasters_dict


    # @property
    # def root_plots_dict(self) -> Dict[str, pg.PlotItem]:
    #     return {k:v['root_plot'] for k,v in self.plots.all_separate_plots.items()} # PlotItem 
    
    @classmethod
    def init_dock_area_builder(cls, n_dock_planning_helper_widgets:int=4, dock_add_locations=None, **param_kwargs):
        """
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

        _obj = cls()

        name:str = 'DockPlanningHelperWindow'

        ## 2023-11-30 - Newest Version using separate rasters:
        # _obj.plots_data, _obj.plots = cls._build_internal_raster_plots(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates, debug_print=True)
        # #TODO 2023-11-30 15:14: - [ ] Unpacking and putting in docks and such not yet finished. Update functions would need to be done separately.
        # rasters_display_outputs = _obj.plots.rasters_display_outputs
        # all_apps = {a_decoder_name:a_raster_setup_tuple.app for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_windows = {a_decoder_name:a_raster_setup_tuple.win for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_plots = {a_decoder_name:a_raster_setup_tuple.plots for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_plots_data = {a_decoder_name:a_raster_setup_tuple.plots_data for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}

        # main_plot_identifiers_list = list(all_windows.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']

        # ## Extract the data items:
        # all_separate_data_all_spots = {a_decoder_name:a_raster_setup_tuple.plots_data.all_spots for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_data_all_scatterplot_tooltips_kwargs = {a_decoder_name:a_raster_setup_tuple.plots_data.all_scatterplot_tooltips_kwargs for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_data_new_sorted_rasters = {a_decoder_name:a_raster_setup_tuple.plots_data.new_sorted_raster for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_data_spikes_dfs = {a_decoder_name:a_raster_setup_tuple.plots_data.spikes_df for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}

        # # Extract the plot/renderable items
        # all_separate_root_plots = {a_decoder_name:a_raster_setup_tuple.plots.root_plot for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_grids = {a_decoder_name:a_raster_setup_tuple.plots.grid for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_scatter_plots = {a_decoder_name:a_raster_setup_tuple.plots.scatter_plot for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
        # all_separate_debug_header_labels = {a_decoder_name:a_raster_setup_tuple.plots.debug_header_label for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}

        # Embedding in docks:
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho DockPlanningHelperWindow')
        # icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
        # if icon is not None:
        #     root_dockAreaWindow.setWindowIcon(icon)

        # _dock_helper_widgets = []
        _dock_helper_widgets_dict = {}
        for i in np.arange(n_dock_planning_helper_widgets):
            dock_id_str: str = f'dock[{i}]'
            a_dock_helper_widget = DockPlanningHelperWidget(dock_title=dock_id_str, dock_id=dock_id_str, defer_show=True)
            # _dock_helper_widgets.append(a_dock_helper_widget)
            _dock_helper_widgets_dict[dock_id_str] = a_dock_helper_widget

        _out_dock_widgets = {}
        dock_configs = {k:CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False) for k, v in _dock_helper_widgets_dict.items()}
        # dock_add_locations = (['left'], ['left'], ['right'], ['right'])
        # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))
        # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))

        if (dock_add_locations is None):
            # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))
            # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), ((lambda a_decoder_name: ['left']), (lambda a_decoder_name: ['bottom']), (lambda a_decoder_name: ['right']), (lambda a_decoder_name: ['bottom', root_dockAreaWindow.find_display_dock('short_LR')]))))
            dock_add_locations = {k:(lambda a_decoder_name: ['bottom']) for k, v in _dock_helper_widgets_dict.items()}
            
            # _dock_helper_widgets

        else:
            assert len(dock_add_locations) == len(dock_configs), f"len(dock_add_locations): {len(dock_add_locations)} != len(dock_configs): {len(dock_configs)}"

        for i, (a_decoder_name, a_widget) in enumerate(_dock_helper_widgets_dict.items()):
            active_dock_add_location_fn = dock_add_locations[a_decoder_name]
            if callable(active_dock_add_location_fn):
                # the value is a lambda-wrapped function that returns a list:
                active_dock_add_location = active_dock_add_location_fn(a_decoder_name)
            else:
                ## the value is just a regular list of string
                active_dock_add_location = active_dock_add_location_fn

            # if (a_decoder_name == 'short_RL'):
            #     short_LR_dock = root_dockAreaWindow.find_display_dock('short_LR')
            #     assert short_LR_dock is not None
            #     dock_add_locations['short_RL'] = ['bottom', short_LR_dock]
            #     print(f'using overriden dock location.')

            # _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name], autoOrientation=False)
            _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_widget, dockSize=(300,600), dockAddLocationOpts=active_dock_add_location, display_config=dock_configs[a_decoder_name], autoOrientation=False)



        # Build callback functions:


        ## Build the utility controls at the bottom:
        utility_controls_ui_dict, ctrls_dock_widgets_dict = _obj._build_utility_controls(root_dockAreaWindow)
        _out_dock_widgets = ctrls_dock_widgets_dict | _out_dock_widgets
        
        
        # Top Info Bar: ______________________________________________________________________________________________________ #
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
        _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow,
                                #   apps=all_apps, all_windows=all_windows, all_separate_plots=all_separate_plots,
                                #   root_plots=all_separate_root_plots, grids=all_separate_grids, scatter_plots=all_separate_scatter_plots, debug_header_labels=all_separate_debug_header_labels,
                                  dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
        _obj.plots_data = RenderPlotsData(name=name, 
                                        #   main_plot_identifiers_list=main_plot_identifiers_list,
                                        #    seperate_all_spots_dict=all_separate_data_all_spots, seperate_all_scatterplot_tooltips_kwargs_dict=all_separate_data_all_scatterplot_tooltips_kwargs,
                                            # seperate_new_sorted_rasters_dict=all_separate_data_new_sorted_rasters, seperate_spikes_dfs_dict=all_separate_data_spikes_dfs,
                                        #    on_update_active_epoch=on_update_active_epoch, on_update_active_scatterplot_kwargs=on_update_active_scatterplot_kwargs,
                                            # **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']},
                                            )
        # _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)
        _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, **utility_controls_ui_dict, **info_labels_widgets_dict, dock_configs=dock_configs, controlled_references=None)
        _obj.params = VisualizationParameters(name=name, use_plaintext_title=False, **param_kwargs)

        # ## Cleanup when done:
        # for a_decoder_name, a_root_plot in _obj.plots.root_plots.items():
        #     a_root_plot.setTitle(title=a_decoder_name)
        #     # a_root_plot.setTitle(title="")
        #     a_left_axis = a_root_plot.getAxis('left')# axisItem
        #     a_left_axis.setLabel(a_decoder_name)
        #     a_left_axis.setStyle(showValues=False)
        #     a_left_axis.setTicks([])
        #     # a_root_plot.hideAxis('bottom')
        #     # a_root_plot.hideAxis('bottom')
        #     a_root_plot.hideAxis('left')
        #     # a_root_plot.setYRange(-0.5, float(_obj.max_n_neurons))
            

        # for a_decoder_name, a_scatter_plot_item in _obj.plots.scatter_plots.items():
        #     a_scatter_plot_item.hideAxis('left')

        # # Hide the debugging labels
        # for a_decoder_name, a_label in _obj.plots.debug_header_labels.items():
        #     # a_label.setText('NEW')
        #     a_label.hide() # hide the labels unless we need them.

        _obj.register_internal_callbacks()

        try:
            ctrl_widgets_dict = _obj.ui
            ctrl_widgets_dict['models_dict']['combined_epoch_stats'] = SimplePandasModel(pd.DataFrame())
            # Create and associate view with model
            ctrl_widgets_dict['views_dict']['combined_epoch_stats'].setModel(ctrl_widgets_dict['models_dict']['combined_epoch_stats'])

        except AttributeError as e:
            # AttributeError: 'NoneType' object has no attribute 'ripple_combined_epoch_stats_df'
            print(f'WARNING: {e}')

        except Exception as e:
            raise e

        

        return _obj



    
    def _build_utility_controls(self, root_dockAreaWindow):
        """ Build the utility controls at the bottom """
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

        ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)

        ctrls_widget = ScrollBarWithSpinBox()
        ctrls_widget.setObjectName("ctrls_widget")
        ctrls_widget.update_range(0, 100)
        ctrls_widget.setValue(10)

        def valueChanged(new_val:int):
            print(f'ScrollBarWithSpinBox valueChanged(new_val: {new_val})')
            self.on_update_epoch_IDX(int(new_val))

        ctrls_widget_connection = ctrls_widget.sigValueChanged.connect(valueChanged)
        ctrl_layout = pg.LayoutWidget()
        ctrl_layout.addWidget(ctrls_widget, row=1, rowspan=1, col=1, colspan=2)
        ctrl_widgets_dict = dict(ctrls_widget=ctrls_widget, ctrls_widget_connection=ctrls_widget_connection)

        # Step 4: Create DataFrame and QTableView
        # df =  selected active_selected_spikes_df # pd.DataFrame(...)  # Replace with your DataFrame
        # model = PandasModel(df)
        # pandasDataFrameTableModel = SimplePandasModel(active_epochs_df.copy())

        # tableView = pg.QtWidgets.QTableView()
        # tableView.setModel(pandasDataFrameTableModel)
        # tableView.setObjectName("pandasTablePreview")
        # # tableView.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)

        # ctrl_widgets_dict['pandasDataFrameTableModel'] = pandasDataFrameTableModel
        # ctrl_widgets_dict['tableView'] = tableView

        # # Step 5: Add TableView to LayoutWidget
        # ctrl_layout.addWidget(tableView, row=2, rowspan=1, col=1, colspan=1)


        # Tabbled table widget:
        tab_widget, views_dict, models_dict = create_tabbed_table_widget(dataframes_dict={'epochs': pd.DataFrame(), 'spikes': pd.DataFrame(), 'combined_epoch_stats': pd.DataFrame()})
        ctrl_widgets_dict['tables_tab_widget'] = tab_widget
        ctrl_widgets_dict['views_dict'] = views_dict
        ctrl_widgets_dict['models_dict'] = models_dict

        # Add the tab widget to the layout
        ctrl_layout.addWidget(tab_widget, row=2, rowspan=1, col=1, colspan=1)
    
        logTextEdit = LogViewer() # QTextEdit subclass
        logTextEdit.setReadOnly(True)
        logTextEdit.setObjectName("logTextEdit")
        # logTextEdit.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)

        ctrl_layout.addWidget(logTextEdit, row=2, rowspan=1, col=2, colspan=1)

        # _out_dock_widgets['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,200), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)
        ctrls_dock_widgets_dict = {}
        ctrls_dock_widgets_dict['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,200), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)

        ui_dict = dict(ctrl_layout=ctrl_layout, **ctrl_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit)
        return ui_dict, ctrls_dock_widgets_dict




    def register_internal_callbacks(self):
        """ registers all internally-owned callback functions. """
        # self.on_idx_changed_callback_function_dict['update_plot_titles_with_stats'] = self.update_plot_titles_with_stats
        pass


    

    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #

    def write_to_log(self, log_messages):
        """ logs text to the text widget at the bottom """
        self.ui.logTextEdit.write_to_log(log_messages)
        # self.ui.logTextEdit.append(log_messages)
        # # Automatically scroll to the bottom
        # self.ui.logTextEdit.verticalScrollBar().setValue(
        #     self.ui.logTextEdit.verticalScrollBar().maximum()
        # )


    def setWindowTitle(self, title: str):
        """ updates the window's title """
        self.ui.root_dockAreaWindow.setWindowTitle(title)


    def set_top_info_bar_visibility(self, is_visible=False):
        """Hides/Shows the top info bar dock """
        LongShortColumnsInfo_dock_layout, LongShortColumnsInfo_dock_Dock = self.plots.dock_widgets['LongShortColumnsInfo_dock']
        # LongShortColumnsInfo_dock_layout.hide() # No use
        # _out_ripple_rasters.ui.long_short_info_layout.hide() # No use
        LongShortColumnsInfo_dock_Dock.setVisible(is_visible)

    def set_bottom_controls_visibility(self, is_visible=False):
        """Hides/Shows the top info bar dock """
        found_dock_layout, found_dock_Dock = self.plots.dock_widgets['bottom_controls']
        # LongShortColumnsInfo_dock_layout.hide() # No use
        # _out_ripple_rasters.ui.long_short_info_layout.hide() # No use
        found_dock_Dock.setVisible(is_visible)







    # ==================================================================================================================== #
    # Core Component Building Classmethods                                                                                 #
    # ==================================================================================================================== #

    # @classmethod
    # def _build_internal_raster_plots(cls, spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, debug_print=True, defer_show=True, **kwargs):
    #     """ 2023-11-30 **DO EM ALL SEPERATELY**

    #     _out_data, _out_plots = _build_internal_raster_plots(spikes_df, active_epochs_df, track_templates, debug_print=True)

    #     History:
    #         Called `_post_modern_debug_plot_directional_template_rasters`


    #     Uses:
    #         paired_separately_sort_neurons

    #     """
    #     from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph # used in `plot_kourosh_activity_style_figure`
    #     from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
    #     from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
    #     from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color
    #     from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons
    #     from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster

    #     ## spikes_df: get the spikes to plot
    #     # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
    #     # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
    #     # track_templates.shared_LR_aclus_only_neuron_IDs

    #     figure_name: str = kwargs.pop('figure_name', 'rasters debugger')

    #     decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

    #     neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
    #     # _unit_qcolors_map, unit_colors_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    #     unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
    #     # `unit_colors_map` is main colors output

    #     included_neuron_ids = np.array(list(unit_colors_map.keys())) # one list for all decoders
    #     n_neurons = len(included_neuron_ids)

    #     print(f'included_neuron_ids: {included_neuron_ids}, n_neurons: {n_neurons}')

    #     # included_neuron_ids = np.sort(np.union1d(track_templates.shared_RL_aclus_only_neuron_IDs, track_templates.shared_LR_aclus_only_neuron_IDs))
    #     # n_neurons = len(included_neuron_ids)

    #     # Get only the spikes for the shared_aclus:
    #     spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(included_neuron_ids)
    #     # spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
    #     spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='new_epoch_IDX', epoch_label_column_name='label') # , override_time_variable_name='t_seconds'
    #     # spikes_df = spikes_df[spikes_df['ripple_id'] != -1]
    #     spikes_df = spikes_df[(spikes_df['new_epoch_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
    #     spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards


    #     # CORRECT: Even: RL, Odd: LR
    #     RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy() # (69, )
    #     LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy() # (64, )

    #     included_any_context_neuron_ids_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (LR_neuron_ids, RL_neuron_ids, LR_neuron_ids, RL_neuron_ids)))

    #     # INDIVIDUAL SORTING for each raster:
    #     # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
    #     sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in decoders_dict.items()} # tuning_curve CoM location
    #     sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids_dict, sortable_values_list_dict=sortable_values_list_dict)

    #     _out_data = RenderPlotsData(name=figure_name, spikes_df=spikes_df, unit_sort_orders_dict=None, included_any_context_neuron_ids_dict=included_any_context_neuron_ids_dict,
    #                                 sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None,
    #                                 unsorted_original_neuron_IDs_lists=deepcopy(unsorted_original_neuron_IDs_lists), unsorted_neuron_IDs_lists=deepcopy(unsorted_neuron_IDs_lists), unsorted_sortable_values_lists=deepcopy(unsorted_sortable_values_lists), unsorted_unit_colors_map=deepcopy(unsorted_unit_colors_map))
    #     _out_plots = RenderPlots(name=figure_name, rasters_display_outputs=None)

    #     # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
    #     _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
    #     _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
    #     _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts
    #     _out_data.unit_sort_orders_dict = {} # empty array

    #     ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
    #     _out_plots.rasters = {}
    #     _out_plots.rasters_display_outputs = {}
    #     for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
    #         title_str = f'{a_decoder_name}'

    #         # an_included_unsorted_neuron_ids = deepcopy(included_any_context_neuron_ids_dict[a_decoder_name])
    #         an_included_unsorted_neuron_ids = deepcopy(unsorted_neuron_IDs_lists[i])
    #         a_sorted_neuron_ids = deepcopy(sorted_neuron_IDs_lists[i])

    #         unit_sort_order, desired_sort_arr = find_desired_sort_indicies(an_included_unsorted_neuron_ids, a_sorted_neuron_ids)
    #         print(f'unit_sort_order: {unit_sort_order}\ndesired_sort_arr: {desired_sort_arr}')
    #         _out_data.unit_sort_orders_dict[a_decoder_name] = deepcopy(unit_sort_order)

    #         # Get only the spikes for the shared_aclus:
    #         a_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(an_included_unsorted_neuron_ids)
    #         a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

    #         _out_plots.rasters_display_outputs[a_decoder_name] = new_plot_raster_plot(a_spikes_df, an_included_unsorted_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=deepcopy(unsorted_unit_colors_map), scatter_plot_kwargs=None, scatter_app_name=f'pho_directional_laps_rasters_{title_str}', defer_show=defer_show, active_context=None)
    #         # an_app, a_win, a_plots, a_plots_data, an_on_update_active_epoch, an_on_update_active_scatterplot_kwargs = _out_plots.rasters_display_outputs[a_decoder_name]


    #     return _out_data, _out_plots


    # ==================================================================================================================== #
    # Registering Output Signals                                                                                           #
    # ==================================================================================================================== #
    def a_debug_callback_fn(self, an_idx: int, an_epoch=None):
        out = io.StringIO()

        curr_epoch_label = self.lookup_label_from_index(an_idx)

        with redirect_stdout(out):
            print(f'=====================================================================================\n\tactive_epoch_IDX: {an_idx} :::', end='\t')

            ## Registered printing functions are called here, and anything they print is written to the textarea at the bottom of the widget.


            print(f'______________________________________________________________________________________________________________________\n')

        self.write_to_log(str(out.getvalue()))


    def update_plot_titles_with_stats(self, an_idx: int):
        """ Updates the titles of each of the four rasters with the appropriate spearman rho value.
        captures: rank_order_results_debug_values || active_epochs_df, formatted_title_strings_dict


        Usages:
            self.params.enable_show_spearman
            self.params.enable_show_pearson
            self.params.enable_show_Z_values

            self.active_epoch_result_df


        """
        is_laps: bool = self.params.is_laps
        use_plaintext_title: bool = self.params.use_plaintext_title
        if not use_plaintext_title:
            formatted_title_strings_dict = DisplayColorsEnum.get_pyqtgraph_formatted_title_dict()

        # curr_epoch_label = a_plotter.lookup_label_from_index(an_idx)
        # ripple_combined_epoch_stats_df = a_plotter.rank_order_results.ripple_combined_epoch_stats_df
        # curr_new_results_df = ripple_combined_epoch_stats_df[ripple_combined_epoch_stats_df.index == curr_epoch_label]

        curr_new_results_df = self.active_epoch_result_df
        for a_decoder_name, a_root_plot in self.plots.root_plots.items():
            # a_real_value = rank_order_results_debug_values[a_decoder_name][0][an_idx]
            a_std_column_name: str = self.decoder_name_to_column_name_prefix_map[a_decoder_name]

            if (curr_new_results_df is not None):
                all_column_names = curr_new_results_df.filter(regex=f'^{a_std_column_name}').columns.tolist()
                active_column_names = []
                # print(active_column_names)
                if self.params.enable_show_spearman:
                    active_column_names = [col for col in all_column_names if col.endswith("_spearman")]
                    if self.params.enable_show_Z_values:
                        active_column_names += [col for col in all_column_names if col.endswith("_spearman_Z")]


                if self.params.enable_show_pearson:
                    active_column_names += [col for col in all_column_names if col.endswith("_pearson")]
                    if self.params.enable_show_Z_values:
                        active_column_names += [col for col in all_column_names if col.endswith("_pearson_Z")]


                active_column_values = curr_new_results_df[active_column_names]
                active_values_dict = active_column_values.iloc[0].to_dict() # {'LR_Long_spearman': -0.34965034965034975, 'LR_Long_pearson': -0.5736588716389961, 'LR_Long_spearman_Z': -0.865774983083525, 'LR_Long_pearson_Z': -1.4243571733839517}
                active_raw_col_val_dict = {k.replace(f'{a_std_column_name}_', ''):v for k,v in active_values_dict.items()} # remove the "LR_Long" prefix so it's just the variable names
            else:
                ## No RankOrderResults
                print(f'WARN: No RankOrderResults')
                active_raw_col_val_dict = {}
                
            active_formatted_col_val_list = [':'.join([generate_html_string(str(k), color='grey', bold=False), generate_html_string(f'{v:0.3f}', color='white', bold=True)]) for k,v in active_raw_col_val_dict.items()]
            final_values_string: str = '; '.join(active_formatted_col_val_list)

            if use_plaintext_title:
                title_str = generate_html_string(f"{a_std_column_name}: {final_values_string}")
            else:
                # Color formatted title:
                a_formatted_title_string_prefix: str = formatted_title_strings_dict[a_std_column_name]
                title_str = generate_html_string(f"{a_formatted_title_string_prefix}: {final_values_string}")

            a_root_plot.setTitle(title=title_str)


   


## Adding callbacks to `DockPlanningHelperWindow` when the slider changes:


# # ==================================================================================================================== #
# # CALLBACKS:                                                                                                           #
# # ==================================================================================================================== #

