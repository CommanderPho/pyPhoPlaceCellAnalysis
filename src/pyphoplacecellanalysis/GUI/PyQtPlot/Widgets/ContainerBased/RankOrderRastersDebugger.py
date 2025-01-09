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

from pyphocorehelpers.print_helpers import generate_html_string

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import PhoDockAreaContainingWindow


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


@metadata_attributes(short_name=None, tags=['gui', 'window', 'figure', '🖼️', '🎨'], input_requires=[], output_provides=[], uses=['_debug_plot_directional_template_rasters', 'add_selected_spikes_df_points_to_scatter_plot'], used_by=['rank_order_debugger'], creation_date='2023-11-17 19:59', related_items=[])
@define(slots=False)
class RankOrderRastersDebugger:
    """ RankOrderRastersDebugger displays four rasters showing the same spikes but sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)


    # Examples ___________________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger

    _out = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, active_epochs_dfe, track_templates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

    # Example 1 __________________________________________________________________________________________________________ #
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import RankOrderRastersDebugger

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
    global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
    global_laps_epochs_df = global_laps.to_dataframe()

    RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict = None
    LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict = None
    _out_laps_rasters: RankOrderRastersDebugger = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, global_laps_epochs_df, track_templates, rank_order_results, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)
    _out_laps_rasters

    
    # Example 2 __________________________________________________________________________________________________________ #    
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
    _out_ripple_rasters: RankOrderRastersDebugger = RankOrderRastersDebugger.init_rank_order_debugger(global_spikes_df, deepcopy(filtered_ripple_simple_pf_pearson_merged_df),
                                                                                                    track_templates, None,
                                                                                                        None, None,
                                                                                                        dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right']))),
                                                                                                        )
    _out_ripple_rasters.set_top_info_bar_visibility(False)

    # Example 3 __________________________________________________________________________________________________________ #
    _out = curr_active_pipeline.display('_display_directional_template_debugger')

    
    
    # Use/Updating _______________________________________________________________________________________________________ #

    Updating Display Epoch:
        The `self.on_update_epoch_IDX(an_epoch_idx=0)` can be used to control which Epoch is displayed, and is synchronized across all four sorts.

    Updating Continuous Displayed Time: 
        `_out_ripple_rasters.programmatically_update_epoch_IDX_from_epoch_start_time(193.65)`

    
    """
    global_spikes_df: pd.DataFrame = field(repr=False)
    active_epochs_df: pd.DataFrame = field(repr=False)
    track_templates: TrackTemplates = field(repr=False)
    rank_order_results: Optional[RankOrderComputationsContainer] = field(repr=False)
    LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)
    RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)


    plots: RenderPlots = field(init=False)
    plots_data: RenderPlotsData = field(init=False, repr=False)
    ui: PhoUIContainer = field(init=False, repr=False)
    params: VisualizationParameters = field(init=False, repr=keys_only_repr)

    active_epoch_IDX: int = field(default=0, repr=True)
    # active_epoch_time_bin_IDX: Optional[int] = field(default=None, repr=True)

    on_idx_changed_callback_function_dict: Dict[str, Callable] = field(default=Factory(dict), repr=False)

    @property
    def decoder_name_to_column_name_prefix_map(self) -> Dict[str, str]:
        return dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], ['LR_Long', 'RL_Long', 'LR_Short', 'RL_Short']))


    @property
    def n_epochs(self) -> int:
        return np.shape(self.active_epochs_df)[0]

    def lookup_label_from_index(self, an_idx: int) -> int:
        """ Looks of the proper epoch "label", as in the value in the 'label' column of active_epochs_df, from a linear index such as that provided by the slider control.

        curr_epoch_label = lookup_label_from_index(a_plotter, an_idx)
        print(f'curr_epoch_label: {curr_epoch_label} :::') ## end line

        """
        curr_epoch_label = self.active_epochs_df['label'].iloc[an_idx] # gets the correct epoch label for the linear IDX
        curr_redundant_label_lookup_label = self.active_epochs_df.label.to_numpy()[an_idx]
        # print(f'curr_redundant_label_lookup_label: {curr_redundant_label_lookup_label} :::') ## end line
        assert str(curr_redundant_label_lookup_label) == str(curr_epoch_label), f"curr_epoch_label: {str(curr_epoch_label)} != str(curr_redundant_label_lookup_label): {str(curr_redundant_label_lookup_label)}"
        return curr_epoch_label


    def find_nearest_time_index(self, target_time: float) -> Optional[int]:
        """ finds the index of the nearest time from the active epochs
        """
        from neuropy.utils.indexing_helpers import find_nearest_time
        df = self.active_epochs_df
        df, closest_index, closest_time, matched_time_difference = find_nearest_time(df=df, target_time=target_time, time_column_name='start', max_allowed_deviation=0.01, debug_print=False)
        # df.iloc[closest_index]
        return closest_index
    

    @property
    def active_epoch_label(self):
        """ returns the epoch 'label' value corresponding to the currently selected `self.active_epoch_IDX`. """
        return self.lookup_label_from_index(an_idx=self.active_epoch_IDX)
    
    @property
    def active_epoch_df(self) -> pd.DataFrame:
        """ returns a the single row of the epoch_df corresponding to `self.active_epoch_IDX`. """
        curr_epoch_label = self.lookup_label_from_index(self.active_epoch_IDX)
        return self.active_epochs_df[self.active_epochs_df.label == curr_epoch_label]

    @property
    def active_epoch_tuple(self) -> tuple:
        """ returns a namedtuple describing the single epoch corresponding to `self.active_epoch_IDX`. """
        # a_df_idx = self.active_epochs_df.index.to_numpy()[self.active_epoch_IDX]
        # curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == a_df_idx)]
        curr_epoch = list(self.active_epoch_df.itertuples(name='EpochTuple'))[0]
        return curr_epoch


    # Data Convenience Accessors _________________________________________________________________________________________ #
    @property
    def combined_epoch_stats_df(self) -> Optional[pd.DataFrame]:
        """ returns combined_epoch_stats_df. """
        if self.rank_order_results is None:
            return None
        is_laps: bool = self.params.is_laps
        if is_laps:
            return self.rank_order_results.laps_combined_epoch_stats_df
        else:
            return self.rank_order_results.ripple_combined_epoch_stats_df
        
    @property
    def active_epoch_result_df(self) -> Optional[pd.DataFrame]:
        """ returns a the combined_epoch_stats_df describing the single epoch corresponding to `self.active_epoch_IDX`. """
        if self.combined_epoch_stats_df is None:
            return None
        # curr_epoch_label = self.lookup_label_from_index(self.active_epoch_IDX)
        # return self.combined_epoch_stats_df[self.combined_epoch_stats_df.label == curr_epoch_label]
        assert np.shape(self.combined_epoch_stats_df)[0] == np.shape(self.active_epochs_df)[0], f"np.shape(self.combined_epoch_stats_df)[0]: {np.shape(self.combined_epoch_stats_df)[0]} != np.shape(self.active_epochs_df)[0]: {np.shape(self.active_epochs_df)[0]}"
        return self.combined_epoch_stats_df.iloc[[self.active_epoch_IDX]] # must pass a LIST of indicies to .iloc[...] so it returns a DataFrame instead of a pd.Series
        

    def get_active_epoch_spikes_df(self) -> pd.DataFrame:
        active_epoch_tuple = self.active_epoch_tuple
        active_epoch_spikes_df: pd.DataFrame = deepcopy(self.global_spikes_df.spikes.time_sliced(active_epoch_tuple.start, active_epoch_tuple.stop))
        return active_epoch_spikes_df

    def get_epoch_active_aclus(self) -> NDArray:
        """ returns a list of aclus active (having at least one spike) in the current epoch (based on `self.active_epoch`) """
        active_epoch_spikes_df: pd.DataFrame = self.get_active_epoch_spikes_df()
        active_epoch_unique_active_aclus = np.unique(active_epoch_spikes_df['aclu'].to_numpy())
        return active_epoch_unique_active_aclus
    
    @property
    def max_n_neurons(self) -> int:
        return np.max([len(v) for v in self.plots_data.unsorted_original_neuron_IDs_lists])


    # Plot Convenience Accessors _________________________________________________________________________________________ #
    @property
    def seperate_new_sorted_rasters_dict(self) -> Dict[str, NewSimpleRaster]:
        return self.plots_data.seperate_new_sorted_rasters_dict


    @property
    def root_plots_dict(self) -> Dict[str, pg.PlotItem]:
        return {k:v['root_plot'] for k,v in self.plots.all_separate_plots.items()} # PlotItem 
    
    
    @property
    def root_dockAreaWindow(self) -> "PhoDockAreaContainingWindow":
        return self.ui.root_dockAreaWindow
    
    @property
    def attached_directional_template_pfs_debugger(self): #-> Optional[TemplateDebugger]:
        """The attached_directional_template_pfs_debugger property."""
        return self.ui.controlled_references.get('directional_template_pfs_debugger', {}).get('obj', None)
 

    @classmethod
    def init_from_rank_order_results(cls, rank_order_results: RankOrderComputationsContainer):
        """
            directional_laps_results = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
            rank_order_results: RankOrderComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['RankOrder']
            minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
            included_qclu_values: float = rank_order_results.included_qclu_values
            print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
            print(f'included_qclu_values: {included_qclu_values}')


        active_epochs_df = deepcopy(rank_order_results.LR_ripple.epochs_df)


        """
        raise NotImplementedError
        # active_epochs_df = deepcopy(rank_order_results.LR_ripple.epochs_df)
        # # active_epochs_df = deepcopy(ripple_result_tuple.active_epochs) # Better?
        # return cls.init_rank_order_debugger(active_epochs_df)


    @classmethod
    def init_rank_order_debugger(cls, global_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, rank_order_results: RankOrderComputationsContainer, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame], LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame], dock_add_locations=None, **param_kwargs):
        """
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()

        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

        _obj = cls(global_spikes_df=global_spikes_df, active_epochs_df=active_epochs_df.copy(), track_templates=track_templates, rank_order_results=rank_order_results,
             RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict=RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict=LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict)

        name:str = 'RankOrderRastersDebugger'

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
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Debug Plot Directional Template Rasters')
        icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
        if icon is not None:
            root_dockAreaWindow.setWindowIcon(icon)

        # decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
        _out_dock_widgets = {}
        dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
                        CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
        # dock_add_locations = (['left'], ['left'], ['right'], ['right'])
        # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))
        # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))

        if (dock_add_locations is None):
            # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))
            dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), ((lambda a_decoder_name: ['left']), (lambda a_decoder_name: ['bottom']), (lambda a_decoder_name: ['right']), (lambda a_decoder_name: ['bottom', root_dockAreaWindow.find_display_dock('short_LR')]))))

        else:
            assert len(dock_add_locations) == len(dock_configs), f"len(dock_add_locations): {len(dock_add_locations)} != len(dock_configs): {len(dock_configs)}"

        for i, (a_decoder_name, a_win) in enumerate(all_windows.items()):
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
            _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(300,600), dockAddLocationOpts=active_dock_add_location, display_config=dock_configs[a_decoder_name], autoOrientation=False)




        # Build callback functions:
        def on_update_active_scatterplot_kwargs(override_scatter_plot_kwargs):
            """ captures: main_plot_identifiers_list, plots, plots_data """
            for _active_plot_identifier in main_plot_identifiers_list:
                # for _active_plot_identifier, a_scatter_plot in plots.scatter_plots.items():
                # new_ax = plots.ax[_active_plot_identifier]
                a_scatter_plot = all_separate_scatter_plots[_active_plot_identifier]
                plots_data = all_separate_plots_data[_active_plot_identifier]
                a_scatter_plot.setData(plots_data.seperate_all_spots_dict[_active_plot_identifier], **(plots_data.seperate_all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {}), **override_scatter_plot_kwargs)

        def on_update_active_epoch(an_epoch_idx, an_epoch):
            """ captures: main_plot_identifiers_list, all_separate_root_plots """
            for _active_plot_identifier in main_plot_identifiers_list:
                new_ax = all_separate_root_plots[_active_plot_identifier]
                print(f'an_epoch: {an_epoch}')
                new_ax.setXRange(an_epoch.start, an_epoch.stop)
                new_ax.setAutoPan(False)
                # new_ax.getAxis('left').setLabel(f'[{an_epoch.label}]')

                # a_scatter_plot = plots.scatter_plots[_active_plot_identifier]


        ## Build the utility controls at the bottom:
        utility_controls_ui_dict, ctrls_dock_widgets_dict = _obj._build_utility_controls(root_dockAreaWindow, active_epochs_df=active_epochs_df, global_spikes_df=global_spikes_df)
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

        root_dockAreaWindow.resize(1440, 360) # set the (width, height)

        ## Build final .plots and .plots_data:
        _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, apps=all_apps, all_windows=all_windows, all_separate_plots=all_separate_plots,
                                  root_plots=all_separate_root_plots, grids=all_separate_grids, scatter_plots=all_separate_scatter_plots, debug_header_labels=all_separate_debug_header_labels,
                                  dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
        _obj.plots_data = RenderPlotsData(name=name, main_plot_identifiers_list=main_plot_identifiers_list,
                                           seperate_all_spots_dict=all_separate_data_all_spots, seperate_all_scatterplot_tooltips_kwargs_dict=all_separate_data_all_scatterplot_tooltips_kwargs, seperate_new_sorted_rasters_dict=all_separate_data_new_sorted_rasters, seperate_spikes_dfs_dict=all_separate_data_spikes_dfs,
                                           on_update_active_epoch=on_update_active_epoch, on_update_active_scatterplot_kwargs=on_update_active_scatterplot_kwargs, **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']})
        # _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)
        _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, **utility_controls_ui_dict, **info_labels_widgets_dict, dock_configs=dock_configs, controlled_references={})
        _obj.params = VisualizationParameters(name=name, is_laps=False, enable_show_spearman=True, enable_show_pearson=False, enable_show_Z_values=True, use_plaintext_title=False, **param_kwargs)

        ## Add Selected Spikes:
        # try:
        #     ## rank_order_results.LR_ripple.selected_spikes_df mode:
        #     if isinstance(LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, pd.DataFrame) and isinstance(RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, pd.DataFrame):
        #         # already a selected_spikes_df! Use it raw!
        #         _obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map = deepcopy(RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        #         _obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map = deepcopy(LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
        #     else:
        #         ## Build the selected spikes df:
        #         (_obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map), (_obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map) = _obj.build_selected_spikes_df(_obj.track_templates, _obj.active_epochs_df,
        #                                                                                                                                                                                                             _obj.RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict,                                                                                                                                                                                                                _obj.LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict)
        #     ## Add the spikes
        #     _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'long_LR')
        #     _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'short_LR')
        #     _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'long_RL')
        #     _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'short_RL')

        # except (IndexError, KeyError, ValueError, TypeError):
        #     print(f'WARN: the selected spikes did not work properly, so none will be shown.')
        #     pass

        cls.try_build_selected_spikes(_obj)

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
            a_root_plot.setYRange(-0.5, float(_obj.max_n_neurons))
            

        # for a_decoder_name, a_scatter_plot_item in _obj.plots.scatter_plots.items():
        #     a_scatter_plot_item.hideAxis('left')

        # Hide the debugging labels
        for a_decoder_name, a_label in _obj.plots.debug_header_labels.items():
            # a_label.setText('NEW')
            a_label.hide() # hide the labels unless we need them.

        _obj.register_internal_callbacks()

        try:
            ctrl_widgets_dict = _obj.ui
            ctrl_widgets_dict['models_dict']['combined_epoch_stats'] = SimplePandasModel(_obj.combined_epoch_stats_df.copy())
            # Create and associate view with model
            ctrl_widgets_dict['views_dict']['combined_epoch_stats'].setModel(ctrl_widgets_dict['models_dict']['combined_epoch_stats'])

        except AttributeError as e:
            # AttributeError: 'NoneType' object has no attribute 'ripple_combined_epoch_stats_df'
            print(f'WARNING: {e}')

        except Exception as e:
            raise e

        

        return _obj



    
    def _build_utility_controls(self, root_dockAreaWindow, active_epochs_df, global_spikes_df):
        """ Build the utility controls at the bottom """
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors

        ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)

        ctrls_widget = ScrollBarWithSpinBox()
        ctrls_widget.setObjectName("ctrls_widget")
        ctrls_widget.update_range(0, (self.n_epochs-1))
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
        tab_widget, views_dict, models_dict = create_tabbed_table_widget(dataframes_dict={'epochs': active_epochs_df.copy(),
                                                                                                        'spikes': global_spikes_df.copy(), 
                                                                                                        'combined_epoch_stats': pd.DataFrame()})
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
    # Programmatically Update Active Epoch Functions                                                                       #
    # ==================================================================================================================== #
    def programmatically_update_epoch_IDX(self, an_epoch_idx: int):
        """ programmatically performs an update to the epoch_IDX 
        """
        assert an_epoch_idx is not None
        assert an_epoch_idx >= 0 # minimum valid epoch
        assert (an_epoch_idx < self.n_epochs) # maximum valid epoch
        _a_ScrollBarWithSpinBox = self.ui.ctrls_widget # ScrollBarWithSpinBox 
        _a_ScrollBarWithSpinBox.setValue(an_epoch_idx)

    
    def programmatically_update_epoch_IDX_from_epoch_start_time(self, target_time: float):
        """ finds and selects the epoch starting nearest to the target_time
        """
        found_IDX = self.find_nearest_time_index(target_time)
        if found_IDX is not None:
            print(f'found_IDX: {found_IDX}')
            self.programmatically_update_epoch_IDX(found_IDX)
        else:
            raise ValueError(f'could not find epoch near target_time: {target_time}.')


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
        # a_df_idx = self.active_epochs_df.index.to_numpy()[an_epoch_idx]
        # print(f'a_df_idx: {a_df_idx}')
        # # curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == (a_df_idx+1))] # this +1 here makes zero sense
        # curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == a_df_idx)] # this +1 here makes zero sense
        
        # an_epoch_label = self.lookup_label_from_index(an_idx=an_epoch_idx)
        
        # curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.lap_id == (an_epoch_idx+1))]
        # curr_epoch = list(curr_epoch_df.itertuples())[0]

        # Update the widget if needed, but block changes
        # self.ui.ctrls_widget
        # self.ui.ctrls_widget.setValue(an_epoch_idx, False)

        self.on_update_active_epoch(an_epoch_idx, an_epoch=self.active_epoch_tuple)

        ## Update scrollbar:
        self.update_plot_titles_with_stats(an_epoch_idx)
        ## Update Table:
        # self.scroll_df_table_view(an_epoch_idx)
        

        ## perform callbacks:
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
        self.ui.logTextEdit.write_to_log(log_messages)
        # self.ui.logTextEdit.append(log_messages)
        # # Automatically scroll to the bottom
        # self.ui.logTextEdit.verticalScrollBar().setValue(
        #     self.ui.logTextEdit.verticalScrollBar().maximum()
        # )


    def setWindowTitle(self, title: str):
        """ updates the window's title """
        self.ui.root_dockAreaWindow.setWindowTitle(title)


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



    def save_figure(self, export_path: Path, export_topmost_scene_element:bool=False, **kwargs):
        """ Exports all four rasters to a specified file path
        
        _out_rank_order_event_raster_debugger.save_figure(export_path=export_path)
        
        export_topmost_scene_element: False - export the PlotItem
            True: export the parent of the PlotItem (includes the grid, a few other things).
        """
        save_paths = []
        # root_plots_dict = {k:v['root_plot'] for k,v in _out_rank_order_event_raster_debugger.plots.all_separate_plots.items()} # PlotItem 

        root_plots_dict = self.root_plots_dict
        # root_plots_dict = self.root_plots_dict
        root_plots_dict['long_LR'].setYRange(-0.5, float(self.max_n_neurons))

        for a_decoder, a_plot in root_plots_dict.items():
            a_plot.setYRange(-0.5, float(self.max_n_neurons))
            if export_topmost_scene_element:
                a_plot = a_plot.parentItem()
            self.get_epoch_active_aclus()
            out_path = export_path.joinpath(f'{a_decoder}_raster.png').resolve()
            export_pyqtgraph_plot(a_plot, savepath=out_path, background=pg.mkColor(0, 0, 0, 0), **kwargs)
            save_paths.append(out_path)
    
        return save_paths




    @function_attributes(short_name=None, tags=['figure', 'export', 'slider', 'debug'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-21 19:49', related_items=[])
    def export_figure_all_slider_values(self, export_path: Union[str,Path], **kwargs):
        """ sweeps the rank_order_event_raster_debugger through its various slider values, exporting all four of its plots as images for each value. 

        Usage:
            export_path = Path(r'~/Desktop/2023-12-19 Exports').resolve()
            all_save_paths = _out_rank_order_event_raster_debugger.export_figure_all_slider_values(export_path=export_path)


        """
        all_save_paths = {}

        for i in np.arange(0, self.n_epochs, 5):
            self.ui.ctrls_widget.setValue(i) ## Adjust the slider, using its callbacks as well to update the displayed epoch.
            
            # _out_rank_order_event_raster_debugger.on_update_epoch_IDX(an_epoch_idx=i)
            active_epoch_label = self.active_epoch_label

            save_paths = []

            for a_decoder, a_plot in self.root_plots_dict.items():
                curr_filename_prefix = f'Epoch{active_epoch_label}_{a_decoder}'
                # a_plot.setYRange(-0.5, float(self.max_n_neurons))
                out_path = export_path.joinpath(f'{curr_filename_prefix}_raster.png').resolve()
                export_pyqtgraph_plot(a_plot, savepath=out_path, background=pg.mkColor(0, 0, 0, 0), **kwargs)
                save_paths.append(out_path)

            all_save_paths[active_epoch_label] = save_paths
        
        return all_save_paths
    

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
            text.setPos(x2, aclu_y_values_dict[aclu]) # the x2 part indicates that we want it aligned to the end of the window (the right-hand-side)

            # # Function to handle click event
            # def on_text_item_clicked(evt):
            #     # Your custom logic here
            #     print("TextItem clicked!")
            #     print(f'\tevt: {evt}')


            # def on_name_clicked(self, q_textitem):
            #     print(q_textitem.text())

            # Connect the click signal to the custom function
            # text.sigMouseClicked.connect(on_text_item_clicked)
            # Connect the click signal to the custom function
            # text.mouseClickEvent = on_text_item_clicked
            # text.mousePressEvent = lambda e: on_name_clicked(text)

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
            for a_plot_item, a_text_item in self.plots.text_items_dict.items():
                a_text_item.scene().removeItem(a_text_item) # example from https://stackoverflow.com/questions/46791395/pyqtgraph-get-text-of-node-and-change-color-on-mouseclick

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
                    active_color = pg.mkColor(a_decoder_color_map[aclu])
                    active_color.setAlphaF(0.95)
                    text.setColor(active_color)
                    # text.setOpacity(0.95/255.)
                else:
                    inactive_color = pg.mkColor("#666666")
                    inactive_color.setAlphaF(0.5)
                    text.setColor(inactive_color) # dark grey (inactive)
                    # text.setOpacity(0.5/255.)
                # text.setVisible(is_aclu_active)


    def scroll_df_table_view(self, row_index: Optional[int]):
        """ scrolls the table to the selected index. 
        
        """
        if row_index is not None:
            model_idx = self.ui.tableView.model().index(row_index, 0)
            # Scroll to the specified row
            self.ui.tableView.scrollTo(model_idx) # not sure this is the right one

            # Select the entire row
            self.ui.tableView.selectRow(model_idx.row())
            # self.ui.tableView.selectRow(row_index)


    # ==================================================================================================================== #
    # Core Component Building Classmethods                                                                                 #
    # ==================================================================================================================== #

    @classmethod
    def _build_internal_raster_plots(cls, spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, debug_print=True, defer_show=True, **kwargs):
        """ 2023-11-30 **DO EM ALL SEPERATELY**

        _out_data, _out_plots = _build_internal_raster_plots(spikes_df, active_epochs_df, track_templates, debug_print=True)

        History:
            Called `_post_modern_debug_plot_directional_template_rasters`


        Uses:
            paired_separately_sort_neurons

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster

        ## spikes_df: get the spikes to plot
        # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
        # track_templates.shared_LR_aclus_only_neuron_IDs

        figure_name: str = kwargs.pop('figure_name', 'rasters debugger')

        decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

        neuron_IDs_lists = [deepcopy(a_decoder.neuron_IDs) for a_decoder in decoders_dict.values()] # [A, B, C, D, ...]
        # _unit_qcolors_map, unit_colors_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
        unit_colors_map, _unit_colors_ndarray_map = build_shared_sorted_neuron_color_maps(neuron_IDs_lists)
        # `unit_colors_map` is main colors output

        included_neuron_ids = np.array(list(unit_colors_map.keys())) # one list for all decoders
        n_neurons = len(included_neuron_ids)

        print(f'included_neuron_ids: {included_neuron_ids}, n_neurons: {n_neurons}')

        # included_neuron_ids = np.sort(np.union1d(track_templates.shared_RL_aclus_only_neuron_IDs, track_templates.shared_LR_aclus_only_neuron_IDs))
        # n_neurons = len(included_neuron_ids)

        # Get only the spikes for the shared_aclus:
        spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(included_neuron_ids)
        # spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
        spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='new_epoch_IDX', epoch_label_column_name='label') # , override_time_variable_name='t_seconds'
        # spikes_df = spikes_df[spikes_df['ripple_id'] != -1]
        spikes_df = spikes_df[(spikes_df['new_epoch_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
        spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards


        # CORRECT: Even: RL, Odd: LR
        RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy() # (69, )
        LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy() # (64, )

        included_any_context_neuron_ids_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (LR_neuron_ids, RL_neuron_ids, LR_neuron_ids, RL_neuron_ids)))

        # INDIVIDUAL SORTING for each raster:
        # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
        sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in decoders_dict.items()} # tuning_curve CoM location
        sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids_dict, sortable_values_list_dict=sortable_values_list_dict)

        _out_data = RenderPlotsData(name=figure_name, spikes_df=spikes_df, unit_sort_orders_dict=None, included_any_context_neuron_ids_dict=included_any_context_neuron_ids_dict,
                                    sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None,
                                    unsorted_original_neuron_IDs_lists=deepcopy(unsorted_original_neuron_IDs_lists), unsorted_neuron_IDs_lists=deepcopy(unsorted_neuron_IDs_lists), unsorted_sortable_values_lists=deepcopy(unsorted_sortable_values_lists), unsorted_unit_colors_map=deepcopy(unsorted_unit_colors_map))
        _out_plots = RenderPlots(name=figure_name, rasters_display_outputs=None)

        # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
        _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
        _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
        _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts
        _out_data.unit_sort_orders_dict = {} # empty array

        ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
        _out_plots.rasters = {}
        _out_plots.rasters_display_outputs = {}
        for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            title_str = f'{a_decoder_name}'

            # an_included_unsorted_neuron_ids = deepcopy(included_any_context_neuron_ids_dict[a_decoder_name])
            an_included_unsorted_neuron_ids = deepcopy(unsorted_neuron_IDs_lists[i])
            a_sorted_neuron_ids = deepcopy(sorted_neuron_IDs_lists[i])

            unit_sort_order, desired_sort_arr = find_desired_sort_indicies(an_included_unsorted_neuron_ids, a_sorted_neuron_ids)
            print(f'unit_sort_order: {unit_sort_order}\ndesired_sort_arr: {desired_sort_arr}')
            _out_data.unit_sort_orders_dict[a_decoder_name] = deepcopy(unit_sort_order)

            # Get only the spikes for the shared_aclus:
            a_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(an_included_unsorted_neuron_ids)
            a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

            _out_plots.rasters_display_outputs[a_decoder_name] = new_plot_raster_plot(a_spikes_df, an_included_unsorted_neuron_ids, unit_sort_order=unit_sort_order, unit_colors_list=deepcopy(unsorted_unit_colors_map), scatter_plot_kwargs=None, scatter_app_name=f'pho_directional_laps_rasters_{title_str}', defer_show=defer_show, active_context=None)
            # an_app, a_win, a_plots, a_plots_data, an_on_update_active_epoch, an_on_update_active_scatterplot_kwargs = _out_plots.rasters_display_outputs[a_decoder_name]


        return _out_data, _out_plots




    # ==================================================================================================================== #
    # Selected Spikes                                                                                                      #
    # ==================================================================================================================== #

    @classmethod
    def try_build_selected_spikes(cls, _obj) -> bool:
        ## Add Selected Spikes:
        try:
            ## rank_order_results.LR_ripple.selected_spikes_df mode:
            if isinstance(_obj.LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict, pd.DataFrame) and isinstance(_obj.RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict, pd.DataFrame):
                # already a selected_spikes_df! Use it raw!
                _obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map = deepcopy(_obj.RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
                _obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map = deepcopy(_obj.LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict).reset_index(drop=True).spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
            else:
                ## Build the selected spikes df:
                (_obj.plots_data.RL_selected_spike_df, _obj.plots_data.RL_neuron_id_to_new_IDX_map), (_obj.plots_data.LR_selected_spike_df, _obj.plots_data.LR_neuron_id_to_new_IDX_map) = _obj.build_selected_spikes_df(_obj.track_templates, _obj.active_epochs_df,
                                                                                                                                                                                                                    _obj.RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict,                                                                                                                                                                                                                _obj.LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict)
            ## Add the spikes
            # _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'long_LR')
            # _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.LR_plots_data, plots=_obj.plots.LR_plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'short_LR')
            # _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'long_RL')
            # _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data.RL_plots_data, plots=_obj.plots.RL_plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'short_RL')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data, plots=_obj.plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'long_LR')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data, plots=_obj.plots, selected_spikes_df=deepcopy(_obj.plots_data.LR_selected_spike_df), _active_plot_identifier = 'short_LR')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data, plots=_obj.plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'long_RL')
            _obj.add_selected_spikes_df_points_to_scatter_plot(plots_data=_obj.plots_data, plots=_obj.plots, selected_spikes_df=deepcopy(_obj.plots_data.RL_selected_spike_df), _active_plot_identifier = 'short_RL')

            return True
        except (IndexError, KeyError, ValueError, TypeError):
            print(f'WARN: the selected spikes did not work properly, so none will be shown.')
            return False



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
    def DEP_add_selected_spikes_df_points_to_scatter_plot(cls, plots_data, plots, selected_spikes_df, _active_plot_identifier = 'long_RL'):
        """ OLD Called after above `build_selected_spikes_df`

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
        # plots_data_dict = plots_data.plots_data_dict # derived
        new_sorted_raster = plots_data['seperate_new_sorted_rasters_dict'][_active_plot_identifier]
        selected_spikes_df = new_sorted_raster.update_spikes_df_visualization_columns(spikes_df=selected_spikes_df)

        ## Build the spots for the raster plot:
        plots_data.all_selected_spots_dict[_active_plot_identifier], plots_data.all_selected_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = new_sorted_raster.build_spikes_all_spots_from_df(spikes_df=selected_spikes_df, should_return_data_tooltips_kwargs=True, generate_debug_tuples=False) # Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(selected_spikes_df, plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)
        # Override the pen for the selected spots, the default renders them looking exactly like normal spikes which is no good:
        for a_point in plots_data.all_selected_spots_dict[_active_plot_identifier]:
            a_point['pen'] = pg.mkPen('#ffffff8e', width=3.5)
            a_point['brush'] = pg.mkBrush('#ffffff2f')


        ## Add the median spikes to the plots:
        a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
        a_scatter_plot.addPoints(plots_data.all_selected_spots_dict[_active_plot_identifier], dataSet='selected_spikes')




        # _out_plots.rasters_display_outputs[a_decoder_name]


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

    @function_attributes(short_name=None, tags=['attached', 'templates', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-10 08:00', related_items=[])
    def plot_attached_directional_templates_pf_debugger(self, curr_active_pipeline):
        """ builds a _display_directional_template_debugger, attaches it to the provided rank_order_event_raster_debugger so it's updated on its callback, and then returns what it created. 
        
        """
        curr_active_pipeline.reload_default_display_functions()
        # epoch_active_aclus = np.array([9,  26,  31,  39,  40,  43,  47,  52,  53,  54,  60,  61,  65,  68,  72,  75,  77,  78,  81,  82,  84,  85,  90,  92,  93,  98, 102]) # some test indicies
        epoch_active_aclus = None
        # epoch_active_aclus = deepcopy(self.get_epoch_active_aclus())
        _out_directional_template_pfs_debugger = curr_active_pipeline.display('_display_directional_template_debugger', included_any_context_neuron_ids=epoch_active_aclus, figure_name=f'<Controlled by RankOrderRastersDebugger>', debug_draw=True, debug_print=True)

        # Hold reference to the controlled plotter:
        if self.ui.controlled_references is None:
            self.ui.controlled_references = {}
        if 'directional_template_pfs_debugger' not in self.ui.controlled_references:
            self.ui.controlled_references['directional_template_pfs_debugger'] = _out_directional_template_pfs_debugger

        def debug_update_paired_directional_template_pfs_debugger(a_plotter, an_idx: int):
            """ captures nothing """
            epoch_active_aclus = deepcopy(a_plotter.get_epoch_active_aclus())
            # update the displayed cells:
            controlled_directional_template_pfs_debugger = a_plotter.ui.controlled_references['directional_template_pfs_debugger']
            directional_template_pfs_debugger_on_update_callback = controlled_directional_template_pfs_debugger.get('ui').on_update_callback
            directional_template_pfs_debugger_on_update_callback(epoch_active_aclus)
        
        self.on_idx_changed_callback_function_dict['debug_update_paired_directional_template_pfs_debugger'] = debug_update_paired_directional_template_pfs_debugger
        
        return _out_directional_template_pfs_debugger, debug_update_paired_directional_template_pfs_debugger


    @function_attributes(short_name=None, tags=['indicator-regions', 'highlight', 'selection'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-10 08:51', related_items=['clear_highlighting_indicator_regions'])
    def add_highlighting_indicator_regions(self, t_start: float, t_stop: float, identifier: str):
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions
    
        epoch_indicator_regions = self.plots.get('epoch_indicator_regions', None)
        if epoch_indicator_regions is None:
            self.plots.epoch_indicator_regions = {} # Initialize

        extant_indicator_regions_dict = self.plots.epoch_indicator_regions.get(identifier, {})
        for a_root_plot, a_rect_item in extant_indicator_regions_dict.items():
            ## remove the item
            a_root_plot.removeItem(a_rect_item)
            a_rect_item.deleteLater()
            # a_root_plot.remove(a_rect_item)
        # print(f'removed all extant')
        self.plots.epoch_indicator_regions[identifier] = {} # clear when done
        for a_decoder_name, a_root_plot in self.plots.root_plots.items():
            self.plots.epoch_indicator_regions[identifier][a_root_plot], epoch_region_label = build_pyqtgraph_epoch_indicator_regions(a_root_plot, t_start=t_start, t_stop=t_stop, epoch_label="", **dict(pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00')), movable=False) # no label
    

	# def select_epoch_time


    @function_attributes(short_name=None, tags=['indicator-regions', 'highlight', 'selection'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-10 08:51', related_items=['add_highlighting_indicator_regions'])
    def clear_highlighting_indicator_regions(self):
        existing_indicator_regions_dict = self.plots.get('epoch_indicator_regions', None)
        if existing_indicator_regions_dict is not None:         
            for identifier, extant_indicator_regions_dict in existing_indicator_regions_dict.items():
                for a_root_plot, a_rect_item in extant_indicator_regions_dict.items():
                    ## remove the item
                    # a_root_plot.remove(a_rect_item)
                    a_root_plot.removeItem(a_rect_item)
                    a_rect_item.deleteLater()
                    # print(f'removing {identifier} for {a_root_plot}...')
            
            self.plots.epoch_indicator_regions = {} # clear all
            # print(f'done clearing')



## Adding callbacks to `RankOrderRastersDebugger` when the slider changes:


# # ==================================================================================================================== #
# # CALLBACKS:                                                                                                           #
# # ==================================================================================================================== #







""" 

Captures: active_epochs_df, rank_order_results, ripple_result_tuple, rank_order_results_debug_values
spikes_df
"""


# ripple_combined_epoch_stats_df = deepcopy(curr_active_pipeline.global_computation_results.computed_data['RankOrder'].ripple_combined_epoch_stats_df)

# def get_epoch_label_row(an_idx: int):
# 	a_label = active_epochs_df.label.to_numpy()[an_idx]
# 	a_row = active_epochs_df[active_epochs_df.label == a_label]
# 	return list(a_row.itertuples())[0], a_label

# def debug_plot_epoch_label_row(a_plotter, an_idx: int):
# 	a_row, a_label = get_epoch_label_row(an_idx=an_idx)
# 	print(f'debug_plot_epoch_label_row(an_idx: {an_idx}):\n\t{print_formatted_active_epochs_df(a_label, a_row)}')


def debug_print_dict_row(a_dict) -> str:
    return '\t'.join([': '.join([str(k), f'{float(v):0.3f}']) for k, v in a_dict.items()])

def print_formatted_active_epochs_df(curr_epoch_label, corresponding_epoch_values_tuple):
    """ 2023-12-13 - NEW - called to print the dataframe at a given index
        ['start', 'stop', 'label', 'duration']
        ['LR_Long_spearman', 'RL_Long_spearman', 'LR_Short_spearman', 'RL_Short_spearman']
        ['LR_Long_Old_Spearman', 'RL_Long_Old_Spearman', 'LR_Short_Old_Spearman', 'RL_Short_Old_Spearman']

        ['LR_Long_Z', 'RL_Long_Z', 'LR_Short_Z', 'RL_Short_Z']
        ['LR_Long_pearson', 'RL_Long_pearson', 'LR_Short_pearson', 'RL_Short_pearson']

        ['LR_Long_ActuallyIncludedAclus', 'LR_Long_rel_num_cells', 'RL_Long_ActuallyIncludedAclus', 'RL_Long_rel_num_cells']
        ['LR_Long_ActuallyIncludedAclus', 'LR_Long_rel_num_cells', 'RL_Long_ActuallyIncludedAclus', 'RL_Long_rel_num_cells']
    """
    print(f'curr_epoch_label: {curr_epoch_label}') # ,\ncorresponding_epoch_values_tuple: {corresponding_epoch_values_tuple}\n'
    # Extracting the fields
    out_str = '\n'.join((
        debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['Index', 'label', 'start', 'stop', 'duration']}),
        # debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['LR_Long_spearman', 'RL_Long_spearman', 'LR_Short_spearman', 'RL_Short_spearman']}),
        debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['LR_Long_spearman', 'LR_Short_spearman', 'RL_Long_spearman', 'RL_Short_spearman']}),
        # debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['LR_Long_Old_Spearman', 'LR_Short_Old_Spearman', 'RL_Long_Old_Spearman', 'RL_Short_Old_Spearman']}),
        # debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['LR_Long_Z', 'LR_Short_Z', 'RL_Long_Z', 'RL_Short_Z']}),
        # debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['LR_Long_Old_Spearman', 'RL_Long_Old_Spearman', 'LR_Short_Old_Spearman', 'RL_Short_Old_Spearman']}),
        # debug_print_dict_row({field: getattr(corresponding_epoch_values_tuple, field) for field in ['LR_Long_Z', 'RL_Long_Z', 'LR_Short_Z', 'RL_Short_Z']}),
        # f'LR_relative_num_cells: {LR_relative_num_cells[an_idx]},\t\t RL_relative_num_cells: {RL_relative_num_cells[an_idx]}',
        # f'LR_template_epoch_actually_included_aclus: {corresponding_epoch_values_tuple.a[an_idx]},\nRL_template_epoch_actually_included_aclus: {RL_template_epoch_actually_included_aclus[an_idx]}',
        f'LR_Long rel: num_cells: {corresponding_epoch_values_tuple.LR_Long_rel_num_cells}, \t aclus: {corresponding_epoch_values_tuple.LR_Long_ActuallyIncludedAclus}',
        f'RL_Long rel: num_cells: {corresponding_epoch_values_tuple.RL_Long_rel_num_cells}, \t aclus: {corresponding_epoch_values_tuple.RL_Long_ActuallyIncludedAclus}',
    ))
    print(out_str)



def debug_update_plot_titles(a_plotter, an_idx: int):
    """ Updates the titles of each of the four rasters with the appropriate spearman rho value.
    captures: rank_order_results_debug_values || active_epochs_df, formatted_title_strings_dict


    Usages:
        a_plotter.params.enable_show_spearman
        a_plotter.params.enable_show_pearson
        a_plotter.params.enable_show_Z_values

    """
    is_laps: bool = a_plotter.params.is_laps
    use_plaintext_title: bool = a_plotter.params.use_plaintext_title
    if not use_plaintext_title:
        formatted_title_strings_dict = DisplayColorsEnum.get_pyqtgraph_formatted_title_dict()

    # curr_epoch_label = a_plotter.lookup_label_from_index(an_idx)
    # ripple_combined_epoch_stats_df = a_plotter.rank_order_results.ripple_combined_epoch_stats_df
    # curr_new_results_df = ripple_combined_epoch_stats_df[ripple_combined_epoch_stats_df.index == curr_epoch_label]

    curr_new_results_df = a_plotter.active_epoch_result_df
    for a_decoder_name, a_root_plot in a_plotter.plots.root_plots.items():
        # a_real_value = rank_order_results_debug_values[a_decoder_name][0][an_idx]
        a_std_column_name: str = a_plotter.decoder_name_to_column_name_prefix_map[a_decoder_name]

        all_column_names = curr_new_results_df.filter(regex=f'^{a_std_column_name}').columns.tolist()
        active_column_names = []
        # print(active_column_names)
        if a_plotter.params.enable_show_spearman:
            active_column_names = [col for col in all_column_names if col.endswith("_spearman")]
            if a_plotter.params.enable_show_Z_values:
                active_column_names += [col for col in all_column_names if col.endswith("_spearman_Z")]


        if a_plotter.params.enable_show_pearson:
            active_column_names += [col for col in all_column_names if col.endswith("_pearson")]
            if a_plotter.params.enable_show_Z_values:
                active_column_names += [col for col in all_column_names if col.endswith("_pearson_Z")]


        active_column_values = curr_new_results_df[active_column_names]
        active_values_dict = active_column_values.iloc[0].to_dict() # {'LR_Long_spearman': -0.34965034965034975, 'LR_Long_pearson': -0.5736588716389961, 'LR_Long_spearman_Z': -0.865774983083525, 'LR_Long_pearson_Z': -1.4243571733839517}
        active_raw_col_val_dict = {k.replace(f'{a_std_column_name}_', ''):v for k,v in active_values_dict.items()} # remove the "LR_Long" prefix so it's just the variable names

        active_formatted_col_val_list = [':'.join([generate_html_string(str(k), color='grey', bold=False), generate_html_string(f'{v:0.3f}', color='white', bold=True)]) for k,v in active_raw_col_val_dict.items()]
        final_values_string: str = '; '.join(active_formatted_col_val_list)

        if use_plaintext_title:
            title_str = generate_html_string(f"{a_std_column_name}: {final_values_string}")
        else:
            # Color formatted title:
            a_formatted_title_string_prefix: str = formatted_title_strings_dict[a_std_column_name]
            title_str = generate_html_string(f"{a_formatted_title_string_prefix}: {final_values_string}")

        a_root_plot.setTitle(title=title_str)


def debug_update_long_short_info_titles(a_plotter, an_idx: int):
    """ Updates the titles of each of the four rasters with the appropriate spearman rho value.
    captures: ripple_result_tuple,
    """

    has_long_short_info_labels = (hasattr(a_plotter.ui, 'long_info_label') and hasattr(a_plotter.ui, 'short_info_label'))
    if has_long_short_info_labels:
        ripple_result_tuple = a_plotter.rank_order_results.ripple_most_likely_result_tuple
        directional_likelihoods_tuple: DirectionalRankOrderLikelihoods = ripple_result_tuple.directional_likelihoods_tuple
        directional_likelihoods_tuple.long_best_direction_indices
        directional_likelihoods_tuple.short_best_direction_indices
        directional_likelihoods_tuple.long_relative_direction_likelihoods

        long_info_string_list = []

        if directional_likelihoods_tuple.long_relative_direction_likelihoods is not None:
            _long_LR_likelihood = directional_likelihoods_tuple.long_relative_direction_likelihoods[an_idx, 0]
            _long_RL_likelihood = directional_likelihoods_tuple.long_relative_direction_likelihoods[an_idx, 1]
            _long_LR_likelihood_str = generate_html_string(f'{float(_long_LR_likelihood):0.3f}', color='black', bold=True)
            _long_RL_likelihood_str = generate_html_string(f'{float(_long_RL_likelihood):0.3f}', color='black', bold=True)
            long_info_string_list.append(generate_html_string(f'LONG LR likelihood: {_long_LR_likelihood_str}'))
            long_info_string_list.append(generate_html_string(f'LONG RL likelihood: {_long_RL_likelihood_str}'))

        if directional_likelihoods_tuple.long_best_direction_indices is not None:
            _long_best_dir_IDX = directional_likelihoods_tuple.long_best_direction_indices[an_idx]
            _long_best_dir_IDX_str = generate_html_string(f'{int(_long_best_dir_IDX)}', color='black', bold=True)
            long_info_string_list.append(generate_html_string(f'LONG best dir IDX: {_long_best_dir_IDX_str}'))

        # original:
        ripple_result_tuple.rank_order_z_score_df
        ripple_result_tuple.active_epochs

        long_best_dir_z_score_value_str = generate_html_string(f'{float(ripple_result_tuple.long_best_dir_z_score_values[an_idx]):0.3f}', color='black', bold=True)
        long_info_string_list.append(generate_html_string(f'LONG best_dir_z_score_value: {long_best_dir_z_score_value_str}'))

        short_best_dir_z_score_value_str = generate_html_string(f'{float(ripple_result_tuple.short_best_dir_z_score_values[an_idx]):0.3f}', color='black', bold=True)
        # j_str = generate_html_string('j', color='red', bold=True)
        a_plotter.ui.long_info_label.setText('\n'.join(long_info_string_list))
        a_plotter.ui.short_info_label.setText(generate_html_string(f'SHORT best_dir_z_score_value: {short_best_dir_z_score_value_str}'))
    else:
        print(f'WARN: debug_update_long_short_info_titles(...) but plotter does not have the `a_plotter.ui.long_info_label`')

def a_debug_callback_fn(a_plotter, an_idx: int, an_epoch=None):
    global epoch_active_aclus, _out_directional_template_pfs_debugger
    out = io.StringIO()
    # _out.on_update_epoch_IDX(an_idx)

    curr_epoch_label = a_plotter.lookup_label_from_index(an_idx)
    rank_order_results = a_plotter.rank_order_results

    # DO LR only:
    LR_ranked_aclus_stats = rank_order_results.LR_ripple.ranked_aclus_stats_dict[curr_epoch_label]
    LR_extra_info = rank_order_results.LR_ripple.extra_info_dict[curr_epoch_label]

    RL_ranked_aclus_stats = rank_order_results.LR_ripple.ranked_aclus_stats_dict[curr_epoch_label]
    RL_extra_info = rank_order_results.RL_ripple.extra_info_dict[curr_epoch_label]


    # ripple_combined_epoch_stats_df = deepcopy(curr_active_pipeline.global_computation_results.computed_data['RankOrder'].ripple_combined_epoch_stats_df)
    curr_new_results_df = a_plotter.rank_order_results.ripple_combined_epoch_stats_df[a_plotter.rank_order_results.ripple_combined_epoch_stats_df.index == curr_epoch_label]

    # RL_extra_info_dict = rank_order_results.RL_ripple.extra_info_dict[curr_epoch_label]

    with redirect_stdout(out):
        print(f'=====================================================================================\n\tactive_epoch_IDX: {an_idx} :::', end='\t')
        # print(f'')
        # print(f'LR_long_relative_real_values: {LR_long_relative_real_values[an_idx]:.4f},\t\t LR_long_relative_real_p_values: {LR_long_relative_real_p_values[an_idx]:.4f}')
        # print(f'LR_short_relative_real_values: {LR_short_relative_real_values[an_idx]:.4f},\t\t LR_short_relative_real_p_values: {LR_short_relative_real_p_values[an_idx]:.4f}')
        # print(f'LR_ripple.long_z_score: {rank_order_results.LR_ripple.long_z_score[an_idx]:.4f},\t\t LR_ripple.short_z_score: {rank_order_results.LR_ripple.short_z_score[an_idx]:.4f}')
        # print(f'RL_long_relative_real_values: {RL_long_relative_real_values[an_idx]:.4f},\t\t RL_long_relative_real_p_values: {RL_long_relative_real_p_values[an_idx]:.4f}')
        # print(f'RL_short_relative_real_values: {RL_short_relative_real_values[an_idx]:.4f},\t\t RL_short_relative_real_p_values: {RL_short_relative_real_p_values[an_idx]:.4f}')
        # print(f'RL_ripple.long_z_score: {rank_order_results.RL_ripple.long_z_score[an_idx]:.4f},\t\t RL_ripple.short_z_score: {rank_order_results.RL_ripple.short_z_score[an_idx]:.4f}')
        # print(f'LR_relative_num_cells: {LR_relative_num_cells[an_idx]},\t\t RL_relative_num_cells: {RL_relative_num_cells[an_idx]}')
        # print(f'LR_template_epoch_actually_included_aclus: {LR_template_epoch_actually_included_aclus[an_idx]},\nRL_template_epoch_actually_included_aclus: {RL_template_epoch_actually_included_aclus[an_idx]}')


        ## Simple key-based indexing:
        print(f'')

        lr_long_columns = curr_new_results_df.filter(regex='^LR_Long').columns.tolist()
        # print(lr_long_columns)
        lr_long_columns_stripped = [col.replace('LR_Long_', '') for col in lr_long_columns]
        # print(lr_long_columns_stripped)

        print(f'curr_new_results_df:\n{curr_new_results_df[lr_long_columns]}')
        # print(render_dataframe(curr_new_results_df))
        # print(f'LR_long_relative_real_values: {LR_long_relative_real_values[an_idx]:.4f},\t\t LR_long_relative_real_p_values: {LR_long_relative_real_p_values[an_idx]:.4f}')
        # print(f'LR_short_relative_real_values: {LR_short_relative_real_values[an_idx]:.4f},\t\t LR_short_relative_real_p_values: {LR_short_relative_real_p_values[an_idx]:.4f}')
        # print(f'LR_ripple.long_z_score: {rank_order_results.LR_ripple.long_z_score[an_idx]:.4f},\t\t LR_ripple.short_z_score: {rank_order_results.LR_ripple.short_z_score[an_idx]:.4f}')
        # print(f'RL_long_relative_real_values: {RL_long_relative_real_values[an_idx]:.4f},\t\t RL_long_relative_real_p_values: {RL_long_relative_real_p_values[an_idx]:.4f}')
        # print(f'RL_short_relative_real_values: {RL_short_relative_real_values[an_idx]:.4f},\t\t RL_short_relative_real_p_values: {RL_short_relative_real_p_values[an_idx]:.4f}')
        # print(f'RL_ripple.long_z_score: {rank_order_results.RL_ripple.long_z_score[an_idx]:.4f},\t\t RL_ripple.short_z_score: {rank_order_results.RL_ripple.short_z_score[an_idx]:.4f}')

        # print(f'LR_relative_num_cells: {LR_extra_info[1]},\t\t RL_relative_num_cells: {RL_extra_info[1]}')
        # print(f'LR_template_epoch_actually_included_aclus: {len(LR_extra_info[1])},\nRL_template_epoch_actually_included_aclus: {len(RL_extra_info[1])}')


        # 'LR_Long_spearman', 'LR_Long_spearman_Z'
        ## Extract Z-score variables:


        # """ captures: active_epochs_df, """
        # # corresponding_epoch_value = active_epochs_df.to_records()[active_epochs_df['label'] == curr_epoch_label]
        # corresponding_epoch_values_tuple = list(a_plotter.active_epochs_df[a_plotter.active_epochs_df['label'] == curr_epoch_label].itertuples(name='EpochRow'))[0] # EpochRow(Index=398, start=1714.3077712343074, stop=1714.6516814583447, label=409, duration=0.3439102240372449, end=1714.6516814583447, LR_Long_spearman=0.5269555552418339, RL_Long_spearman=-0.050011483546781, LR_Short_spearman=0.4606822127204283, RL_Short_spearman=-0.2035100246885261, LR_Long_pearson=0.4836286811698692, RL_Long_pearson=-0.003226348316225221, LR_Short_pearson=0.47186014640172635, RL_Short_pearson=-0.13444915290053647)
        # print_formatted_active_epochs_df(curr_epoch_label, corresponding_epoch_values_tuple)
        # LR_Long_pearson[active_epochs_df['label'] == curr_epoch_label].values
        # print(f'corresponding_epoch_value: {corresponding_epoch_value}')
        print(f'done.\n')
        # ripple_result_tuple.rank_order_z_score_df.label.to_numpy()[an_idx]
        # except BaseException as e:
        # 	print(f'ERR\n\ta_debug_callback_fn(...): e: {e}')
        # 	raise e
        # 	# pass


        # a_row, a_label = get_epoch_label_row(an_idx=an_idx)
        # print(f'!!!! debug_plot_epoch_label_row(an_idx: {an_idx}):\n\ta_label: {a_label}\n\ta_row: {a_row}')

        # display(LR_template_epoch_actually_included_aclus[an_idx])
        # display(RL_template_epoch_actually_included_aclus[an_idx])
        # epoch_active_aclus = np.sort(union_of_arrays(LR_template_epoch_actually_included_aclus[an_idx], RL_template_epoch_actually_included_aclus[an_idx]))
        # print(f'epoch_active_aclus: {epoch_active_aclus}')
        print(f'______________________________________________________________________________________________________________________\n')

    a_plotter.write_to_log(str(out.getvalue()))
    # # update the displayed cells:
    # directional_template_pfs_debugger_on_update_callback = _out_directional_template_pfs_debugger.get('ui').on_update_callback
    # directional_template_pfs_debugger_on_update_callback(epoch_active_aclus)
    # _out_directional_template_pfs_debugger = curr_active_pipeline.display(DirectionalPlacefieldGlobalDisplayFunctions._display_directional_template_debugger, included_any_context_neuron_ids=epoch_active_aclus)

    # rank_order_results.ripple_most_likely_result_tuple.long_short_best_dir_z_score_diff_values[an_idx]
    # display(LR_template_epoch_actually_included_aclus[an_idx])
    # display(RL_template_epoch_actually_included_aclus[an_idx])



# _out_rank_order_event_raster_debugger.on_idx_changed_callback_function_dict['a_debug_callback'] = a_debug_callback_fn
# _out_rank_order_event_raster_debugger.on_idx_changed_callback_function_dict['debug_update_plot_titles_callback'] = debug_update_plot_titles
# # _out_rank_order_event_raster_debugger.on_idx_changed_callback_function_dict['debug_update_paired_directional_template_pfs_debugger'] = debug_update_paired_directional_template_pfs_debugger
# # _out_rank_order_event_raster_debugger.on_idx_changed_callback_function_dict['debug_update_long_short_info_titles'] = debug_update_long_short_info_titles
# # _out_rank_order_event_raster_debugger.on_idx_changed_callback_function_dict['debug_plot_epoch_label_row'] = debug_plot_epoch_label_row



# _out_rank_order_event_raster_debugger.on_update_epoch_IDX(11)