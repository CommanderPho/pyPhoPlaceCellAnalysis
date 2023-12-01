from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
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

    def get_epoch_active_aclus(self) -> np.array:
        """ returns a list of aclus active (having at least one spike) in the current epoch (based on `self.active_epoch`) """
        active_epoch_spikes_df: pd.DataFrame = self.get_active_epoch_spikes_df()
        active_epoch_unique_active_aclus = np.unique(active_epoch_spikes_df['aclu'].to_numpy())
        return active_epoch_unique_active_aclus


    @classmethod
    def init_rank_order_debugger(cls, global_spikes_df: pd.DataFrame, active_epochs_df: pd.DataFrame, track_templates: TrackTemplates, RL_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame], LR_active_epoch_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame]):
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

        name:str = 'RankOrderRastersDebugger'
        # # LR_display_outputs, RL_display_outputs = cls._debug_plot_directional_template_rasters(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates) # `_debug_plot_directional_template_rasters` main plot commmand
        # LR_display_outputs, RL_display_outputs = cls._modern_debug_plot_directional_template_rasters(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates) # `_debug_plot_directional_template_rasters` main plot commmand
        # LR_app, LR_win, LR_plots, LR_plots_data, LR_on_update_active_epoch, LR_on_update_active_scatterplot_kwargs = LR_display_outputs
        # RL_app, RL_win, RL_plots, RL_plots_data, RL_on_update_active_epoch, RL_on_update_active_scatterplot_kwargs = RL_display_outputs

        ## 2023-11-30 - Newest Version using separate rasters:
        _obj.plots_data, _obj.plots = cls._post_modern_debug_plot_directional_template_rasters(_obj.global_spikes_df, _obj.active_epochs_df, _obj.track_templates, debug_print=True)
        #TODO 2023-11-30 15:14: - [ ] Unpacking and putting in docks and such not yet finished. Update functions would need to be done separately.

        # Embedding in docks:
        # root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(RL_win, LR_win, title='Pho Debug Plot Directional Template Rasters')

        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Debug Plot Directional Template Rasters')

        RL_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors)
        LR_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors)

        _out_dock_widgets = {}
        dock_configs = (RL_dock_config, LR_dock_config)
        dock_add_locations = (['left'], ['right'])

        _out_dock_widgets['RL'] = root_dockAreaWindow.add_display_dock(identifier='RL', widget=RL_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[0], display_config=RL_dock_config)
        _out_dock_widgets['LR'] = root_dockAreaWindow.add_display_dock(identifier='LR', widget=LR_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[1], display_config=LR_dock_config)

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






    # ==================================================================================================================== #
    # Update Active Epoch Functions                                                                                        #
    # ==================================================================================================================== #

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

        LR_plots_data: RenderPlotsData = self.plots_data.LR_plots_data
        RL_plots_data: RenderPlotsData = self.plots_data.RL_plots_data

        ## Built flat lists across all four rasters so they aren't broken up into LR/RL when indexing:
        _active_plot_identifiers = list(LR_plots_data.plots_data_dict.keys()) + list(RL_plots_data.plots_data_dict.keys()) # ['long_LR', 'short_LR', 'long_RL', 'short_RL']
        _paired_plots_data = [LR_plots_data, LR_plots_data, RL_plots_data, RL_plots_data]
        _paired_plots = [self.plots.LR_plots, self.plots.LR_plots, self.plots.RL_plots, self.plots.RL_plots]

        assert len(updated_neuron_render_configs_dict) == len(_active_plot_identifiers)
        emphasis_state = SpikeEmphasisState.Default
        # self.plots.text_items_dict = {}

        updated_color_dict_dict = {}

        for _active_plot_identifier, plots_data, plots, updated_neuron_render_configs in zip(_active_plot_identifiers, _paired_plots_data, _paired_plots, updated_neuron_render_configs_dict.values()):
            # plots_data: RenderPlotsData = LR_plots_data

            # plots_data.plots_spikes_df_dict[_active_plot_identifier] = plots_data.plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(plots_data.plots_spikes_df_dict[_active_plot_identifier])
            # plots_data.plots_spikes_df_dict[_active_plot_identifier]

            ## Add the neuron_id labels to the rasters:
            raster_plot_manager = plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager
            a_plot_item = plots.ax[_active_plot_identifier]
            # self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map)

            # updated_color_dict = {cell_id:cell_config.color for cell_id, cell_config in updated_neuron_render_configs.items()} ## TODO: efficiency: pass only the colors that changed instead of all the colors:
            updated_color_dict = {}

            for cell_id, cell_config in updated_neuron_render_configs.items():
                # a_fragile_linear_neuron_IDX = raster_plot_manager.cell_id_to_fragile_linear_neuron_IDX_map[cell_id]
                a_fragile_linear_neuron_IDX = raster_plot_manager.unit_sort_manager.find_neuron_IDXs_from_cell_ids(cell_ids=[cell_id])[0]
                curr_qcolor = cell_config.qcolor

                # Determine if the color changed: Easiest to compare the hex value string:
                did_color_change = (raster_plot_manager.params.neuron_colors_hex[a_fragile_linear_neuron_IDX] != cell_config.color) # the hex color

                # Overwrite the old colors:
                raster_plot_manager.params.neuron_qcolors_map[a_fragile_linear_neuron_IDX] = curr_qcolor
                raster_plot_manager.params.neuron_qcolors[a_fragile_linear_neuron_IDX] = curr_qcolor
                # Overwrite the old secondary/derived colors:
                curr_rgbf_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
                raster_plot_manager.params.neuron_colors[:, a_fragile_linear_neuron_IDX] = curr_rgbf_color[:]
                raster_plot_manager.params.neuron_colors_hex[a_fragile_linear_neuron_IDX] = cell_config.color # the hex color

                if did_color_change:
                    # If the color changed, add it to the changed array:
                    updated_color_dict[cell_id] = cell_config.color

            updated_color_dict_dict[_active_plot_identifier] = updated_color_dict

            #TODO 2023-11-29 20:39: - [ ] rebuild the spikes
            # raster_plot_manager.unit_sort_manager.update_spikes_df_visualization_columns(self.global_spikes_df)
            plots_data.all_spots_dict[_active_plot_identifier], plots_data.all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(plots_data.plots_spikes_df_dict[_active_plot_identifier], plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

            #TODO 2023-11-29 20:40: - [ ] Rebuild the y-axis labels:
            # - make sure configs are updated so the colors used to rebuild are correct.
            # - todo - remove the old items?
            aclus_list = list(raster_plot_manager.params.config_items.keys())
            a_decoder_color_map = {aclu:raster_plot_manager.params.config_items[aclu].curr_state_pen_dict[emphasis_state].color() for aclu in aclus_list} # Recover color from pen:

            ## Get the y-values for the labels
            y_values = raster_plot_manager.unit_sort_manager.fragile_linear_neuron_IDX_to_spatial(raster_plot_manager.unit_sort_manager.find_neuron_IDXs_from_cell_ids(cell_ids=aclus_list))
            aclu_y_values_dict = dict(zip(aclus_list, y_values))
            self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map, aclu_y_values_dict) # ideally we could just add a color update to the update function


        return updated_color_dict_dict


    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #

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


    # ==================================================================================================================== #
    # Cell y-axis Labels                                                                                                   #
    # ==================================================================================================================== #

    @classmethod
    def _build_neuron_y_labels(cls, a_plot_item, a_decoder_color_map, aclu_y_values_dict: Dict):
        """ 2023-11-29 - builds the y-axis text labels for a single one of the four raster plots. """
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
        LR_plots_data: RenderPlotsData = self.plots_data.LR_plots_data
        RL_plots_data: RenderPlotsData = self.plots_data.RL_plots_data

        ## Built flat lists across all four rasters so they aren't broken up into LR/RL when indexing:
        _active_plot_identifiers = list(LR_plots_data.plots_data_dict.keys()) + list(RL_plots_data.plots_data_dict.keys()) # ['long_LR', 'short_LR', 'long_RL', 'short_RL']
        _paired_plots_data = [LR_plots_data, LR_plots_data, RL_plots_data, RL_plots_data]
        _paired_plots = [self.plots.LR_plots, self.plots.LR_plots, self.plots.RL_plots, self.plots.RL_plots]

        emphasis_state = SpikeEmphasisState.Default

        if self.plots.text_items_dict is not None:
            #TODO 2023-11-30 08:33: - [ ] Remove the old labels here if they exist.
            print(f'TODO 2023-11-30 08:33: - [ ] Remove the old labels here if they exist.')

        self.plots.text_items_dict = {}

        for _active_plot_identifier, plots_data, plots in zip(_active_plot_identifiers, _paired_plots_data, _paired_plots):
            ## Add the neuron_id labels to the rasters:
            raster_plot_manager = plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager
            aclus_list = list(raster_plot_manager.params.config_items.keys())
            a_decoder_color_map = {aclu:raster_plot_manager.params.config_items[aclu].curr_state_pen_dict[emphasis_state].color() for aclu in aclus_list} # Recover color from pen:
            a_plot_item = plots.ax[_active_plot_identifier]
            ## Get the y-values for the labels
            y_values = raster_plot_manager.unit_sort_manager.fragile_linear_neuron_IDX_to_spatial(raster_plot_manager.unit_sort_manager.find_neuron_IDXs_from_cell_ids(cell_ids=aclus_list))
            aclu_y_values_dict = dict(zip(aclus_list, y_values))

            self.plots.text_items_dict[a_plot_item] = self._build_neuron_y_labels(a_plot_item, a_decoder_color_map, aclu_y_values_dict)


    @function_attributes(short_name=None, tags=['cell_y_labels', 'update', 'active_epoch'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-29 20:05', related_items=[])
    def update_cell_y_labels(self):
        """ called whenever the window scrolls or changes to reposition the y-axis labels created with self._build_cell_y_labels """
        # Adjust based on the whether the aclu is active or not:
        curr_active_aclus = self.get_epoch_active_aclus()

        LR_plots_data: RenderPlotsData = self.plots_data.LR_plots_data
        RL_plots_data: RenderPlotsData = self.plots_data.RL_plots_data

        ## Built flat lists across all four rasters so they aren't broken up into LR/RL when indexing:
        _active_plot_identifiers = list(LR_plots_data.plots_data_dict.keys()) + list(RL_plots_data.plots_data_dict.keys()) # ['long_LR', 'short_LR', 'long_RL', 'short_RL']
        _paired_plots_data = [LR_plots_data, LR_plots_data, RL_plots_data, RL_plots_data]
        _paired_plots = [self.plots.LR_plots, self.plots.LR_plots, self.plots.RL_plots, self.plots.RL_plots]

        emphasis_state = SpikeEmphasisState.Default

        for _active_plot_identifier, plots_data, plots in zip(_active_plot_identifiers, _paired_plots_data, _paired_plots):
            raster_plot_manager = plots_data.plots_data_dict[_active_plot_identifier].raster_plot_manager
            aclus_list = list(raster_plot_manager.params.config_items.keys())
            a_decoder_color_map = {aclu:raster_plot_manager.params.config_items[aclu].curr_state_pen_dict[emphasis_state].color() for aclu in aclus_list} # Recover color from pen:
            a_plot_item = plots.ax[_active_plot_identifier]
            ## Get the y-values for the labels
            y_values = raster_plot_manager.unit_sort_manager.fragile_linear_neuron_IDX_to_spatial(raster_plot_manager.unit_sort_manager.find_neuron_IDXs_from_cell_ids(cell_ids=aclus_list))
            aclu_y_values_dict = dict(zip(aclus_list, y_values))
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
                    text.setColor(pg.mkColor("#666666"))
                # text.setVisible(is_aclu_active)



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
        long_RL, short_RL = [(a_sort-1) for a_sort in track_templates.decoder_RL_pf_peak_ranks_list] # nope, different sizes: (62,), (69,)
        long_LR, short_LR = [(a_sort-1) for a_sort in track_templates.decoder_LR_pf_peak_ranks_list]
        assert np.shape(long_RL) == np.shape(short_RL), f"{np.shape(long_RL)} != {np.shape(short_RL)}"

        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_long, LR_long, RL_short, LR_short))) # SORTED
        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (None, None, None, None))) # unsorted
        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (np.array(list(reversed(np.arange(len(RL_long))))), None, None, None))) # unsorted
        unit_sort_orders_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (long_LR, long_RL, short_LR, short_RL)))

        # np.arange(len(RL_long)))
        # fake_test_indicies = np.zeros_like(track_templates.long_RL_decoder.neuron_IDs) np.array(list(reversed(np.arange(int(round(len(RL_long)/2.0)))))
        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (fake_test_indicies, None, None, None))) # unsorted


        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (np.array([ 7, 14, 20, 26, 38, 40, 43,  3, 12, 35, 37,  1, 16, 18, 13, 15, 31, 33, 27, 42,  0, 41, 34,  9,  2, 24, 39, 30,  4, 45, 32,  6, 28, 11, 21, 36, 17, 44, 23,  5,  8, 10, 19, 22, 25, 29]),
        #                                                                                    None, None, None))) # unsorted

        ## Get what should be the unit indicies:
        # unit_sort_orders_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (RL_long, LR_long, RL_short, LR_short)))
        unit_unordered_neuron_IDs_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (track_templates.long_LR_decoder.neuron_IDs, track_templates.long_RL_decoder.neuron_IDs, track_templates.short_LR_decoder.neuron_IDs, track_templates.short_RL_decoder.neuron_IDs)))
        unit_ordered_neuron_IDs_dict = {a_decoder_name:neuron_IDs[unit_sort_orders_dict[a_decoder_name]] for a_decoder_name, neuron_IDs in unit_unordered_neuron_IDs_dict.items()}

        for a_decoder_name, neuron_IDs in unit_unordered_neuron_IDs_dict.items():
            print(f'a_decoder_name: {a_decoder_name}')
            print(f'   neuron_IDs: {neuron_IDs}')
            print(f'   unit_sort_orders_dict[a_decoder_name]: {unit_sort_orders_dict[a_decoder_name]}')
            print(f'   a_decoder_name:neuron_IDs[unit_sort_orders_dict[a_decoder_name]]: {neuron_IDs[unit_sort_orders_dict[a_decoder_name]]}')

        print(f'unit_ordered_neuron_IDs_dict: {unit_ordered_neuron_IDs_dict}')

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

        # unit_colors_list_dict = dict(zip(['long_RL', 'long_LR', 'short_RL', 'short_LR'], (deepcopy(RL_unit_colors_list), deepcopy(LR_unit_colors_list), deepcopy(RL_unit_colors_list), deepcopy(LR_unit_colors_list)))) # the colors dict for all four templates
        unit_colors_list_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (deepcopy(LR_unit_colors_list), deepcopy(RL_unit_colors_list), deepcopy(LR_unit_colors_list), deepcopy(RL_unit_colors_list)))) # the colors dict for all four templates

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


        ## Single Figure Mode:
        # merged_display_outputs = plot_multi_sort_raster_browser(deepcopy(spikes_df), included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)
        # app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = merged_display_outputs
        # return merged_display_outputs

        return LR_display_outputs, RL_display_outputs


    @classmethod
    def _post_modern_debug_plot_directional_template_rasters(cls, spikes_df, active_epochs_df, track_templates, debug_print=True, **kwargs):
        """ 2023-11-30 **DO EM ALL SEPERATELY**


        _out_data, _out_plots = _post_modern_debug_plot_directional_template_rasters(spikes_df, active_epochs_df, track_templates, debug_print=True)
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

        figure_name: str = kwargs.pop('figure_name', 'rasters debugger')


        decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }

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
        spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(included_neuron_ids)
        # spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
        spikes_df = spikes_df.spikes.adding_epochs_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX', epoch_label_column_name=None) # , override_time_variable_name='t_seconds'
        # spikes_df = spikes_df[spikes_df['ripple_id'] != -1]
        spikes_df = spikes_df[(spikes_df['new_lap_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
        spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards


        # CORRECT: Even: RL, Odd: LR
        RL_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy() # (69, )
        LR_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy() # (64, )
    
        included_any_context_neuron_ids_dict = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], (LR_neuron_ids, RL_neuron_ids, LR_neuron_ids, RL_neuron_ids)))

        # INDIVIDUAL SORTING:
        sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids_dict)
        # sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts = paired_separately_sort_neurons(decoders_dict,
        #                                                                                                                                                 included_any_context_neuron_ids_dict,
        #                                                                                                                                                 unit_sort_orders_dict)

        _out_data = RenderPlotsData(name=figure_name, spikes_df=spikes_df, unit_sort_orders_dict=None, included_any_context_neuron_ids_dict=included_any_context_neuron_ids_dict, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None)
        _out_plots = RenderPlots(name=figure_name, rasters=None, rasters_display_outputs=None)

        # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
        _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
        _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
        _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts

        ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
        _out_plots.rasters = {}
        _out_plots.rasters_display_outputs = {}
        for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            title_str = f'{a_decoder_name}'

            # an_included_unsorted_neuron_ids = deepcopy(included_any_context_neuron_ids_dict[a_decoder_name])
            a_sorted_neuron_ids = deepcopy(sorted_neuron_IDs_lists[i])
            # an_unit_sort_orders = np.arange(len(a_sorted_neuron_ids))
            an_unit_sort_orders: Dict = dict(sorted(deepcopy(sort_helper_neuron_id_to_sort_IDX_dicts[i]).items()))


            # Get only the spikes for the shared_aclus:
            a_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(a_sorted_neuron_ids)
            a_spikes_df, neuron_id_to_new_IDX_map = a_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards

            a_decoder_color_map: Dict = dict(sorted(deepcopy(sort_helper_neuron_id_to_neuron_colors_dicts[i]).items())) # 34 (n_neurons)
            # a_decoder_color_list = list(a_decoder_color_map.values())
            a_decoder_color_list = list(a_decoder_color_map.values())

            # _out_plots.rasters_display_outputs[a_decoder_name] = plot_multi_sort_raster_browser(a_spikes_df, a_sorted_neuron_ids, unit_sort_orders_dict=an_unit_sort_orders, unit_colors_list_dict=a_decoder_color_map, scatter_app_name=f'pho_directional_laps_rasters_{title_str}', defer_show=False, active_context=None)
            _out_plots.rasters_display_outputs[a_decoder_name] = plot_raster_plot(a_spikes_df, a_sorted_neuron_ids, unit_sort_orders_dict=an_unit_sort_orders, unit_colors_list=a_decoder_color_list, scatter_app_name=f'pho_directional_laps_rasters_{title_str}', defer_show=False, active_context=None)
            
            # an_app, a_win, a_plots, a_plots_data, an_on_update_active_epoch, an_on_update_active_scatterplot_kwargs = _out_plots.rasters_display_outputs[a_decoder_name]

            # _out_plots.rasters[a_decoder_name]

        print('oops')
        # app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = plot_multi_sort_raster_browser(spikes_df, included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)

        return _out_data, _out_plots




    # ==================================================================================================================== #
    # Selected Spikes                                                                                                      #
    # ==================================================================================================================== #
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




