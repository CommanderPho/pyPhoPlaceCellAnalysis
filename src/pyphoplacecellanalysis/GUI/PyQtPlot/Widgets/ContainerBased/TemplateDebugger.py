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
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons, paired_incremental_sort_neurons # _display_directional_template_debugger
from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays


from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon


__all__ = ['TemplateDebugger']



    
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
    # track_templates: TrackTemplates = field(repr=False)
    # RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)
    # LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)

    plots: RenderPlots = field()
    plots_data: RenderPlotsData = field(repr=False)
    ui: PhoUIContainer = field(repr=False)
    params: VisualizationParameters = field(repr=False)
    

    # active_epoch_IDX: int = field(default=0, repr=True)

    # on_idx_changed_callback_function_dict: Dict[str, Callable] = field(default=Factory(dict), repr=False)


    @property
    def track_templates(self) -> TrackTemplates:
        return self.plots_data.track_templates


    # @property
    # def active_epoch_tuple(self) -> tuple:
    #     """ returns a namedtuple describing the single epoch corresponding to `self.active_epoch_IDX`. """
    #     a_df_idx = self.active_epochs_df.index.to_numpy()[self.active_epoch_IDX]
    #     curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == a_df_idx)] # this +1 here makes zero sense
    #     curr_epoch = list(curr_epoch_df.itertuples(name='EpochTuple'))[0]
    #     return curr_epoch

    @property
    def decoders_dict(self) -> Dict:
        return self.track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }


    @classmethod
    def init_templates_debugger(cls, track_templates: TrackTemplates, included_any_context_neuron_ids=None, use_incremental_sorting: bool = False, **kwargs):
        """
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps) # .trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()

        Usage:
        # Recover from the saved global result:
        directional_laps_results = global_computation_results.computed_data['DirectionalLaps']

        assert 'RankOrder' in global_computation_results.computed_data, f"as of 2023-11-30 - RankOrder is required to determine the appropriate 'minimum_inclusion_fr_Hz' to use. Previously None was used."
        rank_order_results = global_computation_results.computed_data['RankOrder'] # RankOrderComputationsContainer
        minimum_inclusion_fr_Hz: float = rank_order_results.minimum_inclusion_fr_Hz
        assert minimum_inclusion_fr_Hz is not None
        if (use_shared_aclus_only_templates):
            track_templates: TrackTemplates = directional_laps_results.get_shared_aclus_only_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # shared-only
        else:
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz) # non-shared-only

        
        
        """
        fignum = kwargs.pop('fignum', None)
        if fignum is not None:
            print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

        defer_render = kwargs.pop('defer_render', False)
        debug_print: bool = kwargs.pop('debug_print', False)

        enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
        use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)
        
        figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
        _out_data = RenderPlotsData(name=figure_name, track_templates=deepcopy(track_templates), out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, sorted_pf_peak_location_list=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
        _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)
        _out_params = VisualizationParameters(name=figure_name, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, use_shared_aclus_only_templates=use_shared_aclus_only_templates, debug_print=debug_print, use_incremental_sorting=use_incremental_sorting, included_any_context_neuron_ids=included_any_context_neuron_ids)
                
        # decoders_dict = track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }


        # build the window with the dock widget in it:
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho Directional Template Debugger: {figure_name}', defer_show=False)
        icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
        if icon is not None:
            root_dockAreaWindow.setWindowIcon(icon)
        # icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico"
        # root_dockAreaWindow.setWindowIcon(pg.QtGui.QIcon(icon_path))

        _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, text_items_dict=None, order_location_lines_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)
        
        root_dockAreaWindow.resize(900, 700)

        _obj = cls(plots=_out_plots, plots_data=_out_data, ui=_out_ui, params=_out_params)

        _obj.buildUI_directional_template_debugger_data()
        update_callback_fn = (lambda included_neuron_ids: _obj.update_directional_template_debugger_data(included_neuron_ids))
        _obj.ui.on_update_callback = update_callback_fn

        # _out_data, _out_plots, _out_ui = TemplateDebugger._subfn_buildUI_directional_template_debugger_data(included_any_context_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict)
        # update_callback_fn = (lambda included_neuron_ids: TemplateDebugger._subfn_update_directional_template_debugger_data(included_neuron_ids, use_incremental_sorting=use_incremental_sorting, debug_print=debug_print, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, _out_data=_out_data, _out_plots=_out_plots, _out_ui=_out_ui, decoders_dict=decoders_dict))
        # _out_ui.on_update_callback = update_callback_fn
        
        return _obj

    # Saving/Exporting to file ___________________________________________________________________________________________ #
    #TODO 2023-11-16 22:16: - [ ] Figure out how to save

    def save_figure(self): # export_file_base_path: Path = Path(f'output').resolve()
        """ captures: epochs_editor, _out_pf1D_heatmaps

        TODO: note output paths are currently hardcoded. Needs to add the animal's context at least. Probably needs to be integrated into pipeline.
        import pyqtgraph as pg
        import pyqtgraph.exporters
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
        """
        ## Get main laps plotter:
        # print_keys_if_possible('_out', _out, max_depth=4)
        # plots = _out['plots']

        ## Already have: epochs_editor, _out_pf1D_heatmaps
        epochs_editor = graphics_output_dict['ui'][0]

        shared_output_file_prefix = f'output/2023-11-20'
        # print(list(plots.keys()))
        # pg.GraphicsLayoutWidget
        main_graphics_layout_widget = epochs_editor.plots.win
        export_file_path = Path(f'{shared_output_file_prefix}_test_main_position_laps_line_plot').with_suffix('.svg').resolve()
        export_pyqtgraph_plot(main_graphics_layout_widget, savepath=export_file_path) # works

        _out_pf1D_heatmaps = graphics_output_dict['plots']
        for a_decoder_name, a_decoder_heatmap_tuple in _out_pf1D_heatmaps.items():
            a_win, a_img = a_decoder_heatmap_tuple
            # a_win.export_image(f'{a_decoder_name}_heatmap.png')
            print(f'a_win: {type(a_win)}')

            # create an exporter instance, as an argument give it the item you wish to export
            exporter = pg.exporters.ImageExporter(a_win.plotItem)
            # exporter = pg.exporters.SVGExporter(a_win.plotItem)
            # set export parameters if needed
            # exporter.parameters()['width'] = 300   # (note this also affects height parameter)

            # save to file
            export_file_path = Path(f'{shared_output_file_prefix}_test_{a_decoder_name}_heatmap').with_suffix('.png').resolve() # '.svg' # .resolve()

            exporter.export(str(export_file_path)) # '.png'
            print(f'exporting to {export_file_path}')
            # .scene()


    

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
        # Get the peak locations for the tuning curves:
        sorted_pf_peak_location_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values()))] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        # track_templates.decoder_peak_location_list

        # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
        _out_data.ref_decoder_name = ref_decoder_name
        _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
        _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
        _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts
        _out_data.sorted_pf_tuning_curves = sorted_pf_tuning_curves
        _out_data.unsorted_included_any_context_neuron_ids = deepcopy(included_any_context_neuron_ids)
        _out_data.sorted_pf_peak_location_list = deepcopy(sorted_pf_peak_location_list)
        return _out_data

    # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`
    @classmethod
    def _subfn_buildUI_directional_template_debugger_data(cls, included_any_context_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, decoders_dict: Dict):
        """ Builds UI """
        _out_data = cls._subfn_rebuild_sort_idxs(decoders_dict, _out_data, use_incremental_sorting=use_incremental_sorting, included_any_context_neuron_ids=included_any_context_neuron_ids)
        # Unpack the updated _out_data:
        sort_helper_neuron_id_to_neuron_colors_dicts = _out_data.sort_helper_neuron_id_to_neuron_colors_dicts
        sorted_pf_tuning_curves = _out_data.sorted_pf_tuning_curves
        sorted_pf_peak_location_list = _out_data.sorted_pf_peak_location_list

        ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
        _out_plots.pf1D_heatmaps = {}
        _out_ui.text_items_dict = {}
        _out_ui.order_location_lines_dict = {}
        
        for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            if use_incremental_sorting:
                title_str = f'{a_decoder_name}_pf1Ds [sort: {_out_data.ref_decoder_name}]'
            else:
                title_str = f'{a_decoder_name}_pf1Ds'

            curr_curves = sorted_pf_tuning_curves[i]
            curr__pf_peak_locations = sorted_pf_peak_location_list[i]

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
            _out_ui.order_location_lines_dict[a_decoder_name] = {} # new dict to hold these items.
            
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
                
                # Add vertical lines
                x_offset = curr__pf_peak_locations[cell_i]
                line_height = 1.0
                line = pg.InfiniteLine(pos=(x_offset, float(cell_i+1)), angle=90, movable=False)
                line.setPen(pg.mkPen('white', width=2))  # Set color and width of the line
                curr_win.addItem(line)
                line.setPos(pg.Point(x_offset, line_height / 2.0)) # Adjust the height of the line if needed
                _out_ui.order_location_lines_dict[a_decoder_name][aclu] = line # add the TextItem to the map


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
    def _subfn_update_directional_template_debugger_data(cls, included_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, decoders_dict: Dict):
        """ Just updates the existing UI, doesn't build new elements.

        ## Needs to update:
        _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name]
        curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name]
        _out_ui.dock_widgets[a_decoder_name] # maybe

        """
        _out_data = cls._subfn_rebuild_sort_idxs(decoders_dict, _out_data, use_incremental_sorting=use_incremental_sorting, included_any_context_neuron_ids=included_neuron_ids)
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


    def buildUI_directional_template_debugger_data(self):
        """Calls `_subfn_buildUI_directional_template_debugger_data` to build the UI and then updates the member variables."""
        self.plots_data, self.plots, self.ui = self._subfn_buildUI_directional_template_debugger_data(self.params.included_any_context_neuron_ids, self.params.use_incremental_sorting, self.params.debug_print, self.params.enable_cell_colored_heatmap_rows, self.plots_data, self.plots, self.ui, self.decoders_dict)

    def update_directional_template_debugger_data(self, included_neuron_ids):
        """Calls `_subfn_update_directional_template_debugger_data` to build the UI and then updates the member variables."""
        self.plots_data, self.plots, self.ui = self._subfn_update_directional_template_debugger_data(included_neuron_ids, self.params.use_incremental_sorting, self.params.debug_print, self.params.enable_cell_colored_heatmap_rows, self.plots_data, self.plots, self.ui, self.decoders_dict)




    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #

    # ==================================================================================================================== #
    # Core Component Building Classmethods                                                                                 #
    # ==================================================================================================================== #

    





