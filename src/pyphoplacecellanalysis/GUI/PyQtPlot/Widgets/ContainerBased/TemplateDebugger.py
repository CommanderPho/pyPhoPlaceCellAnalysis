from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
from nptyping import NDArray
import numpy as np
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.print_helpers import strip_type_str_to_classname

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field, keys_only_repr
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import build_shared_sorted_neuron_color_maps
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multi_sort_raster_browser, plot_raster_plot

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState

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

from pyphoplacecellanalysis.External.pyqtgraph import QtGui
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent, pyqtplot_plot_image


__all__ = ['TemplateDebugger']



    
# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import _debug_plot_directional_template_rasters, build_selected_spikes_df, add_selected_spikes_df_points_to_scatter_plot

@metadata_attributes(short_name=None, tags=['gui', 'template'], input_requires=[], output_provides=[], uses=[], used_by=['_display_directional_template_debugger'], creation_date='2023-12-11 10:24', related_items=[])
@define(slots=False, repr=False)
class TemplateDebugger:
    """ TemplateDebugger displays four 1D heatmaps colored by cell for the tuning curves of PfND. Each shows the same tuning curves but they are sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger

    _out = TemplateDebugger.init_templates_debugger(track_templates) # , included_any_context_neuron_ids


    """
    # track_templates: TrackTemplates = field(repr=False)
    # RL_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)
    # LR_active_epochs_selected_spikes_fragile_linear_neuron_IDX_dict: Union[Dict,pd.DataFrame] = field(repr=False)

    plots: RenderPlots = field(repr=keys_only_repr)
    plots_data: RenderPlotsData = field(repr=keys_only_repr)
    ui: PhoUIContainer = field(repr=keys_only_repr)
    params: VisualizationParameters = field(repr=keys_only_repr)
    

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
    def init_templates_debugger(cls, track_templates: TrackTemplates, included_any_context_neuron_ids=None, use_incremental_sorting:bool=False, enable_pf_peak_indicator_lines:bool=True, **kwargs):
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
        debug_draw: bool = kwargs.pop('debug_draw', False)

        enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
        use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)

        if enable_pf_peak_indicator_lines:
            print(f'WARN: 2023-12-11 - enable_pf_peak_indicator_lines is not yet implemented and the lines are not correctly aligned.')  
                  

        figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
        _out_data = RenderPlotsData(name=figure_name, track_templates=deepcopy(track_templates), out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, sorted_pf_peak_location_list=None, active_pfs_img_extents_dict=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
        _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)
        _out_params = VisualizationParameters(name=figure_name, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, use_shared_aclus_only_templates=use_shared_aclus_only_templates,
                                             debug_print=debug_print, debug_draw=debug_draw, use_incremental_sorting=use_incremental_sorting, enable_pf_peak_indicator_lines=enable_pf_peak_indicator_lines, included_any_context_neuron_ids=included_any_context_neuron_ids, **kwargs)
                
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


    def __repr__(self):
        """ 
        TemplateDebugger(plots: pyphocorehelpers.DataStructure.general_parameter_containers.RenderPlots,
            plots_data: pyphocorehelpers.DataStructure.general_parameter_containers.RenderPlotsData,
            ui: pyphocorehelpers.gui.PhoUIContainer.PhoUIContainer,
            params: pyphocorehelpers.DataStructure.general_parameter_containers.VisualizationParameters
        )
        """
        # content = ", ".join( [f"{a.name}={v!r}" for a in self.__attrs_attrs__ if (v := getattr(self, a.name)) != a.default] )
        # content = ", ".join([f"{a.name}:{strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        content = ",\n\t".join([f"{a.name}: {strip_type_str_to_classname(type(getattr(self, a.name)))}" for a in self.__attrs_attrs__])
        # content = ", ".join([f"{a.name}" for a in self.__attrs_attrs__]) # 'TrackTemplates(long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder, shared_LR_aclus_only_neuron_IDs, is_good_LR_aclus, shared_RL_aclus_only_neuron_IDs, is_good_RL_aclus, decoder_LR_pf_peak_ranks_list, decoder_RL_pf_peak_ranks_list)'
        return f"{type(self).__name__}({content}\n)"


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
    def _subfn_rebuild_sort_idxs(cls, decoders_dict: Dict, _out_data: RenderPlotsData, use_incremental_sorting: bool, included_any_context_neuron_ids: NDArray) -> RenderPlotsData:
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
            # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in decoders_dict.items()} # tuning_curve peak location
            sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in decoders_dict.items()} # tuning_curve CoM location
            sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(decoders_dict, included_any_context_neuron_ids, sortable_values_list_dict=sortable_values_list_dict)

        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        # sorted_pf_tuning_curves_dict = {a_decoder_name:a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_decoder_name, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts.keys(), sort_helper_neuron_id_to_sort_IDX_dicts.values())}
        # Get the peak locations for the tuning curves:
        sorted_pf_peak_location_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values()))] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        # track_templates.decoder_peak_location_list

        # sorted_pf_image_bounds_list = [pyqtplot_build_image_bounds_extent(a_decoder.pf.ratemap.xbins, a_decoder.pf.ratemap.ybins, margin=0.0, debug_print=False) for a_decoder in decoders_dict.values()]
        # pf_xbins_list = [a_decoder.pf.ratemap.xbin for a_decoder in decoders_dict.values()]
        img_extents_dict = {a_decoder_name:[a_decoder.pf.ratemap.xbin[0], 0, (a_decoder.pf.ratemap.xbin[-1]-a_decoder.pf.ratemap.xbin[0]), (float(len(sorted_neuron_IDs_lists[i]))-0.0)] for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()) } # these extents are  (x, y, w, h)
        
        # below uses `sorted_pf_tuning_curves`, `sort_helper_neuron_id_to_neuron_colors_dicts`
        _out_data.ref_decoder_name = ref_decoder_name
        _out_data.sorted_neuron_IDs_lists = sorted_neuron_IDs_lists
        _out_data.sort_helper_neuron_id_to_neuron_colors_dicts = sort_helper_neuron_id_to_neuron_colors_dicts
        _out_data.sort_helper_neuron_id_to_sort_IDX_dicts = sort_helper_neuron_id_to_sort_IDX_dicts
        _out_data.sorted_pf_tuning_curves = sorted_pf_tuning_curves
        _out_data.unsorted_included_any_context_neuron_ids = deepcopy(included_any_context_neuron_ids)
        _out_data.sorted_pf_peak_location_list = deepcopy(sorted_pf_peak_location_list)
        _out_data.active_pfs_img_extents_dict = deepcopy(img_extents_dict)
        return _out_data

    # 2023-11-28 - New Sorting using `paired_incremental_sort_neurons` via `paired_incremental_sorting`
    @classmethod
    def _subfn_buildUI_directional_template_debugger_data(cls, included_any_context_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, _out_params: VisualizationParameters, decoders_dict: Dict):
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
            curr_pf_peak_locations = sorted_pf_peak_location_list[i]
            curr_xbins = deepcopy(a_decoder.pf.ratemap.xbin)
            
            _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(curr_curves, title=title_str, show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)

            # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
            curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img
            if _out_params.debug_draw:
                # Shows the axes if debug_print == true
                curr_win.showAxes(True)
                
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
                if _out_params.enable_pf_peak_indicator_lines:
                    x_offset = curr_pf_peak_locations[cell_i]
                    y_offset = float(cell_i) 
                    line_height = 1.0
                    half_line_height = line_height / 2.0 # to compensate for middle
                    # line = QtGui.QGraphicsLineItem(x_offset, (y_offset - half_line_height), x_offset, (y_offset + half_line_height)) # (xstart, ystart, xend, yend)
                    line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + line_height)) # (xstart, ystart, xend, yend)
                    # line = pg.InfiniteLine(pos=(x_offset, float(cell_i+1)), angle=90, movable=False)
                    line.setPen(pg.mkPen('white', width=2))  # Set color and width of the line
                    curr_win.addItem(line)
                    # line.setPos(pg.Point(x_offset, (y_offset + (line_height / 2.0)))) # Adjust the height of the line if needed
                    _out_ui.order_location_lines_dict[a_decoder_name][aclu] = line # add to the map


            # for x_offset, height in vertical_lines:
            #     line = pg.InfiniteLine(pos=(x_offset, 0), angle=90, movable=False)
            #     line.setPen(pg.mkPen('r', width=2))  # Set color and width of the line
            #     win.addItem(line)

            #     # Adjust the height of the line if needed
            #     # Note: This is a basic implementation. Adjust according to your coordinate system and needs.
            #     line.setPos(pg.Point(x_offset, height / 2.0)) # Adjust the height of the line if needed
                

            ## Build the colored heatmap:
            out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            if debug_print:
                print(f"np.shape(out_colors_heatmap_image_matrix): {np.shape(out_colors_heatmap_image_matrix)}") # (34, 62, 4) - (n_cells, n_pos_bins, n_channels_RGBA)

                # Ensure the data is in the correct range [0, 1]
            out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            if enable_cell_colored_heatmap_rows:
                curr_img.updateImage(out_colors_heatmap_image_matrix) # , xvals=curr_xbins, use the color image only if `enable_cell_colored_heatmap_rows==True`
            _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix

            # Set the extent to map pixels to x-locations
            curr_img.setRect(_out_data.active_pfs_img_extents_dict[a_decoder_name])
    
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
    def _subfn_update_directional_template_debugger_data(cls, included_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, _out_params: VisualizationParameters, decoders_dict: Dict):
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
        sorted_pf_peak_location_list = _out_data.sorted_pf_peak_location_list

        ## Plot the placefield 1Ds as heatmaps and then wrap them in docks and add them to the window:
        assert _out_plots.pf1D_heatmaps is not None
        for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()):
            if use_incremental_sorting:
                title_str = f'{a_decoder_name}_pf1Ds [sort: {_out_data.ref_decoder_name}]'
            else:
                title_str = f'{a_decoder_name}_pf1Ds'

            curr_curves = sorted_pf_tuning_curves[i]
            curr_pf_peak_locations = sorted_pf_peak_location_list[i]
            curr_xbins = deepcopy(a_decoder.pf.ratemap.xbin)
            
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
            

            ## Remove all lines and re-add:
            for aclu, a_line_item in _out_ui.order_location_lines_dict[a_decoder_name].items():
                curr_win.removeItem(a_line_item)
                # a_line_item.deleteLater()
            _out_ui.order_location_lines_dict[a_decoder_name] = {} # clear the dictionary
            

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

                # Add vertical lines
                if _out_params.enable_pf_peak_indicator_lines:
                    x_offset = curr_pf_peak_locations[cell_i]
                    y_offset = float(cell_i) 
                    line_height = 1.0
                    line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + line_height)) # (xstart, ystart, xend, yend)
                    # line.setPen(pg.mkPen('white', width=2))  # Set color and width of the line
                    line.setPen(pg.mkPen(pg.mkColor(a_color_vector), width=2))  # Set color and width of the line
                    curr_win.addItem(line)
                    _out_ui.order_location_lines_dict[a_decoder_name][aclu] = line # add to the map
                    # # Old update-based way:
                    # line = _out_ui.order_location_lines_dict[a_decoder_name][aclu] # QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, line_height)
                    # line.setLine(x_offset, y_offset, x_offset, (y_offset + line_height)) # (xstart, ystart, xend, yend)

            # end `for cell_i, (aclu, a_color_vector)`

            ## Build the colored heatmap:
            out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            # Ensure the data is in the correct range [0, 1]
            out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            if enable_cell_colored_heatmap_rows:
                curr_img.updateImage(out_colors_heatmap_image_matrix) #, xvals=curr_xbins, use the color image only if `enable_cell_colored_heatmap_rows==True`
            _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix
            
            # Set the extent to map pixels to x-locations
            curr_img.setRect(_out_data.active_pfs_img_extents_dict[a_decoder_name])
            
        # end `for i, (a_decoder_name, a_decoder)`

        return _out_data, _out_plots, _out_ui


    def buildUI_directional_template_debugger_data(self):
        """Calls `_subfn_buildUI_directional_template_debugger_data` to build the UI and then updates the member variables."""
        self.plots_data, self.plots, self.ui = self._subfn_buildUI_directional_template_debugger_data(self.params.included_any_context_neuron_ids, self.params.use_incremental_sorting, self.params.debug_print, self.params.enable_cell_colored_heatmap_rows, self.plots_data, self.plots, self.ui, _out_params=self.params, decoders_dict=self.decoders_dict)

    def update_directional_template_debugger_data(self, included_neuron_ids):
        """Calls `_subfn_update_directional_template_debugger_data` to build the UI and then updates the member variables."""
        self.plots_data, self.plots, self.ui = self._subfn_update_directional_template_debugger_data(included_neuron_ids, self.params.use_incremental_sorting, self.params.debug_print, self.params.enable_cell_colored_heatmap_rows, self.plots_data, self.plots, self.ui, _out_params=self.params, decoders_dict=self.decoders_dict)



    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #


# ==================================================================================================================== #
# 2023-12-20 Mouse Tracking Diversion/Failed                                                                           #
# ==================================================================================================================== #

    def mouseMoved(self, a_decoder_name: str, evt):
        """ captures `label` """
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        print(f'mouseMoved(a_decoder_name: {a_decoder_name}, evt: {evt})')
        
        # a_plot_widget, an_image_item = self.plots['pf1D_heatmaps'][a_decoder_name] # (pyqtgraph.widgets.PlotWidget.PlotWidget, pyqtgraph.graphicsItems.ImageItem.ImageItem)
        # # a_view_box = an_image_item.getViewBox()
        # vb = a_plot_widget.vb
        # if a_plot_widget.sceneBoundingRect().contains(pos):
        #     mousePoint = vb.mapSceneToView(pos)
        #     index = int(mousePoint.x())
        #     print(f'\tmousePoint.x(): {mousePoint.x()}, \tindex: {index}')
        #     # if index > 0 and index < len(data1):
        #     #     print(f"<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        #         # if self.label is not None:
        #         #     self.label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        #     # self.vLine.setPos(mousePoint.x())
        #     # self.hLine.setPos(mousePoint.y())


# a_plot_widget.window().setMouseTracking(True)
# a_plot_widget.setMouseEnabled(True)
# from functools import partial

# proxy_dict = {}

# def test_simple_mouseMoved(evt):
#     """ captures `label` """
#     print(f'mouseMoved(evt: {evt})')
    
        
# def test_mouseMoved(a_decoder_name, evt):
#     """ captures `label` """
#     print(f'mouseMoved(a_decoder_name: {a_decoder_name}, evt: {evt})')
#     pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    
#     # a_plot_widget, an_image_item = self.plots['pf1D_heatmaps'][a_decoder_name] # (pyqtgraph.widgets.PlotWidget.PlotWidget, pyqtgraph.graphicsItems.ImageItem.ImageItem)
#     # # a_view_box = an_image_item.getViewBox()
#     # vb = a_plot_widget.vb
#     # if a_plot_widget.sceneBoundingRect().contains(pos):
#     #     mousePoint = vb.mapSceneToView(pos)
#     #     index = int(mousePoint.x())
#     #     print(f'\tmousePoint.x(): {mousePoint.x()}, \tindex: {index}')
        

# a_decoder_name: str = 'long_LR'
# a_plot_widget, an_image_item = plots['pf1D_heatmaps'][a_decoder_name] # (pyqtgraph.widgets.PlotWidget.PlotWidget, pyqtgraph.graphicsItems.ImageItem.ImageItem)
        
# # proxy_dict[a_decoder_name] = pg.SignalProxy(a_plot_widget.scene().sigMouseMoved, rateLimit=30, slot=test_mouseMoved) # partial(all_cells_directional_template_pfs_debugger.mouseMoved, a_decoder_name)

# # proxy_dict[a_decoder_name] = pg.SignalProxy(a_plot_widget.scene().sigMouseMoved, rateLimit=30, slot=partial(test_mouseMoved, a_decoder_name)) # partial(all_cells_directional_template_pfs_debugger.mouseMoved, a_decoder_name)


# a_plot_widget.setMouseTracking(True)
# proxy_dict[a_decoder_name] = pg.SignalProxy(a_plot_widget.scene().sigMouseMoved, rateLimit=30, slot=test_simple_mouseMoved) 

# a_conn = a_plot_widget.scene().sigMouseMoved.connect(test_simple_mouseMoved)



    # def update_plot_titles_with_stats(self, an_idx: int):
    #     """ Updates the titles of each of the four rasters with the appropriate spearman rho value.
    #     captures: rank_order_results_debug_values || active_epochs_df, formatted_title_strings_dict


    #     Usages:
    #         self.params.enable_show_spearman
    #         self.params.enable_show_pearson
    #         self.params.enable_show_Z_values

    #         self.active_epoch_result_df


    #     """
    #     is_laps: bool = self.params.is_laps
    #     use_plaintext_title: bool = self.params.use_plaintext_title
    #     if not use_plaintext_title:
    #         formatted_title_strings_dict = DisplayColorsEnum.get_pyqtgraph_formatted_title_dict()

    #     # curr_epoch_label = a_plotter.lookup_label_from_index(an_idx)
    #     # ripple_combined_epoch_stats_df = a_plotter.rank_order_results.ripple_combined_epoch_stats_df
    #     # curr_new_results_df = ripple_combined_epoch_stats_df[ripple_combined_epoch_stats_df.index == curr_epoch_label]

    #     curr_new_results_df = self.active_epoch_result_df
    #     for a_decoder_name, a_root_plot in self.plots.root_plots.items():
    #         # a_real_value = rank_order_results_debug_values[a_decoder_name][0][an_idx]
    #         a_std_column_name: str = self.decoder_name_to_column_name_prefix_map[a_decoder_name]

    #         all_column_names = curr_new_results_df.filter(regex=f'^{a_std_column_name}').columns.tolist()
    #         active_column_names = []
    #         # print(active_column_names)
    #         if self.params.enable_show_spearman:
    #             active_column_names = [col for col in all_column_names if col.endswith("_spearman")]
    #             if self.params.enable_show_Z_values:
    #                 active_column_names += [col for col in all_column_names if col.endswith("_spearman_Z")]


    #         if self.params.enable_show_pearson:
    #             active_column_names += [col for col in all_column_names if col.endswith("_pearson")]
    #             if self.params.enable_show_Z_values:
    #                 active_column_names += [col for col in all_column_names if col.endswith("_pearson_Z")]


    #         active_column_values = curr_new_results_df[active_column_names]
    #         active_values_dict = active_column_values.iloc[0].to_dict() # {'LR_Long_spearman': -0.34965034965034975, 'LR_Long_pearson': -0.5736588716389961, 'LR_Long_spearman_Z': -0.865774983083525, 'LR_Long_pearson_Z': -1.4243571733839517}
    #         active_raw_col_val_dict = {k.replace(f'{a_std_column_name}_', ''):v for k,v in active_values_dict.items()} # remove the "LR_Long" prefix so it's just the variable names

    #         active_formatted_col_val_list = [':'.join([generate_html_string(str(k), color='grey', bold=False), generate_html_string(f'{v:0.3f}', color='white', bold=True)]) for k,v in active_raw_col_val_dict.items()]
    #         final_values_string: str = '; '.join(active_formatted_col_val_list)

    #         if use_plaintext_title:
    #             title_str = generate_html_string(f"{a_std_column_name}: {final_values_string}")
    #         else:
    #             # Color formatted title:
    #             a_formatted_title_string_prefix: str = formatted_title_strings_dict[a_std_column_name]
    #             title_str = generate_html_string(f"{a_formatted_title_string_prefix}: {final_values_string}")

    #         a_root_plot.setTitle(title=title_str)
            



    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #

    # ==================================================================================================================== #
    # Core Component Building Classmethods                                                                                 #
    # ==================================================================================================================== #

    



#TODO 2023-12-15 22:39: - [ ] Factored out of notebook


# ### TemplateDebugger adding lines:
# aclu_to_idx: Dict = self.rdf.aclu_to_idx
# aclu_to_idx_df: pd.DataFrame = pd.DataFrame({'aclu': list(aclu_to_idx.keys()), 'fragile_linear_idx': list(aclu_to_idx.values())})
# aclu_to_idx_df.to_hdf(file_path, key=f'{key}/rdf/aclu_to_idx_df', format='table', data_columns=True)
# roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
# imv1.addItem(roi)
# directional_laps_results
# ## 2023-10-11 - Get the long/short peak locations
# decoder_peak_coms_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[is_good_aclus] for a_decoder in decoder_args]

# # decoders_list = [directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D]
# decoders_list = [directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D]

# # directional_laps_results.long_LR_one_step_decoder_1D, directional_laps_results.long_RL_one_step_decoder_1D, directional_laps_results.short_LR_one_step_decoder_1D, directional_laps_results.short_RL_one_step_decoder_1D = modified_decoders_list
# directional_laps_results.long_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.long_RL_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_LR_shared_aclus_only_one_step_decoder_1D, directional_laps_results.short_RL_shared_aclus_only_one_step_decoder_1D = modified_decoders_list
# rank_order_results.LR_laps.long_z_score