from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Callable, Union
from attrs import define, field, Factory
import nptyping as ND
from nptyping import NDArray
import numpy as np
import pandas as pd
import neuropy.utils.type_aliases as types

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import build_shared_sorted_neuron_color_maps
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import plot_multi_sort_raster_browser, plot_raster_plot

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
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
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
from pyphoplacecellanalysis.External.pyqtgraph_extensions.PlotWidget.CustomPlotWidget import CustomPlotWidget
from pyphoplacecellanalysis.External.pyqtgraph_extensions.graphicsItems.SelectableTextItem import SelectableTextItem

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder

__all__ = ['TemplateDebugger']


# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import _debug_plot_directional_template_rasters, build_selected_spikes_df, add_selected_spikes_df_points_to_scatter_plot


def build_pf1D_heatmap_with_labels_and_peaks(pf1D_decoder, visible_aclus, plot_item, img_extents_rect, line_height=1.0, a_decoder_aclu_to_color_map: Dict=None):
    """
    Builds a standalone pf1D heatmap with text labels + vertical peak lines for each ACLU.
    """
    import numpy as np
    import pyqtgraph as pg
    from copy import deepcopy
    from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
    # Requires pf1D_decoder.pf.ratemap.pdf_normalized_tuning_curves, pf1D_decoder.pf.ratemap.xbin,
    # pf1D_decoder.pf.ratemap.peak_tuning_curve_center_of_masses, and pf1D_decoder.get_color_for_aclu(aclu).

    assert len(img_extents_rect) == 4, f"len(img_extents_rect): {len(img_extents_rect)} should be 4 and is of type [x, y, width, height]"

    enable_cell_colored_heatmap_rows: bool = (a_decoder_aclu_to_color_map is not None)
    
    curr_img = pg.ImageItem()
    plot_item.addItem(curr_img)

    n_visible_cells = len(visible_aclus)
    all_aclus = deepcopy(pf1D_decoder.neuron_IDs).tolist()
    n_total_cells = len(all_aclus)

    stacked_curves = pf1D_decoder.pf.ratemap.pdf_normalized_tuning_curves
    xbins = pf1D_decoder.pf.ratemap.xbin
    peak_locations = pf1D_decoder.pf.ratemap.peak_tuning_curve_center_of_masses

    temp_color_rows = []
    for cell_i, aclu in enumerate(visible_aclus):
        aclu_idx = all_aclus.index(aclu)
        a_color_vector = a_decoder_aclu_to_color_map[aclu]
        # a_color_vector = pf1D_decoder.get_color_for_aclu(aclu)

        # Text label:
        text = pg.TextItem(text=str(aclu), color=pg.mkColor(a_color_vector), anchor=(1, 0))
        text.setPos(-1.0, cell_i + 1)
        plot_item.addItem(text)

        # Build color row for heatmap:
        base_color = pg.mkColor(a_color_vector)
        row_data = stacked_curves[aclu_idx]
        color_row = DataSeriesColorHelpers.qColorsList_to_NDarray(
            [build_adjusted_color(base_color, value_scale=v) for v in row_data],
            is_255_array=False
        ).T
        temp_color_rows.append(color_row)

        # Vertical line:
        x_offset = peak_locations[aclu_idx]
        y_offset = float(cell_i)
        line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + line_height))
        line.setPen(pg.mkPen(base_color, width=2))
        plot_item.addItem(line)

    out_colors_heatmap_image_matrix = np.stack(temp_color_rows, axis=0)
    out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)

    if enable_cell_colored_heatmap_rows:
        curr_img.updateImage(out_colors_heatmap_image_matrix)

    mod_rect = deepcopy(img_extents_rect)
    if n_visible_cells < n_total_cells:
        mod_rect[-1] = float(n_visible_cells)
    curr_img.setRect(mod_rect)

    return curr_img, out_colors_heatmap_image_matrix





@define(slots=False, eq=False)
class BaseTemplateDebuggingMixin:
    """ TemplateDebugger displays a 1D heatmap colored by cell for the tuning curves of PfND.
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates

        a_lap_id: int = 1
        # time_bin_edges = _out_decoded_unit_specific_time_binned_spike_counts[a_lap_id]
        time_bin_edges = _out_decoded_time_bin_edges[a_lap_id]
        n_epoch_time_bins: int = len(time_bin_edges) - 1
        print(f'a_lap_id: {a_lap_id}, n_epoch_time_bins: {n_epoch_time_bins}')

        ## INPUTS: a_lap_id, an_img_extents, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists
        # Create a main GraphicsLayoutWidget
        win = pg.GraphicsLayoutWidget()

        # Store plot references
        plots = []
        out_pf1D_decoder_template_objects = []


        ## Get data
        active_bin_unit_specific_time_binned_spike_counts = _out_decoded_unit_specific_time_binned_spike_counts[a_lap_id]
        active_lap_active_aclu_spike_counts_list = _out_decoded_active_unit_lists[a_lap_id]
        active_lap_decoded_pos_outputs = _out_decoded_active_p_x_given_n[a_lap_id]
        # Add PlotItems to the layout horizontally
        # Add n_epoch_time_bins plots to the first row
        for a_time_bin_idx in np.arange(n_epoch_time_bins):
            active_bin_active_aclu_spike_counts_dict = active_lap_active_aclu_spike_counts_list[a_time_bin_idx]
            active_bin_active_aclu_spike_count_values = np.array(list(active_bin_active_aclu_spike_counts_dict.values()))
            active_bin_active_aclu_bin_normalized_spike_count_values = active_bin_active_aclu_spike_count_values / np.sum(active_bin_active_aclu_spike_count_values) # relative number of spikes ... todo.. prioritizes high-firing
            
            active_bin_aclus = np.array(list(active_bin_active_aclu_spike_counts_dict.keys()))
            print(f'a_time_bin_idx: {a_time_bin_idx}/{n_epoch_time_bins} - active_bin_aclus: {active_bin_aclus}')
            ## build the plot:
            plot = win.addPlot(title=f"Plot {a_time_bin_idx+1}")
            plots.append(plot)  # Store the reference
            # curr_img, out_colors_heatmap_image_matrix = build_pf1D_heatmap_with_labels_and_peaks(pf1D_decoder=a_decoder, visible_aclus=active_bin_aclus, plot_item=plot, img_extents_rect=an_img_extents, a_decoder_aclu_to_color_map=a_decoder_aclu_to_color_map)
            # _obj = BaseTemplateDebuggingMixin.init_from_decoder(a_decoder=a_decoder, win=win)
            _obj = BaseTemplateDebuggingMixin.init_from_decoder(a_decoder=a_decoder, win=plot)
            _obj.update_base_decoder_debugger_data(included_neuron_ids=active_bin_aclus)
            out_pf1D_decoder_template_objects.append(_obj)  # Store the reference
            
            active_bin_active_aclu_bin_normalized_spike_count_values
            

            if a_time_bin_idx < (n_epoch_time_bins - 1):
                win.nextColumn()  # Move to the next column for horizontal layout
                

        # Show the layout
        win.show()
    
    """
    plots: RenderPlots = field(repr=keys_only_repr)
    plots_data: RenderPlotsData = field(repr=keys_only_repr)
    ui: PhoUIContainer = field(repr=keys_only_repr)
    params: VisualizationParameters = field(repr=keys_only_repr)

    @property
    def pf1D_heatmap(self) -> Tuple[pg.PlotWidget, pg.ImageItem]:
        """ The heatmap plot window and image item """
        return self.plots.pf1D_heatmap
    
    @property
    def decoder(self) -> BasePositionDecoder:
        """The decoder property."""
        return self.plots_data.decoder
    @decoder.setter
    def decoder(self, value: BasePositionDecoder):
        assert self.plots_data is not None
        self.plots_data.decoder = value

    @property
    def order_location_lines(self) -> Dict[types.aclu_index, pg.TextItem]:
        """ the white vertical lines that indicate the peak location of each place cell. """
        return self.ui.order_location_lines

    @classmethod
    def _subfn_rebuild_sort_idxs(cls, decoder: BasePositionDecoder, _out_data: RenderPlotsData, included_any_context_neuron_ids: NDArray) -> RenderPlotsData:
        """ Updates RenderPlotsData """
        
        # INDIVIDUAL SORTING:
        ref_decoder_name: Optional[str] = None
        dummy_decoder_name_key: str = 'only'
        dummy_decoders_dict = {dummy_decoder_name_key: decoder}
        included_any_context_neuron_ids_dict_dict = {dummy_decoder_name_key: included_any_context_neuron_ids}
        # sortable_values_list_dict = {k:deepcopy(np.argmax(a_decoder.pf.ratemap.normalized_tuning_curves, axis=1)) for k, a_decoder in dummy_decoders_dict.items()} # tuning_curve peak location
        sortable_values_list_dict = {k:deepcopy(a_decoder.pf.peak_tuning_curve_center_of_masses) for k, a_decoder in dummy_decoders_dict.items()} # tuning_curve CoM location
        # sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(dummy_decoders_dict, included_any_context_neuron_ids_dict_dict=included_any_context_neuron_ids, sortable_values_list_dict=sortable_values_list_dict)
        sorted_neuron_IDs_lists, sort_helper_neuron_id_to_neuron_colors_dicts, sort_helper_neuron_id_to_sort_IDX_dicts, (unsorted_original_neuron_IDs_lists, unsorted_neuron_IDs_lists, unsorted_sortable_values_lists, unsorted_unit_colors_map) = paired_separately_sort_neurons(dummy_decoders_dict, included_any_context_neuron_ids_dict_dict=included_any_context_neuron_ids_dict_dict, sortable_values_list_dict=sortable_values_list_dict)
        
        ## dict-based way
        sorted_pf_tuning_curves = [a_decoder.pf.ratemap.pdf_normalized_tuning_curves[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values())), :] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(dummy_decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        # Get the peak locations for the tuning curves:
        sorted_pf_peak_location_list = [a_decoder.pf.ratemap.peak_tuning_curve_center_of_masses[np.array(list(a_sort_helper_neuron_id_to_IDX_dict.values()))] for a_decoder, a_sort_helper_neuron_id_to_IDX_dict in zip(dummy_decoders_dict.values(), sort_helper_neuron_id_to_sort_IDX_dicts)]
        img_extents_dict = {a_decoder_name:[a_decoder.pf.ratemap.xbin[0], 0, (a_decoder.pf.ratemap.xbin[-1]-a_decoder.pf.ratemap.xbin[0]), (float(len(sorted_neuron_IDs_lists[i]))-0.0)] for i, (a_decoder_name, a_decoder) in enumerate(dummy_decoders_dict.items()) } # these extents are  (x, y, w, h)
        sorted_neuron_IDs = sorted_neuron_IDs_lists[0]
        
        # Get colors for sorted neurons
        sort_helper_neuron_id_to_neuron_colors = sort_helper_neuron_id_to_neuron_colors_dicts[0]
        sort_helper_neuron_id_to_sort_IDX = sort_helper_neuron_id_to_sort_IDX_dicts[0]

        sorted_pf_tuning_curves = deepcopy(sorted_pf_tuning_curves[0])
        sorted_pf_peak_locations = deepcopy(sorted_pf_peak_location_list[0])

        img_extents = img_extents_dict[dummy_decoder_name_key]
        
        
        ## apply
        _out_data.sorted_neuron_IDs = sorted_neuron_IDs
        _out_data.sort_helper_neuron_id_to_neuron_colors = sort_helper_neuron_id_to_neuron_colors
        _out_data.sort_helper_neuron_id_to_sort_IDX = sort_helper_neuron_id_to_sort_IDX
        _out_data.sorted_pf_tuning_curves = sorted_pf_tuning_curves
        _out_data.unsorted_included_any_context_neuron_ids = deepcopy(included_any_context_neuron_ids)
        _out_data.sorted_pf_peak_locations = deepcopy(sorted_pf_peak_locations)
        _out_data.active_pfs_img_extents = deepcopy(img_extents)
        return _out_data

    @classmethod
    def _subfn_buildUI_base_decoder_debugger_data(cls, included_any_context_neuron_ids, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, _out_params: VisualizationParameters, decoder: BasePositionDecoder, line_height: float = 1.0):
        """ Builds UI """
        _out_data = cls._subfn_rebuild_sort_idxs(decoder, _out_data, included_any_context_neuron_ids)

        # title_str = f'pf1D_heatmap'
        title_str = _out_params.get('title_str', f'pf1D_heatmap')
        curr_curves = _out_data.sorted_pf_tuning_curves
        curr_pf_peak_locations = _out_data.sorted_pf_peak_locations
        
        extant_win = _out_ui.get('win', None)
        # extant_plot_item = _out_ui.get('plot_item', None)
        
        _out_plots.pf1D_heatmap = visualize_heatmap_pyqtgraph(curr_curves, title=title_str, show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=extant_win, defer_show=True)

        curr_win, curr_img = _out_plots.pf1D_heatmap
        if _out_params.debug_draw:
            curr_win.showAxes(True)

        _out_ui.text_items = {}
        _out_ui.order_location_lines = {}

        _temp_curr_out_colors_heatmap_image = []

        for cell_i, aclu in enumerate(_out_data.sorted_neuron_IDs):
            a_color_vector = _out_data.sort_helper_neuron_id_to_neuron_colors[aclu]
            
            text = SelectableTextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0))
            text.setPos(-1.0, (cell_i+1))
            curr_win.addItem(text)
            _out_ui.text_items[aclu] = text

            heatmap_base_color = pg.mkColor(a_color_vector)
            row_data = curr_curves[cell_i, :]
            out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=v) for v in row_data], is_255_array=False).T
            _temp_curr_out_colors_heatmap_image.append(out_colors_row)

            # pf_peak_indicator_lines ____________________________________________________________________________________________ #
            x_offset = curr_pf_peak_locations[cell_i]
            y_offset = float(cell_i)
            line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + line_height))
            line.setPen(pg.mkPen('white', width=2))
            curr_win.addItem(line)
            _out_ui.order_location_lines[aclu] = line

        ## END for cell_i, aclu in enumerate(_out_data.sorted_neuron_IDs)...
        out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
        out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)

        if enable_cell_colored_heatmap_rows:
            curr_img.updateImage(out_colors_heatmap_image_matrix)
        _out_data.out_colors_heatmap_image_matrix = out_colors_heatmap_image_matrix

        curr_img.setRect(_out_data.active_pfs_img_extents)

        return _out_data, _out_plots, _out_ui

    def buildUI_base_decoder_debugger_data(self):
        """Calls `_subfn_buildUI_directional_template_debugger_data` to build the UI"""
        self.plots_data, self.plots, self.ui = self._subfn_buildUI_base_decoder_debugger_data(included_any_context_neuron_ids=self.params.included_any_context_neuron_ids, debug_print=self.params.debug_print, enable_cell_colored_heatmap_rows=self.params.enable_cell_colored_heatmap_rows, _out_data=self.plots_data, _out_plots=self.plots, _out_ui=self.ui, _out_params=self.params, decoder=self.decoder)


    def update_clear_plotted_items(self):
        """ clears the plotted items for re-plotting 
        """
        curr_win, curr_img = self.plots.pf1D_heatmap
        # Update text items and lines
        for aclu, text_item in self.ui.text_items.items():
            curr_win.removeItem(text_item)
            text_item.deleteLater()
        self.ui.text_items.clear()
        
        for aclu, line_item in self.ui.order_location_lines.items():
            curr_win.removeItem(line_item)
        self.ui.order_location_lines.clear()


    def update_base_decoder_debugger_data(self, included_neuron_ids, solo_emphasized_aclus: Optional[List]=None, solo_override_num_spikes_weights: Optional[Dict]=None, solo_override_alpha_weights: Optional[Dict]=None):
        """Updates the visualization with new neuron selections"""
        self.params.solo_emphasized_aclus = solo_emphasized_aclus
        self.params.solo_override_alpha_weights = solo_override_alpha_weights

        if len(included_neuron_ids) == 0:
            print(f'clearing...')
            curr_win, curr_img = self.plots.pf1D_heatmap
            self.update_clear_plotted_items()
            curr_img.clear()
            
        else:
            ## valid:
            _out_data = self._subfn_rebuild_sort_idxs(self.decoder, self.plots_data, included_neuron_ids)
            
            curr_win, curr_img = self.plots.pf1D_heatmap
            self.update_clear_plotted_items()

            _temp_curr_out_colors_heatmap_image = []
            
            for cell_i, aclu in enumerate(_out_data.sorted_neuron_IDs):
                saturation_scale = 1.0
                value_scale_multiplier = 1.0
                alpha_scale_multiplier = 1.0
                spike_scale_size: int = 1
                
                if solo_emphasized_aclus is not None and aclu not in solo_emphasized_aclus:
                    saturation_scale = 0.02
                    value_scale_multiplier = 0.1
                    

                if solo_override_num_spikes_weights is not None:
                    spike_scale_size = solo_override_num_spikes_weights.get(aclu, 1) 

                if solo_override_alpha_weights is not None:
                    alpha_scale_multiplier = solo_override_alpha_weights.get(aclu, 1.0) * alpha_scale_multiplier
                    

                a_color_vector = _out_data.sort_helper_neuron_id_to_neuron_colors[aclu]
                text = SelectableTextItem(f"{int(aclu)}", color=build_adjusted_color(pg.mkColor(a_color_vector), value_scale=value_scale_multiplier, saturation_scale=saturation_scale, alpha_scale=alpha_scale_multiplier), anchor=(1,0))
                text.setPos(-1.0, (cell_i+1))
                curr_win.addItem(text)
                self.ui.text_items[aclu] = text

                heatmap_base_color = pg.mkColor(a_color_vector)
                row_data = _out_data.sorted_pf_tuning_curves[cell_i, :]
                out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=(v * value_scale_multiplier), saturation_scale=saturation_scale, alpha_scale=alpha_scale_multiplier) for v in row_data], is_255_array=False).T
                _temp_curr_out_colors_heatmap_image.append(out_colors_row)

                # pf_peak_indicator_lines ____________________________________________________________________________________________ #
                x_offset = _out_data.sorted_pf_peak_locations[cell_i]
                y_offset = float(cell_i)
                line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + 1.0))
                line.setPen(pg.mkPen(build_adjusted_color(pg.mkColor(a_color_vector), value_scale=value_scale_multiplier, saturation_scale=saturation_scale, alpha_scale=alpha_scale_multiplier), width=(2*spike_scale_size)))
                curr_win.addItem(line)
                self.ui.order_location_lines[aclu] = line
            # END for cell_i, aclu in enumerate(_out_data.sorted_neuron_IDs)...
            
            out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)

            if self.params.enable_cell_colored_heatmap_rows:
                curr_img.updateImage(out_colors_heatmap_image_matrix)
            self.plots_data.out_colors_heatmap_image_matrix = out_colors_heatmap_image_matrix
            
            curr_img.setRect(_out_data.active_pfs_img_extents)


    @classmethod
    def init_from_decoder(cls, a_decoder: BasePositionDecoder, included_all_neuron_ids=None, win=None, plot_item=None, solo_override_alpha_weights=None, title_str: str='test', **kwargs):
        fignum = kwargs.pop('fignum', None)
        if fignum is not None:
            print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

        defer_render = kwargs.pop('defer_render', False)
        debug_print: bool = kwargs.pop('debug_print', False)
        debug_draw: bool = kwargs.pop('debug_draw', False)

        enable_cell_colored_heatmap_rows: bool = kwargs.pop('enable_cell_colored_heatmap_rows', True)
        use_shared_aclus_only_templates: bool = kwargs.pop('use_shared_aclus_only_templates', False)
        

        # plot_item = kwargs.pop('plot_item', None)
        
        
        if included_all_neuron_ids is None:
            included_all_neuron_ids = deepcopy(a_decoder.neuron_IDs) ## all neuron_IDs in the decoder


        figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
        _out_data = RenderPlotsData(name=figure_name, decoder=deepcopy(a_decoder), out_colors_heatmap_image_matrix_dicts={}, sorted_neuron_IDs_lists=None, sort_helper_neuron_id_to_neuron_colors_dicts=None, sort_helper_neuron_id_to_sort_IDX_dicts=None, sorted_pf_tuning_curves=None, sorted_pf_peak_location_list=None, active_pfs_img_extents_dict=None, unsorted_included_any_context_neuron_ids=None, ref_decoder_name=None)
        _out_plots = RenderPlots(name=figure_name, pf1D_heatmaps=None)
        _out_params = VisualizationParameters(name=figure_name, enable_cell_colored_heatmap_rows=enable_cell_colored_heatmap_rows, use_shared_aclus_only_templates=use_shared_aclus_only_templates,
                                             debug_print=debug_print, debug_draw=debug_draw, included_any_context_neuron_ids=included_all_neuron_ids,
                                             solo_emphasized_aclus=None, solo_override_alpha_weights=solo_override_alpha_weights, title_str=title_str, **kwargs)


        if ((win is None) and (plot_item is None)):
            # build the window with the dock widget in it:
            root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho BaseTemplateDebuggingMixin Debugger: {figure_name}', defer_show=False)
            icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
            if icon is not None:
                root_dockAreaWindow.setWindowIcon(icon)
            # icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico"
            # root_dockAreaWindow.setWindowIcon(pg.QtGui.QIcon(icon_path))
            root_dockAreaWindow.resize(900, 700)
            _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, win=root_dockAreaWindow, plot_item=None, text_items_dict=None, order_location_lines_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)

        else:
            # extant plot item already
            app = pg.mkQApp()
            _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=None, win=win, plot_item=plot_item, text_items_dict=None, order_location_lines_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)

        
        ## Initialize Class here:
        _obj = cls(plots=_out_plots, plots_data=_out_data, ui=_out_ui, params=_out_params)

        _obj.buildUI_base_decoder_debugger_data()
        update_callback_fn = (lambda included_neuron_ids, **kwargs: _obj.update_base_decoder_debugger_data(included_neuron_ids, solo_emphasized_aclus=None, **kwargs))
        _obj.ui.on_update_callback = update_callback_fn
        return _obj









@metadata_attributes(short_name=None, tags=['gui', 'template'], input_requires=[], output_provides=[], uses=['visualize_heatmap_pyqtgraph'], used_by=['_display_directional_template_debugger'], creation_date='2023-12-11 10:24', related_items=[])
@define(slots=False, repr=False, eq=False)
class TemplateDebugger:
    """ TemplateDebugger displays four 1D heatmaps colored by cell for the tuning curves of PfND. Each shows the same tuning curves but they are sorted according to four different templates (RL_odd, RL_even, LR_odd, LR_even)
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger

    _out = TemplateDebugger.init_templates_debugger(track_templates) # , included_any_context_neuron_ids


    _out_ui.root_dockAreaWindow
    _out_ui.dock_widgets[a_decoder_name]

    ## Plots:
    _out_plots.pf1D_heatmaps[a_decoder_name] = visualize_heatmap_pyqtgraph(curr_curves, title=title_str, show_value_labels=False, show_xticks=False, show_yticks=False, show_colorbar=False, win=None, defer_show=True) # Sort to match first decoder (long_LR)
    # Adds aclu text labels with appropriate colors to y-axis: uses `sorted_shared_sort_neuron_IDs`:
    curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img


    
    ### Enable emphasizing/demphasizing aclus for TemplateDebugger
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import TemplateDebugger

    template_debugger: TemplateDebugger = _out['obj']

    # a_decoder_idx = 0
    # a_decoder_name = 'long_LR'


    # template_debugger.params.solo_emphasized_aclus = None # remove custom emphasis/demphasis
    # template_debugger.params.solo_emphasized_aclus = [31, 26, 14, 29, 11]


    # plots_data = template_debugger.plots_data
    # template_debugger.plots_data.unsorted_included_any_context_neuron_ids


    # out_colors_heatmap_image_matrix_dicts = plots_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name]
    # sort_helper_neuron_id_to_neuron_colors_dict = plots_data['sort_helper_neuron_id_to_neuron_colors_dicts'][a_decoder_idx] # Only one for all decoders, which is good actually
    # sorted_neuron_IDs_list = plots_data['sorted_neuron_IDs_lists'][a_decoder_idx] # Only one for all decoders, which is good actually

    # out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name]
    #  'sort_helper_neuron_id_to_sort_IDX_dicts'
    # 'sort_helper_neuron_id_to_neuron_colors_dicts'
    # 'sorted_neuron_IDs_lists'
    # sort_helper_neuron_id_to_neuron_colors_dict

    template_debugger.update_cell_emphasis(solo_emphasized_aclus=[31, 26, 14, 29, 11])
    template_debugger.update_cell_emphasis(solo_emphasized_aclus=None)

    template_debugger.params.included_any_context_neuron_ids

    # adjusted_sort_helper_neuron_id_to_neuron_colors_dict = deepcopy(sort_helper_neuron_id_to_neuron_colors_dict) # a list of four dicts in it for some reason??
    # adjusted_sort_helper_neuron_id_to_neuron_colors_dict

    template_debugger.params.solo_emphasized_aclus = [31, 26, 14, 29, 11]
    # demphasized_aclus = ## build from the non-solo_emphasized_aclus
    # len(adjusted_sort_helper_neuron_id_to_neuron_colors_dicts)

    adjusted_sort_helper_neuron_id_to_neuron_colors_dict = {}
    for aclu, a_color in sort_helper_neuron_id_to_neuron_colors_dict.items():
        if aclu in template_debugger.params.solo_emphasized_aclus:
            # original color:
            adjusted_sort_helper_neuron_id_to_neuron_colors_dict[aclu] = deepcopy(a_color)
        else:
            # desaturate the color:
            desaturated_color = build_adjusted_color(deepcopy(a_color), saturation_scale=0.02, value_scale=0.1)
            adjusted_sort_helper_neuron_id_to_neuron_colors_dict[aclu] = desaturated_color
            

        
        
        
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

    @property
    def decoders_dict(self) -> Dict:
        return self.track_templates.get_decoders_dict() # decoders_dict = {'long_LR': track_templates.long_LR_decoder, 'long_RL': track_templates.long_RL_decoder, 'short_LR': track_templates.short_LR_decoder, 'short_RL': track_templates.short_RL_decoder, }


    # @property
    # def active_epoch_tuple(self) -> tuple:
    #     """ returns a namedtuple describing the single epoch corresponding to `self.active_epoch_IDX`. """
    #     a_df_idx = self.active_epochs_df.index.to_numpy()[self.active_epoch_IDX]
    #     curr_epoch_df = self.active_epochs_df[(self.active_epochs_df.index == a_df_idx)] # this +1 here makes zero sense
    #     curr_epoch = list(curr_epoch_df.itertuples(name='EpochTuple'))[0]
    #     return curr_epoch

    # Plot Properties ____________________________________________________________________________________________________ #
    
    @property
    def root_dockAreaWindow(self) -> PhoDockAreaContainingWindow:
        return self.ui.root_dockAreaWindow

    @property
    def dock_widgets(self) -> Dict[types.DecoderName, Tuple[pg.PlotWidget, Dock]]:
        return self.ui.dock_widgets
    
    @property
    def pf1D_heatmaps(self) -> Dict[types.DecoderName, Tuple[pg.PlotWidget, pg.ImageItem]]:
        """ 
        curr_win, curr_img = _out_plots.pf1D_heatmaps[a_decoder_name] # win, img
        
        """
        return self.plots.pf1D_heatmaps

    @property
    def order_location_lines_dict(self) -> Dict[types.DecoderName, Dict[types.aclu_index, pg.TextItem]]:
        """ the white vertical lines that indicate the peak location of each place cell.
        """
        return self.ui.order_location_lines_dict
    


    # Initializer ________________________________________________________________________________________________________ #
    
    @function_attributes(short_name=None, tags=['init', 'buildUI'], input_requires=[], output_provides=[], uses=['buildUI_directional_template_debugger_data'], used_by=[], creation_date='2024-10-21 19:22', related_items=[])
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
            track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only

        
        
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
                                             debug_print=debug_print, debug_draw=debug_draw, use_incremental_sorting=use_incremental_sorting, enable_pf_peak_indicator_lines=enable_pf_peak_indicator_lines, included_any_context_neuron_ids=included_any_context_neuron_ids,
                                             solo_emphasized_aclus=None, **kwargs)
                
        # build the window with the dock widget in it:
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=f'Pho Directional Template Debugger: {figure_name}', defer_show=False)
        icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
        if icon is not None:
            root_dockAreaWindow.setWindowIcon(icon)
        # icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico"
        # root_dockAreaWindow.setWindowIcon(pg.QtGui.QIcon(icon_path))

        _out_ui = PhoUIContainer(name=figure_name, app=app, root_dockAreaWindow=root_dockAreaWindow, text_items_dict=None, order_location_lines_dict=None, dock_widgets=None, dock_configs=None, on_update_callback=None)
        
        root_dockAreaWindow.resize(900, 700)

        ## Initialize Class here:
        _obj = cls(plots=_out_plots, plots_data=_out_data, ui=_out_ui, params=_out_params)

        _obj.buildUI_directional_template_debugger_data()
        update_callback_fn = (lambda included_neuron_ids, **kwargs: _obj.update_directional_template_debugger_data(included_neuron_ids, solo_emphasized_aclus=None, **kwargs))
        _obj.ui.on_update_callback = update_callback_fn
        ## build on-clicked callback:
        _obj._build_internal_callback_functions()        
        print(f'done init')
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

    def save_figure(self, shared_output_file_prefix = f'output/2025-07-21'): # export_file_base_path: Path = Path(f'output').resolve()
        """ captures: epochs_editor, _out_pf1D_heatmaps

        TODO: note output paths are currently hardcoded. Needs to add the animal's context at least. Probably needs to be integrated into pipeline.
        import pyqtgraph as pg
        import pyqtgraph.exporters
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
        """
        ## Get main laps plotter:
        # print_keys_if_possible('_out', _out, max_depth=4)
        # plots = _out['plots']

        # ## Already have: epochs_editor, _out_pf1D_heatmaps
        # epochs_editor = self.ui[0]

        # shared_output_file_prefix = f'output/2023-11-20'
        # # print(list(plots.keys()))
        # # pg.GraphicsLayoutWidget
        # main_graphics_layout_widget = epochs_editor.plots.win
        # export_file_path = Path(f'{shared_output_file_prefix}_test_main_position_laps_line_plot').with_suffix('.svg').resolve()
        # export_pyqtgraph_plot(main_graphics_layout_widget, savepath=export_file_path) # works

        _out_pf1D_heatmaps = self.plots.pf1D_heatmaps
        # _out_pf1D_heatmaps = graphics_output_dict['plots']
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


    def _build_internal_callback_functions(self, debug_print: bool = False):
        """ 
        view_box.scene().sigMouseMoved.connect(_test_on_mouse_moved)
        
        """
        if debug_print:
            print(f'_build_internal_callback_functions()')
        connections = self.ui.setdefault('connections', {})
        sigMouseClickedCallbackDict = connections.setdefault('sigMouseClicked', {})
        sigMouseMovedCallbackDict = connections.setdefault('sigMouseMoved', {})
        # if (sigMouseClickedCallbackDict is not None) and (sigMouseMovedCallbackDict is not None):
        #     print(f'sigMouseMovedCallback and sigMouseMovedCallback already exist! Skipping.')
        # else:
        for a_decoder_name, (curr_win, curr_img) in self.pf1D_heatmaps.items():
            if debug_print:
                print(f'a_decoder_name: {a_decoder_name}')
                print(f'\t curr_win: {curr_win}')
                print(f'\t curr_img: {curr_img}')
            view_box: pg.ViewBox = curr_win.getViewBox()
            if debug_print:
                print(f'\t view_box: {view_box}')
            a_scene: pg.GraphicsScene = view_box.scene()
            if debug_print:
                print(f'\t a_scene: {a_scene}')
            a_scene.setClickRadius(4.0)
            # # mouse Clicked:
            # sigMouseClickedCallback = sigMouseClickedCallbackDict.get(a_decoder_name, None)
            # if sigMouseClickedCallback is not None:
            #     print(f'\tdisconnecting sigMouseClickedCallback for {a_decoder_name}..')
            #     view_box.scene().sigMouseClicked.disconnect(sigMouseClickedCallback) ## disconnect
                    
            # self.ui.connections['sigMouseClicked'][a_decoder_name] = view_box.scene().sigMouseClicked.connect(self.on_mouse_click)

            # mouse Clicked:
            sigMouseClickedCallback = sigMouseClickedCallbackDict.get(a_decoder_name, None)
            if sigMouseClickedCallback is not None:
                if debug_print:
                    print(f'\tdisconnecting sigMouseClickedCallback for {a_decoder_name}..')
                curr_win.sigMouseClicked.disconnect(sigMouseClickedCallback) ## disconnect
                    
            self.ui.connections['sigMouseClicked'][a_decoder_name] = curr_win.sigMouseClicked.connect(self.on_mouse_click)



            # ## mouse Moved:
            # sigMouseMovedCallback = sigMouseMovedCallbackDict.get(a_decoder_name, None)
            # if sigMouseMovedCallback is not None:
            #     print(f'\tdisconnecting sigMouseMovedCallback for {a_decoder_name}..')
            #     view_box.scene().sigMouseMoved.disconnect(sigMouseMovedCallback) ## disconnect
            
            # self.ui.connections['sigMouseMoved'][a_decoder_name] = view_box.scene().sigMouseMoved.connect(self.on_mouse_moved)
            a_scene.setClickRadius(4.0)
            if debug_print:
                print(f'\t "{a_decoder_name}" connections done.')
            
        ## add selection changed callbacks
        for a_decoder_name, a_text_items_dict in self.ui.text_items_dict.items():
            for aclu, a_text_item in a_text_items_dict.items():
                a_text_item.sigSelectedChanged.connect(self.on_change_selection)

        # custom_on_mouse_clicked callback ___________________________________________________________________________________ #

        self.params.on_mouse_clicked_callback_fn_dict = {
            'custom_on_mouse_clicked': self.custom_on_mouse_clicked,
        }
        if debug_print:
            print(f'done _build_internal_callback_functions()')
        

    # ==================================================================================================================== #
    # Extracted Functions:                                                                                                 #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['rebuild'], input_requires=[], output_provides=[], uses=['paired_incremental_sort_neurons'], used_by=['_subfn_buildUI_directional_template_debugger_data', '_subfn_update_directional_template_debugger_data'], creation_date='2024-10-21 19:19', related_items=[])
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
        
        # ymin_ymax_tuple_dict = {a_decoder_name:[(float(cell_i), (float(cell_i) + line_height)) for a_decoder.] for i, (a_decoder_name, a_decoder) in enumerate(decoders_dict.items()) } # these extents are  (x, y, w, h)
        
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
    @function_attributes(short_name=None, tags=['buildUI'], input_requires=[], output_provides=[], uses=['visualize_heatmap_pyqtgraph', 'cls._subfn_rebuild_sort_idxs'], used_by=[], creation_date='2024-10-21 19:20', related_items=[])
    @classmethod
    def _subfn_buildUI_directional_template_debugger_data(cls, included_any_context_neuron_ids, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, _out_params: VisualizationParameters, decoders_dict: Dict, line_height: float = 1.0):
        """ Builds UI """
        print(f'._subfn_buildUI_directional_template_debugger_data(...)')
        _out_data = cls._subfn_rebuild_sort_idxs(decoders_dict, _out_data, use_incremental_sorting=use_incremental_sorting, included_any_context_neuron_ids=included_any_context_neuron_ids)
        # Unpack the updated _out_data:
        sort_helper_neuron_id_to_neuron_colors_dicts = _out_data.sort_helper_neuron_id_to_neuron_colors_dicts
        sorted_pf_tuning_curves = _out_data.sorted_pf_tuning_curves
        sorted_pf_peak_location_list = _out_data.sorted_pf_peak_location_list
        _out_data.active_pfs_ymin_ymax_tuple_list_dict = {}
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
            # curr_win.setObjectName(a_decoder_name)
            curr_win.item_data = {'decoder_name': a_decoder_name, 'decoder_idx': i}
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

            ## build selection helper:            
            # a_decoder_color_map: Dict = sort_helper_neuron_id_to_neuron_colors_dicts[i] # 34 (n_neurons)
            a_decoder_ymin_ymax_tuple_list = [(float(cell_i), (float(cell_i) + line_height)) for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items())]
            _out_data.active_pfs_ymin_ymax_tuple_list_dict[a_decoder_name] = np.array(a_decoder_ymin_ymax_tuple_list)
            

            for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
                # anchor=(1,0) specifies the item's upper-right corner is what setPos specifies. We switch to right vs. left so that they are all aligned appropriately.
                # text = pg.TextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0)) # , angle=15
                text = SelectableTextItem(f"{int(aclu)}", color=pg.mkColor(a_color_vector), anchor=(1,0))
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
                    
                    half_line_height = line_height / 2.0 # to compensate for middle
                    # line = QtGui.QGraphicsLineItem(x_offset, (y_offset - half_line_height), x_offset, (y_offset + half_line_height)) # (xstart, ystart, xend, yend)
                    line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + line_height)) # (xstart, ystart, xend, yend)
                    # line = pg.InfiniteLine(pos=(x_offset, float(cell_i+1)), angle=90, movable=False)
                    line.setPen(pg.mkPen('white', width=2))  # Set color and width of the line
                    curr_win.addItem(line)
                    # line.setPos(pg.Point(x_offset, (y_offset + (line_height / 2.0)))) # Adjust the height of the line if needed
                    _out_ui.order_location_lines_dict[a_decoder_name][aclu] = line # add to the map


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
    def _subfn_update_directional_template_debugger_data(cls, included_neuron_ids: NDArray, use_incremental_sorting: bool, debug_print: bool, enable_cell_colored_heatmap_rows: bool, _out_data: RenderPlotsData, _out_plots: RenderPlots, _out_ui: PhoUIContainer, _out_params: VisualizationParameters, decoders_dict: Dict):
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

        _out_data.included_any_context_neuron_ids = deepcopy(included_neuron_ids)

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
            curr_win.item_data = {'decoder_name': a_decoder_name, 'decoder_idx': i}
            
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
            
            custom_solo_emphasized_aclus = _out_params.get('solo_emphasized_aclus', None)
            custom_solo_visible_aclus = _out_params.get('solo_visible_aclus', None)

            n_total_cells: int = len(curr_pf_peak_locations)
            print(f'n_total_cells: {n_total_cells}')
            if (custom_solo_visible_aclus is None):
                n_visible_cells: int = n_total_cells ## all cells visible
                line_height: float = 1.0
            else:
                # all_aclus = list(a_decoder_color_map.keys())
                # included_visible_aclus = [aclu for aclu in all_aclus if aclu in (custom_solo_emphasized_aclus or [])]
                # included_visible_aclus = [aclu for aclu in _out_data.included_any_context_neuron_ids if aclu in custom_solo_emphasized_aclus]
                # n_visible_cells: int = len(included_visible_aclus)
                print(f'custom_solo_visible_aclus: {custom_solo_visible_aclus}')
                # n_visible_cells: int = np.shape(curr_data)[0]
                n_visible_cells: int = len(np.unique(custom_solo_visible_aclus))
                print(f'n_visible_cells: {n_visible_cells}')
                # line_height: float = (1.0/float(n_total_cells)) * float(n_visible_cells)
                # line_height: float = float(n_total_cells)/float(n_visible_cells)
                line_height: float = 1.0
                print(f'line_height: {line_height}')

            visible_cell_i = 0
            for cell_i, (aclu, a_color_vector) in enumerate(a_decoder_color_map.items()):
                ## Apply emphasis/demphasis:
                
                ## default to full saturation/value scale:
                saturation_scale = 1.0
                value_scale_multiplier = 1.0
                
                if custom_solo_emphasized_aclus is not None:
                    ## apply custom saturation/desaturation here
                    if aclu not in custom_solo_emphasized_aclus:
                        ## demphasize
                        saturation_scale = 0.02
                        value_scale_multiplier = 0.1

                if custom_solo_visible_aclus is not None:
                    ## apply custom visibility here
                    if aclu not in custom_solo_visible_aclus:
                        ## hide
                        continue

                # Create a new text item:
                text = SelectableTextItem(f"{int(aclu)}", color=build_adjusted_color(pg.mkColor(a_color_vector), value_scale=value_scale_multiplier, saturation_scale=saturation_scale), anchor=(1,0))
                # text.setPos(-1.0, (cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
                text.setPos(-1.0, (visible_cell_i+1)) # the + 1 is because the rows are seemingly 1-indexed?
                curr_win.addItem(text)
                _out_ui.text_items_dict[a_decoder_name][aclu] = text # add the TextItem to the map

                # modulate heatmap color for this row (`curr_data[i, :]`):
                heatmap_base_color = pg.mkColor(a_color_vector)
                out_colors_row = DataSeriesColorHelpers.qColorsList_to_NDarray([build_adjusted_color(heatmap_base_color, value_scale=(v * value_scale_multiplier), saturation_scale=saturation_scale) for v in curr_data[cell_i, :]], is_255_array=False).T # (62, 4)
                _temp_curr_out_colors_heatmap_image.append(out_colors_row)

                # Add vertical lines
                if _out_params.enable_pf_peak_indicator_lines:
                    x_offset = curr_pf_peak_locations[cell_i]
                    # y_offset = float(cell_i)
                    y_offset = float(visible_cell_i) 
                    ## INPUTS: line_height
                    
                    line = QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, (y_offset + line_height)) # (xstart, ystart, xend, yend)
                    # line.setPen(pg.mkPen('white', width=2))  # Set color and width of the line
                    line.setPen(pg.mkPen(build_adjusted_color(pg.mkColor(a_color_vector), value_scale=value_scale_multiplier, saturation_scale=saturation_scale), width=2))  # Set color and width of the line
                    curr_win.addItem(line)
                    _out_ui.order_location_lines_dict[a_decoder_name][aclu] = line # add to the map
                    # # Old update-based way:
                    # line = _out_ui.order_location_lines_dict[a_decoder_name][aclu] # QtGui.QGraphicsLineItem(x_offset, y_offset, x_offset, line_height)
                    # line.setLine(x_offset, y_offset, x_offset, (y_offset + line_height)) # (xstart, ystart, xend, yend)
                    
                ## END if _out_params.enable_pf_peak_indicator_lines...
                visible_cell_i += 1
                
            # end `for cell_i, (aclu, a_color_vector)`

            ## Build the colored heatmap:
            out_colors_heatmap_image_matrix = np.stack(_temp_curr_out_colors_heatmap_image, axis=0)
            # Ensure the data is in the correct range [0, 1]
            out_colors_heatmap_image_matrix = np.clip(out_colors_heatmap_image_matrix, 0, 1)
            if enable_cell_colored_heatmap_rows:
                curr_img.updateImage(out_colors_heatmap_image_matrix) #, xvals=curr_xbins, use the color image only if `enable_cell_colored_heatmap_rows==True`
            _out_data['out_colors_heatmap_image_matrix_dicts'][a_decoder_name] = out_colors_heatmap_image_matrix

            # Set the extent to map pixels to x-locations        
            if (n_visible_cells < n_total_cells):
                ## some cells hidden
                _modified_visible_rect = deepcopy(_out_data.active_pfs_img_extents_dict[a_decoder_name])
                _modified_visible_rect[-1] = float(n_visible_cells)
                curr_img.setRect(_modified_visible_rect)                
            else:
                ## normal visibility:
                curr_img.setRect(_out_data.active_pfs_img_extents_dict[a_decoder_name])
            
        # end `for i, (a_decoder_name, a_decoder)`

        return _out_data, _out_plots, _out_ui

    @function_attributes(short_name=None, tags=['buildUI'], input_requires=[], output_provides=[], uses=['_subfn_buildUI_directional_template_debugger_data'], used_by=[], creation_date='2024-10-21 19:18', related_items=[])
    def buildUI_directional_template_debugger_data(self):
        """Calls `_subfn_buildUI_directional_template_debugger_data` to build the UI and then updates the member variables."""
        self.plots_data, self.plots, self.ui = self._subfn_buildUI_directional_template_debugger_data(self.params.included_any_context_neuron_ids, self.params.use_incremental_sorting, self.params.debug_print, self.params.enable_cell_colored_heatmap_rows, self.plots_data, self.plots, self.ui, _out_params=self.params, decoders_dict=self.decoders_dict)

    @function_attributes(short_name=None, tags=['update'], input_requires=[], output_provides=[], uses=['_subfn_update_directional_template_debugger_data'], used_by=[], creation_date='2024-10-21 19:21', related_items=[])
    def update_directional_template_debugger_data(self, included_neuron_ids, solo_emphasized_aclus: Optional[List]=None, solo_visible_aclus: Optional[List]=None):
        """Calls `_subfn_update_directional_template_debugger_data` to build the UI and then updates the member variables."""
        self.params.solo_emphasized_aclus = solo_emphasized_aclus ## reset emphasis on update
        self.params.solo_visible_aclus = solo_visible_aclus ## reset visibility on update
        self.plots_data, self.plots, self.ui = self._subfn_update_directional_template_debugger_data(included_neuron_ids, self.params.use_incremental_sorting, self.params.debug_print, self.params.enable_cell_colored_heatmap_rows, self.plots_data, self.plots, self.ui, _out_params=self.params, decoders_dict=self.decoders_dict)



    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #
    def update_cell_emphasis(self, solo_emphasized_aclus: List):
        """ updates the display of each cell to only emphasize the `solo_emphasized_aclus`, dimming all the others. 
        """
        self.update_directional_template_debugger_data(included_neuron_ids=self.params.included_any_context_neuron_ids, solo_emphasized_aclus=solo_emphasized_aclus)

    def reset_cell_emphasis(self):
        """ resets the emphasis to normal (no special emphasis/demphasis) """
        self.update_cell_emphasis(solo_emphasized_aclus=None)


    def update_cell_visibility(self, solo_visible_aclus: List):
        """ updates the display of each cell to only include the `solo_visible_aclus` as visible, completely hiding all the others. """
        self.update_directional_template_debugger_data(included_neuron_ids=self.params.included_any_context_neuron_ids, solo_visible_aclus=solo_visible_aclus)

    def reset_cell_visibility(self):
        """ resets the emphasis to normal (no special emphasis/demphasis) """
        self.update_cell_visibility(solo_visible_aclus=None)


# ==================================================================================================================== #
# 2023-12-20 Mouse Tracking Diversion/Failed                                                                           #
# ==================================================================================================================== #

    # def mouseMoved(self, a_decoder_name: str, evt):
    #     """ captures `label` """
    #     pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    #     print(f'mouseMoved(a_decoder_name: {a_decoder_name}, evt: {evt})')
        
    #     # a_plot_widget, an_image_item = self.plots['pf1D_heatmaps'][a_decoder_name] # (pyqtgraph.widgets.PlotWidget.PlotWidget, pyqtgraph.graphicsItems.ImageItem.ImageItem)
    #     # # a_view_box = an_image_item.getViewBox()
    #     # vb = a_plot_widget.vb
    #     # if a_plot_widget.sceneBoundingRect().contains(pos):
    #     #     mousePoint = vb.mapSceneToView(pos)
    #     #     index = int(mousePoint.x())
    #     #     print(f'\tmousePoint.x(): {mousePoint.x()}, \tindex: {index}')
    #     #     # if index > 0 and index < len(data1):
    #     #     #     print(f"<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
    #     #         # if self.label is not None:
    #     #         #     self.label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
    #     #     # self.vLine.setPos(mousePoint.x())
    #     #     # self.hLine.setPos(mousePoint.y())


    # def on_mouse_click(self, event, decoder_name=None):
    # def on_mouse_click(self, event):
    def on_mouse_click(self, custom_plot_widget, event):
        debug_print: bool = self.params.debug_enabled and False
        if debug_print:
            print(f'self.on_mouse_click(...)')
            print(f'\tcustom_plot_widget: {custom_plot_widget}')
            print(f'\tevent: {event}')
        on_mouse_clicked_callback_fn_dict = self.params.get('on_mouse_clicked_callback_fn_dict', {})
        for a_callback_name, a_callback_fn in on_mouse_clicked_callback_fn_dict.items():
            a_callback_fn(self, custom_plot_widget, event)
        if debug_print:
            print('\tend.')
        # pos = event.scenePos()
        # print(f'self.on_mouse_click(event: {event})')
        # print(f'on_mouse_click(event: {event}, decoder_name: {decoder_name})')
        
        # if plot.sceneBoundingRect().contains(pos):
        #     mouse_point = plot.vb.mapSceneToView(pos)
        #     print(f"Clicked at: x={mouse_point.x()}, y={mouse_point.y()}")
            

    def on_mouse_moved(self, custom_plot_widget, event_pos):
        if self.params.debug_enabled:
            print(f'self.on_mouse_moved(...)')
            print(f'\tcustom_plot_widget: {custom_plot_widget}')
            print(f'\tevent_pos: {event_pos}')
        on_mouse_moved_callback_fn_dict = self.params.get('on_mouse_moved_callback_fn_dict', {})
        for a_callback_name, a_callback_fn in on_mouse_moved_callback_fn_dict.items():
            a_callback_fn(self, custom_plot_widget, event_pos)
        if self.params.debug_enabled:
            print('\tend.')
        
    @classmethod
    def custom_on_mouse_clicked(cls, self, custom_plot_widget, event):
        """ callback on mouse clicked """
        debug_print: bool = False
        if debug_print:
            print(f'custom_on_mouse_clicked(event: {event})')
        if not hasattr(event, 'scenePos'):
            if debug_print:
                print(f'not MouseClickEvent. skipping.')
            return
        else:    
            pos = event.scenePos() # 'QMouseEvent' object has no attribute 'scenePos'
            if debug_print:
                print(f'\tscenePos: {pos}')
                print(f'\tscreenPos: {event.screenPos()}')
                print(f'\tpos: {event.pos()}')
                
            item_data = custom_plot_widget.item_data
            if debug_print:
                print(f'\titem_data: {item_data}')
            found_decoder_idx = item_data.get('decoder_idx', None)
            found_decoder_name = item_data.get('decoder_name', None)
                        
            if ((found_decoder_idx is None) and (found_decoder_name is None)):
                print(f'WARNING: could not find correct decoder name/idx')
            else:
                if debug_print:
                    print(f'found valid decoder: found_decoder_name: "{found_decoder_name}", found_decoder_idx" {found_decoder_idx}')
                a_win, an_img_item = self.pf1D_heatmaps[found_decoder_name]
                mouse_point = a_win.getViewBox().mapSceneToView(pos)
                if debug_print:
                    print(f"Clicked at: x={mouse_point.x()}, y={mouse_point.y()}")
                found_y_point: float = mouse_point.y()
                ## round down
                found_y_idx: int = int(found_y_point)
                if debug_print:
                    print(f'found_y_idx: {found_y_idx}')
                found_aclu: int = self.plots_data.sorted_neuron_IDs_lists[found_decoder_idx][found_y_idx]
                if debug_print:
                    print(f'found_aclu: {found_aclu}')
                prev_selected_aclus = self.get_any_decoder_selected_aclus().tolist()
                prev_selected_aclus.append(found_aclu)
                self.set_selected_aclus_for_all_decoders(any_selected_aclus=prev_selected_aclus)


    # ==================================================================================================================== #
    # Other Functions                                                                                                      #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['selection', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 01:11', related_items=[])
    def on_change_selection(self, a_text_item, new_is_selected: bool):
        """ called when one of the aclu subplots selection changes 
        """
        if self.params.debug_enabled:
            print(f'on_change_selection(a_text_item: {a_text_item}, new_is_selected: {new_is_selected})')
        pass

    @function_attributes(short_name=None, tags=['selection', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 01:11', related_items=[])
    def get_selected_aclus(self, return_only_selected_aclus: bool=True):
        """ gets the user-selected aclus """
        # is_aclu_selected = []
        selected_aclus_list_dict = {}
        is_aclu_selected_decoder_dict = {}
        for a_decoder_name, a_text_items_dict in self.ui.text_items_dict.items():
            is_aclu_selected_decoder_dict[a_decoder_name] = {}
            selected_aclus_list_dict[a_decoder_name] = []
            for aclu, a_text_item in a_text_items_dict.items():
                if return_only_selected_aclus:
                    if a_text_item.is_selected:
                        selected_aclus_list_dict[a_decoder_name].append(aclu)
                else:
                    is_aclu_selected_decoder_dict[a_decoder_name][aclu] = a_text_item.is_selected

        if return_only_selected_aclus:
            return selected_aclus_list_dict
        else:
            ## return map from aclu to is_selected
            return is_aclu_selected_decoder_dict
        
    @function_attributes(short_name=None, tags=['selection', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 08:43', related_items=[])
    def get_any_decoder_selected_aclus(self) -> NDArray:
        """ gets the user-selected aclus for any decoder
        
        _out.get_any_decoder_selected_aclus()
        
        """
        curr_selected_aclus_dict = self.get_selected_aclus(return_only_selected_aclus=True) # 'long_LR': [45, 24, 18, 35, 32], 'long_RL': [], 'short_LR': [], 'short_RL': []}
        return union_of_arrays(*list(curr_selected_aclus_dict.values()))
    
    @function_attributes(short_name=None, tags=['selection', 'aclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-22 08:44', related_items=[])
    def synchronize_selected_aclus_between_decoders_if_needed(self):
        """ Synchronizes all common selections across each decoder
        
        """
        synchronize_selected_aclus_across_decoders: bool = self.params.setdefault('synchronize_selected_aclus_across_decoders', True)
        curr_selected_aclus_dict = self.get_selected_aclus(return_only_selected_aclus=True) # 'long_LR': [45, 24, 18, 35, 32], 'long_RL': [], 'short_LR': [], 'short_RL': []}
        if synchronize_selected_aclus_across_decoders:
            any_decoder_selectioned_aclus = union_of_arrays(*list(curr_selected_aclus_dict.values()))
            for a_decoder_name, a_text_items_dict in self.ui.text_items_dict.items():
                for aclu in any_decoder_selectioned_aclus:
                    a_text_item = a_text_items_dict.get(aclu, None)
                    if a_text_item is not None:
                        # set the selection
                        a_text_item.perform_update_selected(new_is_selected=True)
                        

    def set_selected_aclus_for_all_decoders(self, any_selected_aclus: NDArray):
        """ forcibly sets the selections across all decoders to only the `any_selected_aclus`
        
        Usage:
            _out.set_selected_aclus_for_all_decoders(any_selected_aclus=[18, 24, 31, 32, 35, 45])
        
        """
        if any_selected_aclus is None:
            any_selected_aclus = [] # no selections 
        if not isinstance(any_selected_aclus, NDArray):
            any_selected_aclus = np.array(any_selected_aclus)

        for a_decoder_name, a_text_items_dict in self.ui.text_items_dict.items():
            for aclu, a_text_item in a_text_items_dict.items():
                # set the selection
                a_text_item.perform_update_selected(new_is_selected=(aclu in any_selected_aclus))
                    

    def clear_selected_aclus_for_all_decoders(self):
        self.set_selected_aclus_for_all_decoders(any_selected_aclus=None)


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