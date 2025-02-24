# ==================================================================================================================== #
# 2025-01-21 - Bin-by-bin decoding examples                                                                            #
# ==================================================================================================================== #
import numpy as np
import pyqtgraph as pg
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import pyphoplacecellanalysis.General.type_aliases as types
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from attrs import field, Factory, define
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr


# @define(slots=False, eq=False)
@metadata_attributes(short_name=None, tags=['pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-24 17:22', related_items=[])
class BinByBinDecodingDebugger:
    """ handles displaying the process of debugging decoding for each time bin
    
    Usage:    
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.BinByBinDecodingDebugger import BinByBinDecodingDebugger 
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates

        # Example usage:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping() 
        global_laps_epochs_df = global_laps.to_dataframe()
        global_laps_epochs_df

        ## INPUTS: 
        time_bin_size: float = 0.250
        a_lap_id: int = 9
        a_decoder_name = 'long_LR'

        ## COMPUTED: 
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)
        (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_spike_counts_and_decoder_outputs(track_templates=track_templates, global_laps_epochs_df=global_laps_epochs_df, spikes_df=global_spikes_df, a_decoder_name=a_decoder_name, time_bin_size=time_bin_size)
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_time_binned_decoder_debug_plots(a_decoder=a_decoder, a_lap_id=a_lap_id, _out_decoded_time_bin_edges=_out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n=_out_decoded_active_p_x_given_n, _out_decoded_unit_specific_time_binned_spike_counts=_out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists=_out_decoded_active_unit_lists, _out_decoded_active_plots_data=_out_decoded_active_plots_data, debug_print=True)
        print(f"Returned window: {win}")
        print(f"Returned decoder objects: {out_pf1D_decoder_template_objects}")


    """
    plots: RenderPlots = field(repr=keys_only_repr)
    plots_data: RenderPlotsData = field(repr=keys_only_repr)
    ui: PhoUIContainer = field(repr=keys_only_repr)
    params: VisualizationParameters = field(repr=keys_only_repr)

    # time_bin_size: float = field(default=0.500) # 500ms
    # spikes_df: pd.DataFrame = field()
    # global_laps_epochs_df: pd.DataFrame = field()
    @classmethod
    def build_spike_counts_and_decoder_outputs(cls, track_templates, global_laps_epochs_df, spikes_df, a_decoder_name='long_LR', time_bin_size=0.500, debug_print=False):
        """
            a_decoder_name: types.DecoderName = 'long_LR'
            
        """
        ## Get a specific decoder
        
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)

        neuron_IDs = deepcopy(a_decoder.neuron_IDs)
        spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(neuron_IDs)
        unique_units = np.unique(spikes_df['aclu'])
        _out_decoded_time_bin_edges = {}
        _out_decoded_unit_specific_time_binned_spike_counts = {}
        _out_decoded_active_unit_lists = {}
        _out_decoded_active_p_x_given_n = {}
        _out_decoded_active_plots_data: Dict[str, RenderPlotsData]  = {}

        for a_row in global_laps_epochs_df.itertuples():
            t_start = a_row.start
            t_end = a_row.stop
            time_bin_edges = np.arange(t_start, (t_end + time_bin_size), time_bin_size)
            n_time_bins = len(time_bin_edges) - 1
            assert n_time_bins > 0
            _out_decoded_time_bin_edges[a_row.lap_id] = time_bin_edges
            unit_specific_time_binned_spike_counts = np.array([
                np.histogram(spikes_df.loc[spikes_df['aclu'] == unit, 't_rel_seconds'], bins=time_bin_edges)[0]
                for unit in unique_units
            ])
            all_lap_active_units_list = []
            active_units_list = []
            for a_time_bin_idx in np.arange(n_time_bins):
                unit_spike_counts = np.squeeze(unit_specific_time_binned_spike_counts[:, a_time_bin_idx])
                normalized_unit_spike_counts = (unit_spike_counts / np.sum(unit_spike_counts))
                active_unit_idxs = np.where(unit_spike_counts > 0)[0]
                active_units = neuron_IDs[active_unit_idxs]
                active_aclu_spike_counts_dict = dict(zip(active_units, unit_spike_counts[active_unit_idxs]))
                active_units_list.append(active_aclu_spike_counts_dict)
                all_lap_active_units_list.extend(active_units)

            _out_decoded_active_unit_lists[a_row.lap_id] = active_units_list
            _out_decoded_unit_specific_time_binned_spike_counts[a_row.lap_id] = unit_specific_time_binned_spike_counts

            # all_lap_active_units_list = np.unique(list(Set(all_lap_active_units_list)))
            all_lap_active_units_list = np.unique(all_lap_active_units_list)
            if debug_print:
                print(f'all_lap_active_units_list: {all_lap_active_units_list}')
            lap_specific_spikes_df = deepcopy(spikes_df).spikes.time_sliced(t_start=t_start, t_stop=t_end).spikes.sliced_by_neuron_id(all_lap_active_units_list)
            lap_specific_spikes_df, neuron_id_to_new_IDX_map = lap_specific_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
            _decoded_pos_outputs = a_decoder.decode(unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, time_bin_size=time_bin_size, output_flat_versions=True, debug_print=False)
            _out_decoded_active_p_x_given_n[a_row.lap_id] = _decoded_pos_outputs
            _out_decoded_active_plots_data[a_row.lap_id] = RenderPlotsData(name=f'lap[{a_row.lap_id}]', spikes_df=lap_specific_spikes_df, active_aclus=all_lap_active_units_list)

        return (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data)


    @classmethod
    def build_time_binned_decoder_debug_plots(cls, a_decoder, a_lap_id, _out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_plots_data, debug_print=False):
        """ builds the plots 
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster

        def _simply_plot_posterior_in_pyqtgraph_plotitem(curr_plot, image, xbin_edges, ybin_edges):
            pg.setConfigOptions(imageAxisOrder='row-major')
            pg.setConfigOptions(antialias=True)
            cmap = pg.colormap.get('jet','matplotlib')
            image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=0.0, debug_print=debug_print)
            curr_plot.hideButtons()
            img_item = pg.ImageItem(image=image, levels=(0,1))
            curr_plot.addItem(img_item, defaultPadding=0.0)
            img_item.setImage(image, rect=image_bounds_extent, autoLevels=False)
            img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
            curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
            curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
            return img_item

        time_bin_edges = _out_decoded_time_bin_edges[a_lap_id]
        n_epoch_time_bins = len(time_bin_edges) - 1
        if debug_print:
            print(f'a_lap_id: {a_lap_id}, n_epoch_time_bins: {n_epoch_time_bins}')

        win = pg.GraphicsLayoutWidget()
        plots = []
        out_pf1D_decoder_template_objects = []
        _out_decoded_active_plots = {}

        plots_data = _out_decoded_active_plots_data[a_lap_id]
        plots_container = RenderPlots(name=a_lap_id, root_plot=None) # [a_lap_id]

        # Epoch Active Spikes, takes up a row _______________________________________________________________ #
        spanning_spikes_raster_plot = win.addPlot(title="spikes_raster Plot", row=0, rowspan=1, col=0, colspan=n_epoch_time_bins)
        spanning_spikes_raster_plot.setTitle("spikes_raster Plot")
        plots_container.root_plot = spanning_spikes_raster_plot
        app, raster_win, plots_container, plots_data = new_plot_raster_plot(plots_data.spikes_df, plots_data.active_aclus, scatter_plot_kwargs=None, win=spanning_spikes_raster_plot, plots_data=plots_data, plots=plots_container,
                                                            scatter_app_name=f'lap_specific_spike_raster', defer_show=True, active_context=None, add_debug_header_label=False)



        _out_decoded_active_plots[a_lap_id] = plots_container
        win.nextRow()

        # Decoded Epoch Posterior (bin-by-bin), takes up a row _______________________________________________________________ #
        spanning_posterior_plot = win.addPlot(title="P_x_given_n Plot", row=1, rowspan=1, col=0, colspan=n_epoch_time_bins)
        spanning_posterior_plot.setTitle("P_x_given_n Plot - Decoded over lap")

        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = _out_decoded_active_p_x_given_n[a_lap_id]
        flat_p_x_given_n = deepcopy(p_x_given_n)
        _simply_plot_posterior_in_pyqtgraph_plotitem(
            curr_plot=spanning_posterior_plot,
            image=flat_p_x_given_n,
            xbin_edges=np.arange(n_epoch_time_bins+1),
            ybin_edges=deepcopy(a_decoder.xbin)
        )
        win.nextRow()

        # Bin-by-bin active spike templates/pf1D fields ______________________________________________________________________ #
        active_bin_unit_specific_time_binned_spike_counts = _out_decoded_unit_specific_time_binned_spike_counts[a_lap_id]
        active_lap_active_aclu_spike_counts_list = _out_decoded_active_unit_lists[a_lap_id]

        for a_time_bin_idx in np.arange(n_epoch_time_bins):
            active_bin_active_aclu_spike_counts_dict = active_lap_active_aclu_spike_counts_list[a_time_bin_idx]
            active_bin_active_aclu_spike_count_values = np.array(list(active_bin_active_aclu_spike_counts_dict.values()))
            active_bin_active_aclu_bin_normalized_spike_count_values = active_bin_active_aclu_spike_count_values / np.sum(
                active_bin_active_aclu_spike_count_values)

            aclu_override_alpha_weights = 0.8 + (0.2 * active_bin_active_aclu_bin_normalized_spike_count_values)
            active_bin_aclus = np.array(list(active_bin_active_aclu_spike_counts_dict.keys()))
            active_solo_override_num_spikes_weights = dict(zip(active_bin_aclus, active_bin_active_aclu_bin_normalized_spike_count_values))
            active_aclu_override_alpha_weights_dict = dict(zip(active_bin_aclus, aclu_override_alpha_weights))
            if debug_print:
                print(f'a_time_bin_idx: {a_time_bin_idx}/{n_epoch_time_bins} - active_bin_aclus: {active_bin_aclus}')

            plot = win.addPlot(title=f"Plot {a_time_bin_idx+1}", row=2, rowspan=1, col=a_time_bin_idx, colspan=1)
            plot.getViewBox().setBorder(color=(200, 200, 200), width=1)
            spanning_posterior_plot.showGrid(x=True, y=True)
            x_axis = spanning_posterior_plot.getAxis('bottom')
            x_axis.setTickSpacing(major=5, minor=1)

            plots.append(plot)
            _obj = BaseTemplateDebuggingMixin.init_from_decoder(a_decoder=a_decoder, win=plot, title_str=f't={a_time_bin_idx}')
            _obj.update_base_decoder_debugger_data(
                included_neuron_ids=active_bin_aclus,
                solo_override_alpha_weights=active_aclu_override_alpha_weights_dict,
                solo_override_num_spikes_weights=active_solo_override_num_spikes_weights
            )
            out_pf1D_decoder_template_objects.append(_obj)

        win.nextRow()
        win.show()
        return win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data)


    @classmethod
    def plot_bin_by_bin_decoding_example(cls, curr_active_pipeline, track_templates, time_bin_size: float = 0.250, a_lap_id: int = 9, a_decoder_name = 'long_LR'):
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_bin_by_bin_decoding_example

                ## INPUTS: time_bin_size
        time_bin_size: float = 0.500 # 500ms

        ## any (generic) directionald decoder
        # neuron_IDs = deepcopy(track_templates.any_decoder_neuron_IDs) # array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  82,  83,  85,  86,  90,  92,  93,  96, 109])

        ## Get a specific decoder
        a_decoder_name: types.DecoderName = 'long_LR'
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)

        
        
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates

        # Example usage:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping() 
        global_laps_epochs_df = global_laps.to_dataframe()
        global_laps_epochs_df



        ## COMPUTED: 
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)
        (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data) = cls.build_spike_counts_and_decoder_outputs(track_templates=track_templates, global_laps_epochs_df=global_laps_epochs_df, spikes_df=global_spikes_df, a_decoder_name=a_decoder_name, time_bin_size=time_bin_size)
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = cls.build_time_binned_decoder_debug_plots(a_decoder=a_decoder, a_lap_id=a_lap_id, _out_decoded_time_bin_edges=_out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n=_out_decoded_active_p_x_given_n, _out_decoded_unit_specific_time_binned_spike_counts=_out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists=_out_decoded_active_unit_lists, _out_decoded_active_plots_data=_out_decoded_active_plots_data, debug_print=True)
        print(f"Returned window: {win}")
        print(f"Returned decoder objects: {out_pf1D_decoder_template_objects}")

        return win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data)
    



    ## OUTPUTS: _out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n



