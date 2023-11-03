# DirectionalTemplatesRastersDebugger
import numpy as np
import pandas as pd
from copy import deepcopy
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _build_default_tick, build_scatter_plot_kwargs
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import RasterScatterPlotManager, UnitSortOrderManager, _build_default_tick, _build_scatter_plotting_managers, _prepare_spikes_df_from_filter_epochs, _subfn_build_and_add_scatterplot_row
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import _plot_multi_sort_raster_browser

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.Render2DScrollWindowPlot import Render2DScrollWindowPlotMixin, ScatterItemData

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


@function_attributes(short_name=None, tags=['directional', 'templates', 'debugger', 'pyqtgraph'], input_requires=[], output_provides=[], uses=['_plot_multi_sort_raster_browser'], used_by=[], creation_date='2023-11-02 14:06', related_items=[])
def _debug_plot_directional_template_rasters(spikes_df, active_epochs_df, track_templates):
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

        odd_display_outputs, even_display_outputs = _debug_plot_directional_template_rasters(global_spikes_df, global_laps_epochs_df, track_templates)

    """
    ## spikes_df: get the spikes to plot
    # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
    # included_neuron_ids = track_templates.shared_aclus_only_neuron_IDs
    # track_templates.shared_LR_aclus_only_neuron_IDs

    included_neuron_ids = np.sort(np.union1d(track_templates.shared_RL_aclus_only_neuron_IDs, track_templates.shared_LR_aclus_only_neuron_IDs))
   
    ## add to the bottom
    # max_idx = (np.nanmax(track_templates.shared_RL_aclus_only_neuron_IDs)-1) # -1 to convert to an index
    
    n_neurons = len(included_neuron_ids)

    # Get only the spikes for the shared_aclus:
    spikes_df = spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids)
    spikes_df = spikes_df.spikes.adding_lap_identity_column(active_epochs_df, epoch_id_key_name='new_lap_IDX')
    spikes_df = spikes_df[(spikes_df['new_lap_IDX'] != -1)] # ['lap', 'maze_relative_lap', 'maze_id']
    spikes_df, neuron_id_to_new_IDX_map = spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards



    even_neuron_ids = track_templates.shared_LR_aclus_only_neuron_IDs.copy()
    odd_neuron_ids = track_templates.shared_RL_aclus_only_neuron_IDs.copy()
    
    # even_long, even_short = [(a_sort-1) for a_sort in track_templates.decoder_LR_pf_peak_ranks_list]
    # odd_long, odd_short = [(a_sort-1) for a_sort in track_templates.decoder_RL_pf_peak_ranks_list]


    even_long, even_short = [(a_sort-1) for a_sort in track_templates.decoder_LR_pf_peak_ranks_list]
    odd_long, odd_short = [(a_sort-1) for a_sort in track_templates.decoder_RL_pf_peak_ranks_list]
    

    # even_long, odd_long, even_short, odd_short = [(a_sort-1) for a_sort in track_templates.decoder_pf_peak_ranks_list]    
    neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None)
    unit_colors_list = neuron_colors_ndarray.copy()

    unit_sort_orders_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (even_long, odd_long, even_short, odd_short)))
    # unit_colors_list_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (unit_colors_list, unit_colors_list, unit_colors_list, unit_colors_list)))

    
    ## Do Even/Odd Separately:
    unit_colors_map = dict(zip(included_neuron_ids, neuron_colors_ndarray.copy().T))
    even_unit_colors_list = np.array([v for k, v in unit_colors_map.items() if k in even_neuron_ids]).T # should be (4, len(shared_RL_aclus_only_neuron_IDs))
    odd_unit_colors_list = np.array([v for k, v in unit_colors_map.items() if k in odd_neuron_ids]).T # should be (4, len(shared_RL_aclus_only_neuron_IDs))
    unit_colors_list_dict = dict(zip(['long_even', 'long_odd', 'short_even', 'short_odd'], (even_unit_colors_list, odd_unit_colors_list, even_unit_colors_list, odd_unit_colors_list)))

    # THE LOGIC MUST BE WRONG HERE. Slicing and dicing each Epoch separately is NOT OKAY. Spikes must be built before-hand. Loser.

    # Even:
    even_names = ['long_even', 'short_even']
    even_unit_sort_orders_dict = {k:v for k, v in unit_sort_orders_dict.items() if k in even_names}
    even_unit_colors_list_dict = {k:v for k, v in unit_colors_list_dict.items() if k in even_names}
    even_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(even_neuron_ids)
    even_spikes_df, even_neuron_id_to_new_IDX_map = even_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
    even_display_outputs = _plot_multi_sort_raster_browser(even_spikes_df, even_neuron_ids, unit_sort_orders_dict=even_unit_sort_orders_dict, unit_colors_list_dict=even_unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters_EVEN', defer_show=False, active_context=None)
    # app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = even_display_outputs

    # Odd:
    odd_names = ['long_odd', 'short_odd']
    odd_unit_sort_orders_dict = {k:v for k, v in unit_sort_orders_dict.items() if k in odd_names}
    odd_unit_colors_list_dict = {k:v for k, v in unit_colors_list_dict.items() if k in odd_names}
    odd_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(odd_neuron_ids)
    odd_spikes_df, odd_neuron_id_to_new_IDX_map = odd_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs() # rebuild the fragile indicies afterwards
    odd_display_outputs = _plot_multi_sort_raster_browser(odd_spikes_df, odd_neuron_ids, unit_sort_orders_dict=odd_unit_sort_orders_dict, unit_colors_list_dict=odd_unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters_ODD', defer_show=False, active_context=None)
    # odd_app, odd_win, odd_plots, odd_plots_data, odd_on_update_active_epoch, odd_on_update_active_scatterplot_kwargs = odd_display_outputs

    # app, win, plots, plots_data, on_update_active_epoch, on_update_active_scatterplot_kwargs = _plot_multi_sort_raster_browser(spikes_df, included_neuron_ids, unit_sort_orders_dict=unit_sort_orders_dict, unit_colors_list_dict=unit_colors_list_dict, scatter_app_name='pho_directional_laps_rasters', defer_show=False, active_context=None)
    return odd_display_outputs, even_display_outputs




@function_attributes(short_name=None, tags=['not-yet-working', 'debug'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-02 17:12', related_items=[])
def debug_plot_selected_spikes(plots, plots_data, shared_directional_aclus_only_neuron_IDs, directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, active_epoch_idx, _active_plot_identifier:str='long_odd'):
    """ overlays the representitive times for each unit within the epoch (such as the median spike time, or the center-of-mass on the spike rasters. 
    
    Usage:
        ## Specific:
        _active_plot_identifier = 'long_odd'
        debug_plot_selected_spikes(plots=odd_plots, plots_data=odd_plots_data, shared_directional_aclus_only_neuron_IDs=track_templates.shared_LR_aclus_only_neuron_IDs, directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, 
                _active_plot_identifier=_active_plot_identifier)

        _active_plot_identifier = 'short_odd'
        debug_plot_selected_spikes(plots=odd_plots, plots_data=odd_plots_data, shared_directional_aclus_only_neuron_IDs=track_templates.shared_LR_aclus_only_neuron_IDs, directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, 
                _active_plot_identifier=_active_plot_identifier)

            
        _active_plot_identifier = 'long_even'
        debug_plot_selected_spikes(plots=even_plots, plots_data=even_plots_data, shared_directional_aclus_only_neuron_IDs=track_templates.shared_RL_aclus_only_neuron_IDs, directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict,
                _active_plot_identifier=_active_plot_identifier)

        _active_plot_identifier = 'short_even'
        debug_plot_selected_spikes(plots=even_plots, plots_data=even_plots_data, shared_directional_aclus_only_neuron_IDs=track_templates.shared_RL_aclus_only_neuron_IDs, directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict=even_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict,
                _active_plot_identifier=_active_plot_identifier)


    """
    # Captures: odd_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict, track_templates.shared_LR_aclus_only_neuron_IDs, active_epoch_idx
    

    # Per group of two (odd/even):
    plots_data.all_selected_spots_dict = {}
    plots_data.all_selected_scatterplot_tooltips_kwargs_dict = {}
    plots_data_dict = plots_data.plots_data_dict # list(plots_data_dict.keys()) # ['long_odd', 'short_odd']


    # an_ax = plots.ax[_active_plot_identifier]
    a_scatter_plot = plots.scatter_plots[_active_plot_identifier]

    # Converts the selected spikes information dict (containing the median/first spikes for each epoch) into a spikes_df capable of being rendered on the raster plot.
    selected_spike_fragile_neuron_IDX = np.squeeze(directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict[active_epoch_idx][:,0]).astype('int')
    selected_spike_aclus = shared_directional_aclus_only_neuron_IDs[selected_spike_fragile_neuron_IDX].astype('int')
    selected_spike_times = np.squeeze(directional_laps_epoch_selected_spikes_fragile_linear_neuron_IDX_dict[active_epoch_idx][:,1]) # (n_cells, 2)
    selected_spike_epoch_idx = np.full_like(selected_spike_fragile_neuron_IDX, active_epoch_idx).astype('int')

    selected_spike_df = pd.DataFrame({'t': selected_spike_times, 'fragile_linear_neuron_IDX': selected_spike_fragile_neuron_IDX, 'aclu': selected_spike_aclus, 'epoch_IDX': selected_spike_epoch_idx})
    selected_spike_df['t_rel_seconds'] = selected_spike_df['t']
    selected_spike_df['neuron_type'] = False # stupid workaround
    selected_spike_df['flat_spike_idx'] = -1 # stupid workaround

    # Update the dataframe
    active_spikes_shit = deepcopy(selected_spike_df)
    active_spikes_shit = plots_data_dict[_active_plot_identifier].unit_sort_manager.update_spikes_df_visualization_columns(active_spikes_shit)

    ## Build the spots for the raster plot:
    plots_data.all_selected_spots_dict[_active_plot_identifier], plots_data.all_selected_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] = Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(active_spikes_shit, plots_data_dict[_active_plot_identifier].raster_plot_manager.config_fragile_linear_neuron_IDX_map, should_return_data_tooltips_kwargs=True)

    # Override the pen for the selected spots, the default renders them looking exactly like normal spikes which is no good:
    for a_point in plots_data.all_selected_spots_dict[_active_plot_identifier]:
        a_point['pen'] = pg.mkPen('#FFFFFF', width=4.5)


    # normal_all_spots_dict = plots_data.all_spots_dict[_active_plot_identifier]
    selected_all_spots_dict = plots_data.all_selected_spots_dict[_active_plot_identifier]
    # merged_all_spots_dict = (plots_data.all_spots_dict[_active_plot_identifier] + plots_data.all_selected_spots_dict[_active_plot_identifier])

    ## Add the median spikes to the plots:
    a_scatter_plot.addPoints(selected_all_spots_dict)
    
