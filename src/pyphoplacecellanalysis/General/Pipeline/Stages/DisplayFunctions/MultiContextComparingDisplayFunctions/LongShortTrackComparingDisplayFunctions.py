from copy import deepcopy
from enum import Enum, unique # for PlacefieldOverlapMetricMode
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, FigureBase # FigureBase: both Figure and SubFigure
from flexitext import flexitext ## flexitext for formatted matplotlib text

from neuropy.core.neuron_identities import PlotStringBrevityModeEnum, NeuronType  # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.plotting.figure import Fig # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.plotting.ratemaps import plot_ratemap_1D # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.utils.matplotlib_helpers import build_or_reuse_figure # used for `_make_pho_jonathan_batch_plots(...)`
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter # for `_plot_long_short_firing_rate_indicies`
from neuropy.utils.matplotlib_helpers import fit_both_axes
from neuropy.utils.matplotlib_helpers import draw_epoch_regions # plot_expected_vs_observed

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.indexing_helpers import Paginator
from pyphocorehelpers.print_helpers import generate_html_string # used for `plot_long_short_surprise_difference_plot`

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.color_helpers import convert_pen_brush_to_matplot_kwargs

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer  # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedFigureController
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget # used by RateRemappingPaginatedFigureController
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import make_fr
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult # used in _display_short_long_pf1D_comparison
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results, _find_any_context_neurons, build_neurons_color_map # for plot_short_v_long_pf1D_comparison
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import build_replays_custom_scatter_markers, CustomScatterMarkerMode # used in _make_pho_jonathan_batch_plots
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _build_neuron_type_distribution_color # used in _make_pho_jonathan_batch_plots
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # for PlacefieldOverlapMetricMode
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1D_placecell_validation # for _plot_pho_jonathan_batch_plot_single_cell
from neuropy.utils.matplotlib_helpers import FormattedFigureText
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines, add_track_shapes
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager

# ==================================================================================================================== #
# Long/Short Epoch Configs                                                                                             #
# ==================================================================================================================== #



@unique
class PlacefieldOverlapMetricMode(ExtendedEnum):
    """Docstring for PlacefieldOverlapMetricMode."""
    POLY = "POLY"
    CONVOLUTION = "CONVOLUTION"
    PRODUCT = "PRODUCT"
    REL_ENTROPY = "REL_ENTROPY"


def build_extra_cell_info_label_string(row) -> str:
    """ used in `_display_jonathan_interactive_replay_firing_rate_comparison` to format the extra info labels for each aclu like its firing rate indices. """
    row_dict = dict(row._asdict())
    has_instantaneous_version = np.all(np.isin(list(row_dict.keys()), ['laps_frs_index', 'laps_inst_frs_index', 'replays_frs_index', 'replays_inst_frs_index', 'non_replays_frs_index', 'non_replays_inst_frs_index']))
    if has_instantaneous_version:
        # if have inst
        # pre-format row output values:
        row = {k:f"<size:8><weight:bold>{round(v, 2)}</></>" for k,v in row_dict.items()}
        return '\n'.join([f"<size:9><weight:bold>fri</></> (epochs: bin|inst)", 
         f"laps: {row['laps_frs_index']}|{row['laps_inst_frs_index']}",
         f"replays: {row['replays_frs_index']}|{row['replays_inst_frs_index']}",
         f"non_replays: {row['non_replays_frs_index']}|{row['non_replays_inst_frs_index']}",
        ])
        # return '\n'.join([f"<size:12><weight:bold>fri</></> (epochs|bin|inst)", 
        #  f"fri[laps]: {row['laps_frs_index']}|inst: {row['laps_inst_frs_index']}",
        #  f"fri[replays]: {row['replays_frs_index']}|inst: {row['replays_inst_frs_index']}",
        #  f"fri[non_replays]: {row['non_replays_frs_index']}|inst: {row['non_replays_inst_frs_index']}",
        # ])
    else:
        return '\n'.join([f"{k}: {round(v, 3)}" for k,v in row_dict.items()])




@function_attributes(short_name=None, tags=['pyqtgraph', 'helper', 'long_short', 'regions', 'rectangles'], input_requires=['pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers.build_pyqtgraph_epoch_indicator_regions'], output_provides=[], uses=[], used_by=[], creation_date='2023-04-19 19:04')
def _helper_add_long_short_session_indicator_regions(win, long_epoch, short_epoch):
    """Add session indicators to pyqtgraph plot for the long and the short epoch

            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.LongShortTrackComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _helper_add_long_short_session_indicator_regions

            long_epoch = curr_active_pipeline.filtered_epochs[long_epoch_name]
            short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
            long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(win, long_epoch, short_epoch)

            long_epoch_linear_region, long_epoch_region_label = long_epoch_indicator_region_items
            short_epoch_linear_region, short_epoch_region_label = short_epoch_indicator_region_items
    """
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions # Add session indicators to pyqtgraph plot

    long_short_display_config_manager = LongShortDisplayConfigManager()
    long_epoch_config = long_short_display_config_manager.long_epoch_config.as_pyqtgraph_kwargs()
    short_epoch_config = long_short_display_config_manager.short_epoch_config.as_pyqtgraph_kwargs()

    long_epoch_indicator_region_items = build_pyqtgraph_epoch_indicator_regions(win, t_start=long_epoch.t_start, t_stop=long_epoch.t_stop, **long_epoch_config)
    short_epoch_indicator_region_items = build_pyqtgraph_epoch_indicator_regions(win, t_start=short_epoch.t_start, t_stop=short_epoch.t_stop, **short_epoch_config)
    return long_epoch_indicator_region_items, short_epoch_indicator_region_items



class LongShortTrackComparingDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ LongShortTrackComparingDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='jonathan_interactive_replay_firing_rate_comparison', tags=['display','interactive','jonathan', 'firing_rate', 'pyqtgraph'], input_requires=[], output_provides=[], uses=['_make_jonathan_interactive_plot'], used_by=[], creation_date='2023-04-11 03:14', is_global=True)
    def _display_jonathan_interactive_replay_firing_rate_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, included_neuron_types=None, require_placefield=True, save_figure=True, **kwargs):
            """ Jonathan's interactive display. Currently hacked up to directly compute the results to display within this function
                Internally calls `_make_jonathan_interactive_plot(...)`

                Usage:
                active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
                curr_active_pipeline.display('_display_jonathan_interactive_replay_firing_rate_comparison', active_identifying_session_ctx)

            """
            if include_includelist is None:
                include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            if included_neuron_types is None:
                included_neuron_types = NeuronType.from_any_string_series(['pyr'])


            long_epoch_name = include_includelist[0] # 'maze1_PYR'
            short_epoch_name = include_includelist[1] # 'maze2_PYR'
            if len(include_includelist) > 2:
                global_epoch_name = include_includelist[-1] # 'maze_PYR'
            else:
                print(f'WARNING: no global_epoch detected.')
                global_epoch_name = '' # None
                

            print(f'include_includelist: {include_includelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')
            pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
            pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
            pf1d = computation_results[global_epoch_name]['computed_data']['pf1D']

            ## Proper global-computations based way:
            sess = owning_pipeline_reference.sess
            aclu_to_idx = global_computation_results.computed_data['jonathan_firing_rate_analysis'].rdf.aclu_to_idx
            rdf = global_computation_results.computed_data['jonathan_firing_rate_analysis'].rdf.rdf
            irdf = global_computation_results.computed_data['jonathan_firing_rate_analysis'].irdf.irdf
            pos_df = global_computation_results.sess.position.to_dataframe()

            ## time_binned_unit_specific_binned_spike_rate mode:
            time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate['time_bins']
            time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate['time_binned_unit_specific_binned_spike_rate']
            # ## instantaneous_unit_specific_spike_rate mode:
            # time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['time_bins']
            # time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['instantaneous_unit_specific_spike_rate_values']

            neuron_replay_stats_df = deepcopy(global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df)

            if 'neuron_type' not in neuron_replay_stats_df.columns:
                ## Add neuron type to the replay stats dataframe:
                neuron_replay_stats_df['neuron_type'] = [sess.neurons.aclu_to_neuron_type_map[aclu] for aclu in neuron_replay_stats_df.index.to_numpy()]

            # Filter by the included neuron types:
            neuron_replay_stats_df = neuron_replay_stats_df[np.isin([v.value for v in neuron_replay_stats_df['neuron_type']], [v.value for v in included_neuron_types])]

            if require_placefield:
                ## Require placefield presence on either the long or the short
                neuron_replay_stats_df = neuron_replay_stats_df[np.logical_or(neuron_replay_stats_df['has_long_pf'], neuron_replay_stats_df['has_short_pf'])]

            graphics_output_dict, neuron_df = _make_jonathan_interactive_plot(sess, time_bins, neuron_replay_stats_df, time_binned_unit_specific_binned_spike_rate, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=True)
            graphics_output_dict['plot_data'] = {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate}

            return graphics_output_dict

    @function_attributes(short_name='batch_pho_jonathan_replay_firing_rate_comparison', tags=['display','jonathan', 'firing_rate', 'matplotlib', 'batch', 'inefficient', 'slow'], input_requires=[], output_provides=[], uses=['_make_pho_jonathan_batch_plots'], used_by=[], creation_date='2023-04-11 03:14', is_global=True)
    def _display_batch_pho_jonathan_replay_firing_rate_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True, **kwargs):
            """ Stacked Jonathan-style firing-rate-across-epochs-plot. Pho's batch adaptation of the primary elements from Jonathan's interactive display.
                Usage:

                    %matplotlib qt
                    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

                    graphics_output_dict = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', active_identifying_session_ctx)
                    fig, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
                    neuron_df, rdf, aclu_to_idx, irdf = plot_data['df'], plot_data['rdf'], plot_data['aclu_to_idx'], plot_data['irdf']
                    # Grab the output axes:
                    curr_axs_dict = axs[0]
                    curr_firing_rate_ax, curr_lap_spikes_ax, curr_placefield_ax = curr_axs_dict['firing_rate'], curr_axs_dict['lap_spikes'], curr_axs_dict['placefield'] # Extract variables from the `curr_axs_dict` dictionary to the local workspace

            """
            if include_includelist is None:
                include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_includelist[0] # 'maze1_PYR'
            short_epoch_name = include_includelist[1] # 'maze2_PYR'
            assert len(include_includelist) > 2
            global_epoch_name = include_includelist[-1] # 'maze_PYR'
            print(f'include_includelist: {include_includelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')

            ## TODO: move this computation elsewhere, this is BAD:
            long_results = computation_results[long_epoch_name]
            short_results = computation_results[short_epoch_name]
            global_results = computation_results[global_epoch_name]
        
            ## Add three columns to global_results.sess.spikes_df, indicating whether each spike is included in the filtered_spikes_df for the (long, short, global) pf1Ds
            if 'is_included_long_pf1D' not in global_results.sess.spikes_df.columns:
                global_results.sess.spikes_df['is_included_long_pf1D'] = False
                global_results.sess.spikes_df.loc[np.isin(global_results.sess.spikes_df.index, long_results.computed_data.pf1D.filtered_spikes_df.index),'is_included_long_pf1D'] = True
            if 'is_included_short_pf1D' not in global_results.sess.spikes_df.columns:
                global_results.sess.spikes_df['is_included_short_pf1D'] = False
                global_results.sess.spikes_df.loc[np.isin(global_results.sess.spikes_df.index, short_results.computed_data.pf1D.filtered_spikes_df.index),'is_included_short_pf1D'] = True
            if 'is_included_global_pf1D' not in global_results.sess.spikes_df.columns:
                global_results.sess.spikes_df['is_included_global_pf1D'] = False
                global_results.sess.spikes_df.loc[np.isin(global_results.sess.spikes_df.index, global_results.computed_data.pf1D.filtered_spikes_df.index),'is_included_global_pf1D'] = True

            use_filtered_positions: bool = kwargs.pop('use_filtered_positions', False)
            cell_spikes_dfs_dict, aclu_to_fragile_linear_idx_map = _build_spikes_df_interpolated_props(global_results, should_interpolate_to_filtered_positions=use_filtered_positions) # cell_spikes_dfs_list is indexed by aclu_to_fragile_linear_idx_map
            time_variable_name = global_results.sess.spikes_df.spikes.time_variable_name
            
            # pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
            # pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
            pf1D_all = global_results['computed_data']['pf1D'] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

            ## Proper global-computations based way:
            sess = owning_pipeline_reference.sess
            active_identifying_session_ctx = sess.get_context()
            t_split = sess.paradigm[0][0,1] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

            aclu_to_idx = global_computation_results.computed_data['jonathan_firing_rate_analysis'].rdf.aclu_to_idx
            rdf = global_computation_results.computed_data['jonathan_firing_rate_analysis'].rdf.rdf
            irdf = global_computation_results.computed_data['jonathan_firing_rate_analysis'].irdf.irdf
            # pos_df = global_computation_results.sess.position.to_dataframe()
            ## time_binned_unit_specific_binned_spike_rate mode:
            time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate['time_bins']
            time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate['time_binned_unit_specific_binned_spike_rate']
            # ## instantaneous_unit_specific_spike_rate mode:
            # time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['time_bins']
            # time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['instantaneous_unit_specific_spike_rate_values']
            neuron_replay_stats_df = global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df
            # compare_firing_rates(rdf, irdf)

            n_max_plot_rows = kwargs.pop('n_max_plot_rows', 6)
            show_inter_replay_frs = kwargs.pop('show_inter_replay_frs', True)
            included_unit_neuron_IDs = kwargs.pop('included_unit_neuron_IDs', None)
            
            # Get the provided context or use the session context:
            active_context = kwargs.get('active_context', active_identifying_session_ctx)
            kwargs['active_context'] = active_context

            active_context = active_context.adding_context_if_missing(display_fn_name='batch_pho_jonathan_replay_firing_rate_comparison')
            curr_fig_num = kwargs.pop('fignum', None)
            if curr_fig_num is None:
                ## Set the fig_num, if not already set:
                curr_fig_num = f'long|short fr indicies_{active_context.get_description(separator="/")}'
            kwargs['fignum'] = curr_fig_num


            ## Have to pull out the rate-remapping stats for each neuron_id
            try:
                curr_long_short_fr_indicies_analysis = global_computation_results.computed_data['long_short_fr_indicies_analysis']

                # extract one set of keys for the aclus
                _curr_aclus = list(curr_long_short_fr_indicies_analysis['laps_frs_index'].keys())
                _curr_frs_indicies_dict = {k:v.values() for k,v in curr_long_short_fr_indicies_analysis.items() if k in ['laps_frs_index', 'laps_inst_frs_index', 'replays_frs_index', 'replays_inst_frs_index', 'non_replays_frs_index', 'non_replays_inst_frs_index']} # extract the values
                long_short_fr_indicies_df = pd.DataFrame(_curr_frs_indicies_dict, index=_curr_aclus)
                
                # build the labels for each cell using `build_extra_cell_info_label_string(...)`:
                optional_cell_info_labels = {aclu:build_extra_cell_info_label_string(row) for aclu, row in zip(_curr_aclus, long_short_fr_indicies_df.itertuples(name='ExtraCellInfoLabels', index=False))}

            except BaseException:
                # set optional cell info labels to None
                optional_cell_info_labels = {}

            graphics_output_dict: MatplotlibRenderPlots = _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, time_binned_unit_specific_binned_spike_rate, pf1D_all, aclu_to_idx, rdf, irdf,
                show_inter_replay_frs=show_inter_replay_frs, n_max_plot_rows=n_max_plot_rows, included_unit_neuron_IDs=included_unit_neuron_IDs, cell_spikes_dfs_dict=cell_spikes_dfs_dict, time_variable_name=time_variable_name, defer_render=defer_render, optional_cell_info_labels=optional_cell_info_labels,
                use_filtered_positions=use_filtered_positions, **kwargs)

            final_context = active_context
            graphics_output_dict['context'] = final_context
            graphics_output_dict['plot_data'] |= {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis'].time_binned_unit_specific_spike_rate,
                'time_variable_name':time_variable_name, 'fignum':curr_fig_num}

            def _perform_write_to_file_callback():
                ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
                return owning_pipeline_reference.output_figure(final_context, graphics_output_dict.figures[0])
            
            if save_figure:
                active_out_figure_paths = _perform_write_to_file_callback()
            else:
                active_out_figure_paths = []

            graphics_output_dict['saved_figures'] = active_out_figure_paths
            
            return graphics_output_dict

    @function_attributes(short_name='short_long_pf1D_comparison', tags=['long_short','1D','placefield'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_short_v_long_pf1D_comparison', 'determine_long_short_pf1D_indicies_sort_by_peak'], used_by=[], creation_date='2023-04-26 06:12', is_global=True)
    def _display_short_long_pf1D_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None, **kwargs):
            """ Displays a figure for comparing the 1D placefields across-epochs (between the short and long tracks). By default renders the second track's placefield flipped over the x-axis and hatched. 
                Usage:

                    %matplotlib qt
                    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

                    graphics_output_dict = curr_active_pipeline.display('_display_short_long_pf1D_comparison', active_identifying_session_ctx)
                    fig, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
                    

            """
            reuse_axs_tuple = kwargs.pop('reuse_axs_tuple', None)
            # reuse_axs_tuple = None # plot fresh
            # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
            # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
            single_figure = kwargs.pop('single_figure', True)
            debug_print = kwargs.pop('debug_print', False)

            active_config_name = kwargs.pop('active_config_name', None)
            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())

            # Plot 1D Keywoard args:
            shared_kwargs = kwargs.pop('shared_kwargs', {})
            long_kwargs = kwargs.pop('long_kwargs', {})
            short_kwargs = kwargs.pop('short_kwargs', {})

            shared_kwargs['active_context'] = active_context

            # Allow overriding the sortby for both long and short if a top-level sortby kwarg is passed:
            override_sortby = kwargs.pop('sortby', None)
            if (override_sortby is not None and isinstance(override_sortby, str)):
                print(f'override_sortby: {override_sortby} is being used.')
                # long_sortby = long_kwargs.get('sortby', None)
                # short_sortby = short_kwargs.get('sortby', None)
                # assert (not (long_sortby is not None and isinstance(long_sortby, str)) and (short_sortby is not None and isinstance(short_sortby, str))) # assert no valid sortby values are set for the individual kwargs
                long_kwargs['sortby'] = override_sortby
                short_kwargs['sortby'] = override_sortby

            if include_includelist is None:
                include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_includelist[0] # 'maze1_PYR'
            short_epoch_name = include_includelist[1] # 'maze2_PYR'
            assert len(include_includelist) > 2
            global_epoch_name = include_includelist[-1] # 'maze_PYR'
            if debug_print:
                print(f'include_includelist: {include_includelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')           
    
            long_results = computation_results[long_epoch_name]['computed_data']
            short_results = computation_results[short_epoch_name]['computed_data']

            # curr_any_context_neurons = _find_any_context_neurons(*[owning_pipeline_reference.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in [long_epoch_name, short_epoch_name]])
            curr_any_context_neurons = _find_any_context_neurons(*[a_result.pf1D.ratemap.neuron_ids for a_result in [long_results, short_results]])

            if included_any_context_neuron_ids is None:
                included_any_context_neuron_ids = curr_any_context_neurons
            else:
                # include only the specified neuron_ids:
                included_any_context_neuron_ids = curr_any_context_neurons[np.isin(curr_any_context_neurons, included_any_context_neuron_ids)]

            # SORT BY LONG PEAK LOCATION: Determine the sort indicies to align the placefields by the position on the long:
            ### to use this mode, you must pass the string 'peak_long' for both long_kwargs['sortby'] and short_kwargs['sortby']
            long_sortby = long_kwargs.get('sortby', None)
            short_sortby = short_kwargs.get('sortby', None)
            if (long_sortby is not None and isinstance(long_sortby, str)) and (short_sortby is not None and isinstance(short_sortby, str)):
                assert long_sortby == 'peak_long'
                assert short_sortby == 'peak_long'
                new_all_aclus_sort_indicies = determine_long_short_pf1D_indicies_sort_by_peak(owning_pipeline_reference, included_any_context_neuron_ids, debug_print=False)
                # shared_kwargs['sortby'] = (shared_kwargs.get('sortby', None) or new_all_aclus_sort_indicies)
                long_kwargs['sortby'] = new_all_aclus_sort_indicies # (long_kwargs.get('sortby', None) or 
                short_kwargs['sortby'] = new_all_aclus_sort_indicies # (short_kwargs.get('sortby', None) or 
                print(f'DEBUG: new_all_aclus_sort_indicies: {new_all_aclus_sort_indicies}')
            
            (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, included_any_context_neuron_ids, reuse_axs_tuple=reuse_axs_tuple, single_figure=single_figure,
                shared_kwargs=shared_kwargs, long_kwargs=long_kwargs, short_kwargs=short_kwargs, debug_print=debug_print, **kwargs)

            if single_figure:
                final_context = active_context.adding_context('display_fn', display_fn_name='display_short_long_pf1D_comparison')
            else:
                base_final_context = active_context.adding_context('display_fn', display_fn_name='display_short_long_pf1D_comparison')
                final_context = (base_final_context.overwriting_context(track='long'), base_final_context.overwriting_context(track='short'),) # final context is a tuple of contexts

            def _perform_write_to_file_callback():
                ## 2023-05-31 - Reference Output of matplotlib figure to file, along with building appropriate context.
                if single_figure:
                    return owning_pipeline_reference.output_figure(final_context, fig_short_pf_1D)
                else:
                    owning_pipeline_reference.output_figure(final_context[0], fig_long_pf_1D)
                    owning_pipeline_reference.output_figure(final_context[1], fig_short_pf_1D)

            if save_figure:
                active_out_figure_paths = _perform_write_to_file_callback()
            else:
                active_out_figure_paths = []

            graphics_output_dict = MatplotlibRenderPlots(name='display_short_long_pf1D_comparison', figures=(fig_long_pf_1D, fig_short_pf_1D), axes=(ax_long_pf_1D, ax_short_pf_1D), plot_data={}, context=final_context, saved_figures=active_out_figure_paths)
            graphics_output_dict['plot_data'] = {'included_any_context_neuron_ids': included_any_context_neuron_ids, 'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}

            return graphics_output_dict

    @function_attributes(short_name='short_long_pf1D_scalar_overlap_comparison', tags=['display','long_short'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-08 12:44', is_global=True)
    def _display_short_long_pf1D_scalar_overlap_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, **kwargs):
            """ Displays a figure for comparing the scalar comparison quantities computed for 1D placefields across-epochs (between the short and long tracks)
                This currently renders as a colorful bar-graph with one bar for each aclu

                Usage:

                    %matplotlib qt
                    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

                    graphics_output_dict = curr_active_pipeline.display('_display_short_long_pf1D_scalar_overlap_comparison', active_identifying_session_ctx)
                    fig, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
                    neuron_df, rdf, aclu_to_idx, irdf = plot_data['df'], plot_data['rdf'], plot_data['aclu_to_idx'], plot_data['irdf']
                    # Grab the output axes:
                    curr_axs_dict = axs[0]
                    curr_firing_rate_ax, curr_lap_spikes_ax, curr_placefield_ax = curr_axs_dict['firing_rate'], curr_axs_dict['lap_spikes'], curr_axs_dict['placefield'] # Extract variables from the `curr_axs_dict` dictionary to the local workspace

            """
            overlap_metric_mode = kwargs.pop('overlap_metric_mode', PlacefieldOverlapMetricMode.POLY)
            if not isinstance(overlap_metric_mode, PlacefieldOverlapMetricMode):
                overlap_metric_mode = PlacefieldOverlapMetricMode.init(overlap_metric_mode)

            reuse_axs_tuple = kwargs.pop('reuse_axs_tuple', None)            
            # reuse_axs_tuple = None # plot fresh
            # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
            # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
            single_figure = kwargs.pop('single_figure', True)
            debug_print = kwargs.pop('debug_print', False)

            if include_includelist is None:
                include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_includelist[0] # 'maze1_PYR'
            short_epoch_name = include_includelist[1] # 'maze2_PYR'
            assert len(include_includelist) > 2
            global_epoch_name = include_includelist[-1] # 'maze_PYR'
            if debug_print:
                print(f'include_includelist: {include_includelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')           
    
            short_long_pf_overlap_analyses_results = global_computation_results['computed_data']['short_long_pf_overlap_analyses']
            pf_neurons_diff = short_long_pf_overlap_analyses_results['short_long_neurons_diff'] # get shared neuron info:
            n_neurons = pf_neurons_diff.shared.n_neurons
            shared_fragile_neuron_IDXs = pf_neurons_diff.shared.shared_fragile_neuron_IDXs
            neurons_colors_array = build_neurons_color_map(n_neurons, sortby=shared_fragile_neuron_IDXs, cmap=None)

            if overlap_metric_mode.name == PlacefieldOverlapMetricMode.POLY.name:
                poly_overlap_df = short_long_pf_overlap_analyses_results['poly_overlap_df']
                fig, ax = plot_short_v_long_pf1D_scalar_overlap_comparison(poly_overlap_df, pf_neurons_diff, neurons_colors_array, reuse_axs_tuple=reuse_axs_tuple, single_figure=single_figure, overlap_metric_mode=overlap_metric_mode, debug_print=debug_print)
            elif overlap_metric_mode.name == PlacefieldOverlapMetricMode.CONVOLUTION.name:
                conv_overlap_dict = short_long_pf_overlap_analyses_results['conv_overlap_dict']
                conv_overlap_scalars_df = short_long_pf_overlap_analyses_results['conv_overlap_scalars_df']
                fig, ax = plot_short_v_long_pf1D_scalar_overlap_comparison(conv_overlap_scalars_df, pf_neurons_diff, neurons_colors_array, reuse_axs_tuple=reuse_axs_tuple, single_figure=single_figure, overlap_metric_mode=overlap_metric_mode, debug_print=debug_print)
            elif overlap_metric_mode.name == PlacefieldOverlapMetricMode.PRODUCT.name:
                prod_overlap_dict = short_long_pf_overlap_analyses_results['product_overlap_dict']
                product_overlap_scalars_df = short_long_pf_overlap_analyses_results['product_overlap_scalars_df']
                fig, ax = plot_short_v_long_pf1D_scalar_overlap_comparison(product_overlap_scalars_df, pf_neurons_diff, neurons_colors_array, reuse_axs_tuple=reuse_axs_tuple, single_figure=single_figure, overlap_metric_mode=overlap_metric_mode, debug_print=debug_print)
            elif overlap_metric_mode.name == PlacefieldOverlapMetricMode.REL_ENTROPY.name:
                relative_entropy_overlap_dict = short_long_pf_overlap_analyses_results['relative_entropy_overlap_dict']
                relative_entropy_overlap_scalars_df = short_long_pf_overlap_analyses_results['relative_entropy_overlap_scalars_df']
                fig, ax = plot_short_v_long_pf1D_scalar_overlap_comparison(relative_entropy_overlap_scalars_df, pf_neurons_diff, neurons_colors_array, reuse_axs_tuple=reuse_axs_tuple, single_figure=single_figure, overlap_metric_mode=overlap_metric_mode, debug_print=debug_print)
            else:
                raise NotImplementedError
            
            final_context = owning_pipeline_reference.sess.get_context().adding_context('display_fn', display_fn_name='display_short_long_pf1D_scalar_overlap_comparison')
    
            def _perform_write_to_file_callback():
                return owning_pipeline_reference.output_figure(final_context, fig)
            
            if save_figure:
                active_out_figure_paths = _perform_write_to_file_callback(kwargs.pop('fig_save_parent_path', None))
            else:
                active_out_figure_paths = []
            graphics_output_dict = MatplotlibRenderPlots(name='_display_short_long_pf1D_poly_overlap_comparison', figures=(fig), axes=(ax), context=final_context, plot_data={'colors': neurons_colors_array}, saved_figures=active_out_figure_paths)
            # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}

            return graphics_output_dict

    @function_attributes(short_name='short_long_firing_rate_index_comparison', tags=['display','long_short','short_long','firing_rate', 'fr_index'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['_plot_long_short_firing_rate_indicies'], used_by=[], creation_date='2023-04-11 08:08', is_global=True)
    def _display_short_long_firing_rate_index_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True, **kwargs):
            """ Displays a figure for comparing the 1D placefields across-epochs (between the short and long tracks)
                Usage:

                    %matplotlib qt
                    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

                    graphics_output_dict = curr_active_pipeline.display('_display_batch_pho_jonathan_replay_firing_rate_comparison', active_identifying_session_ctx)
                    fig, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
                    neuron_df, rdf, aclu_to_idx, irdf = plot_data['df'], plot_data['rdf'], plot_data['aclu_to_idx'], plot_data['irdf']
                    # Grab the output axes:
                    curr_axs_dict = axs[0]
                    curr_firing_rate_ax, curr_lap_spikes_ax, curr_placefield_ax = curr_axs_dict['firing_rate'], curr_axs_dict['lap_spikes'], curr_axs_dict['placefield'] # Extract variables from the `curr_axs_dict` dictionary to the local workspace

            """
            debug_print = kwargs.pop('debug_print', False)

            # Plot long|short firing rate index:
            long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis']
            x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
            active_context = long_short_fr_indicies_analysis_results['active_context']
            
            final_context = active_context.adding_context('display_fn', display_fn_name='display_long_short_laps')
            fig = _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, final_context, debug_print=debug_print)
            
            if not defer_render:
                fig.show()
        
            def _perform_write_to_file_callback():
                return owning_pipeline_reference.output_figure(final_context, fig)
            
            if save_figure:
                active_out_figure_paths = _perform_write_to_file_callback()
            else:
                active_out_figure_paths = []
                
            graphics_output_dict = MatplotlibRenderPlots(name='display_short_long_firing_rate_index_comparison', figures=(fig), axes=tuple(fig.axes), plot_data={}, context=final_context, saved_figures=active_out_figure_paths)
            # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}            
            return graphics_output_dict

    @function_attributes(short_name=None, tags=['display', 'long_short', 'laps', 'position', 'behavior', 'needs_footer', '1D'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-29 18:20', related_items=[], is_global=True)
    def _display_long_short_laps(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True, **kwargs):
            """ Displays a figure displaying the 1D laps detected for both the long and short tracks.
                Usage:

                    %matplotlib qt
                    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

                    graphics_output_dict = curr_active_pipeline.display('_display_long_short_laps', active_identifying_session_ctx)
                    fig, axs, plot_data = graphics_output_dict['fig'], graphics_output_dict['axs'], graphics_output_dict['plot_data']
                    

            """
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            # long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
            from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import plot_laps_2d
            fig, out_axes_list = plot_laps_2d(global_session, legacy_plotting_mode=False)
            out_axes_list[0].set_title('Estimated Laps')
            fig.canvas.manager.set_window_title('Estimated Laps')

            final_context = owning_pipeline_reference.sess.get_context().adding_context('display_fn', display_fn_name='display_long_short_laps')

            def _perform_write_to_file_callback():
                return owning_pipeline_reference.output_figure(final_context, fig)
            
            if save_figure:
                active_out_figure_paths = _perform_write_to_file_callback()
            else:
                active_out_figure_paths = []
                            
            graphics_output_dict = MatplotlibRenderPlots(name='_display_long_short_laps', figures=(fig,), axes=out_axes_list, plot_data={}, context=final_context, saved_figures=active_out_figure_paths)
            return graphics_output_dict

    @function_attributes(short_name='long_and_short_firing_rate_replays_v_laps', tags=['display','long_short','firing_rate'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['_plot_session_long_short_track_firing_rate_figures'], used_by=[], creation_date='2023-06-08 10:22', is_global=True)
    def _display_long_and_short_firing_rate_replays_v_laps(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True, **kwargs):
        """ Displays two figures, one for the long and one for the short track, that compare the firing rates during running (laps) and those during decoded replays.
            Usage:
            
            
            Option: this relies on the global result `jonathan_firing_rate_analysis_result.neuron_replay_stats_df`, but otherwise it could be made non-global as it does operate on separate epochs independently ('maze1', 'maze2').

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult
        if not isinstance(global_computation_results.computed_data.jonathan_firing_rate_analysis, JonathanFiringRateAnalysisResult):
            jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
        else:
            jonathan_firing_rate_analysis_result = global_computation_results.computed_data.jonathan_firing_rate_analysis

        (fig_L, ax_L, active_display_context_L), (fig_S, ax_S, active_display_context_S), _perform_write_to_file_callback = _plot_session_long_short_track_firing_rate_figures(owning_pipeline_reference, jonathan_firing_rate_analysis_result, defer_render=defer_render)
        
        if save_figure:
            active_out_figure_paths = _perform_write_to_file_callback()
        else:
            active_out_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='long_and_short_firing_rate_replays_v_laps', figures=(fig_L, fig_S), axes=(ax_L, ax_S), context=(active_display_context_L, active_display_context_S), plot_data={'context': (active_display_context_L, active_display_context_S)}, saved_figures=active_out_figure_paths)
        return graphics_output_dict

    @function_attributes(short_name='running_and_replay_speeds_over_time', tags=['speed', 'laps', 'replay', 'velocity', 'time', 'running', 'fit'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-07 21:13', related_items=[], is_global=True)
    def _display_running_and_replay_speeds_over_time(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True, **kwargs):
        """ plots the animal's running speed and the decoded replay velocities (as computed by the Radon transform method) across the recording session. 
        Renders a vertical stack of two subplots.
        
        %matplotlib qt
        _out = curr_active_pipeline.display('_display_running_and_replay_speeds_over_time', curr_active_pipeline.get_session_context())
        _out
        
        TODO 2023-06-07 - Do I need to set up defer_render:bool=True for non-interactive plotting (like when writing to a file)?

        """
        def _subfn_add_replay_velocities(df, ax):
            """ plots the replay velocities from the dataframe on the ax """
            df['center'] = (df['stop'] + df['start'])/2.0
            for index, row in df.iterrows():
                start = row['start']
                stop = row['stop']
                center = row['center']
                
                # Single Version:
                # velocity = row['velocity']
                # ax.plot([start, stop], [velocity, velocity], label=row['label'], marker='s', markersize=4.5, color='k') # , linewidth=2.5

                # LONG/SHORT Version:
                velocity_L = row['velocity_LONG']
                ax.plot([start, stop], [velocity_L, velocity_L], label=f"{row['label']}_Long", marker='s', markersize=3.5, color='g') # , linewidth=2.5
                velocity_S = row['velocity_SHORT']
                ax.plot([start, stop], [velocity_S, velocity_S], label=f"{row['label']}_Short", marker='s', markersize=3.5, color='r') # , linewidth=2.5
                # Draw directed line
                head_length = 40.0
                # arrow_start = (start, velocity_L)
                # arrow_end = (stop, velocity_S)
                arrow_start = (center, velocity_L)
                arrow_end = (center, velocity_S) # - (head_length * 0.5) subtract off half the head-length so the arrow ends at the point
                arrow_dx = arrow_end[0] - arrow_start[0]
                arrow_dy = arrow_end[1] - arrow_start[1]
                ax.arrow(*arrow_start, arrow_dx, arrow_dy, head_width=20.0, head_length=head_length, fc='k', ec='k')
                
            # Set labels and title
            ax.set_xlabel('time')
            ax.set_ylabel('Velocity')
            ax.set_title('Replay Velocities over Time')

            # Display legend
            # ax.legend()

            return plt.gcf(), ax

        def _subfn_perform_plot(pos_df, replay_result_df, maze_epochs):
            # Create subplots grid
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

            # Plotting Running Speed over Time
            pos_df.plot(x='t', y=['lin_pos', 'speed'], title='Running Speed over Time', ax=ax1)
            epochs_collection, epoch_labels = draw_epoch_regions(maze_epochs, ax1, defer_render=False, debug_print=False)

            # plot replay velocities:
            _subfn_add_replay_velocities(replay_result_df, ax2)

            # Adjust spacing between subplots
            plt.tight_layout()

            # Show the combined plot
            if not defer_render:
                plt.show()
            return fig, (ax1, ax2), {'epochs_collection': epochs_collection, 'epoch_labels': epoch_labels}
            
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        ### Extract Relevant Data from owning_pipeline_reference:

        # Running Speed:
        # Look at lap speed over time
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        global_session.position.compute_higher_order_derivatives() # make sure the higher order derivatives are computed
        running_pos_df = global_session.position.to_dataframe()

        ## long_short_decoding_analyses:
        curr_long_short_decoding_analyses = global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
        ## Extract variables from results object:
        replay_result_df = deepcopy(curr_long_short_decoding_analyses.long_results_obj.active_filter_epochs.to_dataframe())
        maze_epochs = owning_pipeline_reference.sess.epochs
            
        fig, (ax1, ax2), plot_data_dict = _subfn_perform_plot(running_pos_df, replay_result_df, maze_epochs=maze_epochs)
        ax1.set_xlim(maze_epochs.t_start, maze_epochs.t_stop) # clip the x-lims to the maze epochs
        
        # output approach copied from `_display_long_short_laps`
        fig.canvas.manager.set_window_title('Running vs. Replay Speeds over time')
        final_context = owning_pipeline_reference.sess.get_context().adding_context('display_fn', display_fn_name='running_and_replay_speeds_over_time')


        def _perform_write_to_file_callback():
            return owning_pipeline_reference.output_figure(final_context, fig)
        
        if save_figure:
            active_out_figure_paths = _perform_write_to_file_callback()
        else:
            active_out_figure_paths = []
            
        graphics_output_dict = MatplotlibRenderPlots(name='_display_running_and_replay_speeds_over_time', figures=(fig,), axes=[ax1, ax2], plot_data=plot_data_dict, context=final_context, saved_figures=active_out_figure_paths)
        return graphics_output_dict
    
    @function_attributes(short_name='long_short_expected_v_observed_firing_rate', tags=['display','long_short','firing_rate', 'expected','observed'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-08 10:48', is_global=True)
    def _display_long_short_expected_v_observed_firing_rate(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, included_neuron_IDs=None, defer_render=False, save_figure=True, **kwargs):
        """ Displays expected v observed firing rate for each cell independently

        """
        def _subfn_prepare_plot_expected_vs_observed(curr_active_pipeline, included_neuron_IDs, defer_render:bool):
            """ 2023-06-01 - Sets up the `plot_expected_vs_observed` plot and exports it. 
            Captures: 'save_figure'
            
            Usage:
                fig, axes, final_context, active_out_figure_paths = _subfn_prepare_plot_expected_vs_observed(curr_active_pipeline)
            """
            from pyphocorehelpers.geometry_helpers import map_value # _prepare_plot_expected_vs_observed

            ## long_short_decoding_analyses:
            curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
            ## Extract variables from results object:
            long_one_step_decoder_1D, short_one_step_decoder_1D, long_replays, short_replays, global_replays, long_shared_aclus_only_decoder, short_shared_aclus_only_decoder, shared_aclus, long_short_pf_neurons_diff, n_neurons, long_results_obj, short_results_obj, is_global = curr_long_short_decoding_analyses.long_decoder, curr_long_short_decoding_analyses.short_decoder, curr_long_short_decoding_analyses.long_replays, curr_long_short_decoding_analyses.short_replays, curr_long_short_decoding_analyses.global_replays, curr_long_short_decoding_analyses.long_shared_aclus_only_decoder, curr_long_short_decoding_analyses.short_shared_aclus_only_decoder, curr_long_short_decoding_analyses.shared_aclus, curr_long_short_decoding_analyses.long_short_pf_neurons_diff, curr_long_short_decoding_analyses.n_neurons, curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj, curr_long_short_decoding_analyses.is_global

            ## TODO: add these to `expected_v_observed_result``:
            decoder_1D_LONG = long_results_obj.original_1D_decoder
            decoder_1D_SHORT = short_results_obj.original_1D_decoder
            assert (decoder_1D_LONG.neuron_IDs == decoder_1D_SHORT.neuron_IDs).all()
            neuron_IDs = decoder_1D_LONG.neuron_IDs.copy()
            assert (decoder_1D_LONG.neuron_IDXs == decoder_1D_SHORT.neuron_IDXs).all()
            neuron_IDXs = decoder_1D_LONG.neuron_IDXs.copy()
            
            if included_neuron_IDs is None:
                included_neuron_IDs = neuron_IDs
            else:
                assert len(included_neuron_IDs) > 0, f"{included_neuron_IDs} should not be empty."
                is_included_neuronID = np.isin(neuron_IDs, included_neuron_IDs)
                neuron_IDs = neuron_IDs[is_included_neuronID]
                neuron_IDXs = neuron_IDXs[is_included_neuronID]
                

            ## Get global 'long_short_post_decoding' results:
            curr_long_short_post_decoding = curr_active_pipeline.global_computation_results.computed_data['long_short_post_decoding']
            expected_v_observed_result = curr_long_short_post_decoding.expected_v_observed_result


            num_epochs = len(expected_v_observed_result.num_timebins_in_epoch)
            

            ## Various sets of display args that can be used:
            # display_kwargs = dict(x_variable='time', variable='decode_tbins_obs_exp_diff')
            # t_SHARED = expected_v_observed_result.Flat_decoder_time_bin_centers.copy()
            # y_LONG = expected_v_observed_result.Flat_all_epochs_computed_expected_cell_num_spikes_LONG.copy()
            # y_SHORT = expected_v_observed_result.Flat_all_epochs_computed_expected_cell_num_spikes_SHORT.copy()


            # display_kwargs = dict(x_variable='time', variable='obs_exp_diff_Max')
            # t_SHARED = expected_v_observed_result.Flat_epoch_time_bins_mean.copy()
            # y_LONG = expected_v_observed_result.all_epochs_computed_observed_from_expected_difference_maximum_LONG.copy()
            # y_SHORT = expected_v_observed_result.all_epochs_computed_observed_from_expected_difference_maximum_SHORT.copy()


            # t_SHARED = Flat_epoch_time_bins_mean.copy() # time mode
            # display_kwargs = dict(x_variable='time', variable='obs_exp_diff_ptp')


            display_kwargs = dict(x_variable='epoch_idx', variable='obs_exp_diff_ptp')
            t_SHARED = np.arange(num_epochs) # one for each epoch if not using time
            y_LONG = expected_v_observed_result.observed_from_expected_diff_ptp_LONG.copy()
            y_SHORT = expected_v_observed_result.observed_from_expected_diff_ptp_SHORT.copy()
            
            # display_kwargs = dict(x_variable='epoch_idx', variable='obs_exp_diff_mean')
            # y_LONG = observed_from_expected_diff_mean_LONG.copy()
            # y_SHORT = observed_from_expected_diff_mean_SHORT.copy()
            # assert y_LONG.shape[0] == t_SHARED.shape


            ## Settings
            # # sharey = False
            # sharey=True
            # shift_offset = 0 # num aclus to offset
            # # y_scale = "log" # ax.set_yscale()
            # y_scale = "linear" 

            if display_kwargs['x_variable'] == 'epoch_idx':
                # Map the times to the epoch index axes:
                map_value_time_to_epoch_idx_space = lambda v: map_value(v, (expected_v_observed_result.Flat_epoch_time_bins_mean[0], expected_v_observed_result.Flat_epoch_time_bins_mean[-1]), (0, (num_epochs-1))) # same map
                track_epochs_index_space = deepcopy(curr_active_pipeline.sess.epochs)
                track_epochs_index_space._df[['start','stop','duration']] = map_value_time_to_epoch_idx_space(track_epochs_index_space._df[['start','stop','duration']]) # convert epochs to array index space        
                track_epochs = track_epochs_index_space # index space
            elif display_kwargs['x_variable'] == 'time':
                track_epochs = curr_active_pipeline.sess.epochs # time space
            else:
                raise NotImplementedError
                

            final_context = curr_active_pipeline.sess.get_context().adding_context('display_fn', display_fn_name='plot_expected_vs_observed').adding_context('display_kwargs', **display_kwargs)
            # print(f'num_neurons: {expected_v_observed_result.num_neurons}')
            # # active_num_rows = min(num_neurons, 20)
            # active_num_rows = expected_v_observed_result.num_neurons

            # # figsize=(32,16)
            # figsize=(4, 13) #(24, 8)
            
            fig, axes = plot_expected_vs_observed(t_SHARED, y_SHORT, y_LONG, neuron_IDXs=neuron_IDXs, neuron_IDs=neuron_IDs, track_epochs=track_epochs, sharey=True, figsize=(4, 13), max_num_rows=10, y_scale="linear")
            if not defer_render:
                plt.show()

            if save_figure:
                active_out_figure_paths = owning_pipeline_reference.output_figure(final_context, fig)
            else:
                active_out_figure_paths = []

            return fig, axes, final_context, active_out_figure_paths

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        fig, axes, final_context, active_out_figure_paths = _subfn_prepare_plot_expected_vs_observed(owning_pipeline_reference, included_neuron_IDs=included_neuron_IDs, defer_render=defer_render)

        graphics_output_dict = MatplotlibRenderPlots(name='long_short_expected_v_observed_firing_rate', figures=(fig,), axes=(axes,), context=final_context, plot_data={'context': final_context, 'path': active_out_figure_paths})
        return graphics_output_dict

    @function_attributes(short_name=None, tags=['long_short_stacked_epoch_slices', 'epoch', 'needs_improvement', 'inefficient'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_decoded_epoch_slices_paginated'], used_by=[], creation_date='2023-06-02 14:12', is_global=True)
    def _display_long_and_short_stacked_epoch_slices(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, included_epoch_indicies=None, defer_render=False, save_figure=True, **kwargs):
        """ Plots two figures showing the entire stack of decoded epochs for both the long and short, including their Radon transformed lines if that information is available.

        """
        def _subfn_prepare_plot_long_and_short_stacked_epoch_slices(curr_active_pipeline, included_epoch_indicies=None, defer_render=True, save_figure=True):
            """ 2023-06-01 - 
            
            ## TODO 2023-06-02 NOW, NEXT: this might not work in 'AGG' mode because it tries to render it with QT, but we can see.
            
            Usage:
                (pagination_controller_L, pagination_controller_S), (fig_L, fig_S), (ax_L, ax_S), (final_context_L, final_context_S), (active_out_figure_paths_L, active_out_figure_paths_S) = _subfn_prepare_plot_long_and_short_stacked_epoch_slices(curr_active_pipeline, defer_render=False)
            """
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices_paginated

            ## long_short_decoding_analyses:
            curr_long_short_decoding_analyses = curr_active_pipeline.global_computation_results.computed_data['long_short_leave_one_out_decoding_analysis']
            ## Extract variables from results object:
            long_results_obj, short_results_obj = curr_long_short_decoding_analyses.long_results_obj, curr_long_short_decoding_analyses.short_results_obj
            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

            pagination_controller_L, active_out_figure_paths_L, final_context_L = plot_decoded_epoch_slices_paginated(curr_active_pipeline, long_results_obj, curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj'), included_epoch_indicies=included_epoch_indicies, save_figure=save_figure)
            fig_L = pagination_controller_L.plots.fig
            ax_L = fig_L.get_axes()
            if defer_render:
                widget_L = pagination_controller_L.ui.mw # MatplotlibTimeSynchronizedWidget
                widget_L.close()
                pagination_controller_L = None

            
            pagination_controller_S, active_out_figure_paths_S, final_context_S = plot_decoded_epoch_slices_paginated(curr_active_pipeline, short_results_obj, curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='short_results_obj'), included_epoch_indicies=included_epoch_indicies, save_figure=save_figure)
            fig_S = pagination_controller_S.plots.fig
            ax_S = fig_S.get_axes()
            if defer_render:
                widget_S = pagination_controller_S.ui.mw # MatplotlibTimeSynchronizedWidget
                widget_S.close()
                pagination_controller_S = None

            return (pagination_controller_L, pagination_controller_S), (fig_L, fig_S), (ax_L, ax_S), (final_context_L, final_context_S), (active_out_figure_paths_L, active_out_figure_paths_S)

        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        pagination_controllers, figs, axs, ctxts, out_figure_paths = _subfn_prepare_plot_long_and_short_stacked_epoch_slices(owning_pipeline_reference, included_epoch_indicies=included_epoch_indicies, defer_render=defer_render, save_figure=save_figure)
        graphics_output_dict = MatplotlibRenderPlots(name='long_short_stacked_epoch_slices', figures=figs, axes=axs, context=ctxts, plot_data={'context': ctxts, 'path': out_figure_paths})
        if not defer_render:
            graphics_output_dict.plot_data['controllers'] = pagination_controllers
            
        return graphics_output_dict




# ==================================================================================================================== #
# Private Display Helpers                                                                                              #
# ==================================================================================================================== #



# ==================================================================================================================== #
def _temp_draw_jonathan_ax(t_split, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, colors=None, fig=None, ax=None, active_aclu:int=0, custom_replay_markers=None, include_horizontal_labels=True, include_vertical_labels=True, should_render=False):
    """ Draws the time binned firing rates and the replay firing rates for a single cell

        This is the top half of each pho-jonathan-style single cell plot

        custom_replay_markers:
            # The colors for each point indicating the percentage of participating cells that belong to which track.
                - More long_only -> more red
                - More short_only -> more blue


    Black circles are replays where this aclu was not active. They'll all be at y=0 because the cell didn't fire during them.
    Green circles are replays where this aclu did fire, and their y-height indicates the cell's firing rate during that replay.
    orange circles are this aclu's inter_replay_frs
    translucent lilac (purple) curve is this aclu's time-binned firing rate
    


    Usage:

        index = new_index
        active_aclu = int(joined_df.index[index])
        _temp_draw_jonathan_ax(ax[0,1])

        t_split = sess.paradigm[0][0,1]
        _temp_draw_jonathan_ax(t_split, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, colors=colors, fig=None, ax=ax[0,1], active_aclu=active_aclu)

    Historical:
        used to take sess: DataSession as first argument and then access `sess.paradigm[0][0,1]` internally. On 2022-11-27 refactored to take this time `t_split` directly and no longer require session

    TODO:
        The `colors` argument is only used to plot the irdf (which only happens if `show_inter_replay_frs == True`), and seems uneeded. Could be removed through entire call-tree.

    """
    assert ax is not None
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'];

    show_replay_neuron_participation_distribution_labels = False
    # print(f"selected neuron has index: {index} aclu: {active_aclu}")

    # this redraws ax
    ax.clear()

    plot_replays_kwargs = {}
    is_aclu_active_in_replay = np.array([active_aclu in replay_active_aclus for replay_active_aclus in rdf.active_aclus]) # .shape (743,)
    centers = (rdf["start"].values + rdf["end"].values)/2
    heights = make_fr(rdf)[:, aclu_to_idx[active_aclu]]

    if custom_replay_markers is not None:
        ### New 2022-11-28 Custom Scatter Marker Mode (using `custom_replay_markers`):
        assert isinstance(custom_replay_markers, list)
        for replay_idx, curr_out_plot_kwargs in enumerate(custom_replay_markers):
            # if replay_idx < 5:
            if is_aclu_active_in_replay[replay_idx]:
                for i, out_plot_kwarg in enumerate(curr_out_plot_kwargs):
                    # this should be only iterate through the two separate paths to be plotted
                    ax.plot(centers[replay_idx], heights[replay_idx], markersize=5, **out_plot_kwarg, zorder=7) # , label=f'replay[{replay_idx}]'
            else:
                # don't do the fancy custom makers for the inactive (zero firing for this aclu) replay points:
                plot_replays_kwargs = {
                    'marker':'o',
                    's': 3,
                    'c': 'black'
                }
                ax.scatter(centers, heights, **plot_replays_kwargs, zorder=5) # , label=f'replay[{replay_idx}]'
                # pass # don't plot at all

    else:
        # else no custom replay markers
        extra_plots_replays_kwargs_list = None
        if 'neuron_type_distribution_color_RGB' in rdf.columns:
            ### Single-SCATTER MODE:
            # direct color mode:
            # plot_replays_kwargs['c'] = rdf.neuron_type_distribution_color.values.tolist()
            # plot_replays_kwargs['edgecolors'] = 'black'
            # plot_replays_kwargs = {
            #     'marker':'o',
            #     's': 5,
            #     'c': rdf.neuron_type_distribution_color_RGB.values.tolist(),
            #     # 'edgecolors': 'black',
            #     # 'linewidths': 2.0,
            #     # 'fillstyle': 'left'
            # }
            # scalar colors with colormap mode:
            # plot_replays_kwargs['cmap'] = 'PiYG' # 'coolwarm' # 'PiYG'
            # plot_replays_kwargs['edgecolors'] = 'black'

            ### edge indicator mode:
            plot_replays_kwargs = {'marker':'o',
                's': 5,
                'c': 'black',
                'edgecolors': rdf.neuron_type_distribution_color_RGB.values.tolist(),
                'linewidths': 5,
                'alpha': 0.5
            }

            # ### MULTI-SCATTER MODE: this doesn't really work well and wasn't finished.
            # plot_replays_kwargs = {'marker':'o',
            #     's': _marker_shared,
            #     'c': 'black',
            #     'edgecolors': rdf.neuron_type_distribution_color_RGB.values.tolist(),
            #     'linewidths': 5,
            #     'alpha': 0.1
            # }
            # secondary_plot_replays_kwargs = {
            #     's': _marker_shared+_marker_long_only,
            #     'c': 'green',
            #     'alpha': 0.9
            # }
            # third_plot_replays_kwargs = {
            #     's': _marker_shared+_marker_long_only+_marker_short_only,
            #     'c': 'red',
            #     'alpha': 0.9
            # }
            # extra_plots_replays_kwargs_list = [secondary_plot_replays_kwargs, third_plot_replays_kwargs]


            # NOTE: 'markeredgewidth' was renamed to 'linewidths'
            # ax.plot(centers, heights, '.', **plot_replays_kwargs)
            ax.scatter(centers, heights, **plot_replays_kwargs)
            if extra_plots_replays_kwargs_list is not None:
                for curr_plot_kwargs in extra_plots_replays_kwargs_list:
                    ax.scatter(centers, heights, **curr_plot_kwargs, zorder=5) # double stroke style
                    # for plot command instead of scatter
                    # curr_plot_kwargs['markersize'] = curr_plot_kwargs.popitem('s', None)
                    # ax.plot(centers, heights, **curr_plot_kwargs) # double stroke style


    if show_replay_neuron_participation_distribution_labels:
        n_replays = np.shape(rdf)[0]
        _percent_long_only = rdf.num_long_only_neuron_participating.values
        _percent_shared = rdf.num_shared_neuron_participating.values
        _percent_short_only = rdf.num_short_only_neuron_participating.values
        # for i, txt in enumerate(n):
        for i in np.arange(n_replays):
            if is_aclu_active_in_replay[i]:
                # only add the text for active replays for this cell (those where it had non-zero firing):
                txt = f'{_percent_long_only[i]}|{_percent_shared[i]}|{_percent_short_only[i]}'
                ax.annotate(txt, (centers.to_numpy()[i], heights[i]), fontsize=6)

    if show_inter_replay_frs:
        # this would show the inter-replay firing times in orange it's frankly distracting
        centers = (irdf["start"] + irdf["end"])/2
        heights = make_fr(irdf)[:, aclu_to_idx[active_aclu]]
        ax.plot(centers, heights, '.', color=colors[1]+"80", zorder=4, label='inter_replay_frs')

    if include_horizontal_labels:
        ax.set_title(f"Replay firing rates for neuron {active_aclu}")
        ax.set_xlabel("Time of replay (s)")

    if include_vertical_labels:
        ax.set_ylabel("Firing Rate (Hz)")

    # Pho's firing rate additions:
    try:
        t = time_bins
        v = unit_specific_time_binned_firing_rates[active_aclu].to_numpy() # index directly by ACLU
        if v is not None:
            # Plot the continuous firing rates
            ax.plot(t, v, color='#aaaaff8c', zorder=2, label='time_binned_frs') # this color is a translucent lilac (purple) color)
    except KeyError:
        print(f'non-placefield neuron. Skipping.')
        t, v = None, None
        pass


    ## Get the track configs for the colors:
    long_short_display_config_manager = LongShortDisplayConfigManager()
    long_epoch_config = long_short_display_config_manager.long_epoch_config.as_matplotlib_kwargs()
    short_epoch_config = long_short_display_config_manager.short_epoch_config.as_matplotlib_kwargs()
    
    # Highlight the two epochs with their characteristic colors ['r','b'] - ideally this would be at the very back
    x_start, x_stop = ax.get_xlim()
    ax.axvspan(x_start, t_split, color=long_epoch_config['facecolor'], alpha=0.2, zorder=0)
    ax.axvspan(t_split, x_stop, color=short_epoch_config['facecolor'], alpha=0.2, zorder=0)

    # Draw the vertical epoch splitter line:
    required_epoch_bar_height = ax.get_ylim()[-1]
    ax.vlines(t_split, ymin = 0, ymax=required_epoch_bar_height, color=(0,0,0,.25), zorder=25) # divider should be in very front

    if should_render:
        if fig is None:
            fig = plt.gcf()

        fig.canvas.draw()

def _temp_draw_jonathan_spikes_on_track(ax, pos_df, single_neuron_spikes):
    """ this plots where the neuron spiked on the track

    Usage:
        single_neuron_spikes = sess.spikes_df[sess.spikes_df.aclu == aclu]
        _temp_draw_jonathan_spikes_on_track(ax[1,1], pos_df, single_neuron_spikes)
    """
    ax.clear()
    ax.plot(pos_df.t, pos_df.x, color=[.75, .75, .75])

    ax.plot(single_neuron_spikes.t_rel_seconds, single_neuron_spikes.x, 'k.', ms=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Position")
    ax.set_title("Animal position on track")

# ==================================================================================================================== #
def _make_jonathan_interactive_plot(sess, time_bins, neuron_replay_stats_df, unit_specific_time_binned_firing_rates, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False):

    # ==================================================================================================================== #
    ## Plotting/Graphics:
    fig, ax = plt.subplots(2,2, figsize=(12.11,4.06));
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'];

    graphics_output_dict = {'fig': fig, 'axs': ax, 'colors': colors}

    # plotting for ax[0,0] _______________________________________________________________________________________________ #
    ax[0,0].axis("equal");

    # I initially set the boundaries like this so I would know where to put the single-track cells
    # I'm sure there's a better way, though
    ylim = (-58.34521620102153, 104.37547397480944)
    xlim = (-97.76920925869598, 160.914964866984)

    # this fills in the nan's in the single-track cells so that they get plotted at the edges
    # plotting everything in one go makes resizing points later simpler
    neuron_replay_stats_df.long_pf_peak_x.fillna(xlim[0] + 1, inplace=True) # xlim[0] + 1 is the extreme edge of the plot
    neuron_replay_stats_df.short_pf_peak_x.fillna(ylim[0] + 1, inplace=True)

    remap_scatter = ax[0,0].scatter(neuron_replay_stats_df.long_pf_peak_x, neuron_replay_stats_df.short_pf_peak_x, s=7, picker=True, c=[colors[c] for c in neuron_replay_stats_df["has_na"]]);
    ax[0,0].set_ylim(ylim);
    ax[0,0].set_xlim(xlim);
    ax[0,0].xaxis.set_tick_params(labelbottom=False)
    ax[0,0].yaxis.set_tick_params(labelleft=False)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])

    ax[0,0].set_xlabel("Distance along long track")
    ax[0,0].set_ylabel("Distance along short track")
    ax[0,0].set_title("Peak tuning on short vs. long track")

    graphics_output_dict['remap_scatter'] = remap_scatter

    # plotting for ax[1,0]: ______________________________________________________________________________________________ #
    diff_scatter = ax[1,0].scatter(neuron_replay_stats_df.non_replay_diff, neuron_replay_stats_df.replay_diff, s=7, picker=True);
    # ax[1,0].set_xlabel("Firing rate along long track")
    # ax[1,0].set_ylabel("Firing rate along short track")
    ax[1,0].set_title("Firing rate on short vs. long track")

    graphics_output_dict['diff_scatter'] = diff_scatter

    #TODO
    # diff_scatter = ax[1,0].scatter(scaled_participation, d_activity, s=7, picker=True);

    g_index = 0 # this stands for global index
    # it keeps track of the index of the neuron we have selected
    # this is the index in the dataframe (if you were using `iloc`), and not the ACLU

    # pos_df = sess.position.to_dataframe()

    def on_index_change(new_index):
        'This gets called when the selected neuron changes; it updates the graphs'

        index = new_index
        aclu = int(neuron_replay_stats_df.index[index])
        print(f"selected neuron has index: {index} aclu: {aclu}")

        # this changes the size of the neuron in ax[0,0]
        remap_scatter.set_sizes([7 if i!= index else 30 for i in range(len(neuron_replay_stats_df))])

        # this changes the size of the neuron in ax[1,0]
        diff_scatter.set_sizes([7 if i!= index else 30 for i in range(len(neuron_replay_stats_df))])

        ## New ax[0,1] draw method:
        t_split = sess.paradigm[0][0,1]
        _temp_draw_jonathan_ax(t_split, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, colors=colors, fig=fig, ax=ax[0,1], active_aclu=aclu, should_render=True)

        # this plots where the neuron spiked on the track
        single_neuron_spikes = sess.spikes_df[sess.spikes_df.aclu == aclu]
        _temp_draw_jonathan_spikes_on_track(ax[1,1], pos_df, single_neuron_spikes)

        fig.canvas.draw()


    def on_keypress(event):
        global g_index
        if event.key=='tab':
            g_index += 1
            g_index %= len(neuron_replay_stats_df)
        elif event.key=='b':
            g_index -= 1
            g_index %= len(neuron_replay_stats_df)
        on_index_change(g_index)


    def on_pick(event):
        on_index_change(int(event.ind[0]))

    on_index_change(g_index)

    graphics_output_dict['on_index_change'] = {'callback': on_index_change, 'g_index': g_index}


    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    return graphics_output_dict, neuron_replay_stats_df
# ==================================================================================================================== #

def _build_spikes_df_interpolated_props(global_results, should_interpolate_to_filtered_positions:bool=False):
    """ Interpolates the spikes_df's spike positions and other properties from the measured positions, etc. 

    IMPORTANT: the position to be used for interpolation for each spike depends on whether we're only using the filtered positions or not.
        2023-09-22 - as of now, deciding to NOT use filtered positions so the spike dots will render appropriately for the endcaps.

    """
    # Group by the aclu (cluster indicator) column
    cell_grouped_spikes_df = global_results.sess.spikes_df.groupby(['aclu'])
    cell_spikes_dfs = [cell_grouped_spikes_df.get_group(a_neuron_id) for a_neuron_id in global_results.sess.spikes_df.spikes.neuron_ids] # a list of dataframes for each neuron_id
    aclu_to_fragile_linear_idx_map = {a_neuron_id:i for i, a_neuron_id in enumerate(global_results.sess.spikes_df.spikes.neuron_ids)}
    # get position variables usually used within pfND.setup(...) - self.t, self.x, self.y:
    ndim = global_results.computed_data.pf1D.ndim
    if should_interpolate_to_filtered_positions:
        # restrict to only the filtered positions. I think this is usually NOT what we want.
        pos_df = global_results.computed_data.pf1D.filtered_pos_df
    else:
        # pos_df = global_results.computed_data.pf1D.pos_df
        pos_df = global_results.computed_data.pf1D.position.to_dataframe()

    t = pos_df.t.to_numpy()
    x = pos_df.x.to_numpy()
    if (ndim > 1):
        y = pos_df.y.to_numpy()
    else:
        y = None

    # spk_pos, spk_t = [], []
    # re-interpolate given the updated spks
    for cell_df in cell_spikes_dfs:
        cell_spike_times = cell_df[global_results.sess.spikes_df.spikes.time_variable_name].to_numpy()
        spk_x = np.interp(cell_spike_times, t, x) # TODO: shouldn't we already have interpolated spike times for all spikes in the dataframe?

        # update the dataframe 'x', 'y' properties:
        cell_df.loc[:, 'x'] = spk_x
        if (ndim > 1):
            spk_y = np.interp(cell_spike_times, t, y) # TODO: shouldn't we already have interpolated spike times for all spikes in the dataframe?
            cell_df.loc[:, 'y'] = spk_y
            # spk_pos.append([spk_x, spk_y])        
        # else:
        #     # otherwise only 1D:
        #     spk_pos.append([spk_x])
            
        # spk_t.append(cell_spike_times)

    # spk_pos[0][0].shape # (214,)
    # returns (spk_t, spk_pos) arrays that can be used to plot spikes
    # return cell_spikes_dfs_list, aclu_to_fragile_linear_idx_map #, (spk_t, spk_pos)
    return {a_neuron_id:cell_spikes_dfs[i] for i, a_neuron_id in enumerate(global_results.sess.spikes_df.spikes.neuron_ids)}, aclu_to_fragile_linear_idx_map # return a dict instead


def _simple_plot_spikes(ax, a_spk_t, a_spk_pos, spikes_color_RGB=(1, 0, 0), spikes_alpha=0.2, **kwargs):
    spikes_color_RGBA = [*spikes_color_RGB, spikes_alpha]
    spike_plot_kwargs = ({'linestyle':'none', 'markersize':5.0, 'marker': '.', 'markerfacecolor':spikes_color_RGBA, 'markeredgecolor':spikes_color_RGBA, 'zorder':10} | kwargs)
    ax.plot(a_spk_t, a_spk_pos, color=spikes_color_RGBA, **(spike_plot_kwargs or {})) # , color=[*spikes_color, spikes_alpha]
    return ax

def _plot_general_all_spikes(ax_activity_v_time, active_spikes_df, time_variable_name='t', spikes_alpha=0.9, defer_render=True):
    """ Plots all spikes for a given cell from that cell's complete `active_spikes_df`
    There are three different classes of spikes: all (black), long (red), short (blue)
    
    
    Usage:

        curr_aclu_axs = axs[-2]
        ax_activity_v_time = curr_aclu_axs['lap_spikes']

    ## Serving to replace:
    active_epoch_placefields1D.plotRaw_v_time(placefield_cell_index, ax=ax_activity_v_time, spikes_alpha=spikes_alpha,
            position_plot_kwargs={'color': '#393939c8', 'linewidth': 1.0, 'zorder':5},
            spike_plot_kwargs=spike_plot_kwargs, should_include_labels=False
        ) # , spikes_color=spikes_color, spikes_alpha=spikes_alpha
    """

    # ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_spikes_df[global_results.sess.spikes_df.spikes.time_variable_name].values, active_spikes_df['x'].values, spikes_color_RGB=(0, 0, 0), spikes_alpha=1.0) # all
    ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_spikes_df[time_variable_name].values, active_spikes_df['x'].values, spikes_color_RGB=(0.1, 0.1, 0.1), spikes_alpha=spikes_alpha) # all

    active_long_spikes_df = active_spikes_df[active_spikes_df.is_included_long_pf1D]
    ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_long_spikes_df[time_variable_name].values, active_long_spikes_df['x'].values, spikes_color_RGB=(1, 0, 0), spikes_alpha=spikes_alpha, zorder=15)

    active_short_spikes_df = active_spikes_df[active_spikes_df.is_included_short_pf1D]
    ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_short_spikes_df[time_variable_name].values, active_short_spikes_df['x'].values, spikes_color_RGB=(0, 0, 1), spikes_alpha=spikes_alpha, zorder=15)

    # active_global_spikes_df = active_spikes_df[active_spikes_df.is_included_global_pf1D]
    # ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_global_spikes_df[time_variable_name].values, active_global_spikes_df['x'].values, spikes_color_RGB=(0, 1, 0), spikes_alpha=1.0, zorder=25, markersize=2.5)

    if not defer_render:
        fig = ax_activity_v_time.get_figure().get_figure() # For SubFigure
        fig.canvas.draw()

    return ax_activity_v_time


@function_attributes(short_name='_plot_pho_jonathan_batch_plot_single_cell', tags=['private', 'matplotlib'], input_requires=[], output_provides=[], uses=['plot_1D_placecell_validation', '_temp_draw_jonathan_ax', '_plot_general_all_spikes'], used_by=['_make_pho_jonathan_batch_plots'], creation_date='2023-04-11 08:06')
def _plot_pho_jonathan_batch_plot_single_cell(t_split, time_bins, unit_specific_time_binned_firing_rates, pf1D_all, rdf_aclu_to_idx, rdf, irdf, show_inter_replay_frs, pf1D_aclu_to_idx, aclu, curr_fig, colors, debug_print=False, **kwargs):
    """ Plots a single cell's plots for a stacked Jonathan-style firing-rate-across-epochs-plot
    Internally calls `plot_1D_placecell_validation`, `_temp_draw_jonathan_ax`, and `_plot_general_all_spikes`

    Used by:
        `_make_pho_jonathan_batch_plots`

    Historical:
        used to take sess: DataSession as first argument and then access `sess.paradigm[0][0,1]` internally. On 2022-11-27 refactored to take this time `t_split` directly and no longer require session


    """
    # short_title_string = f'{aclu:02d}'

    formatted_cell_label_string = (f"<size:22><weight:bold>{aclu:02d}</></>")
    
    ## Optional additional information about the cell to be rendered next to its aclu, like it's firing rate indicies, etc:
    optional_cell_info_labels_dict = kwargs.get('optional_cell_info_labels', {})
    optional_cell_info_labels = optional_cell_info_labels_dict.get(aclu, None) # get the single set of optional labels for this aclu


    # the index passed into `plot_1D_placecell_validation(...)` must be in terms of the `pf1D_all` ratemap that's provided. the `rdf_aclu_to_idx` does NOT work and will result in indexing errors
    # pf1D_aclu_to_idx = {aclu:i for i, aclu in enumerate(pf1D_all.ratemap.neuron_ids)}

    # Not sure if this is okay, but it's possible that the aclu isn't in the ratemap, in which case currently we'll just skip plotting?
    cell_linear_fragile_IDX = pf1D_aclu_to_idx.get(aclu, None)
    if cell_linear_fragile_IDX is None:
        print(f'WARNING: aclu {aclu} is not present in the `pf1D_all` ratemaps. Which contain aclus: {pf1D_all.ratemap.neuron_ids}') #TODO 2023-07-07 20:55: - [ ] Note this is hit all the time, not sure what it's supposed to warn about
    else:
        cell_neuron_extended_ids = pf1D_all.ratemap.neuron_extended_ids[cell_linear_fragile_IDX]
        # print(f'aclu: {aclu}, cell_neuron_extended_ids: {cell_neuron_extended_ids}')
        # subtitle_string = f'(shk <size:10><weight:bold>{cell_neuron_extended_ids.shank}</></>, clu <size:10><weight:bold>{cell_neuron_extended_ids.cluster}</></>)'
        try:
            subtitle_string = f'shk <size:10><weight:bold>{cell_neuron_extended_ids.shank}</></>, clu <size:10><weight:bold>{cell_neuron_extended_ids.cluster}</></>\nqclu <size:10><weight:bold>{cell_neuron_extended_ids.quality}</></>\ntype <size:10><weight:bold>{cell_neuron_extended_ids.cell_type}</></>'
        except AttributeError: # 'NeuronExtendedIdentityTuple' object has no attribute 'quality'
            # Older definition that is missing the quality and neuron type:
            subtitle_string = f'shk <size:10><weight:bold>{cell_neuron_extended_ids.shank}</></>, clu <size:10><weight:bold>{cell_neuron_extended_ids.cluster}</></>'
            

        # print(f'\tsubtitle_string: {subtitle_string}')
        formatted_cell_label_string = f'{formatted_cell_label_string}\n<size:9>{subtitle_string}</>'
    
    if optional_cell_info_labels is not None:
        print(f'has optional_cell_info_labels: {optional_cell_info_labels}')
        optional_cell_info_labels_string: str = optional_cell_info_labels # already should be a string
        # formatted_cell_label_string = f'{formatted_cell_label_string}\n<size:9>{optional_cell_info_labels_string}</>' # single label mode
        optional_formatted_cell_label_string = f'<size:9>{optional_cell_info_labels_string}</>' # separate single mode
    else:
        optional_formatted_cell_label_string = ''

    # cell_linear_fragile_IDX = rdf_aclu_to_idx[aclu] # get the cell_linear_fragile_IDX from aclu
    # title_string = ' '.join(['pf1D', f'Cell {aclu:02d}'])
    # subtitle_string = ' '.join([f'{pf1D_all.config.str_for_display(False)}'])
    # if debug_print:
    #     print(f'\t{title_string}\n\t{subtitle_string}')
    from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1D_placecell_validation
    
    # gridspec mode:
    curr_fig.set_facecolor('0.65') # light grey

    # num_gridspec_columns = 8 # hardcoded
    # gs_kw = dict(width_ratios=np.repeat(1, num_gridspec_columns).tolist(), height_ratios=[1, 1], wspace=0.0, hspace=0.0)
    # gs_kw['width_ratios'][-1] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others

    # gs = curr_fig.add_gridspec(2, num_gridspec_columns, **gs_kw) # layout figure is usually a gridspec of (1,8)
    # curr_ax_firing_rate = curr_fig.add_subplot(gs[0, :-1]) # the whole top row except the last element (to match the firing rates below)
    # curr_ax_cell_label = curr_fig.add_subplot(gs[0, -1]) # the last element of the first row contains the labels that identify the cell
    # curr_ax_lap_spikes = curr_fig.add_subplot(gs[1, :-1]) # all up to excluding the last element of the row
    # curr_ax_placefield = curr_fig.add_subplot(gs[1, -1], sharey=curr_ax_lap_spikes) # only the last element of the row


    # # New Gridspec - Both left and right columns:
    # num_gridspec_columns = 10 # hardcoded
    # gs_kw = dict(width_ratios=np.repeat(1, num_gridspec_columns).tolist(), height_ratios=[1, 1], wspace=0.0, hspace=0.0)
    # gs_kw['width_ratios'][0] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others
    # gs_kw['width_ratios'][-2] = 0.2 # make the last column (containing the 1D placefield plot) a fraction of the width of the others
    # gs_kw['width_ratios'][-1] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others

    # gs = curr_fig.add_gridspec(2, num_gridspec_columns, **gs_kw) # layout figure is usually a gridspec of (1,8)
    # curr_ax_firing_rate = curr_fig.add_subplot(gs[0, 1:-2]) # the whole top row except the last element (to match the firing rates below)
    # curr_ax_cell_label = curr_fig.add_subplot(gs[0, 0]) # the last element of the first row contains the labels that identify the cell
    # curr_ax_extra_information_labels = curr_fig.add_subplot(gs[0, -2:]) # the last two element of the first row contains the labels that identify the cell
    # curr_ax_lap_spikes = curr_fig.add_subplot(gs[1, 1:-2]) # all up to excluding the last element of the row
    # curr_ax_placefield = curr_fig.add_subplot(gs[1, -2], sharey=curr_ax_lap_spikes) # only the last element of the row
    

    # New Gridspec - Both left and right columns:
    num_gridspec_columns = 9 # hardcoded
    gs_kw = dict(width_ratios=np.repeat(1, num_gridspec_columns).tolist(), height_ratios=[1, 1], wspace=0.0, hspace=0.0)
    # gs_kw['width_ratios'][0] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others
    # gs_kw['width_ratios'][1] = 0.0 # make the last column (containing the 1D placefield plot) a fraction of the width of the others
    # gs_kw['width_ratios'][-1] = 0.1 # make the last column (containing the 1D placefield plot) a fraction of the width of the others

    gs = curr_fig.add_gridspec(2, num_gridspec_columns, **gs_kw) # layout figure is usually a gridspec of (1,8)
    curr_ax_firing_rate = curr_fig.add_subplot(gs[0, 1:-1], label=f'ax_firing_rate[{aclu:02d}]') # the whole top row except the last element (to match the firing rates below)
    curr_ax_cell_label = curr_fig.add_subplot(gs[0, 0], label=f'ax_cell_label[{aclu:02d}]') # the last element of the first row contains the labels that identify the cell
    curr_ax_extra_information_labels = curr_fig.add_subplot(gs[1, 0], label=f'ax_extra_info_labels[{aclu:02d}]') # the last two element of the first row contains the labels that identify the cell

    curr_ax_lap_spikes = curr_fig.add_subplot(gs[1, 1:-1], label=f'ax_lap_spikes[{aclu:02d}]') # all up to excluding the last element of the row
    # curr_ax_left_placefield = curr_fig.add_subplot(gs[1, 1], sharey=curr_ax_lap_spikes) # only the last element of the row
    curr_ax_placefield = curr_fig.add_subplot(gs[1, -1], sharey=curr_ax_lap_spikes, label=f'ax_pf1D[{aclu:02d}]') # only the last element of the row


    text_formatter = FormattedFigureText()
    text_formatter.left_margin = 0.5
    text_formatter.top_margin = 0.5
    # text_formatter.left_margin = 0.025
    # text_formatter.top_margin = 0.00
    
    title_axes_kwargs = dict(ha="center", va="center", xycoords='axes fraction') # , ma="left"
    # title_axes_kwargs = dict(ha="left", va="bottom", ma="left", xycoords='axes fraction')
    # Setup the aclu number ("title axis"):
    # title_axes_kwargs = dict(ha="center", va="center", fontsize=22, color="black")
    # curr_ax_cell_label.text(0.5, 0.5, short_title_string, transform=curr_ax_cell_label.transAxes, **title_axes_kwargs)
    # flexitext version:
    title_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, formatted_cell_label_string, ax=curr_ax_cell_label, **title_axes_kwargs)
    # curr_ax_cell_label.set_facecolor('0.95')
    extra_information_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, optional_formatted_cell_label_string, xycoords='axes fraction', ax=curr_ax_extra_information_labels, ha="center", va="center")

    # # # `flexitext` version:
    
    # plt.title('')
    # plt.suptitle('')
    # fig = plt.gcf() # get figure to setup the margins on.
    # text_formatter.setup_margins(curr_fig)
    # # flexitext(text_formatter.left_margin, text_formatter.top_margin, '<size:22><color:crimson, weight:bold>long ($L$)</>|<color:royalblue, weight:bold>short($S$)</> <weight:bold>firing rate indicies</></>', va="bottom", xycoords="figure fraction")
    # footer_text_obj = flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")


    curr_ax_cell_label.axis('off')
    curr_ax_extra_information_labels.axis('off')

    custom_replay_scatter_markers_plot_kwargs_list = kwargs.pop('custom_replay_scatter_markers_plot_kwargs_list', None)
    # Whether to plot the orange horizontal indicator lines that show where spikes occur. Slows down plots a lot.
    should_plot_spike_indicator_points_on_placefield = kwargs.pop('should_plot_spike_indicator_points_on_placefield', False)
    
    ## New ax[0,1] draw method:
    _temp_draw_jonathan_ax(t_split, time_bins, unit_specific_time_binned_firing_rates, rdf_aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, colors=colors, fig=curr_fig, ax=curr_ax_firing_rate, active_aclu=aclu,
                        include_horizontal_labels=False, include_vertical_labels=False, should_render=False, custom_replay_markers=custom_replay_scatter_markers_plot_kwargs_list)
    # curr_ax_firing_rate includes only bottom and left spines, and only y-axis ticks and labels
    curr_ax_firing_rate.set_xticklabels([])
    curr_ax_firing_rate.spines['top'].set_visible(False)
    curr_ax_firing_rate.spines['right'].set_visible(False)
    # curr_ax_firing_rate.spines['bottom'].set_visible(False)
    # curr_ax_firing_rate.spines['left'].set_visible(False)
    curr_ax_firing_rate.get_xaxis().set_ticks([])
    # curr_ax_firing_rate.get_yaxis().set_ticks([])

    # this plots where the neuron spiked on the track
    curr_ax_lap_spikes.set_xticklabels([])
    curr_ax_lap_spikes.set_yticklabels([])
    curr_ax_lap_spikes.axis('off')

    curr_ax_placefield.set_xticklabels([])
    curr_ax_placefield.set_yticklabels([])
    curr_ax_placefield.sharey(curr_ax_lap_spikes)


    ## I think that `plot_1D_placecell_validation` is used to plot the position v time and the little placefield on the right
    _ = plot_1D_placecell_validation(pf1D_all, cell_linear_fragile_IDX, extant_fig=curr_fig, extant_axes=(curr_ax_lap_spikes, curr_ax_placefield),
            **({'should_include_labels': False, 'should_plot_spike_indicator_points_on_placefield': should_plot_spike_indicator_points_on_placefield, 
                'should_plot_spike_indicator_lines_on_trajectory': False, 'spike_indicator_lines_alpha': 0.2,
                'spikes_color':(0.1, 0.1, 0.1), 'spikes_alpha':0.1, 'should_include_spikes': False} | kwargs))

    # Custom All Spikes: Note that I set `'should_include_spikes': False` in call to `plot_1D_placecell_validation` above so the native spikes from that function aren't plotted
    cell_spikes_dfs_dict = kwargs.get('cell_spikes_dfs_dict', None)
    time_variable_name = kwargs.get('time_variable_name', None)
    if cell_spikes_dfs_dict is not None:
        assert time_variable_name is not None, f"if cell_spikes_dfs_dict is passed time_variable_name must also be passed"
        # active_spikes_df = cell_spikes_dfs[cellind]
        active_spikes_df = cell_spikes_dfs_dict[aclu]
        curr_ax_lap_spikes = _plot_general_all_spikes(curr_ax_lap_spikes, active_spikes_df, time_variable_name=time_variable_name, defer_render=True)

    t_start, t_end = curr_ax_lap_spikes.get_xlim()
    curr_ax_firing_rate.set_xlim((t_start, t_end)) # We don't want to clip to only the spiketimes for this cell, we want it for all cells, or even when the recording started/ended
    curr_ax_lap_spikes.sharex(curr_ax_firing_rate) # Sync the time axes of the laps and the firing rates

    return {'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield, 'labels': curr_ax_cell_label}


@function_attributes(short_name='_make_pho_jonathan_batch_plots', tags=['private', 'matplotlib', 'active','jonathan'], input_requires=[], output_provides=[], uses=['_plot_pho_jonathan_batch_plot_single_cell', 'build_replays_custom_scatter_markers', '_build_neuron_type_distribution_color', 'build_or_reuse_figure'], used_by=['_display_batch_pho_jonathan_replay_firing_rate_comparison'], creation_date='2023-04-11 08:06')
def _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, included_unit_neuron_IDs=None, marker_split_mode=CustomScatterMarkerMode.TriSplit, n_max_plot_rows:int=4, optional_cell_info_labels=None, debug_print=False, defer_render=False, **kwargs) -> MatplotlibRenderPlots:
    """ Stacked Jonathan-style firing-rate-across-epochs-plot
    Internally calls `_plot_pho_jonathan_batch_plot_single_cell`
        n_max_plot_rows: the maximum number of rows to plot


    # The colors for each point indicating the percentage of participating cells that belong to which track.
        - More long_only -> more red
        - More short_only -> more blue


    """
    if included_unit_neuron_IDs is None:
        n_all_neuron_IDs = np.shape(neuron_replay_stats_df)[0] 
        n_max_plot_rows = min(n_all_neuron_IDs, n_max_plot_rows) # don't allow more than the possible number of neuronIDs
        included_unit_neuron_IDs = [int(neuron_replay_stats_df.index[i]) for i in np.arange(n_max_plot_rows)]
    else:
        # truncate to n_max_plot_rows if needed:
        actual_num_unit_neuron_IDs = min(len(included_unit_neuron_IDs), n_max_plot_rows) # only include the possible rows
        if (actual_num_unit_neuron_IDs < len(included_unit_neuron_IDs)):
            print(f'WARNING: truncating included_unit_neuron_IDs of length {len(included_unit_neuron_IDs)} to length {actual_num_unit_neuron_IDs} due to n_max_plot_rows: {n_max_plot_rows}...')
            included_unit_neuron_IDs = included_unit_neuron_IDs[:actual_num_unit_neuron_IDs]

    # the index passed into plot_1D_placecell_validation(...) must be in terms of the pf1D_all ratemap that's provided. the rdf_aclu_to_idx does not work and will result in indexing errors
    _temp_aclu_to_fragile_linear_neuron_IDX = {aclu:i for i, aclu in enumerate(pf1D_all.ratemap.neuron_ids)} 

    actual_num_subfigures = min(len(included_unit_neuron_IDs), n_max_plot_rows) # only include the possible rows 
    active_context = kwargs.get('active_context', None)


    ## Figure Setup:
    fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
    subfigs = fig.subfigures(actual_num_subfigures, 1, wspace=0.07)
    ##########################

    rdf, (_percent_long_only, _percent_shared, _percent_short_only, _percent_short_long_diff) = _build_neuron_type_distribution_color(rdf) # for building partially filled scatter plot points.
    

    # Build custom replay markers:
    custom_replay_scatter_markers_plot_kwargs_list = build_replays_custom_scatter_markers(rdf, marker_split_mode=marker_split_mode, debug_print=debug_print)
    kwargs['custom_replay_scatter_markers_plot_kwargs_list'] = custom_replay_scatter_markers_plot_kwargs_list

    # Set 'optional_cell_info_labels' on the kwargs passed to `_plot_pho_jonathan_batch_plot_single_cell`
    kwargs['optional_cell_info_labels'] = optional_cell_info_labels

    # # Build all spikes interpolated positions/dfs:
    # cell_spikes_dfs_dict = kwargs.get('cell_spikes_dfs_dict', None)
    # cell_grouped_spikes_df, cell_spikes_dfs = _build_spikes_df_interpolated_props(global_results)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    axs_list = []

    ## IDEA: to change the display order, keep `_temp_aclu_to_fragile_linear_neuron_IDX` the same and just modify the order of aclu values iterated over
    # _temp_aclu_to_subfig_idx

    for i, aclu in enumerate(included_unit_neuron_IDs):

        is_first_row = (i==0)
        is_last_row = (i == (n_max_plot_rows-1))

        # aclu = int(neuron_replay_stats_df.index[i])
        if debug_print:
            print(f"selected neuron has index: {i} aclu: {aclu}")
        
        try:
            curr_fig = subfigs[i] # TypeError: 'SubFigure' object is not subscriptable
        except TypeError as e:
            # TypeError: 'SubFigure' object is not subscriptable ->  # single subfigure, not subscriptable
            curr_fig = subfigs
        except Exception as e:
            # Unhandled exception
            raise e
        
        curr_single_cell_out_dict = _plot_pho_jonathan_batch_plot_single_cell(t_split, time_bins, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs, _temp_aclu_to_fragile_linear_neuron_IDX, aclu, curr_fig, colors, debug_print=debug_print, **kwargs)

        # output the axes created:
        axs_list.append(curr_single_cell_out_dict)

    if isinstance(subfigs, FigureBase):
        subfigs = [subfigs] # wrap it to be a single item list

    # graphics_output_dict = {'fig': fig, 'subfigs': subfigs, 'axs': axs_list, 'colors': colors}
    graphics_output_dict = MatplotlibRenderPlots(name='make_pho_jonathan_batch_plots', figures=(fig,), subfigs=subfigs, axes=axs_list, plot_data={'colors': colors})

    if not defer_render:
        fig.show()
    return graphics_output_dict


# ==================================================================================================================== #




# ==================================================================================================================== #

@mpl.rc_context(Fig.get_mpl_style(style='figPublish'))
def plot_short_v_long_pf1D_comparison(long_results, short_results, curr_any_context_neurons, reuse_axs_tuple=None, single_figure=False, shared_kwargs=None, long_kwargs=None, short_kwargs=None, title_string=None, subtitle_string=None, should_plot_vertical_track_bounds_lines=False, should_plot_linear_track_shapes=False, debug_print=False):
    """ Produces a figure to compare the 1D placefields on the long vs. the short track. 
    
    single_figure:bool - if True, both long and short are plotted on the same axes of a single shared figure. Otherwise seperate figures are used for each
    should_plot_vertical_track_bounds_lines: bool - if True, vertical lines representing the bounds of the linear track are rendered
    should_plot_linear_track_shapes: bool - if True, plots 2D linear tracks on the figure
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.LongShortTrackComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_short_v_long_pf1D_comparison

        long_results = curr_active_pipeline.computation_results['maze1_PYR'].computed_data
        short_results = curr_active_pipeline.computation_results['maze2_PYR'].computed_data
        curr_any_context_neurons = _find_any_context_neurons(*[curr_active_pipeline.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in ['maze1_PYR', 'maze2_PYR']])
        reuse_axs_tuple=None # plot fresh
        # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
        # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
        (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, curr_any_context_neurons, reuse_axs_tuple=reuse_axs_tuple, single_figure=True)

    """
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_vertical_track_bounds_lines, add_track_shapes
    
    if shared_kwargs is None:
        shared_kwargs = {}
    if long_kwargs is None:
        long_kwargs = {}
    if short_kwargs is None:
        short_kwargs = {}

    # Shared/Common kwargs:
    plot_ratemap_1D_kwargs = (dict(pad=1, brev_mode=PlotStringBrevityModeEnum.NONE, normalize=True, debug_print=debug_print, normalize_tuning_curve=True, skip_figure_titles=single_figure) | shared_kwargs)
    active_context = shared_kwargs.get('active_context', None)
    print(f'active_context: {active_context}')
    if single_figure:
        plot_ratemap_1D_kwargs['skip_figure_titles'] = True

    flat_stack_mode:bool = shared_kwargs.get('flat_stack_mode', False)

    y_baseline_offset = 0.0 # 0.5 does not work uniform offset to be added to all pfmaps so that the negative-flipped one isn't cut off
    single_cell_pfmap_processing_fn_identity = lambda i, aclu, pfmap: (0.5 * pfmap) + y_baseline_offset # scale down by 1/2 so that both it and the flipped version fit on the same axis
    single_cell_pfmap_processing_fn_flipped_y = lambda i, aclu, pfmap: (-0.5 * pfmap) + y_baseline_offset # flip over the y-axis
    
        
    if flat_stack_mode:
        y_lims_offset = None
        ytick_location_shift:float = 0.0 # overrides with zero
    else:
        if single_figure:
            y_lims_offset = -0.5 # shift the y-lims down by (-0.5 * pad) so it isn't cut off
            ytick_location_shift:float = 0.0 # overrides with zero
        else:
            y_lims_offset = None
            ytick_location_shift:float = 0.5 # default
        
    plot_ratemap_1D_kwargs['ytick_location_shift'] = ytick_location_shift

    # single_cell_pfmap_processing_fn_identity = lambda i, aclu, pfmap: pfmap # flip over the y-axis
    # single_cell_pfmap_processing_fn_flipped_y = lambda i, aclu, pfmap: -1.0 * pfmap # flip over the y-axis
    # y_lims_offset = None
    
    n_neurons = len(curr_any_context_neurons)
    shared_fragile_neuron_IDXs = np.arange(n_neurons)
    
    ## sort has to be done here on `shared_fragile_neuron_IDXs`, as this is used for sortby below for both.


    # neurons_colors_array = build_neurons_color_map(n_neurons, sortby=shared_fragile_neuron_IDXs, cmap=None, included_unit_indicies=None, included_unit_neuron_IDs=curr_any_context_neurons)
    if debug_print:
        print(f'n_neurons: {n_neurons}')
        print(f'shared_fragile_neuron_IDXs: {shared_fragile_neuron_IDXs}.\t np.shape: {np.shape(shared_fragile_neuron_IDXs)}')
        print(f'curr_any_context_neurons: {curr_any_context_neurons}.\t np.shape: {np.shape(curr_any_context_neurons)}')

    if reuse_axs_tuple is not None:
        if not single_figure:
            assert len(reuse_axs_tuple) == 2
            ax_long_pf_1D, ax_short_pf_1D = reuse_axs_tuple
            fig_long_pf_1D = ax_long_pf_1D.get_figure()
            fig_short_pf_1D = ax_short_pf_1D.get_figure()
            PhoActiveFigureManager2D.reshow_figure_if_needed(fig_long_pf_1D) # TODO 2023-06-16 13:35: - [ ] Do I need to disable these for defer_render=True?
            PhoActiveFigureManager2D.reshow_figure_if_needed(fig_short_pf_1D)
        else:
            # single figure
            if isinstance(reuse_axs_tuple, tuple):
                ax_long_pf_1D = reuse_axs_tuple[0]
            else:
                # hopefully an Axis directly
                ax_long_pf_1D = reuse_axs_tuple
            # for code reuse the ax_short_pf_1D = ax_long_pf_1D, fig_short_pf_1D = fig_long_pf_1D are set after plotting the long anyway
            
    else:
        if debug_print:
            print(f'reuse_axs_tuple is None. Making new figures/axes')
        ax_long_pf_1D, ax_short_pf_1D = None, None
        fig_long_pf_1D, fig_short_pf_1D = None, None
        
    ## Need to determine the same sort for both of them?

    # Long/Short Specific (Distinguishing) kwargs:
    long_kwargs = (plot_ratemap_1D_kwargs | {'sortby': shared_fragile_neuron_IDXs, 'included_unit_neuron_IDs': curr_any_context_neurons, 'fignum': None,  'ax': ax_long_pf_1D, 'curve_hatch_style': None, 'single_cell_pfmap_processing_fn': single_cell_pfmap_processing_fn_identity} | long_kwargs)
    ax_long_pf_1D, long_sort_ind, long_neurons_colors_array = plot_ratemap_1D(long_results.pf1D.ratemap, **long_kwargs)
    fig_long_pf_1D = ax_long_pf_1D.get_figure()
    
    if single_figure:
        ax_short_pf_1D = ax_long_pf_1D # Set the axes for the short to that that was just plotted on by the long
        fig_short_pf_1D = fig_long_pf_1D
    
    # Long/Short Specific (Distinguishing) kwargs:
    short_kwargs = (plot_ratemap_1D_kwargs | {'sortby': shared_fragile_neuron_IDXs, 'included_unit_neuron_IDs': curr_any_context_neurons, 'fignum': None, 'ax': ax_short_pf_1D, 'curve_hatch_style': {'hatch':'///', 'edgecolor':'k'}, 'single_cell_pfmap_processing_fn': single_cell_pfmap_processing_fn_flipped_y} | short_kwargs)
    ax_short_pf_1D, short_sort_ind, short_neurons_colors_array = plot_ratemap_1D(short_results.pf1D.ratemap, **short_kwargs, name=f"short")
    fig_short_pf_1D = ax_short_pf_1D.get_figure()
    
    # Do set_ylim before calling `add_vertical_track_bounds_lines(...)` to make sure full vertical span is used
    if y_lims_offset is not None:
        ax_long_pf_1D.set_ylim((np.array(ax_long_pf_1D.get_ylim()) + y_lims_offset))
        if not single_figure:
            ax_short_pf_1D.set_ylim((np.array(ax_short_pf_1D.get_ylim()) + y_lims_offset)) # TODO: I think this is right

    if single_figure:
        if (title_string is not None) or (subtitle_string is not None):
            perform_update_title_subtitle(fig=fig_long_pf_1D, ax=ax_long_pf_1D, title_string=title_string, subtitle_string=subtitle_string, active_context=active_context, use_flexitext_titles=True)
        else:
            fig_long_pf_1D.suptitle('Long vs. Short (hatched)')
            
        # Plot the track bounds:
        if should_plot_vertical_track_bounds_lines:
            long_track_line_collection, short_track_line_collection = add_vertical_track_bounds_lines(grid_bin_bounds=deepcopy(long_results.pf1D.config.grid_bin_bounds), ax=ax_long_pf_1D, include_long=True, include_short=True)
    
        if should_plot_linear_track_shapes:
            long_rects_outputs, short_rects_outputs = add_track_shapes(grid_bin_bounds=deepcopy(long_results.pf1D.config.grid_bin_bounds), ax=ax_long_pf_1D, include_long=True, include_short=True)
    else:
        fig_long_pf_1D.suptitle('Long')
        fig_short_pf_1D.suptitle('Short')
        ax_short_pf_1D.set_xlim(ax_long_pf_1D.get_xlim())

        # Plot the track bounds:
        if should_plot_vertical_track_bounds_lines:
            long_track_line_collection, _ = add_vertical_track_bounds_lines(grid_bin_bounds=deepcopy(long_results.pf1D.config.grid_bin_bounds), ax=ax_long_pf_1D, include_long=True, include_short=False) # only long
            _, short_track_line_collection = add_vertical_track_bounds_lines(grid_bin_bounds=deepcopy(short_results.pf1D.config.grid_bin_bounds), ax=ax_short_pf_1D, include_long=False, include_short=True) # only short

        if should_plot_linear_track_shapes:
            long_rects_outputs, _ = add_track_shapes(grid_bin_bounds=deepcopy(long_results.pf1D.config.grid_bin_bounds), ax=ax_long_pf_1D, include_long=True, include_short=False) # only long
            _, short_rects_outputs = add_track_shapes(grid_bin_bounds=deepcopy(short_results.pf1D.config.grid_bin_bounds), ax=ax_short_pf_1D, include_long=False, include_short=True) # only short
            

        # ax_long_pf_1D.sharex(ax_short_pf_1D)
        
    if not should_plot_vertical_track_bounds_lines:
        long_track_line_collection = None 
        short_track_line_collection = None
        
    if not should_plot_linear_track_shapes:
        long_rects_outputs = None
        short_rects_outputs = None

    # Could return: long_track_line_collection, short_track_line_collection
    # graphics_output_dict = MatplotlibRenderPlots(name='display_short_long_pf1D_comparison', figures=(fig_long_pf_1D, fig_short_pf_1D), axes=(ax_long_pf_1D, ax_short_pf_1D), plot_data={}, context=final_context, saved_figures=active_out_figure_paths)
    # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}
    # long_plots = {'long_track_line_collection': long_track_line_collection}
    
    return (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array)


@mpl.rc_context(Fig.get_mpl_style(style='figPublish'))
def plot_short_v_long_pf1D_scalar_overlap_comparison(overlap_scalars_df, pf_neurons_diff, neurons_colors_array, reuse_axs_tuple=None, single_figure=False, overlap_metric_mode=PlacefieldOverlapMetricMode.POLY, variant_name='', debug_print=False):
    """ Produces a figure containing a bar chart to compare *a scalar value* the 1D placefields on the long vs. the short track. 
    poly_overlap_df: pd.DataFrame - computed by compute_polygon_overlap(...)
    pf_neurons_diff: pd.DataFrame - 
    single_figure:bool - if True, both long and short are plotted on the same axes of a single shared figure. Otherwise seperate figures are used for each
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.LongShortTrackComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_short_v_long_pf1D_scalar_overlap_comparison

        long_results = curr_active_pipeline.computation_results['maze1_PYR'].computed_data
        short_results = curr_active_pipeline.computation_results['maze2_PYR'].computed_data
        curr_any_context_neurons = _find_any_context_neurons(*[curr_active_pipeline.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in ['maze1_PYR', 'maze2_PYR']])
        reuse_axs_tuple=None # plot fresh
        # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
        # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
        (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_scalar_overlap_comparison(long_results, short_results, curr_any_context_neurons, reuse_axs_tuple=reuse_axs_tuple, single_figure=True)

    """
    from neuropy.utils.matplotlib_helpers import add_value_labels # for adding small labels beside each point indicating their ACLU

    if not isinstance(overlap_metric_mode, PlacefieldOverlapMetricMode):
        overlap_metric_mode = PlacefieldOverlapMetricMode.init(overlap_metric_mode)

    curr_any_context_neurons = pf_neurons_diff.either
    n_neurons = pf_neurons_diff.shared.n_neurons
    shared_fragile_neuron_IDXs = pf_neurons_diff.shared.shared_fragile_neuron_IDXs

    if debug_print:
        print(f'n_neurons: {n_neurons}')
        print(f'shared_fragile_neuron_IDXs: {shared_fragile_neuron_IDXs}.\t np.shape: {np.shape(shared_fragile_neuron_IDXs)}')
        print(f'curr_any_context_neurons: {curr_any_context_neurons}.\t np.shape: {np.shape(curr_any_context_neurons)}')

    if overlap_metric_mode.name == PlacefieldOverlapMetricMode.POLY.name:
        freq_series = overlap_scalars_df.poly_overlap
        lowercase_desc = 'poly'
        titlecase_desc = 'Poly'
    elif overlap_metric_mode.name == PlacefieldOverlapMetricMode.CONVOLUTION.name:
        freq_series = overlap_scalars_df.conv_overlap
        lowercase_desc = 'conv'
        titlecase_desc = 'Conv'
    elif overlap_metric_mode.name == PlacefieldOverlapMetricMode.PRODUCT.name:
        freq_series = overlap_scalars_df.prod_overlap
        # freq_series = overlap_scalars_df.prod_overlap_peak_max
        lowercase_desc = 'prod'
        titlecase_desc = 'Prod'
    elif overlap_metric_mode.name == PlacefieldOverlapMetricMode.REL_ENTROPY.name:
        freq_series = overlap_scalars_df.short_long_relative_entropy
        lowercase_desc = 'rel_entropy'
        titlecase_desc = 'RelEntropy'
    else:
        raise NotImplementedError

    x_labels = overlap_scalars_df.index.to_numpy()

    neurons_color_tuples_list = [tuple(neurons_colors_array[:-1, color_idx]) for color_idx in np.arange(np.shape(neurons_colors_array)[1])]

    # Plot the figure.
    fig = plt.figure(figsize=(12, 8), num=f'pf1D_{lowercase_desc}_overlap{variant_name}', clear=True)
    
    ax = freq_series.plot(kind='bar', color=neurons_color_tuples_list)
    ax.set_title(f'1D Placefield Short vs. Long {titlecase_desc} Overlap')
    ax.set_xlabel('Cell ID (aclu)')
    ax.set_ylabel(f'{titlecase_desc} Overlap')
    ax.set_xticklabels(x_labels)
    
    # Call the function above. All the magic happens there.
    add_value_labels(ax, labels=x_labels) # 

    return fig, ax


@function_attributes(short_name='long_short_fr_indicies', tags=['private', 'long_short', 'long_short_firing_rate', 'firing_rate', 'display', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=['_display_short_long_firing_rate_index_comparison'], creation_date='2023-03-28 14:20')
def _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, active_context, neurons_colors=None, debug_print=False, is_centered = False, enable_hover_labels=True, enable_tiny_point_labels=True):
    """ Plot long|short firing rate index 
    Each datapoint is a neuron.

    used in `_display_short_long_firing_rate_index_comparison()`

    Parameters:
        is_centered: bool - if True, the spines are centered at (0, 0)
        enable_hover_labels = True # add interactive point hover labels using mplcursors
        enable_tiny_point_labels = True # add static tiny aclu labels beside each point

    """
    if enable_hover_labels:
        import mplcursors # for hover tooltips that specify the aclu of the selected point

    # from neuropy.utils.matplotlib_helpers import add_value_labels # for adding small labels beside each point indicating their ACLU

    if isinstance(x_frs_index, dict):
        # convert to pd.Series
        x_frs_index = pd.Series(x_frs_index.values(), index=x_frs_index.keys(), copy=False)
    if isinstance(y_frs_index, dict):
        # convert to pd.Series
        y_frs_index = pd.Series(y_frs_index.values(), index=y_frs_index.keys(), copy=False)
                
    if neurons_colors is not None:
        if isinstance(neurons_colors, dict):
            point_colors = [neurons_colors[aclu] for aclu in list(x_frs_index.keys())]
        else:
            # otherwise assumed to be an array with the same length as the number of points
            assert isinstance(neurons_colors, np.ndarray)
            assert np.shape(point_colors)[0] == 4 # (4, n_neurons)
            assert np.shape(point_colors)[1] == len(x_frs_index)
            point_colors = neurons_colors
            # point_colors = [f'{i}' for i in list(x_frs_index.keys())] 
    else:
        point_colors = None
    
    fig, ax = plt.subplots(figsize=(8.5, 7.25), num=f'long|short fr indicies_{active_context.get_description(separator="/")}', clear=True)

    xlabel_kwargs = {}
    ylabel_kwargs = {}
    if is_centered:
        xlabel_kwargs = dict(loc='left')
        ylabel_kwargs = dict(loc='bottom')

    scatter_plot = ax.scatter(x_frs_index.values, y_frs_index.values, c=point_colors) # , s=10, alpha=0.5
    plt.xlabel('Replay Firing Rate Index $\\frac{L_{R}-S_{R}}{L_{R} + S_{R}}$', fontsize=16, **xlabel_kwargs)
    plt.ylabel('Laps Firing Rate Index $\\frac{L_{\\theta}-S_{\\theta}}{L_{\\theta} + S_{\\theta}}$', fontsize=16, **ylabel_kwargs)


    ## Non-flexitext version:
    # plt.title('long ($L$)|short($S$) firing rate indicies')
    # plt.suptitle(f'{active_context.get_description(separator="/")}')

    # `flexitext` version:
    text_formatter = FormattedFigureText()
    plt.title('')
    plt.suptitle('')
    fig = plt.gcf() # get figure to setup the margins on.
    text_formatter.setup_margins(fig)
    flexitext(text_formatter.left_margin, text_formatter.top_margin, '<size:22><color:crimson, weight:bold>long ($L$)</>|<color:royalblue, weight:bold>short($S$)</> <weight:bold>firing rate indicies</></>', va="bottom", xycoords="figure fraction")
    footer_text_obj = flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")

    # fig.set_size_inches([8.5, 7.25]) # size figure so the x and y labels aren't cut off

    if enable_hover_labels or enable_tiny_point_labels:
        point_hover_labels = [f'{i}' for i in list(x_frs_index.keys())] # point_hover_labels will be added as tooltip annotations to the datapoints

        if enable_tiny_point_labels:
            # add static tiny labels beside each point
            for i, (x, y, label) in enumerate(zip(x_frs_index.values, y_frs_index.values, point_hover_labels)):
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(2,2), ha='left', va='bottom', fontsize=8) # , color=rect.get_facecolor()

        if enable_hover_labels:
            # add hover labels:
            # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
            # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib/21654635#21654635
            # add hover labels using mplcursors
            mplcursors.cursor(scatter_plot, hover=True).connect("add", lambda sel: sel.annotation.set_text(point_hover_labels[sel.index]))

    # Set the x and y axes to standard limits for easy visual comparison across sessions
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    
    if is_centered:
        ## The "spines" are the vertical and horizontal "axis" bars where the tick marks are drawn.
        ax.spines[['left', 'bottom']].set_position('center')
        ax.spines[['top', 'right']].set_visible(False)
    else:
        # Hide the right and top spines (box components)
        ax.spines[['right', 'top']].set_visible(False)
        

    # # Call the function above. All the magic happens there.
    # add_value_labels(ax, labels=x_labels) # 

    return fig


# ==================================================================================================================== #
# 2023-04-19 Surprise                                                                                                  #
# ==================================================================================================================== #

@function_attributes(short_name='plot_long_short_expected_vs_observed_firing_rates', tags=['pyqtgraph','long_short'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 17:26', is_global=True)
def plot_long_short_expected_vs_observed_firing_rates(long_results_obj, short_results_obj, limit_aclus=None):
    """ 2023-03-28 4:30pm - Expected vs. Observed Firing Rates for each cell and each epoch 
    
    Usage:
        win, plots_tuple, legend = plot_long_short_expected_vs_observed_firing_rates(long_results_obj, short_results_obj, limit_aclus=[20])

    """
    num_cells = long_results_obj.original_1D_decoder.num_neurons
    num_epochs = long_results_obj.active_filter_epochs.n_epochs
    # make a separate symbol_brush color for each cell:
    cell_color_symbol_brush = [pg.intColor(i,hues=9, values=3, alpha=180) for i, aclu in enumerate(long_results_obj.original_1D_decoder.neuron_IDs)] # maxValue=128
    # All properties in common:
    win = pg.plot()
     # win.setWindowTitle('Long v. Short - Leave-one-out Expected vs. Observed Firing Rates')
    win.setWindowTitle('Long v. Short - Leave-one-out Expected vs. Observed Num Spikes')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())
    # restrict the aclus to display to limit_aclus
    if limit_aclus is None:
        limit_aclus = long_results_obj.original_1D_decoder.neuron_IDs
    # check whether the neuron_ID is included:
    is_neuron_ID_active = np.isin(long_results_obj.original_1D_decoder.neuron_IDs, limit_aclus)    
    # restrict to the limit indicies
    active_neuron_IDs = np.array(long_results_obj.original_1D_decoder.neuron_IDs)[is_neuron_ID_active]
    active_neuron_IDXs =  np.array(long_results_obj.original_1D_decoder.neuron_IDXs)[is_neuron_ID_active]

    plots_tuple = tuple([{}, {}])
    label_prefix_list = ['long', 'short']
    long_short_symbol_list = ['t', 't1'] # note: 's' is a square. 'o', 't1': triangle pointing upwards
    
    for long_or_short_idx, a_results_obj in enumerate((long_results_obj, short_results_obj)):
        label_prefix = label_prefix_list[long_or_short_idx]
        # print(F'long_or_short_idx: {long_or_short_idx = }, label_prefix: {label_prefix =}')
        plots = plots_tuple[long_or_short_idx]
        curr_symbol = long_short_symbol_list[long_or_short_idx]
        
        ## add scatter plots on top
        for unit_IDX, aclu in zip(active_neuron_IDXs, active_neuron_IDs):
            # find only the time bins when the cell fires:
            curr_epoch_is_cell_active = np.logical_not(a_results_obj.is_non_firing_time_bin)[unit_IDX, :]
            # Use mean time_bin and surprise for each epoch
            curr_epoch_time_bins = a_results_obj.flat_all_epochs_decoded_epoch_time_bins[unit_IDX, curr_epoch_is_cell_active]
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_firing_rates[unit_IDX, curr_epoch_is_cell_active] # measured firing rates (Hz) 
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num measured spikes 
            curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num spikes diff
            # curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_firing_rates[unit_IDX, :] # firing rate diff
            plots[aclu] = win.plot(x=curr_epoch_time_bins, y=curr_epoch_data, pen=cell_color_symbol_brush[unit_IDX], symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX], name=f'{label_prefix}[{aclu}]', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128)
            legend.addItem(plots[aclu], f'{label_prefix}[{aclu}]')
            ## Make error bars
            # err = pg.ErrorBarItem(x=curr_epoch_time_bins, y=data.mean(axis=1), height=data.std(axis=1), beam=0.5, pen={'color':'w', 'width':2})
            # win.addItem(err)

    win.graphicsItem().setLabel(axis='left', text='Short v. Long - Expected vs. Observed # Spikes')
    win.graphicsItem().setLabel(axis='bottom', text='time')
    return win, plots_tuple, legend

@function_attributes(short_name='plot_long_short_any_values', tags=['pyqtgraph','long_short'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 17:27')
def plot_long_short_any_values(long_results_obj, short_results_obj, x, y, limit_aclus=None):
    """ 2023-03-28 4:31pm - Any values, specified by a lambda function for each cell and each epoch 

        x_fn = lambda a_results_obj: a_results_obj.all_epochs_decoded_epoch_time_bins_mean[:,0]
        # y_fn = lambda a_results_obj: a_results_obj.all_epochs_all_cells_one_left_out_posterior_to_scrambled_pf_surprises_mean
        # y_fn = lambda a_results_obj: a_results_obj.all_epochs_all_cells_one_left_out_posterior_to_pf_surprises_mean
        y_fn = lambda a_results_obj: a_results_obj.all_epochs_computed_one_left_out_posterior_to_pf_surprises

        # (time_bins, neurons), (epochs, neurons), (epochs)
        # all_epochs_computed_one_left_out_posterior_to_pf_surprises, all_epochs_computed_cell_one_left_out_posterior_to_pf_surprises_mean, all_epochs_all_cells_one_left_out_posterior_to_pf_surprises_mean
        win, plots_tuple, legend = plot_long_short_any_values(long_results_obj, short_results_obj, x=x_fn, y=y_fn, limit_aclus=[20])

    """
    num_cells = long_results_obj.original_1D_decoder.num_neurons
    num_epochs = long_results_obj.active_filter_epochs.n_epochs
    # make a separate symbol_brush color for each cell:
    cell_color_symbol_brush = [pg.intColor(i,hues=9, values=3, alpha=180) for i, aclu in enumerate(long_results_obj.original_1D_decoder.neuron_IDs)] # maxValue=128
    # All properties in common:
    win = pg.plot()
    win.setWindowTitle('Short v. Long - Leave-one-out Custom Surprise Plot')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())
    # restrict the aclus to display to limit_aclus
    if limit_aclus is None:
        limit_aclus = long_results_obj.original_1D_decoder.neuron_IDs
    # check whether the neuron_ID is included:
    is_neuron_ID_active = np.isin(long_results_obj.original_1D_decoder.neuron_IDs, limit_aclus)    
    # restrict to the limit indicies
    active_neuron_IDs = np.array(long_results_obj.original_1D_decoder.neuron_IDs)[is_neuron_ID_active]
    active_neuron_IDXs =  np.array(long_results_obj.original_1D_decoder.neuron_IDXs)[is_neuron_ID_active]

    plots_tuple = tuple([{}, {}])
    label_prefix_list = ['long', 'short']
    long_short_symbol_list = ['t', 'o'] # note: 's' is a square. 'o', 't1': triangle pointing upwards
    
    for long_or_short_idx, a_results_obj in enumerate((long_results_obj, short_results_obj)):
        label_prefix = label_prefix_list[long_or_short_idx]
        # print(F'long_or_short_idx: {long_or_short_idx = }, label_prefix: {label_prefix =}')
        plots = plots_tuple[long_or_short_idx]
        curr_symbol = long_short_symbol_list[long_or_short_idx]
        
        ## add scatter plots on top
        for unit_IDX, aclu in zip(active_neuron_IDXs, active_neuron_IDs):
            # find only the time bins when the cell fires:
            curr_epoch_is_cell_active = np.logical_not(a_results_obj.is_non_firing_time_bin)[unit_IDX, :]
            # Use mean time_bin and surprise for each epoch
            curr_epoch_time_bins = a_results_obj.flat_all_epochs_decoded_epoch_time_bins[unit_IDX, curr_epoch_is_cell_active]
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_firing_rates[unit_IDX, curr_epoch_is_cell_active] # measured firing rates (Hz) 
            # curr_epoch_data = a_results_obj.flat_all_epochs_measured_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num measured spikes 
            # curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_spike_counts[unit_IDX, curr_epoch_is_cell_active] # num spikes diff
            # curr_epoch_data = a_results_obj.flat_all_epochs_difference_from_expected_cell_firing_rates[unit_IDX, :] # firing rate diff
            print(f'curr_epoch_time_bins.shape: {np.shape(curr_epoch_time_bins)}')
            curr_epoch_data = y(a_results_obj) # [unit_IDX, curr_epoch_is_cell_active]
            print(f'np.shape(curr_epoch_data): {np.shape(curr_epoch_data)}')
            curr_epoch_data = curr_epoch_data[unit_IDX, curr_epoch_is_cell_active]
            plots[aclu] = win.plot(x=curr_epoch_time_bins, y=curr_epoch_data, pen=None, symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX], name=f'{label_prefix}[{aclu}]', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128)
            legend.addItem(plots[aclu], f'{label_prefix}[{aclu}]')

    win.graphicsItem().setLabel(axis='left', text='Short v. Long - Surprise (Custom)')
    win.graphicsItem().setLabel(axis='bottom', text='time')
    return win, plots_tuple, legend

@function_attributes(short_name='plot_long_short', tags=['pyqtgraph','long_short'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-19 17:26')
def plot_long_short(long_results_obj, short_results_obj):
    win = pg.plot()
    win.setWindowTitle('Short v. Long - Leave-one-out All Cell Average Surprise Outputs')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())

    ax_long = win.plot(x=long_results_obj.all_epochs_decoded_epoch_time_bins_mean[:,0], y=long_results_obj.all_epochs_all_cells_computed_surprises_mean, pen=None, symbol='o', symbolBrush=pg.intColor(0,6,maxValue=128), name=f'long') #  symbolBrush=pg.intColor(i,6,maxValue=128)
    legend.addItem(ax_long, f'long')
    ax_short = win.plot(x=short_results_obj.all_epochs_decoded_epoch_time_bins_mean[:,0], y=short_results_obj.all_epochs_all_cells_computed_surprises_mean, pen=None, symbol='o', symbolBrush=pg.intColor(1,6,maxValue=128), name=f'short') #  symbolBrush=pg.intColor(i,6,maxValue=128)
    legend.addItem(ax_short, f'short')

    win.graphicsItem().setLabel(axis='left', text='Short v. Long - Leave-one-out All Cell Average Surprise')
    win.graphicsItem().setLabel(axis='bottom', text='time')
    return win, (ax_long, ax_short), legend

def plot_long_short_surprise_difference_plot(curr_active_pipeline, long_results_obj, short_results_obj, long_epoch_name, short_epoch_name):
    """ 2023-05-17 - Refactored into display functions file from notebook.
    
    Usage: 
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_long_short_surprise_difference_plot
        win, plots = plot_long_short_surprise_difference_plot(curr_active_pipeline, long_results_obj, short_results_obj, long_epoch_name, short_epoch_name)
    
    """
    # Private Subfunctions _______________________________________________________________________________________________ #
    def _subfn_add_difference_plot_series(win, plots, result_df_grouped, series_suffix, **kwargs):
        """ captures nothing
        modifies `plots` """
        x=result_df_grouped.time_bin_centers.to_numpy()
        y=result_df_grouped['surprise_diff'].to_numpy()
        series_id_str = f'difference_{series_suffix}'
        plots[series_id_str] = win.plot(x=x, y=y, name=series_id_str, alpha=0.5, **kwargs) #  symbolBrush=pg.intColor(i,6,maxValue=128) , symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX]

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #

    # make a separate symbol_brush color for each cell:
    # cell_color_symbol_brush = [pg.intColor(i,hues=9, values=3, alpha=180) for i, aclu in enumerate(long_results_obj.original_1D_decoder.neuron_IDs)] # maxValue=128
    # All properties in common:
    win = pg.plot() # PlotWidget
    win.setWindowTitle('Long Sanity Check - Leave-one-out Custom Surprise Plot')
    # legend_size = (80,60) # fixed size legend
    legend_size = None # auto-sizing legend to contents
    legend = pg.LegendItem(legend_size, offset=(-1,0)) # do this instead of # .addLegend
    legend.setParentItem(win.graphicsItem())

    plots = {}
    label_prefix_list = ['normal', 'scrambled']
    long_short_symbol_list = ['t', 'o'] # note: 's' is a square. 'o', 't1': triangle pointing upwards0

    # Use mean time_bin and surprise for each epoch
    # plots['normal'] = win.plot(x=valid_time_bin_indicies, y=one_left_out_posterior_to_pf_surprises_mean, pen=None, symbol='t', symbolBrush=pg.intColor(1,6,maxValue=128), name=f'normal', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128) , symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX]
    # plots['scrambled'] = win.plot(x=valid_time_bin_indicies, y=one_left_out_posterior_to_scrambled_pf_surprises_mean, pen=None, symbol='t', symbolBrush=pg.intColor(2,6,maxValue=128), name=f'scrambled', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128) , symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX]

    # curr_surprise_difference = one_left_out_posterior_to_scrambled_pf_surprises_mean - one_left_out_posterior_to_pf_surprises_mean

    # x=valid_time_bin_indicies
    # y=curr_surprise_difference
    # x=result_df_grouped.time_bin_indices.to_numpy()

    
    _subfn_add_difference_plot_series(win, plots, long_results_obj.result_df_grouped, series_suffix='_long', **dict(pen=None, symbol='t', symbolBrush=pg.intColor(2,6,maxValue=128), clickable=True, hoverable=True, hoverSize=7))

    _subfn_add_difference_plot_series(win, plots, short_results_obj.result_df_grouped, series_suffix='_short', **dict(pen=None, symbol='o', symbolBrush=pg.intColor(3,6,maxValue=128), clickable=True))

    # dict(pen=None, symbol='t', symbolBrush=pg.intColor(2,6,maxValue=128))


    # x=result_df_grouped.time_bin_centers.to_numpy()
    # y=result_df_grouped['surprise_diff'].to_numpy()
    # plots['difference'] = win.plot(x=x, y=y, pen=None, symbol='t', symbolBrush=pg.intColor(2,6,maxValue=128), name=f'difference', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128) , symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX]

    # long_results_obj.result, long_results_obj.result_df, long_results_obj.result_df_grouped

    # short_results_obj.result, short_results_obj.result_df, short_results_obj.result_df_grouped


    for k, v in plots.items():
        legend.addItem(v, f'{k}')

    win.graphicsItem().setLabel(axis='left', text='Normal v. Random - Surprise (Custom)')
    win.graphicsItem().setLabel(axis='bottom', text='time')

    win.showGrid(True, True)  # Show grid for reference

    # Emphasize the y=0 crossing by drawing a horizontal line at y=0
    vline = pg.InfiniteLine(pos=0, angle=0, movable=False, pen=pg.mkPen(color='w', width=2, style=pg.QtCore.Qt.DashLine))
    win.addItem(vline)

    # Add session indicators to pyqtgraph plot
    long_epoch = curr_active_pipeline.filtered_epochs[long_epoch_name]
    short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
    long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(win, long_epoch, short_epoch)

    # epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(win, t_start=curr_active_pipeline.filtered_epochs[long_epoch_name].t_start, t_stop=curr_active_pipeline.filtered_epochs[long_epoch_name].t_stop, epoch_label='long', **dict(pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00')))
    # epoch_linear_region, epoch_region_label = build_pyqtgraph_epoch_indicator_regions(win, t_start=curr_active_pipeline.filtered_epochs[short_epoch_name].t_start, t_stop=curr_active_pipeline.filtered_epochs[short_epoch_name].t_stop, epoch_label='short', **dict(pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00')))

    i_str = generate_html_string('i', color='white', bold=True)
    j_str = generate_html_string('j', color='red', bold=True)
    title_str = generate_html_string(f'JSD(p_x_given_n, pf[{i_str}]) - JSD(p_x_given_n, pf[{j_str}]) where {j_str} non-firing')
    win.setTitle(title_str)

    win.setWindowTitle('Long Sanity Check - Leave-one-out Custom Surprise Plot - JSD')

    return win, plots




# ==================================================================================================================== #
# 2023-05-02 Laps/Replay Rate Remapping 1D Index Line                                                                  #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['matplotlib', 'long_short_fr_indicies_analysis', 'rate_remapping'], input_requires=['long_short_fr_indicies_analysis'], output_provides=[], uses=[], used_by=[], creation_date='2023-05-03 23:12')
def plot_rr_aclu(aclu: list, rr_laps: np.ndarray, rr_replays: np.ndarray, rr_neuron_types: np.ndarray, sort=None, fig=None, axs=None, defer_render=False):
    """ Plots rate remapping (rr) values computed from `long_short_fr_indicies_analysis`
    Renders a vertical stack (one for each aclu) of 1D number lines ranging from -1 ("Long Only") to 1 ("Short Only"), where 0 means equal rates on both long and short.
        It plots two points for each of these, a triangle corresponding to that cell's rr_laps and a open circle for the rr_replays.
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.LongShortTrackComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_rr_aclu

        rr_aclus = np.array(list(curr_active_pipeline.global_computation_results.computed_data.long_short_fr_indicies_analysis.y_frs_index.keys()))
        rr_neuron_type = [global_session.neurons.aclu_to_neuron_type_map[aclu] for aclu in rr_aclus]
        rr_laps = np.array(list(curr_active_pipeline.global_computation_results.computed_data.long_short_fr_indicies_analysis.y_frs_index.values()))
        rr_replays = np.array(list(curr_active_pipeline.global_computation_results.computed_data.long_short_fr_indicies_analysis.x_frs_index.values()))
        rr_skew = rr_laps / rr_replays

        n_debug_limit = 100
        fig, ax = plot_rr_aclu([str(aclu) for aclu in rr_aclus[:n_debug_limit]], rr_laps=rr_laps[:n_debug_limit], rr_replays=rr_replays[:n_debug_limit])

    """
    def _subfn_remove_all_ax_features(ax):
        """ removes the outer box (spines), the x- and y-axes from the ax"""
        # Set axis limits and ticks
        ax.set_xlim(-1, 1)
        # ax.set_xticks([-1, 0, 1])

        # Remove y-axis and spines
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.xaxis.set_visible(False)
        ax.spines['bottom'].set_visible(False)


    def _subfn_draw_tick_and_label(ax, x, y, label, sub_label, tick_half_height=0.5, supress_labels=False):
        # Add sub-label for "Long Only"
        ax.plot([x, x], [y-tick_half_height, y+tick_half_height], color='black', linewidth=1.5) # draw tick mark
        label_start_y = y-(tick_half_height + 0.05)
        
        if (not supress_labels) and label is not None:
            ax.text(x, label_start_y, label, fontsize=12, ha='center')
        if (not supress_labels) and sub_label is not None:
            ax.text(x, label_start_y-0.1, sub_label, fontsize=8, ha='center')
        

    def _subfn_draw_single_aclu_num_line(ax, aclu, aclu_y: float, rr_laps: float, rr_replays: float, rr_neuron_types = None, supress_labels=False):
        """ plots a single aclu's plot centered vertically at `aclu_y` """
        # Add thick black baseline line segment along y=0 between -1 and 1
        ax.plot([-1, 1], [aclu_y, aclu_y], color='black', linewidth=2)

        # Add tick labels and sub-labels
        _subfn_draw_tick_and_label(ax, x=-1, y=aclu_y, label='Long Only', sub_label='(short fr = 0)', tick_half_height=0.2, supress_labels=supress_labels)
        _subfn_draw_tick_and_label(ax, x=0, y=aclu_y, label=None, sub_label=None, tick_half_height=0.1, supress_labels=supress_labels) # 'Equal'
        _subfn_draw_tick_and_label(ax, x=1, y=aclu_y, label='Short Only', sub_label='(long fr = 0)', tick_half_height=0.2, supress_labels=supress_labels)

        # Add markers for rr_laps and rr_replays
        ax.plot(rr_laps, aclu_y, marker=r'$\theta$', markersize=10, color='black', label='rr_laps')
        ax.plot(rr_replays, aclu_y, marker='o', markersize=10, fillstyle='none', color='black', label='rr_replays')

        # Add text for aclu to the left of the plot
        ax.text(-1.2, 0.5, aclu, fontsize=20, va='center')
        
        _subfn_remove_all_ax_features(ax)
        ax.set_ylim(-0.5, 0.5)

    if not isinstance(aclu, np.ndarray):
        aclu = np.array(aclu) # convert to ndarray
    n_aclus = len(aclu)
    aclu_indicies = np.arange(n_aclus)
    

    if (fig is None) or (axs is None):
        ## Create a new figure and set of subplots
        fig, axs = plt.subplots(nrows=len(rr_replays))
    else:
        assert len(axs) >= n_aclus, f"make sure that there are enough axes to display each aclu provided."
        # Clear existing axes:
        for ax in axs:
            ax.clear()


    ## Sort process (local sort):
    sort_indicies = np.arange(n_aclus) # default sort is the one passed in unless otherwise specified.    
    if sort is not None:
        if isinstance(sort, str):
            if sort == 'rr_replays':                
                ## Sort on 'rr_replays'
                sort_indicies = np.argsort(rr_replays)
            else:
                raise NotImplementedError # only 'rr_replays' sort is implemented.
        elif isinstance(sort, np.ndarray):
            # a set of sort indicies
            sort_indicies = sort
        else:
            # Unknown or no sort
            pass
    else:
        # No sort
        pass

    ## Sort all the variables:
    aclu = aclu[sort_indicies]
    rr_replays = rr_replays[sort_indicies]
    rr_laps = rr_laps[sort_indicies]

    if rr_neuron_types is None:
        rr_neuron_types = [None] * len(aclu) # a list containing all None values
        assert len(rr_neuron_types) == len(aclu)

    else:
        # if rr_neuron_types is not None:
        assert len(rr_neuron_types) == len(aclu)
        rr_neuron_types = rr_neuron_types[sort_indicies]

    ## Main Loop:
    for ax, an_aclu_index, an_aclu, an_rr_laps, an_rr_replays, an_rr_neuron_types in zip(axs, aclu_indicies, aclu, rr_laps, rr_replays, rr_neuron_types):
        is_last_iteration = (an_aclu_index == aclu_indicies[-1]) # labels will be surpressed on all but the last iteration
        _subfn_draw_single_aclu_num_line(ax, an_aclu, aclu_y=0.0, rr_laps=an_rr_laps, rr_replays=an_rr_replays, rr_neuron_types=an_rr_neuron_types, supress_labels=(not is_last_iteration)) # (float(an_aclu_index)*0.5)

    # Add title to the plot
    # ax.set_title('Long-Short Equity')

    # Show plot
    if not defer_render:            
        plt.show()

    return fig, axs, sort_indicies




@metadata_attributes(short_name=None, tags=['rate_remapping', 'matplotlib', 'long_short_fr_indicies_analysis', 'paginated'], input_requires=[], output_provides=[], uses=['plot_rr_aclu'], used_by=[], creation_date='2023-05-09 11:29', related_items=[])
class RateRemappingPaginatedFigureController(PaginatedFigureController):
    """2023-05-09 - Aims to refactor `build_figure_and_control_widget_from_paginator`, a series of nested functions, into a stateful class

    At its core uses `plot_rr_aclu` to plot the rate remapping number lines
    
    
    
    Usage:
        import matplotlib.pyplot as plt
        %matplotlib qt
        from pyphocorehelpers.indexing_helpers import Paginator
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.LongShortTrackComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import RateRemappingPaginatedFigureController

        a_paginator = Paginator.init_from_data((rr_aclus, rr_laps, rr_replays), max_num_columns=1, max_subplots_per_page=20, data_indicies=None, last_figure_subplots_same_layout=False)
        _out_rr_pagination_controller = RateRemappingPaginatedFigureController.init_from_paginator(a_paginator, a_name='TestRateRemappingPaginatedFigureController')
        _out_rr_pagination_controller
    """
    
    @classmethod
    def init_from_paginator(cls, a_paginator, a_name:str = 'RateRemappingPaginatedFigureController', plot_function_name='plot_rr_aclu', active_context=None):
        """ 
        Usage:

            from pyphocorehelpers.indexing_helpers import Paginator
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import RateRemappingPaginatedFigureController
            ## Paginated multi-plot
            # Provide a tuple or list containing equally sized sequences of items:
            a_paginator = Paginator.init_from_data((rr_aclus, rr_laps, rr_replays, rr_neuron_type), max_num_columns=1, max_subplots_per_page=20, data_indicies=None, last_figure_subplots_same_layout=False)

            ## Build GUI components:
            active_identifying_session_ctx = curr_active_pipeline.sess.get_context()
            _out_rr_pagination_controller = RateRemappingPaginatedFigureController.init_from_paginator(a_paginator, a_name='TestRateRemappingPaginatedFigureController', active_context=active_identifying_session_ctx)

        """
        new_obj = cls(params=VisualizationParameters(name=a_name), plots_data=RenderPlotsData(name=a_name, paginator=a_paginator), plots=RenderPlots(name=a_name), ui=PhoUIContainer(name=a_name, connections=PhoUIContainer(name=a_name)))
        # new_obj.ui.connections = PhoUIContainer(name=name)
        num_slices = a_paginator.max_num_items_per_page
        # Setup equivalent to that performed in `pyphoplacecellanalysis.Pho2D.stacked_epoch_slices.stacked_epoch_basic_setup`
        new_obj.params.name = a_name
        new_obj.params.window_title = plot_function_name
        new_obj.params.num_slices = num_slices # not sure if needed
        new_obj.params.active_identifying_figure_ctx = active_context
        
        # new_obj.params._debug_test_max_num_slices = debug_test_max_num_slices
        # new_obj.params.active_num_slices = min(num_slices, new_obj.params._debug_test_max_num_slices)
        
        # new_obj.params.single_plot_fixed_height = single_plot_fixed_height
        # new_obj.params.all_plots_height = float(new_obj.params.active_num_slices) * float(new_obj.params.single_plot_fixed_height)

        ## Real setup:
        new_obj.configure()
        new_obj.initialize()
        return new_obj
    

    @classmethod
    def init_from_rr_data(cls, rr_aclus, rr_laps, rr_replays, rr_neuron_type, max_subplots_per_page:int=20, a_name:str = 'RateRemappingPaginatedFigureController', plot_function_name='plot_rr_aclu', active_context=None):
        """ initializes directly from the rate-remapping data arrays. Builds the paginator.
        
        The Paginator can be accessed via `self.plots_data.paginator` if needed.
        
        Usage:
        
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import RateRemappingPaginatedFigureController

            ## Paginated multi-plot
            active_identifying_session_ctx = curr_active_pipeline.sess.get_context()
            _out_rr_pagination_controller = RateRemappingPaginatedFigureController.init_from_rr_data(rr_aclus, rr_laps, rr_replays, rr_neuron_type, max_subplots_per_page=20, a_name='TestRateRemappingPaginatedFigureController', active_context=active_identifying_session_ctx)
            a_paginator = _out_rr_pagination_controller.plots_data.paginator
        """
        ## Paginated multi-plot
        a_paginator = Paginator.init_from_data((rr_aclus, rr_laps, rr_replays, rr_neuron_type), max_num_columns=1, max_subplots_per_page=max_subplots_per_page, data_indicies=None, last_figure_subplots_same_layout=False)

        ## Build GUI components:
        new_obj = cls.init_from_paginator(a_paginator, a_name=a_name, plot_function_name=plot_function_name, active_context=active_context)
        return new_obj
    



    def configure(self, **kwargs):
        """ assigns and computes needed variables for rendering. """
        self.params.debug_print = kwargs.pop('debug_print', False)

    def initialize(self, **kwargs):
        """ sets up Figures """
        # self.fig, self.axs = plt.subplots(nrows=len(rr_replays))
        self._build_figure_widget_from_paginator()
        ## Setup Selectability
        self._subfn_helper_setup_selectability()
        ## Setup on_click callback:
        if not self.params.has_attr('callback_id') or self.params.get('callback_id', None) is None:
            self.params.callback_id = self.plots.fig.canvas.mpl_connect('button_press_event', self.on_click) ## TypeError: unhashable type: 'DecodedEpochSlicesPaginatedFigureController'

        ## 2. Update:
        self.on_paginator_control_widget_jump_to_page(page_idx=0)
        _a_connection = self.ui.mw.ui.paginator_controller_widget.jump_to_page.connect(self.on_paginator_control_widget_jump_to_page) # bind connection
        self.ui.connections['paginator_controller_widget_jump_to_page'] = _a_connection


    def update(self, **kwargs):
        """ called to specifically render data on the figure. """
        pass

    def on_close(self):
        """ called when the figure is closed. """
        pass
    

    def _build_figure_widget_from_paginator(self):
        """ builds new CustomMatplotlibWidget in self.ui.mw to hold the matplotlib plot """
        ## Build Widget to hold the matplotlib plot:
        self.ui.mw = CustomMatplotlibWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=False)

        ## Add the PaginationControlWidget
        self._subfn_helper_add_pagination_control_widget(self.plots_data.paginator, self.ui.mw, defer_render=False)

        ## Setup figure by building axes:
        self.plots.fig = self.ui.mw.getFigure()

        ## LIMITATION: Only works for 1D subplot configurations:
        # NOTE: fig.add_subplot(nrows, ncols, index) 
        self.plots.axs = [self.plots.fig.add_subplot(self.plots_data.paginator.max_num_items_per_page, 1, i+1) for i in np.arange(self.plots_data.paginator.max_num_items_per_page)] # here we're aiming to approximate the `plt.subplots(nrows=len(included_page_data_indicies)) # one row for each page` setup

        self.ui.mw.draw()
        self.ui.mw.show()

    def on_paginator_control_widget_jump_to_page(self, page_idx: int):
        """ Update captures `a_paginator`, 'mw' """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_rr_aclu
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_figure_basename_from_display_context, session_context_to_relative_path

        # print(f'on_paginator_control_widget_jump_to_page(page_idx: {page_idx})')
        # included_page_data_indicies, (curr_page_rr_aclus, curr_page_rr_laps, curr_page_rr_replays, *curr_page_rr_extras_tuple) = self.paginator.get_page_data(page_idx=page_idx)
        included_page_data_indicies, (curr_page_rr_aclus, curr_page_rr_laps, curr_page_rr_replays, curr_page_rr_neuron_type) = self.paginator.get_page_data(page_idx=page_idx)
        
        # if self.params.active_identifying_figure_ctx is not None:
        #     active_identifying_ctx = self.params.active_identifying_figure_ctx.adding_context(collision_prefix='_RateRemapping_plot_test', display_fn_name='plot_rr_aclu', plot_result_set='shared', page=f'{page_idx+1}of{self.paginator.num_pages}', aclus=f"{included_page_data_indicies}")
        # else:
        #     active_identifying_ctx = None

        # print(f'\tincluded_page_data_indicies: {included_page_data_indicies}')
        self.plots.fig = self.ui.mw.getFigure()
        self.plots.axs = self.ui.mw.axes
        # print(f'axs: {axs}')
        self.plots.fig, self.plots.axs, sort_indicies = plot_rr_aclu([str(aclu) for aclu in curr_page_rr_aclus], rr_laps=curr_page_rr_laps, rr_replays=curr_page_rr_replays, rr_neuron_types=curr_page_rr_neuron_type, fig=self.plots.fig, axs=self.plots.axs)
        # print(f'\t done.')

        self.perform_update_titles_from_context(page_idx=page_idx, included_page_data_indicies=included_page_data_indicies, collision_prefix='_RateRemapping_plot_test', display_fn_name='plot_rr_aclu', plot_result_set='shared')

        # Update selections for all axes on this page:
        self.perform_update_selections()

        self.ui.mw.draw()


# ==================================================================================================================== #
# 2023-05-25 - Long_replay|Long_laps and Short_replay|Short_laps plots                                                 #
# ==================================================================================================================== #

def _plot_session_long_short_track_firing_rate_figures(curr_active_pipeline, jonathan_firing_rate_analysis_result, defer_render=False):
    """ 2023-05-25 - Plots a comparison of the lap vs. replay firing rates for a single track.
    
    Inputs:
        `curr_active_pipeline`: is needed to register_output_file(...)
        
    Example:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import _plot_session_long_short_track_firing_rate_figures
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult

        jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
        long_plots, short_plots = _plot_session_long_short_track_firing_rate_figures(curr_active_pipeline, jonathan_firing_rate_analysis_result, figures_parent_out_path=None)
        (fig_L, ax_L, active_display_context_L), (fig_S, ax_S, active_display_context_S) = long_plots, short_plots

    """
    def _plot_single_track_firing_rate_compare(x_frs_dict, y_frs_dict, active_context, neurons_colors=None, is_centered = False):
        """ 2023-05-25 - Plot long_replay|long_laps firing rate index 
        Each datapoint is a neuron.

        is_centered: bool - if True, the spines are centered at (0, 0)

        Copied from same file as `_display_short_long_firing_rate_index_comparison()`
        
        Captures: `defer_show`
        """
        if neurons_colors is not None:
            if isinstance(neurons_colors, dict):
                point_colors = [neurons_colors[aclu] for aclu in list(x_frs_dict.keys())]
            else:
                # otherwise assumed to be an array with the same length as the number of points
                assert isinstance(neurons_colors, np.ndarray)
                assert np.shape(point_colors)[0] == 4 # (4, n_neurons)
                assert np.shape(point_colors)[1] == len(x_frs_dict)
                point_colors = neurons_colors
                # point_colors = [f'{i}' for i in list(x_frs_index.keys())] 
        else:
            point_colors = None
            # point_colors = 'black'
        
        point_hover_labels = [f'{i}' for i in list(x_frs_dict.keys())] # point_hover_labels will be added as tooltip annotations to the datapoints
        fig, ax = plt.subplots(figsize=(8.5, 7.25), num=f'track_replay|track_laps frs_{active_context.get_description(separator="/")}', clear=True)
        
        # TODO 2023-05-25 - build the display context:
        active_display_context = active_context.adding_context('display_fn', display_fn_name='plot_single_track_firing_rate_compare')

        xlabel_kwargs = {}
        ylabel_kwargs = {}
        if is_centered:
            xlabel_kwargs = dict(loc='left')
            ylabel_kwargs = dict(loc='bottom')

        scatter_plot = ax.scatter(x_frs_dict.values(), y_frs_dict.values(), c=point_colors) # , s=10, alpha=0.5
        plt.xlabel('Replay Firing Rate (Hz)', fontsize=16, **xlabel_kwargs)
        plt.ylabel('Laps Firing Rate (Hz)', fontsize=16, **ylabel_kwargs)
        
        # Non-flexitext title:
        # plt.title('Computed track_replay|track_laps firing rate')
        # plt.suptitle(f'{active_display_context.get_description(separator="/")}')

        # `flexitext` version:
        text_formatter = FormattedFigureText()
        plt.title('')
        plt.suptitle('')
        text_formatter.setup_margins(fig)

        ## Need to extract the track name ('maze1') for the title in this plot. 
        track_name = active_context.get_description(subset_includelist=['filter_name'], separator=' | ') # 'maze1'
        # TODO: do we want to convert this into "long" or "short"?
        flexitext(text_formatter.left_margin, text_formatter.top_margin, f'<size:22><weight:bold>{track_name}</> replay|laps <weight:bold>firing rate</></>', va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")

        # add static tiny labels for the neuron_id beside each data point
        text_kwargs = dict(textcoords="offset points", xytext=(0,0)) # (2,2)
        texts = [ax.annotate(label, (x, y), ha='left', va='bottom', fontsize=8, **text_kwargs) for i, (x, y, label) in enumerate(zip(x_frs_dict.values(), y_frs_dict.values(), point_hover_labels))]
        # texts = [ax.text(x, y, label, ha='center', va='center', fontsize=8) for i, (x, y, label) in enumerate(zip(x_frs_dict.values(), y_frs_dict.values(), point_hover_labels))]
        
        ## get current axes:
        # ax = plt.gca()

        if is_centered:
            ## The "spines" are the vertical and horizontal "axis" bars where the tick marks are drawn.
            ax.spines[['left', 'bottom']].set_position('center')
            ax.spines[['top', 'right']].set_visible(False)
        else:
            # Hide the right and top spines (box components)
            ax.spines[['right', 'top']].set_visible(False)
    

        if not defer_render:
            fig.show()

        return fig, ax, active_display_context

        
    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)] # filter_name is wrong for long_epoch_context


    ## Long Track Replay|Laps FR Figure
    neuron_replay_stats_df = jonathan_firing_rate_analysis_result.neuron_replay_stats_df.dropna(subset=['long_replay_mean', 'long_non_replay_mean'], inplace=False)
    x_frs = {k:v for k,v in neuron_replay_stats_df['long_replay_mean'].items()}
    y_frs = {k:v for k,v in neuron_replay_stats_df['long_non_replay_mean'].items()}
    fig_L, ax_L, active_display_context_L = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=long_epoch_context)


    ## Short Track Replay|Laps FR Figure
    neuron_replay_stats_df = jonathan_firing_rate_analysis_result.neuron_replay_stats_df.dropna(subset=['short_replay_mean', 'short_non_replay_mean'], inplace=False)
    x_frs = {k:v for k,v in neuron_replay_stats_df['short_replay_mean'].items()}
    y_frs = {k:v for k,v in neuron_replay_stats_df['short_non_replay_mean'].items()}
    fig_S, ax_S, active_display_context_S = _plot_single_track_firing_rate_compare(x_frs, y_frs, active_context=short_epoch_context)

    ## Fit both the axes:
    fit_both_axes(ax_L, ax_S)

    def _perform_write_to_file_callback():
        active_out_figure_paths_L = curr_active_pipeline.output_figure(active_display_context_L, fig_L)
        active_out_figure_paths_S = curr_active_pipeline.output_figure(active_display_context_S, fig_S)
        return (active_out_figure_paths_L + active_out_figure_paths_S)
    return (fig_L, ax_L, active_display_context_L), (fig_S, ax_S, active_display_context_S), _perform_write_to_file_callback



# ==================================================================================================================== #
# 2023-06-01 - New Averaged Long vs. Short Expected Firing Rates plots                                                 #
# ==================================================================================================================== #

def plot_expected_vs_observed(t_SHARED, y_SHORT, y_LONG, neuron_IDXs, neuron_IDs, track_epochs, sharey=True, figsize=(4, 13), max_num_rows:int=50, shift_offset:int=0, y_scale = "linear"):
    """ 2023-05-31 - plots the expected firing rates for the decoded postions for both the long and short decoder vs. the observed
    
    plots one subplot for each neuron. 

        t_SHARED, y_LONG, y_SHORT


    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import plot_expected_vs_observed
    
    
    """
    num_neurons:int = len(neuron_IDXs)
    assert len(neuron_IDXs) == len(neuron_IDs)

    active_num_rows = min(num_neurons, max_num_rows)
    # the y-min and max across all cells and time bins:
    global_y_max_SHORT = np.max([np.max(v) for v in y_SHORT]) # 0.9280231494040394
    global_y_min_SHORT = np.min([np.min(v) for v in y_SHORT]) # -4.850176354048617
    print(f'global_y_min_SHORT: {global_y_min_SHORT}, global_y_max_SHORT: {global_y_max_SHORT}')

    global_y_max_LONG = np.max([np.max(v) for v in y_LONG]) # 0.9280231494040394
    global_y_min_LONG = np.min([np.min(v) for v in y_LONG]) # -4.850176354048617
    print(f'global_y_min_LONG: {global_y_min_LONG}, global_y_max_LONG: {global_y_max_LONG}')

    # the y-min and max across all cells and time bins:
    global_y_max = np.max([global_y_max_LONG, global_y_max_SHORT]) # 0.9280231494040394
    # global_y_min = np.min([global_y_min_LONG, global_y_min_SHORT]) # -4.850176354048617
    global_y_min = 0.0 # y min is overriden to 0 for peak-to-peak (ptp) mode
    print(f'global_y_min: {global_y_min}, global_y_max: {global_y_max}')

    # global_x_min, global_x_max = Flat_decoder_time_bin_centers_LONG[0], Flat_decoder_time_bin_centers_LONG[-1]
    global_x_min, global_x_max = t_SHARED[0], t_SHARED[-1]


    fig, axes = plt.subplots(ncols=1, nrows=active_num_rows, sharex=True, sharey=sharey, figsize=figsize) # , sharey=True
    # Set the x-axis and y-axis limits of the first subplot, which because sharex=True and sharey=True will set the rest of the plots too. NOTE: setting the axis limits FIRST disables autoscaling which cause problems when adding the epoch region indicator
    axes[0].set_xlim([global_x_min, global_x_max])
    axes[0].set_ylim([global_y_min, global_y_max])

    for i, ax in enumerate(axes):
        shifted_i = shift_offset + i
        neuron_IDX = neuron_IDXs[shifted_i]
        neuron_id = neuron_IDs[shifted_i]
        
        # convert y-axis to Logarithmic scale
        ax.set_yscale(y_scale)

        # ax.scatter(Flat_decoder_time_bin_centers, np.concatenate([all_epochs_computed_observed_from_expected_difference[decoded_epoch_idx][neuron_IDX, :] for decoded_epoch_idx in np.arange(decoder_result.num_filter_epochs)]), marker="o",  s=2)
        # ax.scatter(Flat_decoder_time_bin_centers, Flat_all_epochs_computed_expected_cell_num_spikes[neuron_IDX], marker="o",  s=2, label=f'long[{neuron_id}]')
        ax.axhline(y=0.0, linewidth=1, color='k') # the y=0.0 line
        # Per-epoch result:
        # ax.scatter(Flat_decoder_time_bin_centers_LONG, Flat_all_epochs_computed_expected_cell_num_spikes_LONG[neuron_IDX], marker="o",  s=2, label=f'Long[{neuron_id}]')
        # ax.scatter(Flat_decoder_time_bin_centers_SHORT, Flat_all_epochs_computed_expected_cell_num_spikes_SHORT[neuron_IDX], marker="o",  s=2, label=f'Short[{neuron_id}]')

        ## Add in the mean epoch difference:
        _s_long = ax.scatter(t_SHARED, y_LONG[neuron_IDX, :], marker="s",  s=5, label=f'E<Long[{neuron_id}]>', alpha=0.8)
        _s_short = ax.scatter(t_SHARED, y_SHORT[neuron_IDX, :], marker="s",  s=5, label=f'E<Short[{neuron_id}]>', alpha=0.8)

        ax.set_ylabel(f'{neuron_id}')
        
        epochs_collection, epoch_labels = draw_epoch_regions(track_epochs, ax, defer_render=False, debug_print=False)
        
    # axes[-1].set_xlabel('time')
    axes[-1].set_xlabel('replay idx')
    # axes[-1].legend() # show the legend

    axes[0].legend((_s_long, _s_short), ('E<Long>', 'E<Short>'), loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0, frameon=False)
    # plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center')

    # (top=0.921, bottom=0.063, left=0.06, right=0.986, hspace=1.0, wspace=0.2)
    # plt.tight_layout()
    plt.suptitle('Expected vs. Observed Firing Rate Differences (by Replay Epoch)', wrap=True)
    return fig, axes



@function_attributes(short_name=None, tags=[], conforms_to=[], input_requires=[], output_provides=[], uses=['JonathanFiringRateAnalysisResult'], used_by=['_display_short_long_pf1D_comparison'], creation_date='2023-06-16 13:01')
def determine_long_short_pf1D_indicies_sort_by_peak(curr_active_pipeline, curr_any_context_neurons, debug_print=False):
    """ Builds proper sort indicies for '_display_short_long_pf1D_comparison'
    Captures Nothing

    curr_any_context_neurons: 
    
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_pf1D, short_pf1D, global_pf1D = long_results.pf1D, short_results.pf1D, global_results.pf1D
    ## Builds proper sort indicies for '_display_short_long_pf1D_comparison'
    long_ratemap = long_pf1D.ratemap
    short_ratemap = short_pf1D.ratemap
    # gets the curr_any_context_neurons: aclus of neurons to sort
    curr_any_context_neurons = _find_any_context_neurons(*[k.neuron_ids for k in [long_ratemap, short_ratemap]])
    
    """
    def _subfn_sort_desired(extant_arr, desired_sort_arr):
        """ 
        Want to find the set of sort indicies that can be applied to extant_arr s.t.
        (extant_arr[out_sort_idxs] == desired_sort_arr)
        
        INEFFICIENT: O^n^2
        
        Usage:
        
            new_all_aclus_sort_indicies = _sort_desired(active_2d_plot.neuron_ids, all_sorted_aclus)
            assert len(new_all_aclus_sort_indicies) == len(active_2d_plot.neuron_ids), f"need to have one new_all_aclus_sort_indicies value for each neuron_id"
            assert np.all(active_2d_plot.neuron_ids[new_all_aclus_sort_indicies] == all_sorted_aclus), f"must sort "
            new_all_aclus_sort_indicies
        """
        missing_aclu_indicies = np.isin(extant_arr, desired_sort_arr, invert=True)
        missing_aclus = extant_arr[missing_aclu_indicies] # array([ 3,  4,  8, 13, 24, 34, 56, 87])
        if len(missing_aclus) > 0:
            desired_sort_arr = np.concatenate((desired_sort_arr, missing_aclus)) # the final desired output order of aclus. Want to compute the indicies that are required to sort an ordered array of indicies in this order
            ## TODO: what about entries in desired_sort_arr that might be missing in extant_arr?? Hopefully never happens.
        assert len(desired_sort_arr) == len(extant_arr), f"need to have one all_sorted_aclu value for each neuron_id but len(desired_sort_arr): {len(desired_sort_arr)} and len(extant_arr): {len(extant_arr)}"
        # sort_idxs = np.array([desired_sort_arr.tolist().index(v) for v in extant_arr])
        sort_idxs = np.array([extant_arr.tolist().index(v) for v in desired_sort_arr])
        assert len(sort_idxs) == len(extant_arr), f"need to have one new_all_aclus_sort_indicies value for each neuron_id"
        assert np.all(extant_arr[sort_idxs] == desired_sort_arr), f"must sort: extant_arr[sort_idxs]: {extant_arr[sort_idxs]}\n desired_sort_arr: {desired_sort_arr}"
        return sort_idxs, desired_sort_arr

    #### BEGIN FUNCTION BODY ###
    

    ## Sorts aclus using `neuron_replay_stats_df`'s columns (  neuron_replay_stats_df = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis'].neuron_replay_stats_df -- this is produced by '_perform_jonathan_replay_firing_rate_analyses') :
    if not isinstance(curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis, JonathanFiringRateAnalysisResult):
        jonathan_firing_rate_analysis_result = JonathanFiringRateAnalysisResult(**curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.to_dict())
    else:
        jonathan_firing_rate_analysis_result = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis

    neuron_replay_stats_df = jonathan_firing_rate_analysis_result.neuron_replay_stats_df.copy()
    _sorted_neuron_stats_df = neuron_replay_stats_df.sort_values(by=["long_pf_peak_x", "short_pf_peak_x", 'neuron_IDX'], ascending=[True, True, True], inplace=False).copy() # also did test_df = neuron_replay_stats_df.sort_values(by=['long_pf_peak_x'], inplace=False, ascending=True).copy()
    _sorted_neuron_stats_df = _sorted_neuron_stats_df[np.isin(_sorted_neuron_stats_df.index, curr_any_context_neurons)] # clip to only those neurons included in `curr_any_context_neurons`
    _sorted_aclus = _sorted_neuron_stats_df.index.to_numpy()
    _sorted_neuron_IDXs = _sorted_neuron_stats_df.neuron_IDX.to_numpy()
    if debug_print:
        print(f'_sorted_aclus: {_sorted_aclus}')
        print(f'_sorted_neuron_IDXs: {_sorted_neuron_IDXs}')

    ## Use this sort for the 'curr_any_context_neurons' sort order:
    new_all_aclus_sort_indicies, desired_sort_arr = _subfn_sort_desired(curr_any_context_neurons, _sorted_aclus)
    if debug_print:
        print(f'new_all_aclus_sort_indicies: {new_all_aclus_sort_indicies}')
    return new_all_aclus_sort_indicies



