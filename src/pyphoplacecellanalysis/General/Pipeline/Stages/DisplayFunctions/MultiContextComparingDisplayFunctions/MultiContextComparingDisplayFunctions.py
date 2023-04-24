from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, FigureBase # FigureBase: both Figure and SubFigure

from neuropy.utils.dynamic_container import overriding_dict_with # required for _display_2d_placefield_result_plot_raw
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.plotting.figure import Fig # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.plotting.ratemaps import plot_ratemap_1D # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.utils.matplotlib_helpers import build_or_reuse_figure # used for `_make_pho_jonathan_batch_plots(...)`
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter # for `_plot_long_short_firing_rate_indicies`

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer  # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.Widgets.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget # for context_nested_docks/single_context_nested_docks

# MOVED IN TO `_single_context_nested_docks`
# from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import make_fr
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results, _find_any_context_neurons, build_neurons_color_map # for plot_short_v_long_pf1D_comparison
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import build_replays_custom_scatter_markers # used in _make_pho_jonathan_batch_plots
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _build_neuron_type_distribution_color # used in _make_pho_jonathan_batch_plots


from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1D_placecell_validation # for _make_pho_jonathan_batch_plots

from enum import unique # for PlacefieldOverlapMetricMode
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # for PlacefieldOverlapMetricMode

@unique
class PlacefieldOverlapMetricMode(ExtendedEnum):
    """Docstring for PlacefieldOverlapMetricMode."""
    POLY = "POLY"
    CONVOLUTION = "CONVOLUTION"
    PRODUCT = "PRODUCT"
    REL_ENTROPY = "REL_ENTROPY"


class MultiContextComparingDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ MultiContextComparingDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='context_nested_docks', tags=['display','docks','pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:14')
    def _display_context_nested_docks(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
        """ Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18)
        NOTE: Ignores `active_config` because context_nested_docks is for all contexts

        Input:
            owning_pipeline_reference: A reference to the pipeline upon which this display function is being called

        Usage:

        display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_filtered_session_ctx, enable_gui=False, debug_print=False) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
        master_dock_win = display_output['master_dock_win']
        app = display_output['app']
        out_items = display_output['out_items']

        """
        assert owning_pipeline_reference is not None
        #
        if include_whitelist is None:
            include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        out_items = {}
        master_dock_win, app, out_items = _context_nested_docks(owning_pipeline_reference, active_config_names=include_whitelist, **overriding_dict_with(lhs_dict={'enable_gui': False, 'debug_print': False}, **kwargs))

        # return master_dock_win, app, out_items
        return {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}

    @function_attributes(short_name='jonathan_interactive_replay_firing_rate_comparison', tags=['display','interactive','jonathan', 'firing_rate', 'pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:14')
    def _display_jonathan_interactive_replay_firing_rate_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
            """ Jonathan's interactive display. Currently hacked up to directly compute the results to display within this function
                Internally calls `_make_jonathan_interactive_plot(...)`

                Usage:
                active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
                curr_active_pipeline.display('_display_jonathan_replay_firing_rate_comparison', active_identifying_session_ctx)

            """
            if include_whitelist is None:
                include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_whitelist[0] # 'maze1_PYR'
            short_epoch_name = include_whitelist[1] # 'maze2_PYR'
            if len(include_whitelist) > 2:
                global_epoch_name = include_whitelist[-1] # 'maze_PYR'
            else:
                print(f'WARNING: no global_epoch detected.')
                global_epoch_name = '' # None

            print(f'include_whitelist: {include_whitelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')
            pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
            pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
            pf1d = computation_results[global_epoch_name]['computed_data']['pf1D']

            ## Proper global-computations based way:
            sess = owning_pipeline_reference.sess
            aclu_to_idx = global_computation_results.computed_data['jonathan_firing_rate_analysis']['rdf']['aclu_to_idx']
            rdf = global_computation_results.computed_data['jonathan_firing_rate_analysis']['rdf']['rdf']
            irdf = global_computation_results.computed_data['jonathan_firing_rate_analysis']['irdf']['irdf']
            pos_df = global_computation_results.sess.position.to_dataframe()

            ## time_binned_unit_specific_binned_spike_rate mode:
            time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']['time_bins']
            time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']['time_binned_unit_specific_binned_spike_rate']
            # ## instantaneous_unit_specific_spike_rate mode:
            # time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['time_bins']
            # time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['instantaneous_unit_specific_spike_rate_values']

            neuron_replay_stats_df = global_computation_results.computed_data['jonathan_firing_rate_analysis']['neuron_replay_stats_df']

            graphics_output_dict, neuron_df = _make_jonathan_interactive_plot(sess, time_bins, neuron_replay_stats_df, time_binned_unit_specific_binned_spike_rate, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False)
            graphics_output_dict['plot_data'] = {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']}

            return graphics_output_dict

    @function_attributes(short_name='batch_pho_jonathan_interactive_replay_firing_rate_comparison', tags=['display','interactive','jonathan', 'firing_rate', 'matplotlib', 'batch'], input_requires=[], output_provides=[], uses=['_make_pho_jonathan_batch_plots'], used_by=[], creation_date='2023-04-11 03:14')
    def _display_batch_pho_jonathan_replay_firing_rate_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
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
            if include_whitelist is None:
                include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_whitelist[0] # 'maze1_PYR'
            short_epoch_name = include_whitelist[1] # 'maze2_PYR'
            assert len(include_whitelist) > 2
            global_epoch_name = include_whitelist[-1] # 'maze_PYR'
            print(f'include_whitelist: {include_whitelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')

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

            # cell_spikes_dfs_list, aclu_to_fragile_linear_idx_map = _build_spikes_df_interpolated_props(global_results) # cell_spikes_dfs_list is indexed by aclu_to_fragile_linear_idx_map
            cell_spikes_dfs_dict, aclu_to_fragile_linear_idx_map = _build_spikes_df_interpolated_props(global_results) # cell_spikes_dfs_list is indexed by aclu_to_fragile_linear_idx_map
            time_variable_name = global_results.sess.spikes_df.spikes.time_variable_name

            # pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
            # pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
            pf1D_all = global_results['computed_data']['pf1D'] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

            ## Proper global-computations based way:
            sess = owning_pipeline_reference.sess
            active_identifying_session_ctx = sess.get_context()
            t_split = sess.paradigm[0][0,1] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

            aclu_to_idx = global_computation_results.computed_data['jonathan_firing_rate_analysis']['rdf']['aclu_to_idx']
            rdf = global_computation_results.computed_data['jonathan_firing_rate_analysis']['rdf']['rdf']
            irdf = global_computation_results.computed_data['jonathan_firing_rate_analysis']['irdf']['irdf']
            # pos_df = global_computation_results.sess.position.to_dataframe()
            ## time_binned_unit_specific_binned_spike_rate mode:
            time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']['time_bins']
            time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']['time_binned_unit_specific_binned_spike_rate']
            # ## instantaneous_unit_specific_spike_rate mode:
            # time_bins = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['time_bins']
            # time_binned_unit_specific_binned_spike_rate = global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_instantaneous_unit_specific_spike_rate']['instantaneous_unit_specific_spike_rate_values']
            neuron_replay_stats_df = global_computation_results.computed_data['jonathan_firing_rate_analysis']['neuron_replay_stats_df']
            # compare_firing_rates(rdf, irdf)

            n_max_plot_rows = kwargs.pop('n_max_plot_rows', 6)
            show_inter_replay_frs = kwargs.pop('show_inter_replay_frs', True)
            included_unit_neuron_IDs = kwargs.pop('included_unit_neuron_IDs', None)

            active_context = active_identifying_session_ctx.adding_context(collision_prefix='fn', fn_name='batch_pho_jonathan_interactive_replay_firing_rate_comparison')
            curr_fig_num = kwargs.pop('fignum', None)
            if curr_fig_num is None:
                ## Set the fig_num, if not already set:
                curr_fig_num = f'long|short fr indicies_{active_context.get_description(separator="/")}'
            kwargs['fignum'] = curr_fig_num

            graphics_output_dict = _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, time_binned_unit_specific_binned_spike_rate, pf1D_all, aclu_to_idx, rdf, irdf,
                show_inter_replay_frs=show_inter_replay_frs, n_max_plot_rows=n_max_plot_rows, included_unit_neuron_IDs=included_unit_neuron_IDs, cell_spikes_dfs_dict=cell_spikes_dfs_dict, time_variable_name=time_variable_name, **kwargs)


            graphics_output_dict['plot_data'] = {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate'],
                'time_variable_name':time_variable_name}

            return graphics_output_dict


    def _display_short_long_pf1D_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
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


            # Plot 1D Keywoard args:
            shared_kwargs = kwargs.pop('shared_kwargs', None)
            long_kwargs = kwargs.pop('long_kwargs', None)
            short_kwargs = kwargs.pop('short_kwargs', None)

            if include_whitelist is None:
                include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_whitelist[0] # 'maze1_PYR'
            short_epoch_name = include_whitelist[1] # 'maze2_PYR'
            assert len(include_whitelist) > 2
            global_epoch_name = include_whitelist[-1] # 'maze_PYR'
            if debug_print:
                print(f'include_whitelist: {include_whitelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')           
    
            long_results = computation_results[long_epoch_name]['computed_data']
            short_results = computation_results[short_epoch_name]['computed_data']
            # curr_any_context_neurons = _find_any_context_neurons(*[owning_pipeline_reference.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in [long_epoch_name, short_epoch_name]])
            curr_any_context_neurons = _find_any_context_neurons(*[a_result.pf1D.ratemap.neuron_ids for a_result in [long_results, short_results]])

            (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, curr_any_context_neurons, reuse_axs_tuple=reuse_axs_tuple, single_figure=single_figure,
                shared_kwargs=shared_kwargs, long_kwargs=long_kwargs, short_kwargs=short_kwargs, debug_print=debug_print)

            graphics_output_dict = MatplotlibRenderPlots(name='display_short_long_pf1D_comparison', figures=(fig_long_pf_1D, fig_short_pf_1D), axes=(ax_long_pf_1D, ax_short_pf_1D), plot_data={})
            graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}

            return graphics_output_dict


    def _display_short_long_pf1D_scalar_overlap_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
            """ Displays a figure for comparing the scalar comparison quantities computed for 1D placefields across-epochs (between the short and long tracks)
                This currently renders as a bar-graph

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
            overlap_metric_mode = kwargs.pop('overlap_metric_mode', PlacefieldOverlapMetricMode.POLY)
            if not isinstance(overlap_metric_mode, PlacefieldOverlapMetricMode):
                overlap_metric_mode = PlacefieldOverlapMetricMode.init(overlap_metric_mode)

            reuse_axs_tuple = kwargs.pop('reuse_axs_tuple', None)            
            # reuse_axs_tuple = None # plot fresh
            # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
            # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
            single_figure = kwargs.pop('single_figure', True)
            debug_print = kwargs.pop('debug_print', False)

            if include_whitelist is None:
                include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            long_epoch_name = include_whitelist[0] # 'maze1_PYR'
            short_epoch_name = include_whitelist[1] # 'maze2_PYR'
            assert len(include_whitelist) > 2
            global_epoch_name = include_whitelist[-1] # 'maze_PYR'
            if debug_print:
                print(f'include_whitelist: {include_whitelist}\nlong_epoch_name: {long_epoch_name}, short_epoch_name: {short_epoch_name}, global_epoch_name: {global_epoch_name}')           
    

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
            
            graphics_output_dict = MatplotlibRenderPlots(name='_display_short_long_pf1D_poly_overlap_comparison', figures=(fig), axes=(ax), plot_data={'colors': neurons_colors_array})
            # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}

            return graphics_output_dict

    @function_attributes(short_name='short_long_firing_rate_index_comparison', tags=['display','short_long','firing_rate', 'fr_index'], input_requires=[], output_provides=[], uses=['_plot_long_short_firing_rate_indicies'], used_by=[], creation_date='2023-04-11 08:08')
    def _display_short_long_firing_rate_index_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
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
            fig_save_parent_path = kwargs.pop('fig_save_parent_path', Path(r'E:\Dropbox (Personal)\Active\Kamran Diba Lab\Results from 2023-01-20 - LongShort Firing Rate Indicies'))            
            debug_print = kwargs.pop('debug_print', False)

            # Plot long|short firing rate index:
            long_short_fr_indicies_analysis_results = global_computation_results.computed_data['long_short_fr_indicies_analysis']
            x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
            active_context = long_short_fr_indicies_analysis_results['active_context']
            fig, _temp_full_fig_save_path = _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, active_context, fig_save_parent_path=fig_save_parent_path, debug_print=debug_print)

            graphics_output_dict = MatplotlibRenderPlots(name='display_short_long_firing_rate_index_comparison', figures=(fig), axes=tuple(fig.axes), plot_data={})
            # graphics_output_dict['plot_data'] = {'sort_indicies': (long_sort_ind, short_sort_ind), 'colors':(long_neurons_colors_array, short_neurons_colors_array)}

            return graphics_output_dict



# ==================================================================================================================== #
# Private Display Helpers                                                                                              #
# ==================================================================================================================== #
def _single_context_nested_docks(curr_active_pipeline, active_config_name, app, master_dock_win, enable_gui=False, debug_print=True):
        """ 2022-08-18 - Called for each config name in context_nested_docks's for loop.


        """
        

        out_display_items = dict()

        # Get relevant variables for this particular context:
        # curr_active_pipeline is set above, and usable here
        # sess = curr_active_pipeline.filtered_sessions[active_config_name]
        active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None)

        curr_active_config = curr_active_pipeline.active_configs[active_config_name]
        # curr_active_display_config = curr_active_config.plotting_config

        ## Build the active context by starting with the session context:
        # active_identifying_session_ctx = sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        ## Add the filter to the active context
        # active_identifying_session_ctx.add_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
        # active_identifying_session_ctx = curr_active_pipeline.filtered_contexts[active_config_name]
        active_identifying_filtered_session_ctx = curr_active_pipeline.filtered_contexts[active_config_name] # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

        # ==================================================================================================================== #
        ## Figure Formatting Config GUI (FigureFormatConfigControls):
        def on_finalize_figure_format_config(updated_figure_format_config):
                if debug_print:
                    print('on_finalize_figure_format_config')
                    print(f'\t {updated_figure_format_config}')
                # figure_format_config = updated_figure_format_config
                pass

        ## Finally, add the display function to the active context
        active_identifying_ctx = active_identifying_filtered_session_ctx.adding_context('display_fn', display_fn_name='figure_format_config_widget')
        active_identifying_ctx_string = active_identifying_ctx.get_description(separator='|') # Get final discription string:
        if debug_print:
            print(f'active_identifying_ctx_string: {active_identifying_ctx_string}')

        if enable_gui:
            from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks
            figure_format_config_widget = FigureFormatConfigControls(config=curr_active_config)
            figure_format_config_widget.figure_format_config_finalized.connect(on_finalize_figure_format_config)
            figure_format_config_widget.show() # even without .show() being called, the figure still appears

            ## Get the figure_format_config from the figure_format_config widget:
            figure_format_config = figure_format_config_widget.figure_format_config

            master_dock_win.add_display_dock(identifier=active_identifying_ctx_string, widget=figure_format_config_widget, display_config=CustomDockDisplayConfig(showCloseButton=False))
            out_display_items[active_identifying_ctx] = (figure_format_config_widget)

        else:

            # out_display_items[active_identifying_ctx] = None
             out_display_items[active_identifying_ctx] = (PhoUIContainer(figure_format_config=curr_active_config))

        # ==================================================================================================================== #
        ## 2D Position Decoder Section (DecoderPlotSelectorWidget):
        active_identifying_ctx = active_identifying_filtered_session_ctx.adding_context('display_fn', display_fn_name='2D Position Decoder')
        active_identifying_ctx_string = active_identifying_ctx.get_description(separator='|') # Get final discription string:
        if debug_print:
            print(f'active_identifying_ctx_string: {active_identifying_ctx_string}')

        if enable_gui:
            decoder_plot_widget = DecoderPlotSelectorWidget()
            decoder_plot_widget.show()
            master_dock_win.add_display_dock(identifier=active_identifying_ctx_string, widget=decoder_plot_widget, display_config=CustomDockDisplayConfig(showCloseButton=True))
            out_display_items[active_identifying_ctx] = (decoder_plot_widget)
        else:
            out_display_items[active_identifying_ctx] = None

        # ==================================================================================================================== #
        ## GUI Placefields (pyqtplot_plot_image_array):

        # Get the decoders from the computation result:
        # active_one_step_decoder = computation_result.computed_data['pf2D_Decoder'] # doesn't actually require the Decoder, could just use computation_result.computed_data['pf2D']
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        active_identifying_ctx = active_identifying_filtered_session_ctx.adding_context('display_fn', display_fn_name='pyqtplot_plot_image_array')
        active_identifying_ctx_string = active_identifying_ctx.get_description(separator='|') # Get final discription string:
        if debug_print:
            print(f'active_identifying_ctx_string: {active_identifying_ctx_string}')

        if enable_gui:
            ## Build the widget:
            app, pyqtplot_pf2D_parent_root_widget, pyqtplot_pf2D_root_render_widget, pyqtplot_pf2D_plot_array, pyqtplot_pf2D_img_item_array, pyqtplot_pf2D_other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy, app=app, parent_root_widget=None, root_render_widget=None, max_num_columns=8)
            pyqtplot_pf2D_parent_root_widget.show()
            master_dock_win.add_display_dock(identifier=active_identifying_ctx_string, widget=pyqtplot_pf2D_parent_root_widget, display_config=CustomDockDisplayConfig(showCloseButton=True))
            out_display_items[active_identifying_ctx] = (pyqtplot_pf2D_parent_root_widget, pyqtplot_pf2D_root_render_widget, pyqtplot_pf2D_plot_array, pyqtplot_pf2D_img_item_array, pyqtplot_pf2D_other_components_array)
        else:
            out_display_items[active_identifying_ctx] = None

        return active_identifying_filtered_session_ctx, out_display_items
        # END single_context_nested_docks(...)

def _context_nested_docks(curr_active_pipeline, active_config_names, enable_gui=False, debug_print=True):
    """ 2022-08-18 - builds a series of nested contexts for each active_config

    Usage:
        master_dock_win, app, out_items = context_nested_docks(curr_active_pipeline, enable_gui=False, debug_print=True)
    """
    # include_whitelist = curr_active_pipeline.active_completed_computation_result_names # ['maze', 'sprinkle']

    if enable_gui:
        master_dock_win, app = DockAreaWrapper._build_default_dockAreaWindow(title='active_global_window', defer_show=False)
        master_dock_win.resize(1920, 1024)
    else:
        master_dock_win = None
        app = None

    out_items = {}
    for a_config_name in active_config_names:
        active_identifying_session_ctx, out_display_items = _single_context_nested_docks(curr_active_pipeline=curr_active_pipeline, active_config_name=a_config_name, app=app, master_dock_win=master_dock_win, enable_gui=enable_gui, debug_print=debug_print)
        out_items[a_config_name] = (active_identifying_session_ctx, out_display_items)

    return master_dock_win, app, out_items

# ==================================================================================================================== #
def _temp_draw_jonathan_ax(t_split, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, colors=None, fig=None, ax=None, active_aclu:int=0, custom_replay_markers=None, include_horizontal_labels=True, include_vertical_labels=True, should_render=False):
    """ Draws the time binned firing rates and the replay firing rates for a single cell


        custom_replay_markers:
            # The colors for each point indicating the percentage of participating cells that belong to which track.
                - More long_only -> more red
                - More short_only -> more blue


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
                    ax.plot(centers[replay_idx], heights[replay_idx], markersize=5, **out_plot_kwarg, zorder=7)
            else:
                # don't do the fancy custom makers for the inactive (zero firing for this aclu) replay points:
                plot_replays_kwargs = {
                    'marker':'o',
                    's': 3,
                    'c': 'black'
                }
                ax.scatter(centers, heights, **plot_replays_kwargs, zorder=5)
                # pass # don't plot at all

    else:
        
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
        ax.plot(centers, heights, '.', color=colors[1]+"80", zorder=4)

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
            ax.plot(t, v, color='#aaaaff8c', zorder=2) # this color is a translucent lilac (purple) color)
    except KeyError:
        print(f'non-placefield neuron. Skipping.')
        t, v = None, None
        pass


    # Highlight the two epochs with their characteristic colors ['r','b'] - ideally this would be at the very back
    x_start, x_stop = ax.get_xlim()
    ax.axvspan(x_start, t_split, color='red', alpha=0.2, zorder=0)
    ax.axvspan(t_split, x_stop, color='blue', alpha=0.2, zorder=0)

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

def _build_spikes_df_interpolated_props(global_results):
    # Group by the aclu (cluster indicator) column
    cell_grouped_spikes_df = global_results.sess.spikes_df.groupby(['aclu'])
    cell_spikes_dfs = [cell_grouped_spikes_df.get_group(a_neuron_id) for a_neuron_id in global_results.sess.spikes_df.spikes.neuron_ids] # a list of dataframes for each neuron_id
    aclu_to_fragile_linear_idx_map = {a_neuron_id:i for i, a_neuron_id in enumerate(global_results.sess.spikes_df.spikes.neuron_ids)}
    # get position variables usually used within pfND.setup(...) - self.t, self.x, self.y:
    ndim = global_results.computed_data.pf1D.ndim
    pos_df = global_results.computed_data.pf1D.filtered_pos_df
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

def _plot_general_all_spikes(ax_activity_v_time, active_spikes_df, time_variable_name='t', defer_render=True):
    """ Plots all spikes for a given cell from that cell's complete `active_spikes_df`
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
    ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_spikes_df[time_variable_name].values, active_spikes_df['x'].values, spikes_color_RGB=(0.1, 0.1, 0.1), spikes_alpha=1.0) # all

    active_long_spikes_df = active_spikes_df[active_spikes_df.is_included_long_pf1D]
    ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_long_spikes_df[time_variable_name].values, active_long_spikes_df['x'].values, spikes_color_RGB=(1, 0, 0), spikes_alpha=1.0, zorder=15)

    active_short_spikes_df = active_spikes_df[active_spikes_df.is_included_short_pf1D]
    ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_short_spikes_df[time_variable_name].values, active_short_spikes_df['x'].values, spikes_color_RGB=(0, 0, 1), spikes_alpha=1.0, zorder=15)

    # active_global_spikes_df = active_spikes_df[active_spikes_df.is_included_global_pf1D]
    # ax_activity_v_time = _simple_plot_spikes(ax_activity_v_time, active_global_spikes_df[time_variable_name].values, active_global_spikes_df['x'].values, spikes_color_RGB=(0, 1, 0), spikes_alpha=1.0, zorder=25, markersize=2.5)

    if not defer_render:
        fig = ax_activity_v_time.get_figure().get_figure() # For SubFigure
        fig.canvas.draw()

    return ax_activity_v_time


@function_attributes(short_name='_plot_pho_jonathan_batch_plot_single_cell', tags=['private'], input_requires=[], output_provides=[], uses=['plot_1D_placecell_validation', '_temp_draw_jonathan_ax', '_plot_general_all_spikes'], used_by=['_make_pho_jonathan_batch_plots'], creation_date='2023-04-11 08:06')
def _plot_pho_jonathan_batch_plot_single_cell(t_split, time_bins, unit_specific_time_binned_firing_rates, pf1D_all, rdf_aclu_to_idx, rdf, irdf, show_inter_replay_frs, pf1D_aclu_to_idx, aclu, curr_fig, colors, debug_print=False, **kwargs):
    """ Plots a single cell's plots for a stacked Jonathan-style firing-rate-across-epochs-plot
    Internally calls `plot_1D_placecell_validation`, `_temp_draw_jonathan_ax`, and `_plot_general_all_spikes`

    Used by:
        `_make_pho_jonathan_batch_plots`

    Historical:
        used to take sess: DataSession as first argument and then access `sess.paradigm[0][0,1]` internally. On 2022-11-27 refactored to take this time `t_split` directly and no longer require session


    """
    # cell_linear_fragile_IDX = rdf_aclu_to_idx[aclu] # get the cell_linear_fragile_IDX from aclu
    # title_string = ' '.join(['pf1D', f'Cell {aclu:02d}'])
    # subtitle_string = ' '.join([f'{pf1D_all.config.str_for_display(False)}'])
    # if debug_print:
    #     print(f'\t{title_string}\n\t{subtitle_string}')

    short_title_string = f'{aclu:02d}'

    # gridspec mode:
    curr_fig.set_facecolor('0.75')

    num_gridspec_columns = 8 # hardcoded
    gs_kw = dict(width_ratios=np.repeat(1, num_gridspec_columns).tolist(), height_ratios=[1, 1], wspace=0.0, hspace=0.0)
    gs_kw['width_ratios'][-1] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others

    gs = curr_fig.add_gridspec(2, num_gridspec_columns, **gs_kw) # layout figure is usually a gridspec of (1,8)
    curr_ax_firing_rate = curr_fig.add_subplot(gs[0, :-1]) # the whole top row except the last element (to match the firing rates below)
    curr_ax_cell_label = curr_fig.add_subplot(gs[0, -1]) # the last element of the first row contains the labels that identify the cell
    curr_ax_lap_spikes = curr_fig.add_subplot(gs[1, :-1]) # all up to excluding the last element of the row
    curr_ax_placefield = curr_fig.add_subplot(gs[1, -1], sharey=curr_ax_lap_spikes) # only the last element of the row

    # Setup title axis:
    title_axes_kwargs = dict(ha="center", va="center", fontsize=18, color="black")
    curr_ax_cell_label.text(0.5, 0.5, short_title_string, transform=curr_ax_cell_label.transAxes, **title_axes_kwargs)
    curr_ax_cell_label.axis('off')

    custom_replay_scatter_markers_plot_kwargs_list = kwargs.pop('custom_replay_scatter_markers_plot_kwargs_list', None)

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

    # the index passed into plot_1D_placecell_validation(...) must be in terms of the pf1D_all ratemap that's provided. the rdf_aclu_to_idx does not work and will result in indexing errors
    # pf1D_aclu_to_idx = {aclu:i for i, aclu in enumerate(pf1D_all.ratemap.neuron_ids)}

    # Not sure if this is okay, but it's possible that the aclu isn't in the ratemap, in which case currently we'll just skip plotting?
    cell_linear_fragile_IDX = pf1D_aclu_to_idx.get(aclu, None)
    if cell_linear_fragile_IDX is None:
        print(f'WARNING: aclu {aclu} is not present in the pf1D_all ratemaps. Which contain aclus: {pf1D_all.ratemap.neuron_ids}')
    _ = plot_1D_placecell_validation(pf1D_all, cell_linear_fragile_IDX, extant_fig=curr_fig, extant_axes=(curr_ax_lap_spikes, curr_ax_placefield),
            **({'should_include_labels': False, 'should_plot_spike_indicator_points_on_placefield': False, 'spike_indicator_lines_alpha': 0.2,
                'spikes_color':(0.1, 0.1, 0.1), 'spikes_alpha':0.1, 'should_include_spikes': False} | kwargs))

    # Custom All Spikes: Note that I set `'should_include_spikes': False` in call to `plot_1D_placecell_validation` above so the native spikes aren't plotted
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




@function_attributes(short_name='_make_pho_jonathan_batch_plots', tags=['private'], input_requires=[], output_provides=[], uses=['_plot_pho_jonathan_batch_plot_single_cell'], used_by=[], creation_date='2023-04-11 08:06')
def _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, included_unit_neuron_IDs=None, n_max_plot_rows:int=4, debug_print=False, **kwargs):
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



    ## Figure Setup:
    fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
    # fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    subfigs = fig.subfigures(actual_num_subfigures, 1, wspace=0.07)

    

    ##########################

    rdf, (_percent_long_only, _percent_shared, _percent_short_only, _percent_short_long_diff) = _build_neuron_type_distribution_color(rdf)

    # Build custom replay markers:
    custom_replay_scatter_markers_plot_kwargs_list = build_replays_custom_scatter_markers(rdf, debug_print=debug_print)
    kwargs['custom_replay_scatter_markers_plot_kwargs_list'] = custom_replay_scatter_markers_plot_kwargs_list


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
            curr_fig = subfigs[i] 
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

    graphics_output_dict = {'fig': fig, 'subfigs': subfigs, 'axs': axs_list, 'colors': colors}
    fig.show()
    return graphics_output_dict


# ==================================================================================================================== #




# ==================================================================================================================== #

@mpl.rc_context(Fig.get_mpl_style(style='figPublish'))
def plot_short_v_long_pf1D_comparison(long_results, short_results, curr_any_context_neurons, reuse_axs_tuple=None, single_figure=False, shared_kwargs=None, long_kwargs=None, short_kwargs=None, debug_print=False):
    """ Produces a figure to compare the 1D placefields on the long vs. the short track. 
    
    single_figure:bool - if True, both long and short are plotted on the same axes of a single shared figure. Otherwise seperate figures are used for each
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.MultiContextComparingDisplayFunctions import plot_short_v_long_pf1D_comparison

        long_results = curr_active_pipeline.computation_results['maze1_PYR'].computed_data
        short_results = curr_active_pipeline.computation_results['maze2_PYR'].computed_data
        curr_any_context_neurons = _find_any_context_neurons(*[curr_active_pipeline.computation_results[k].computed_data.pf1D.ratemap.neuron_ids for k in ['maze1_PYR', 'maze2_PYR']])
        reuse_axs_tuple=None # plot fresh
        # reuse_axs_tuple=(ax_long_pf_1D, ax_short_pf_1D)
        # reuse_axs_tuple=(ax_long_pf_1D, ax_long_pf_1D) # plot only on long axis
        (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array) = plot_short_v_long_pf1D_comparison(long_results, short_results, curr_any_context_neurons, reuse_axs_tuple=reuse_axs_tuple, single_figure=True)

    """
    if shared_kwargs is None:
        shared_kwargs = {}
    if long_kwargs is None:
        long_kwargs = {}
    if short_kwargs is None:
        short_kwargs = {}

    # Shared/Common kwargs:
    plot_ratemap_1D_kwargs = (dict(pad=2, brev_mode=PlotStringBrevityModeEnum.NONE, normalize=True, debug_print=debug_print, normalize_tuning_curve=True) | shared_kwargs)
    

    single_cell_pfmap_processing_fn_identity = lambda i, aclu, pfmap: pfmap # flip over the y-axis
    single_cell_pfmap_processing_fn_flipped_y = lambda i, aclu, pfmap: -1.0 * pfmap # flip over the y-axis


    n_neurons = len(curr_any_context_neurons)
    shared_fragile_neuron_IDXs = np.arange(n_neurons)
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
            PhoActiveFigureManager2D.reshow_figure_if_needed(fig_long_pf_1D)
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
    
    if single_figure:
        fig_long_pf_1D.suptitle('Long vs. Short (hatched)')
    else:
        fig_long_pf_1D.suptitle('Long')
        fig_short_pf_1D.suptitle('Short')
        ax_short_pf_1D.set_xlim(ax_long_pf_1D.get_xlim())
        # ax_long_pf_1D.sharex(ax_short_pf_1D)
        
    return (fig_long_pf_1D, ax_long_pf_1D, long_sort_ind, long_neurons_colors_array), (fig_short_pf_1D, ax_short_pf_1D, short_sort_ind, short_neurons_colors_array)




@mpl.rc_context(Fig.get_mpl_style(style='figPublish'))
def plot_short_v_long_pf1D_scalar_overlap_comparison(overlap_scalars_df, pf_neurons_diff, neurons_colors_array, reuse_axs_tuple=None, single_figure=False, overlap_metric_mode=PlacefieldOverlapMetricMode.POLY, variant_name='', debug_print=False):
    """ Produces a figure containing a bar chart to compare *a scalar value* the 1D placefields on the long vs. the short track. 
    poly_overlap_df: pd.DataFrame - computed by compute_polygon_overlap(...)
    pf_neurons_diff: pd.DataFrame - 
    single_figure:bool - if True, both long and short are plotted on the same axes of a single shared figure. Otherwise seperate figures are used for each
    
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.MultiContextComparingDisplayFunctions import plot_short_v_long_pf1D_scalar_overlap_comparison

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


def _test_plot_conv(long_xbins, long_curve, short_xbins, short_curve, x, overlap_curves): # , t_full, m_full
    """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.MultiContextComparingDisplayFunctions import _test_plot_conv
        long_curve = long_curves[i]
        short_curve = short_curves[i] 
    """
    # convolved_result = m_full_subset

    if isinstance(overlap_curves, dict):
        overlap_plot_dict = overlap_curves
    elif isinstance(overlap_curves, (tuple, list)):
        overlap_plot_dict = {}
        # overlap_plot_list = []
        # labels = []
        for i, a_curve in enumerate(overlap_curves):
            # overlap_plot_list.append(x)
            # overlap_plot_list.append(a_curve)
            overlap_plot_dict[f'overlap[{i}]'] = a_curve

    else:
        # overlap_plot_list = (overlap_curves) # make a single item array
        overlap_plot_dict['Conv'] = overlap_curves
        # labels = ['Conv']

    ### Plot the input, repsonse function, and analytic result
    f1, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,num='Analytic', sharex=True, sharey=True)
    ax1.plot(long_xbins, long_curve, label='Long pf1D'),ax1.set_xlabel('Position'),ax1.set_ylabel('Long'),ax1.legend()
    ax2.plot(short_xbins, short_curve, label='Short pf1D'),ax2.set_xlabel('Position'),ax2.set_ylabel('Short'),ax2.legend()
    # ax3.plot(*overlap_plot_list, label=labels) # , label='Conv'
    for a_label, an_overlap_curve in overlap_plot_dict.items():
        ax3.plot(x, an_overlap_curve, label=a_label) # , label='Conv'

    ax3.set_xlabel('Position'),ax3.set_ylabel('Convolved'),ax3.legend()

    # ### Plot the discrete convolution agains analytic
    # f2, ax4 = plt.subplots(nrows=1)
    # # ax4.scatter(t_same[::2],m_same[::2],label='Discrete Convolution (Same)')
    # ax4.scatter(t_full[::2],m_full[::2],label='Discrete Convolution (Full)',facecolors='none',edgecolors='k')
    # # ax4.scatter(t_full_subset[::2], convolved_result[::2], label='Discrete Convolution (Valid)', facecolors='none', edgecolors='r')
    # ax4.plot(t,m,label='Analytic Solution'),ax4.set_xlabel('Time'),ax4.set_ylabel('Signal'),ax4.legend()
    # plt.show()
    return MatplotlibRenderPlots(name='', figures=(f1), axes=((ax1,ax2,ax3)))


# def _test_plot_conv(t, t_response, t_full_subset, m_full_subset, t_full, m_full):
#     s = long_curves[i]
#     r = short_curves[i] 
#     m = m_full_subset

#     ### Plot the input, repsonse function, and analytic result
#     f1, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1,num='Analytic', sharex=True)
#     ax1.plot(t,s,label='Input'),ax1.set_xlabel('Time'),ax1.set_ylabel('Signal'),ax1.legend()
#     ax2.plot(t_response,r,label='Response'),ax2.set_xlabel('Time'),ax2.set_ylabel('Signal'),ax2.legend()
#     ax3.plot(t_full_subset, m_full_subset, label='Output'),ax3.set_xlabel('Time'),ax3.set_ylabel('Signal'),ax3.legend()

#     ### Plot the discrete convolution agains analytic
#     f2, ax4 = plt.subplots(nrows=1)
#     # ax4.scatter(t_same[::2],m_same[::2],label='Discrete Convolution (Same)')
#     ax4.scatter(t_full[::2],m_full[::2],label='Discrete Convolution (Full)',facecolors='none',edgecolors='k')
#     ax4.scatter(t_full_subset[::2], m_full_subset[::2], label='Discrete Convolution (Valid)', facecolors='none', edgecolors='r')
#     ax4.plot(t,m,label='Analytic Solution'),ax4.set_xlabel('Time'),ax4.set_ylabel('Signal'),ax4.legend()
#     plt.show()




@function_attributes(short_name='long_short_fr_indicies', tags=['private', 'long_short', 'long_short_firing_rate', 'firing_rate', 'display', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=['_display_short_long_firing_rate_index_comparison'], creation_date='2023-03-28 14:20')
def _plot_long_short_firing_rate_indicies(x_frs_index, y_frs_index, active_context, fig_save_parent_path=None, neurons_colors=None, debug_print=False):
    """ Plot long|short firing rate index 
    Each datapoint is a neuron.

    used in `_display_short_long_firing_rate_index_comparison()`

    """
    import mplcursors # for hover tooltips that specify the aclu of the selected point

    # from neuropy.utils.matplotlib_helpers import add_value_labels # for adding small labels beside each point indicating their ACLU

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

    point_hover_labels = [f'{i}' for i in list(x_frs_index.keys())] # point_hover_labels will be added as tooltip annotations to the datapoints
    fig, ax = plt.subplots(figsize=(8.5, 7.25), num=f'long|short fr indicies_{active_context.get_description(separator="/")}', clear=True)

    scatter_plot = ax.scatter(x_frs_index.values(), y_frs_index.values(), c=point_colors) # , s=10, alpha=0.5
    plt.xlabel('Replay Firing Rate Index $\\frac{L_{R}-S_{R}}{L_{R} + S_{R}}$', fontsize=16)
    plt.ylabel('Laps Firing Rate Index $\\frac{L_{\\theta}-S_{\\theta}}{L_{\\theta} + S_{\\theta}}$', fontsize=16)
    plt.title('Computed long ($L$)|short($S$) firing rate indicies')
    plt.suptitle(f'{active_context.get_description(separator="/")}')
    # fig = plt.gcf()
    # fig.set_size_inches([8.5, 7.25]) # size figure so the x and y labels aren't cut off

    # add static tiny labels beside each point
    for i, (x, y, label) in enumerate(zip(x_frs_index.values(), y_frs_index.values(), point_hover_labels)):
        # x = x_frs_index.values()[i]
        # y = y_frs_index.values()[i]
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(2,2), ha='left', va='bottom', fontsize=8) # , color=rect.get_facecolor()

    # add hover labels:
    # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
    # https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib/21654635#21654635
    # add hover labels using mplcursors
    mplcursors.cursor(scatter_plot, hover=True).connect("add", lambda sel: sel.annotation.set_text(point_hover_labels[sel.index]))

    ## get current axes:
    # ax = plt.gca()

    # Set the x and y axes to standard limits for easy visual comparison across sessions
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    # # Call the function above. All the magic happens there.
    # add_value_labels(ax, labels=x_labels) # 

    temp_fig_filename = f'{active_context.get_description()}.png'
    if debug_print:
        print(f'temp_fig_filename: {temp_fig_filename}')
    if fig_save_parent_path is None:
        fig_save_parent_path = Path.cwd()

    _temp_full_fig_save_path = fig_save_parent_path.joinpath(temp_fig_filename)

    with ProgressMessagePrinter(_temp_full_fig_save_path, 'Saving', 'plot_long_short_firing_rate_indicies results'):
        fig.savefig(fname=_temp_full_fig_save_path, transparent=True)
    fig.show()

    return fig, _temp_full_fig_save_path
    # return MatplotlibRenderPlots(name='', figures=(fig), axes=(ax))




# ==================================================================================================================== #
# 2023-04-19 Surprise                                                                                                  #
# ==================================================================================================================== #


@function_attributes(short_name=None, tags=['pyqtgraph', 'helper', 'long_short', 'regions', 'rectangles'], input_requires=['pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers.build_pyqtgraph_epoch_indicator_regions'], output_provides=[], uses=[], used_by=[], creation_date='2023-04-19 19:04')
def _helper_add_long_short_session_indicator_regions(win, long_epoch, short_epoch):
    """Add session indicators to pyqtgraph plot for the long and the short epoch

            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.MultiContextComparingDisplayFunctions import _helper_add_long_short_session_indicator_regions

            long_epoch = curr_active_pipeline.filtered_epochs[long_epoch_name]
            short_epoch = curr_active_pipeline.filtered_epochs[short_epoch_name]
            long_epoch_indicator_region_items, short_epoch_indicator_region_items = _helper_add_long_short_session_indicator_regions(win, long_epoch, short_epoch)

            long_epoch_linear_region, long_epoch_region_label = long_epoch_indicator_region_items
            short_epoch_linear_region, short_epoch_region_label = short_epoch_indicator_region_items
    """
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import build_pyqtgraph_epoch_indicator_regions # Add session indicators to pyqtgraph plot
    long_epoch_config = dict(epoch_label='long', pen=pg.mkPen('#0b0049'), brush=pg.mkBrush('#0099ff42'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'))
    short_epoch_config = dict(epoch_label='short', pen=pg.mkPen('#490000'), brush=pg.mkBrush('#f5161659'), hoverBrush=pg.mkBrush('#fff400'), hoverPen=pg.mkPen('#00ff00'))
    
    long_epoch_indicator_region_items = build_pyqtgraph_epoch_indicator_regions(win, t_start=long_epoch.t_start, t_stop=long_epoch.t_stop, **long_epoch_config)
    short_epoch_indicator_region_items = build_pyqtgraph_epoch_indicator_regions(win, t_start=short_epoch.t_start, t_stop=short_epoch.t_stop, **short_epoch_config)
    return long_epoch_indicator_region_items, short_epoch_indicator_region_items

@function_attributes(short_name='plot_long_short_expected_vs_observed_firing_rates', tags=['pyqtgraph','long_short'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-28 17:26')
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
     # win.setWindowTitle('Short v. Long - Leave-one-out Expected vs. Observed Firing Rates')
    win.setWindowTitle('Short v. Long - Leave-one-out Expected vs. Observed Num Spikes')
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
            plots[aclu] = win.plot(x=curr_epoch_time_bins, y=curr_epoch_data, pen=None, symbol=curr_symbol, symbolBrush=cell_color_symbol_brush[unit_IDX], name=f'{label_prefix}[{aclu}]', alpha=0.5) #  symbolBrush=pg.intColor(i,6,maxValue=128)
            legend.addItem(plots[aclu], f'{label_prefix}[{aclu}]')

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

