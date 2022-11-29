from enum import Enum
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from neuropy.utils.dynamic_container import overriding_dict_with # required for _display_2d_placefield_result_plot_raw
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.plotting.figure import Fig # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from neuropy.plotting.ratemaps import plot_ratemap_1D # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.plotting.figure_management import PhoActiveFigureManager2D # for plot_short_v_long_pf1D_comparison (_display_short_long_pf1D_comparison)
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer  # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.Widgets.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import make_fr
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results, _find_any_context_neurons, build_neurons_color_map # for plot_short_v_long_pf1D_comparison
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import build_replays_custom_scatter_markers # used in _make_pho_jonathan_batch_plots
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _build_neuron_type_distribution_color # used in _make_pho_jonathan_batch_plots


from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1D_placecell_validation # for _make_pho_jonathan_batch_plots



class PlacefieldOverlapMetricMode(Enum):
    """Docstring for PlacefieldOverlapMetricMode."""
    POLY = "POLY"
    CONVOLUTION = "CONVOLUTION"
    PRODUCT = "PRODUCT"

    @classmethod
    def init(cls, name):
        if name == cls.POLY.name:
            return cls.POLY
        elif name == cls.CONVOLUTION.name:
            return cls.CONVOLUTION
        elif name == cls.PRODUCT.name:
            return cls.PRODUCT
        else:
            raise NotImplementedError
    




class MultiContextComparingDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ MultiContextComparingDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

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
            # pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
            # pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
            pf1D_all = computation_results[global_epoch_name]['computed_data']['pf1D'] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

            ## Proper global-computations based way:
            sess = owning_pipeline_reference.sess
            t_split = sess.paradigm[0][0,1] # passed to _make_pho_jonathan_batch_plots(t_split, ...)

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
            # compare_firing_rates(rdf, irdf)

            n_max_plot_rows = kwargs.get('n_max_plot_rows', 6)
            show_inter_replay_frs = kwargs.get('show_inter_replay_frs', True)
            included_unit_neuron_IDs = kwargs.get('included_unit_neuron_IDs', None)

            
            graphics_output_dict = _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, time_binned_unit_specific_binned_spike_rate, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, n_max_plot_rows=n_max_plot_rows, included_unit_neuron_IDs=included_unit_neuron_IDs)
            graphics_output_dict['plot_data'] = {'df': neuron_replay_stats_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf, 'time_binned_unit_specific_spike_rate': global_computation_results.computed_data['jonathan_firing_rate_analysis']['time_binned_unit_specific_spike_rate']}

            return graphics_output_dict


    def _display_short_long_pf1D_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
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

            else:
                raise NotImplementedError
            
            graphics_output_dict = MatplotlibRenderPlots(name='_display_short_long_pf1D_poly_overlap_comparison', figures=(fig), axes=(ax), plot_data={'colors': neurons_colors_array})
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

def _plot_pho_jonathan_batch_plot_single_cell(t_split, time_bins, unit_specific_time_binned_firing_rates, pf1D_all, rdf_aclu_to_idx, rdf, irdf, show_inter_replay_frs, pf1D_aclu_to_idx, aclu, curr_fig, colors, debug_print=False, **kwargs):
    """ Plots a single cell's plots for a stacked Jonathan-style firing-rate-across-epochs-plot
    Internally calls `plot_1D_placecell_validation` and `_temp_draw_jonathan_ax`

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
    if aclu in pf1D_aclu_to_idx:
        cell_linear_fragile_IDX = pf1D_aclu_to_idx[aclu]
        _ = plot_1D_placecell_validation(pf1D_all, cell_linear_fragile_IDX, extant_fig=curr_fig, extant_axes=(curr_ax_lap_spikes, curr_ax_placefield),
                **({'should_include_labels': False, 'should_plot_spike_indicator_points_on_placefield': False, 'spike_indicator_lines_alpha': 0.2, 'spikes_color':(0.1, 0.1, 0.1), 'spikes_alpha':0.5} | kwargs))
        t_start, t_end = curr_ax_lap_spikes.get_xlim()
        curr_ax_firing_rate.set_xlim((t_start, t_end)) # We don't want to clip to only the spiketimes for this cell, we want it for all cells, or even when the recording started/ended
        curr_ax_lap_spikes.sharex(curr_ax_firing_rate) # Sync the time axes of the laps and the firing rates

    else:
        print(f'WARNING: aclu {aclu} is not present in the pf1D_all ratemaps. Which contain aclus: {pf1D_all.ratemap.neuron_ids}')
        cell_linear_fragile_IDX = None
        _ = plot_1D_placecell_validation(pf1D_all, cell_linear_fragile_IDX, extant_fig=curr_fig, extant_axes=(curr_ax_lap_spikes, curr_ax_placefield),
                **({'should_include_labels': False, 'should_plot_spike_indicator_points_on_placefield': False, 'spike_indicator_lines_alpha': 0.2, 'spikes_color':(0.1, 0.1, 0.1), 'spikes_alpha':0.5} | kwargs))
        t_start, t_end = curr_ax_lap_spikes.get_xlim()
        curr_ax_firing_rate.set_xlim((t_start, t_end)) # We don't want to clip to only the spiketimes for this cell, we want it for all cells, or even when the recording started/ended
        curr_ax_lap_spikes.sharex(curr_ax_firing_rate) # Sync the time axes of the laps and the firing rates

    return {'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield, 'labels': curr_ax_cell_label}

# from neuropy.utils.misc import RowColTuple # for _make_pho_jonathan_batch_plots_advanced
# from neuropy.utils.matplotlib_helpers import build_or_reuse_figure, _build_variable_max_value_label, _determine_best_placefield_2D_layout, _build_neuron_identity_label # for _make_pho_jonathan_batch_plots_advanced

# def _make_pho_jonathan_batch_plots_advanced(sess, time_bins, neuron_replay_stats_df, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, included_unit_indicies=None, included_unit_neuron_IDs=None,
#     subplots:RowColTuple=(40, 3), fig_column_width:float=8.0, fig_row_height:float=1.0, resolution_multiplier:float=1.0, max_screen_figure_size=(None, None), fignum=1, fig=None, missing_aclu_string_formatter=None, debug_print=False):
#     """ An attempt to allow standardized pagination and such in `_make_pho_jonathan_batch_plots(...)` by converting from `plot_ratemap_2D(...)`'s page form and such. 

#     NOT YET FINISHED
    
#     Parameters
#     ----------
#     subplots : tuple, optional
#         number of cells within each figure window. If cells exceed the number of subplots, then cells are plotted in successive figure windows of same size, by default (10, 8)
#     fignum : int, optional
#         figure number to start from, by default None
#     fig_subplotsize: tuple, optional
#         fig_subplotsize: the size of a single subplot. used to compute the figure size

#     """
#     # last_figure_subplots_same_layout = False
#     last_figure_subplots_same_layout = True
#     # missing_aclu_string_formatter: a lambda function that takes the current aclu string and returns a modified string that reflects that this aclu value is missing from the current result (e.g. missing_aclu_string_formatter('3') -> '3 <shared>')
#     if missing_aclu_string_formatter is None:
#         missing_aclu_string_formatter = lambda curr_extended_id_string: f'{curr_extended_id_string}-'

#     active_maps, title_substring, included_unit_indicies = _help_plot_ratemap_neuronIDs(ratemap, included_unit_indicies=included_unit_indicies, included_unit_neuron_IDs=included_unit_neuron_IDs, plot_variable=plot_variable, debug_print=debug_print)

#     # ==================================================================================================================== #
    
#     ## BEGIN FACTORING OUT:
#     ## NEW COMBINED METHOD, COMPUTES ALL PAGES AT ONCE:
#     if resolution_multiplier is None:
#         resolution_multiplier = 1.0
#     nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=ratemap.xbin, ybin=ratemap.ybin, included_unit_indicies=included_unit_indicies, subplots=subplots, fig_column_width=fig_column_width, fig_row_height=fig_row_height, resolution_multiplier=resolution_multiplier, max_screen_figure_size=max_screen_figure_size, last_figure_subplots_same_layout=last_figure_subplots_same_layout, debug_print=debug_print)
    
#     if fignum is None:
#         if f := plt.get_fignums():
#             fignum = f[-1] + 1
#         else:
#             fignum = 1

#     figures, page_gs, graphics_obj_dicts = [], [], []
#     for fig_ind in range(nfigures):
#         # Dynamic Figure Sizing: 
#         curr_fig_page_grid_size = page_grid_sizes[fig_ind]
#         active_figure_size = page_figure_sizes[fig_ind]
        
#         ## Figure Setup:
#         fig = build_or_reuse_figure(fignum=fignum, fig=fig, fig_idx=fig_ind, figsize=active_figure_size, dpi=None, clear=True, tight_layout=False)
        
#         # grid_rect = (0.01, 0.05, 0.98, 0.9) # (left, bottom, width, height) 
#         # grid_rect = 111
#         # grid = ImageGrid(fig, grid_rect,  # similar to subplot(211)
#         #         nrows_ncols=(curr_fig_page_grid_size.num_rows, curr_fig_page_grid_size.num_columns),
#         #         axes_pad=0.05,
#         #         label_mode="1",
#         #         share_all=True,
#         #         aspect=True,
#         #         cbar_location="top",
#         #         cbar_mode=curr_cbar_mode,
#         #         cbar_size="7%",
#         #         cbar_pad="1%",
#         #         )
#         # page_gs.append(grid)

#         num_gridspec_columns = 8 # hardcoded
#         gs_kw = dict(width_ratios=np.repeat(1, num_gridspec_columns).tolist(), height_ratios=[1, 1], wspace=0.0, hspace=0.0)
#         gs_kw['width_ratios'][-1] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others

#         ## NOTE: previously expected a subfigure to use instead of the main figure:
#         gs = fig.add_gridspec(2, num_gridspec_columns, **gs_kw) # layout figure is usually a gridspec of (1,8)
#         curr_ax_firing_rate = fig.add_subplot(gs[0, :-1]) # the whole top row except the last element (to match the firing rates below)
#         curr_ax_cell_label = fig.add_subplot(gs[0, -1]) # the last element of the first row contains the labels that identify the cell
#         curr_ax_lap_spikes = fig.add_subplot(gs[1, :-1]) # all up to excluding the last element of the row
#         curr_ax_placefield = fig.add_subplot(gs[1, -1], sharey=curr_ax_lap_spikes) # only the last element of the row

#         title_string = f'2D Placemaps {title_substring} ({len(ratemap.neuron_ids)} good cells)'

#         fig.suptitle(title_string)
#         figures.append(fig)
#         graphics_obj_dicts.append({}) # New empty dict

#     # New page-based version:
#     for page_idx in np.arange(num_pages):
#         if debug_print:
#             print(f'page_idx: {page_idx}')
        
#         active_page_grid = page_gs[page_idx]
#         active_graphics_obj_dict = graphics_obj_dicts[page_idx]
#         # print(f'active_page_grid: {active_page_grid}')
            
#         for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
#             # Need to convert to page specific:
#             curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
#             curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
#             curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
#             # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')
           
#             if curr_included_unit_index is not None:
#                 # valid neuron ID, access like normal
#                 pfmap = active_maps[curr_included_unit_index]
#                 # normal (non-shared mode)
#                 curr_ratemap_relative_neuron_IDX = curr_included_unit_index
#                 curr_neuron_ID = ratemap.neuron_ids[curr_ratemap_relative_neuron_IDX]
                
#                 ## Labeling:
#                 formatted_max_value_string = None 
#                 final_title_str = _build_neuron_identity_label(neuron_extended_id=ratemap.neuron_extended_ids[curr_ratemap_relative_neuron_IDX], brev_mode=brev_mode, formatted_max_value_string=formatted_max_value_string, use_special_overlayed_title=use_special_overlayed_title)

#             else:
#                 # invalid neuron ID, generate blank entry
#                 curr_ratemap_relative_neuron_IDX = None # This neuron_ID doesn't correspond to a neuron_IDX in the current ratemap, so we'll mark this value as None
#                 curr_neuron_ID = included_unit_neuron_IDs[a_linear_index]

#                 pfmap = np.zeros((np.shape(active_maps)[1], np.shape(active_maps)[2])) # fully allocated new array of zeros
#                 curr_extended_id_string = f'{curr_neuron_ID}' # get the aclu value (which is all that's known about the missing cell and use that as the curr_extended_id_string
#                 final_title_str = missing_aclu_string_formatter(curr_extended_id_string)

#             # Get the axis to plot on:
#             curr_ax = active_page_grid[curr_page_relative_linear_index]
            
#             ## Plot the main heatmap for this pfmap:
#             # curr_im, curr_title_anchored_text = _plot_single_tuning_map_2D(ratemap.xbin, ratemap.ybin, pfmap, ratemap.occupancy, final_title_str=final_title_str, drop_below_threshold=drop_below_threshold, brev_mode=brev_mode, plot_mode=plot_mode,
#             #                                 ax=curr_ax, max_value_formatter=max_value_formatter, use_special_overlayed_title=use_special_overlayed_title, bg_rendering_mode=bg_rendering_mode)


#             curr_single_cell_out_dict = _plot_pho_jonathan_batch_plot_single_cell(sess, time_bins, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs, i, aclu, curr_fig, colors, debug_print=debug_print, **kwargs)

            
#             # active_graphics_obj_dict[curr_neuron_ID] = {'axs': [curr_ax], 'image': curr_im, 'title_obj': curr_title_anchored_text}
#             active_graphics_obj_dict[curr_neuron_ID] = curr_single_cell_out_dict


#             # if curr_ratemap_relative_neuron_IDX is not None:
#                 # This means this neuron is included in the current ratemap:
#                 ## Decision: Only do these extended plotting things when the neuron_IDX is included/valid.
                

#         ## Remove the unused axes if there are any:
#         # Note that this makes use of the fact that curr_page_relative_linear_index is left maxed-out after the above loop finishes executing.
#         num_axes_to_remove = (len(active_page_grid) - 1) - curr_page_relative_linear_index
#         if (num_axes_to_remove > 0):
#             for a_removed_linear_index in np.arange(curr_page_relative_linear_index+1, len(active_page_grid)):
#                 removal_ax = active_page_grid[a_removed_linear_index]
#                 fig.delaxes(removal_ax)

#         # Apply subplots adjust to fix margins:
#         plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        
#     return figures, page_gs, graphics_obj_dicts


def _make_pho_jonathan_batch_plots(t_split, time_bins, neuron_replay_stats_df, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, included_unit_neuron_IDs=None, n_max_plot_rows:int=4, debug_print=False, **kwargs):
    """ Stacked Jonathan-style firing-rate-across-epochs-plot
    Internally calls `_plot_pho_jonathan_batch_plot_single_cell`
        n_max_plot_rows: the maximum number of rows to plot


    # The colors for each point indicating the percentage of participating cells that belong to which track.
        - More long_only -> more red
        - More short_only -> more blue


    """
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        
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
    subfigs = fig.subfigures(actual_num_subfigures, 1, wspace=0.07)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    

    ##########################

    # _long_to_short_balances = rdf['neuron_type_distribution_color_scalar'].values

    # ## Test the building marker for specific aclu:

    # # custom_markers_dict_list = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False) for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # # scatter_plot_kwargs_list, scatter_markerstyles_list, scatter_marker_paths_list = custom_markers_dict_list['plot_kwargs'], custom_markers_dict_list['markerstyles'], custom_markers_dict_list['paths'] # Extract variables from the `custom_markers_dict_list` dictionary to the local workspace

    # custom_markers_tuple_list = [build_custom_scatter_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False) for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # scatter_markerstyles_list = [a_tuple[1] for a_tuple in custom_markers_tuple_list]

    # # Break into two parts <list<tuple[2]<MarkerStyle>> -> list<MarkerStyle>, List<MarkerStyle>
    # scatter_markerstyles_0_list = [a_tuple[0] for a_tuple in scatter_markerstyles_list]
    # scatter_markerstyles_1_list = [a_tuple[1] for a_tuple in scatter_markerstyles_list]

    # len(scatter_markerstyles_0_list) # 743 - n_replays
    # # out_plot_kwargs_array = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False, debug_print=False)[0] for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # # out_plot_kwargs_array

    # out_scatter_marker_paths_array = [_build_marker(long, shared, short, long_to_short_balance, is_tri_mode=False)[1] for long, shared, short, long_to_short_balance in list(zip(_percent_long_only, _percent_shared, _percent_short_only, _long_to_short_balances))]
    # out_scatter_marker_paths_array


    rdf, (_percent_long_only, _percent_shared, _percent_short_only, _percent_short_long_diff) = _build_neuron_type_distribution_color(rdf)

    # Build custom replay markers:
    custom_replay_scatter_markers_plot_kwargs_list = build_replays_custom_scatter_markers(rdf, debug_print=debug_print)
    kwargs['custom_replay_scatter_markers_plot_kwargs_list'] = custom_replay_scatter_markers_plot_kwargs_list

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

    graphics_output_dict = {'fig': fig, 'subfigs': subfigs, 'axs': axs_list, 'colors': colors}
    fig.show()
    return graphics_output_dict

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
    """ Produces a figure to compare *a scalar value* the 1D placefields on the long vs. the short track. 
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

    def add_value_labels(ax, spacing=5, labels=None):
        """Add labels to the end (top) of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """

        # For each bar: Place a label
        for i, rect in enumerate(ax.patches):
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            if labels is None:
                label = "{:.1f}".format(y_value)
                # # Use cell ID (given by x position) as the label
                label = "{}".format(x_value)
            else:
                label = str(labels[i])
                
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va,                      # Vertically align label differently for positive and negative values.
                color=rect.get_facecolor(),
                rotation=90)                      
                                            # 

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