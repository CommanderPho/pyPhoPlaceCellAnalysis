import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuropy.utils.dynamic_container import overriding_dict_with # required for _display_2d_placefield_result_plot_raw
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer  # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.Widgets.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import take_difference, take_difference_nonzero, make_fr

from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1D_placecell_validation # for _make_pho_jonathan_batch_plots

# BAD DOn'T DO THIS:
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _final_compute_jonathan_replay_fr_analyses


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

    def _display_jonathan_replay_firing_rate_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
            """ Jonathan's interactive display. Currently hacked up to directly compute the results to display within this function

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

            final_jonathan_df = global_computation_results.computed_data['jonathan_firing_rate_analysis']['final_jonathan_df']

            graphics_output_dict, neuron_df = _make_jonathan_interactive_plot(sess, time_bins, final_jonathan_df, time_binned_unit_specific_binned_spike_rate, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False)
            # output_dict = {'fig': fig, 'axs': ax, 'colors': colors}
            graphics_output_dict['plot_data'] = {'df': neuron_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf}
            
            return graphics_output_dict

    def _display_batch_pho_jonathan_replay_firing_rate_comparison(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_whitelist=None, **kwargs):
            """ Stacked Jonathan-style firing-rate-across-epochs-plot. Pho's batch adaptation of the primary elements from Jonathan's interactive display. Currently hacked up to directly compute the results to display within this function
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
            pf1d_long = computation_results[long_epoch_name]['computed_data']['pf1D']
            pf1d_short = computation_results[short_epoch_name]['computed_data']['pf1D']
            pf1D_all = computation_results[global_epoch_name]['computed_data']['pf1D']


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
            final_jonathan_df = global_computation_results.computed_data['jonathan_firing_rate_analysis']['final_jonathan_df']
            # compare_firing_rates(rdf, irdf)

            graphics_output_dict = _make_pho_jonathan_batch_plots(sess, time_bins, final_jonathan_df, time_binned_unit_specific_binned_spike_rate, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=True, n_max_plot_rows=6)

            # output_dict = {'fig': fig, 'axs': ax, 'colors': colors}
            graphics_output_dict['plot_data'] = {'df': final_jonathan_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf}
            
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


# Jonathan's visualizations:
# def compare_firing_rates(rdf, irdf, show_nonzero=True):
#     x1 = take_difference(irdf)
#     y1 = take_difference(rdf)
#     fig, ax = plt.subplots()

#     ax.plot(x1,y1,'.',label="naieve difference")

#     if show_nonzero:
#         x2 = take_difference_nonzero(irdf)
#         y2 = take_difference_nonzero(rdf)
        
#         # x2 = take_difference_adjust_for_time(irdf)
#         # y2 = take_difference_nonzero(rdf)

#         ax.plot(x2,y2,'.', label="nonzero difference")
#         ax.plot(np.vstack([x1,x2]), np.vstack([y1,y2]), color = (0,0,0,.1))


#     ax.set_xlabel("Mean FR change in all non-replay time (Hz)");
#     ax.set_ylabel("Mean FR change in replay time (Hz)");
#     ax.set_title("Firing rates are correlated in replay and non-replay time");
#     if show_nonzero:
#         ax.legend()
    
    # plt.axis("equal");
    


# # this cell really works best in qt
# %matplotlib qt


def _temp_draw_jonathan_ax(sess, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, colors=None, fig=None, ax=None, active_aclu:int=0, include_horizontal_labels=True, include_vertical_labels=True, should_render=False):
    """ Draws the time binned firing rates and the replay firing rates for a single cell

    Usage:

    index = new_index
    active_aclu = int(joined_df.index[index])
    _temp_draw_jonathan_ax(ax[0,1])

    _temp_draw_jonathan_ax(sess, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, colors=colors, fig=None, ax=ax[0,1], active_aclu=active_aclu)


    """
    assert ax is not None
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    
    # print(f"selected neuron has index: {index} aclu: {active_aclu}")

    # this redraws ax
    ax.clear()

    centers = (rdf["start"] + rdf["end"])/2
    heights = make_fr(rdf)[:, aclu_to_idx[active_aclu]]
    ax.plot(centers, heights, '.')

    if show_inter_replay_frs:
        # this would show the inter-replay firing times in orange it's frankly distracting
        centers = (irdf["start"] + irdf["end"])/2
        heights = make_fr(irdf)[:, aclu_to_idx[active_aclu]]
        ax.plot(centers, heights, '.', color=colors[1]+"80")

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
            ax.plot(t, v, color='#aaaaff8c') # this color is a translucent lilac (purple) color)
    except KeyError:
        print(f'non-placefield neuron. Skipping.')
        t, v = None, None
        pass

    required_epoch_bar_height = ax.get_ylim()[-1]
    # Draw the vertical epoch splitter line:
    ax.vlines(sess.paradigm[0][0,1], ymin = 0, ymax=required_epoch_bar_height, color=(0,0,0,.25))

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


def _make_jonathan_interactive_plot(sess, time_bins, final_jonathan_df, unit_specific_time_binned_firing_rates, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False):

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
    final_jonathan_df.long.fillna(xlim[0] + 1, inplace=True) # xlim[0] + 1 is the extreme edge of the plot
    final_jonathan_df.short.fillna(ylim[0] + 1, inplace=True)

    remap_scatter = ax[0,0].scatter(final_jonathan_df.long, final_jonathan_df.short, s=7, picker=True, c=[colors[c] for c in final_jonathan_df["has_na"]]);
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
    diff_scatter = ax[1,0].scatter(final_jonathan_df.non_replay_diff, final_jonathan_df.replay_diff, s=7, picker=True);
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
        aclu = int(final_jonathan_df.index[index])
        print(f"selected neuron has index: {index} aclu: {aclu}")
        
        # this changes the size of the neuron in ax[0,0]
        remap_scatter.set_sizes([7 if i!= index else 30 for i in range(len(final_jonathan_df))])

        # this changes the size of the neuron in ax[1,0]
        diff_scatter.set_sizes([7 if i!= index else 30 for i in range(len(final_jonathan_df))])

        ## New ax[0,1] draw method:
        _temp_draw_jonathan_ax(sess, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, colors=colors, fig=fig, ax=ax[0,1], active_aclu=aclu, should_render=True)

        # this plots where the neuron spiked on the track
        single_neuron_spikes = sess.spikes_df[sess.spikes_df.aclu == aclu]
        _temp_draw_jonathan_spikes_on_track(ax[1,1], pos_df, single_neuron_spikes)
        
        fig.canvas.draw()


    def on_keypress(event):
        global g_index
        if event.key=='tab':
            g_index += 1
            g_index %= len(final_jonathan_df)
        elif event.key=='b':
            g_index -= 1
            g_index %= len(final_jonathan_df)
        on_index_change(g_index)


    def on_pick(event):
        on_index_change(int(event.ind[0]))

    on_index_change(g_index)

    graphics_output_dict['on_index_change'] = {'callback': on_index_change, 'g_index': g_index}


    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    return graphics_output_dict, final_jonathan_df





def _make_pho_jonathan_batch_plots(sess, time_bins, final_jonathan_df, unit_specific_time_binned_firing_rates, pf1D_all, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, n_max_plot_rows:int=4, debug_print=False, **kwargs):
    """ Stacked Jonathan-style firing-rate-across-epochs-plot
    Internally calls `plot_1D_placecell_validation` and `_temp_draw_jonathan_ax`

        n_max_plot_rows: the maximum number of rows to plot


    """
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    subfigs = fig.subfigures(n_max_plot_rows, 1, wspace=0.07)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'];

    num_gridspec_columns = 8
    axs_list = []

    for i in np.arange(n_max_plot_rows):
        is_first_row = (i==0)
        is_last_row = (i == (n_max_plot_rows-1))
        aclu = int(final_jonathan_df.index[i])
        if debug_print:
            print(f"selected neuron has index: {i} aclu: {aclu}")
        title_string = ' '.join(['pf1D', f'Cell {aclu:02d}'])
        subtitle_string = ' '.join([f'{pf1D_all.config.str_for_display(False)}'])
        short_title_string = f'{aclu:02d}'
        if debug_print:
            print(f'\t{title_string}\n\t{subtitle_string}')
        

        # gridspec mode:
        curr_fig = subfigs[i]
        curr_fig.set_facecolor('0.75')



        gs_kw = dict(width_ratios=np.repeat(1, num_gridspec_columns).tolist(), height_ratios=[1, 1], wspace=0.0, hspace=0.0)
        gs_kw['width_ratios'][-1] = 0.3 # make the last column (containing the 1D placefield plot) a fraction of the width of the others

        gs = curr_fig.add_gridspec(2, num_gridspec_columns, **gs_kw) # layout figure is usually a gridspec of (1,8)
        # gs.update(wspace=0, hspace=0.0) # set the spacing between axes.
        # curr_ax_firing_rate = curr_fig.add_subplot(gs[0, :]) # the whole top row
        # curr_ax_lap_spikes = curr_fig.add_subplot(gs[1, :-1]) # all up to excluding the last element of the row
        # curr_ax_placefield = curr_fig.add_subplot(gs[1, -1], sharey=curr_ax_lap_spikes) # only the last element of the row
        curr_ax_firing_rate = curr_fig.add_subplot(gs[0, :-1]) # the whole top row except the last element (to match the firing rates below)
        curr_ax_cell_label = curr_fig.add_subplot(gs[0, -1]) # the last element of the first row contains the labels that identify the cell
        curr_ax_lap_spikes = curr_fig.add_subplot(gs[1, :-1]) # all up to excluding the last element of the row
        curr_ax_placefield = curr_fig.add_subplot(gs[1, -1], sharey=curr_ax_lap_spikes) # only the last element of the row
        
        # Setup title axis:
        title_axes_kwargs = dict(ha="center", va="center", fontsize=18, color="black")
        curr_ax_cell_label.text(0.5, 0.5, short_title_string, transform=curr_ax_cell_label.transAxes, **title_axes_kwargs)
        curr_ax_cell_label.axis('off')

        ## New ax[0,1] draw method:
        _temp_draw_jonathan_ax(sess, time_bins, unit_specific_time_binned_firing_rates, aclu_to_idx, rdf, irdf, show_inter_replay_frs=show_inter_replay_frs, colors=colors, fig=curr_fig, ax=curr_ax_firing_rate, active_aclu=aclu,
                            include_horizontal_labels=False, include_vertical_labels=False, should_render=True)
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
        _ = plot_1D_placecell_validation(pf1D_all, i, extant_fig=curr_fig, extant_axes=(curr_ax_lap_spikes, curr_ax_placefield), **({'should_include_labels': False, 'should_plot_spike_indicator_points_on_placefield': False, 'spike_indicator_lines_alpha': 0.2} | kwargs))

        t_start, t_end = curr_ax_lap_spikes.get_xlim()
        curr_ax_firing_rate.set_xlim((t_start, t_end)) # We don't want to clip to only the spiketimes for this cell, we want it for all cells, or even when the recording started/ended
        curr_ax_lap_spikes.sharex(curr_ax_firing_rate) # Sync the time axes of the laps and the firing rates
        
        # output the axes created:
        axs_list.append({'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield, 'labels': curr_ax_cell_label})

    graphics_output_dict = {'fig': fig, 'subfigs': subfigs, 'axs': axs_list, 'colors': colors}
    fig.show()
    return graphics_output_dict