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

# BAD DOn'T DO THIS:
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import _final_compute_jonathan_replay_fr_analyses

class MultiContextComparingDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ MultiContextComparingDisplayFunctions
    These display functions compare results across several contexts.

    """

    def _display_context_nested_docks(owning_pipeline_reference, computation_results, active_configs, include_whitelist=None, **kwargs):
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

    def _display_jonathan_replay_firing_rate_comparison(owning_pipeline_reference, computation_results, active_configs, include_whitelist=None, **kwargs):
            """ Jonathan's interactive display. Currently hacked up to directly compute the results to display within this function

                Usage:
                active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
                curr_active_pipeline.display('_display_jonathan_replay_firing_rate_comparison', active_identifying_session_ctx)

            """
            if include_whitelist is None:
                include_whitelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

            # ['maze1_PYR', 'maze2_PYR']
            pf1d_long = computation_results['maze1_PYR']['computed_data']['pf1D']
            pf1d_short = computation_results['maze2_PYR']['computed_data']['pf1D']

            try:
                # pf2D_Decoder = computation_results['maze_PYR']['computed_data']['pf2D_Decoder']
                active_firing_rate_trends = computation_results['maze_PYR']['computed_data']['firing_rate_trends']

                ## time_binned_unit_specific_binned_spike_rate mode:
                # time_bins = active_firing_rate_trends.all_session_spikes.time_binning_container.centers
                # time_binned_unit_specific_binned_spike_rate = active_firing_rate_trends.all_session_spikes.time_binned_unit_specific_binned_spike_rate

                ## instantaneous_unit_specific_spike_rate mode:
                neuron_IDs = np.unique(computation_results['maze_PYR'].sess.spikes_df.aclu)
                # neuron_IDXs = np.arange(len(neuron_IDs))
                instantaneous_unit_specific_spike_rate = active_firing_rate_trends.all_session_spikes.instantaneous_unit_specific_spike_rate
                # instantaneous_unit_specific_spike_rate = computation_results['maze_PYR']['computed_data']['firing_rate_trends'].all_session_spikes.instantaneous_unit_specific_spike_rate
                instantaneous_unit_specific_spike_rate_values = pd.DataFrame(instantaneous_unit_specific_spike_rate.magnitude, columns=neuron_IDs) # builds a df with times along the rows and aclu values along the columns in the style of unit_specific_binned_spike_counts
                time_bins = instantaneous_unit_specific_spike_rate.times.magnitude # .shape (3429,)
                time_binned_unit_specific_binned_spike_rate = instantaneous_unit_specific_spike_rate_values # .shape (3429, 71)

            except KeyError:
                # except ValueError:
                # print(f'non-placefield neuron. Skipping.')
                time_bins, time_binned_unit_specific_binned_spike_rate = {}, {}


            # ## Compute for all the session spikes first:
            sess = owning_pipeline_reference.sess
            # BAD DOn'T DO THIS:
            rdf, aclu_to_idx, irdf, aclu_to_idx_irdf = _final_compute_jonathan_replay_fr_analyses(sess, sess.replay)
            pos_df = sess.position.to_dataframe()

            ## TODO: This is the proper way once global computations work:
            # aclu_to_idx = computation_result.computed_data['jonathan_firing_rate_analysis']['rdf']['aclu_to_idx']
            # rdf = computation_result.computed_data['jonathan_firing_rate_analysis']['rdf']['rdf'],
            # irdf = computation_result.computed_data['jonathan_firing_rate_analysis']['irdf']['irdf']
            # pos_df = computation_result.sess.position.to_dataframe()
            # compare_firing_rates(rdf, irdf)

            graphics_output_dict, neuron_df = _make_jonathan_interactive_plot(sess, time_bins, time_binned_unit_specific_binned_spike_rate, pf1d_short, pf1d_long, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False)
            # output_dict = {'fig': fig, 'axs': ax, 'colors': colors}
            graphics_output_dict['plot_data'] = {'df': neuron_df, 'rdf':rdf, 'aclu_to_idx':aclu_to_idx, 'irdf':irdf}
            
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


def _temp_draw_jonathan_ax(sess, time_bins, unit_specific_time_binned_firing_rates, joined_df, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False, ax=None, active_aclu:int=0):
    """ 

    Usage:

    index = new_index
    active_aclu = int(joined_df.index[index])
    _temp_draw_jonathan_ax(ax[0,1])

    """
    assert ax is not None
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'];
    
    # print(f"selected neuron has index: {index} aclu: {active_aclu}")

    # this redraws ax
    ax.clear()

    # Draw the vertical epoch splitter line:
    ax.vlines(sess.paradigm[0][0,1], ymin = 0, ymax=60, color=(0,0,0,.25))

    centers = (rdf["start"] + rdf["end"])/2
    heights = make_fr(rdf)[:, aclu_to_idx[active_aclu]]
    ax.plot(centers, heights, '.')

    if show_inter_replay_frs:
        # this would show the inter-replay firing times in orange it's frankly distracting
        centers = (irdf["start"] + irdf["end"])/2
        heights = make_fr(irdf)[:, aclu_to_idx[active_aclu]]
        ax.plot(centers, heights, '.', color=colors[1]+"80")

    ax.set_title(f"Replay firing rates for neuron {active_aclu}")
    ax.set_xlabel("Time of replay (s)")
    ax.set_ylabel("Firing Rate (Hz)")

    # Pho's firing rate additions:
    try:
        # found_idx = pf2D_Decoder.neuron_IDs.index(aclu)
        # print(f'found_idx: {found_idx}')
        t = time_bins
        v = unit_specific_time_binned_firing_rates[active_aclu].to_numpy() # index directly by ACLU
        # print(f't: {np.shape(t)}, v: {np.shape(v)}')

    except KeyError:
        print(f'non-placefield neuron. Skipping.')
        t, v = None, None
        pass

    if v is not None:
        # Plot the continuous firing rates
        ax.plot(t, v)

    plt.gcf().canvas.draw()
    # fig.canvas.draw()








def _subfn_computations_make_jonathan_interactive_plot(unit_specific_time_binned_firing_rates, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf):
    # ==================================================================================================================== #
    ## Calculating:

    ## The actual firing rate we want:
    
    # unit_specific_time_binned_firing_rates = pf2D_Decoder.unit_specific_time_binned_spike_counts.astype(np.float32) / pf2D_Decoder.time_bin_size
    print(f'np.shape(unit_specific_time_binned_firing_rates): {np.shape(unit_specific_time_binned_firing_rates)}')

    # calculations for ax[0,0] ___________________________________________________________________________________________ #
    # below we find where the tuning curve peak was for each cell in each context and store it in a dataframe
    # pf1d_long = computation_results['maze1_PYR']['computed_data']['pf1D']
    long_peaks = [pf1d_long.xbin_centers[np.argmax(x)] for x in pf1d_long.ratemap.tuning_curves]
    long_df = pd.DataFrame(long_peaks, columns=['long'], index=pf1d_long.cell_ids)

    # pf1d_short = computation_results['maze2_PYR']['computed_data']['pf1D']
    short_peaks = [pf1d_short.xbin_centers[np.argmax(x)] for x in pf1d_short.ratemap.tuning_curves]
    short_df = pd.DataFrame(short_peaks, columns=['short'],index=pf1d_short.cell_ids)

    # df keeps most of the interesting data for these plots
    # at this point, it has columns 'long' and 'short' holding the peak tuning curve positions for each context
    # the index of this dataframe are the ACLU's for each neuron; this is why `how='outer'` works.
    df = long_df.join(short_df, how='outer')
    df["has_na"] = df.isna().any(axis=1)

    # calculations for ax[1,0] ___________________________________________________________________________________________ #
    non_replay_diff = take_difference_nonzero(irdf)
    replay_diff = take_difference_nonzero(rdf)
    df["non_replay_diff"] = [non_replay_diff[aclu_to_idx[aclu]] for aclu in df.index]
    df["replay_diff"] = [replay_diff[aclu_to_idx[aclu]] for aclu in df.index]

    return df



def _make_jonathan_interactive_plot(sess, time_bins, unit_specific_time_binned_firing_rates, pf1d_short, pf1d_long, pos_df, aclu_to_idx, rdf, irdf, show_inter_replay_frs=False):

    # ==================================================================================================================== #
    ## Calculating:
    df = _subfn_computations_make_jonathan_interactive_plot(unit_specific_time_binned_firing_rates, pf1d_short, pf1d_long, aclu_to_idx, rdf, irdf)

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
    df.long.fillna(xlim[0] + 1, inplace=True) # xlim[0] + 1 is the extreme edge of the plot
    df.short.fillna(ylim[0] + 1, inplace=True)

    remap_scatter = ax[0,0].scatter(df.long, df.short, s=7, picker=True, c=[colors[c] for c in df["has_na"]]);
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
    diff_scatter = ax[1,0].scatter(df.non_replay_diff, df.replay_diff, s=7, picker=True);
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
        aclu = int(df.index[index])
        print(f"selected neuron has index: {index} aclu: {aclu}")
        
        # this changes the size of the neuron in ax[0,0]
        remap_scatter.set_sizes([7 if i!= index else 30 for i in range(len(df))])

        # this changes the size of the neuron in ax[1,0]
        diff_scatter.set_sizes([7 if i!= index else 30 for i in range(len(df))])

        # this redraws ax
        ax[0,1].clear()

        ax[0,1].vlines(sess.paradigm[0][0,1], ymin = 0, ymax=60, color=(0,0,0,.25))

        centers = (rdf["start"] + rdf["end"])/2
        heights = make_fr(rdf)[:, aclu_to_idx[aclu]]
        ax[0,1].plot(centers, heights, '.')

        if show_inter_replay_frs:
            # this would show the inter-replay firing times in orange
            # it's frankly distracting
            centers = (irdf["start"] + irdf["end"])/2
            heights = make_fr(irdf)[:, aclu_to_idx[aclu]]
            ax[0,1].plot(centers, heights, '.', color=colors[1]+ "80")

        ax[0,1].set_title(f"Replay firing rates for neuron {aclu}")
        ax[0,1].set_xlabel("Time of replay (s)")
        ax[0,1].set_ylabel("Firing Rate (Hz)")
        # Pho's firing rate additions:
        try:
            # found_idx = pf2D_Decoder.neuron_IDs.index(aclu)
            # print(f'found_idx: {found_idx}')
            t = time_bins
            v = unit_specific_time_binned_firing_rates[aclu].to_numpy() # index directly by ACLU
            # print(f't: {np.shape(t)}, v: {np.shape(v)}')

        except KeyError:
        # except ValueError:
            print(f'non-placefield neuron. Skipping.')
            t, v = None, None
            pass

        if v is not None:
            # Plot the continuous firing rates
            ax[0,1].plot(t, v, color='#aaaaff8c') # this color is a translucent lilac (purple) color



        # this plots where the neuron spiked
        ax[1,1].clear()
        ax[1,1].plot(pos_df.t, pos_df.x, color=[.75, .75, .75])
        single_neuron_spikes = sess.spikes_df[sess.spikes_df.aclu == aclu]
        ax[1,1].plot(single_neuron_spikes.t_rel_seconds, single_neuron_spikes.x, 'k.', ms=1)
        ax[1,1].set_xlabel("t (s)")
        ax[1,1].set_ylabel("Position")
        ax[1,1].set_title("Animal position on track")


        fig.canvas.draw()


    def on_keypress(event):
        global g_index
        if event.key=='tab':
            g_index += 1
            g_index %= len(df)
        elif event.key=='b':
            g_index -= 1
            g_index %= len(df)
        on_index_change(g_index)


    def on_pick(event):
        on_index_change(int(event.ind[0]))

    on_index_change(g_index)

    graphics_output_dict['on_index_change'] = {'callback': on_index_change, 'g_index': g_index}


    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    return graphics_output_dict, df


