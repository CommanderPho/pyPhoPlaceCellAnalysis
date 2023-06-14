from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from neuropy.utils.dynamic_container import overriding_dict_with # required for _display_2d_placefield_result_plot_raw

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer  # for context_nested_docks/single_context_nested_docks

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.Widgets.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget # for context_nested_docks/single_context_nested_docks

# MOVED IN TO `_single_context_nested_docks`
# from pyphoplacecellanalysis.GUI.Qt.Widgets.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array # for context_nested_docks/single_context_nested_docks


class MultiContextComparingDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ MultiContextComparingDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='grid_bin_bounds_validation', tags=['grid_bin_bounds','validation','pandas'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-14 18:17', related_items=[], is_global=True)
    def _display_grid_bin_bounds_validation(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, **kwargs):
        """ Renders a single figure that shows the 1D linearized position from several different sources to ensure sufficient overlap. Useful for validating that the grid_bin_bounds are chosen reasonably.

        """
        assert owning_pipeline_reference is not None
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        long_grid_bin_bounds, short_grid_bin_bounds, global_grid_bin_bounds = [owning_pipeline_reference.computation_results[a_name].computation_config['pf_params'].grid_bin_bounds for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # print(long_grid_bin_bounds, short_grid_bin_bounds, global_grid_bin_bounds)
        
        long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        
        long_pos_df, short_pos_df, global_pos_df, all_pos_df = [a_sess.position.to_dataframe() for a_sess in (long_session, short_session, global_session, owning_pipeline_reference.sess)]
        combined_pos_df = deepcopy(all_pos_df)
        combined_pos_df.loc[long_pos_df.index, 'long_lin_pos'] = long_pos_df.lin_pos
        combined_pos_df.loc[short_pos_df.index, 'short_pos_df'] = short_pos_df.lin_pos
        combined_pos_df.loc[global_pos_df.index, 'global_lin_pos'] = global_pos_df.lin_pos

        # ['long_lin_pos', 'short_pos_df', 'global_lin_pos']
        # combined_pos_df
        title = 'grid_bin_bounds validation across epochs'
        
        # Plot it:
        combined_pos_df.plot(x='t', y=['x', 'lin_pos', 'long_lin_pos', 'short_pos_df', 'global_lin_pos'], title='grid_bin_bounds validation across epochs')
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title(title)
        fig.canvas.manager.set_window_title(title)
        
        final_context = owning_pipeline_reference.sess.get_context().adding_context('display_fn', display_fn_name='_display_grid_bin_bounds_validation')
        
        out_man = owning_pipeline_reference.get_output_manager()
        
        graphics_output_dict = MatplotlibRenderPlots(name='_display_grid_bin_bounds_validation', figures=(fig,), axes=(ax,), plot_data={}, context=final_context, saved_figures=[])
        return graphics_output_dict
 
    @function_attributes(short_name='context_nested_docks', tags=['display','docks','pyqtgraph', 'interactive'], is_global=True, input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:14')
    def _display_context_nested_docks(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, **kwargs):
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
        if include_includelist is None:
            include_includelist = owning_pipeline_reference.active_completed_computation_result_names # ['maze', 'sprinkle']

        out_items = {}
        master_dock_win, app, out_items = _context_nested_docks(owning_pipeline_reference, active_config_names=include_includelist, **overriding_dict_with(lhs_dict={'enable_gui': False, 'debug_print': False}, **kwargs))

        # return master_dock_win, app, out_items
        return {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}

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
    # include_includelist = curr_active_pipeline.active_completed_computation_result_names # ['maze', 'sprinkle']

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
