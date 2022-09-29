from neuropy.utils.dynamic_container import overriding_dict_with # required for _display_2d_placefield_result_plot_raw
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder

from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1d_placecell_validations

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer  # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array # for context_nested_docks/single_context_nested_docks


class DefaultDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    
    def _display_1d_placefield_validations(computation_result, active_config, **kwargs):
        """ Renders all of the flat 1D place cell validations with the yellow lines that trace across to their horizontally drawn placefield (rendered on the right of the plot) """
        out_figures_list = plot_1d_placecell_validations(computation_result.computed_data['pf1D'], active_config.plotting_config, **overriding_dict_with(lhs_dict={'modifier_string': 'lap_only', 'should_save': False}, **kwargs))
        return out_figures_list

    def _display_2d_placefield_result_plot_raw(computation_result, active_config, **kwargs):
        """ produces a stupid figure """
        out_figures_list = computation_result.computed_data['pf2D'].plot_raw(**overriding_dict_with(lhs_dict={'label_cells': True}, **kwargs)); # Plots an overview of each cell all in one figure
        return out_figures_list


    def _display_context_nested_docks(computation_result, active_config, **kwargs):
        """ Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18) 
        NOTE: Ignores `active_config` because context_nested_docks is for all contexts
        
        Usage:
        
        display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_filtered_session_ctx, enable_gui=False, debug_print=False) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
        master_dock_win = display_output['master_dock_win']
        app = display_output['app']
        out_items = display_output['out_items']

        """
        owning_pipeline_reference = kwargs.get('owning_pipeline', None) # A reference to the pipeline upon which this display function is being called
        assert owning_pipeline_reference is not None
        # 
        out_items = {}
        master_dock_win, app, out_items = context_nested_docks(owning_pipeline_reference, **overriding_dict_with(lhs_dict={'enable_gui': False, 'debug_print': False}, **kwargs))
        
        # return master_dock_win, app, out_items
        return {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}



# ==================================================================================================================== #
# Private Display Helpers                                                                                              #
# ==================================================================================================================== #
def single_context_nested_docks(curr_active_pipeline, active_config_name, app, master_dock_win, enable_gui=False, debug_print=True):
        """ 2022-08-18 - Called for each config name in context_nested_docks's for loop.
        
        
        """
        out_display_items = dict()
        
        # Get relevant variables for this particular context:
        # curr_active_pipeline is set above, and usable here
        sess = curr_active_pipeline.filtered_sessions[active_config_name]

        # active_computation_results = curr_active_pipeline.computation_results[active_config_name]
        # active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
        # active_computation_config = curr_active_pipeline.computation_results[active_config_name].computation_config
        # active_computation_errors = curr_active_pipeline.computation_results[active_config_name].accumulated_errors
        # active_pf_1D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
        # active_pf_2D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D']    
        # active_pf_1D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_dt', None)
        # active_pf_2D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_dt', None)
        # active_firing_rate_trends = curr_active_pipeline.computation_results[active_config_name].computed_data.get('firing_rate_trends', None)
        active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None)
        # active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
        # active_extended_stats = curr_active_pipeline.computation_results[active_config_name].computed_data.get('extended_stats', None)
        # active_eloy_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('EloyAnalysis', None)
        # active_simpler_pf_densities_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('SimplerNeuronMeetingThresholdFiringAnalysis', None)
        # active_ratemap_peaks_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', None)
        # active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
        # active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()
        # curr_spikes_df = sess.spikes_df

        curr_active_config = curr_active_pipeline.active_configs[active_config_name]
        # curr_active_display_config = curr_active_config.plotting_config

        ## Build the active context by starting with the session context:
        active_identifying_session_ctx = sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        ## Add the filter to the active context
        active_identifying_session_ctx.add_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

        def on_finalize_figure_format_config(updated_figure_format_config):
                if debug_print:
                    print('on_finalize_figure_format_config')
                    print(f'\t {updated_figure_format_config}')
                # figure_format_config = updated_figure_format_config
                pass
                
        ## Finally, add the display function to the active context
        active_identifying_ctx = active_identifying_session_ctx.adding_context('display_fn', display_fn_name='figure_format_config_widget')
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
        
        ## Finally, add the display function to the active context
        active_identifying_ctx = active_identifying_session_ctx.adding_context('display_fn', display_fn_name='2D Position Decoder')
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

        

        # Get the decoders from the computation result:
        # active_one_step_decoder = computation_result.computed_data['pf2D_Decoder'] # doesn't actually require the Decoder, could just use computation_result.computed_data['pf2D']            
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        active_identifying_ctx = active_identifying_session_ctx.adding_context('display_fn', display_fn_name='pyqtplot_plot_image_array')
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
        
        return active_identifying_session_ctx, out_display_items
        # END single_context_nested_docks(...)
        
        
def context_nested_docks(curr_active_pipeline, enable_gui=False, debug_print=True):
    """ 2022-08-18 - builds a series of nested contexts for each active_config 
    
    Usage:
        master_dock_win, app, out_items = context_nested_docks(curr_active_pipeline, enable_gui=False, debug_print=True)
    """
    active_config_names = curr_active_pipeline.active_completed_computation_result_names # ['maze', 'sprinkle']
    
    if enable_gui:
        master_dock_win, app = DockAreaWrapper._build_default_dockAreaWindow(title='active_global_window', defer_show=False)
        master_dock_win.resize(1920, 1200)
    else:
        master_dock_win = None
        app = None

    out_items = {}
    for a_config_name in active_config_names:
        active_identifying_session_ctx, out_display_items = single_context_nested_docks(curr_active_pipeline=curr_active_pipeline, active_config_name=a_config_name, app=app, master_dock_win=master_dock_win, enable_gui=enable_gui, debug_print=debug_print)
        out_items[a_config_name] = (active_identifying_session_ctx, out_display_items)
        
    return master_dock_win, app, out_items
