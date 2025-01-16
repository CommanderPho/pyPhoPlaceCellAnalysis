from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuropy.analyses.placefields import PfND
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

## Required for _display_2d_placefield_result_plot_ratemaps_2D and _display_normal
from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import _display_add_computation_param_text_box, _save_displayed_figure_if_needed

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_all_placefields
from neuropy.plotting.ratemaps import BackgroundRenderingOptions # for _plot_latent_recursive_pfs_depth_level
from neuropy.utils.matplotlib_helpers import enumTuningMap2DPlotVariables # for getting the variant name from the dict
from neuropy.utils.mixins.unwrap_placefield_computation_parameters import unwrap_placefield_computation_parameters
from neuropy.utils.mixins.dict_representable import overriding_dict_with

from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots
from pyphocorehelpers.function_helpers import function_attributes
# from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, add_bin_ticks, build_binned_imageItem
from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout, _scale_current_placefield_to_acceptable_range
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import display_all_pf_2D_pyqtgraph_binned_image_rendering

# For _display_recurrsive_latent_placefield_comparisons
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.Pho2D.matplotlib.CustomMatplotlibWidget import CustomMatplotlibWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig # for context_nested_docks/single_context_nested_docks
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomCyclicColorsDockDisplayConfig, NamedColorScheme

from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _find_any_context_neurons


class DefaultRatemapDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing Bayesian Decoder performance. """
    
    @function_attributes(short_name='1d_placefields', tags=['display', 'placefields', '1D', 'matplotlib'], input_requires=["computation_result.computed_data['pf1D']"], output_provides=[], uses=['Pf1D.plot_ratemaps_1D(...)'], used_by=[], creation_date='2023-04-11 03:05')
    def _display_1d_placefields(computation_result, active_config, owning_pipeline=None, active_context=None, defer_display=False, **kwargs):
        from neuropy.core.neuron_identities import PlotStringBrevityModeEnum
        assert active_context is not None
        assert owning_pipeline is not None

        ## Finally, add the display function to the active context
        active_display_fn_identifying_ctx = active_context.adding_context('display_fn', display_fn_name='1d_placefields') # using short name instead of full name here
        # _build_safe_kwargs
        pf1D: PfND = computation_result.computed_data['pf1D']
        ax_pf_1D = pf1D.plot_ratemaps_1D(active_context=active_display_fn_identifying_ctx, brev_mode=kwargs.pop('brev_mode', PlotStringBrevityModeEnum.MINIMAL), **kwargs)
        active_figure = plt.gcf()
        
        ## Setup the plot title and add the session information:
        session_identifier = computation_result.sess.get_description() # 'sess_bapun_RatN_Day4_2019-10-15_11-30-06'
        fig_label = f'1d_placefields | {session_identifier} | {active_figure.number}'
        # print(f'fig_label: {fig_label}')
        active_figure.set_label(fig_label)
        active_figure.canvas.manager.set_window_title(fig_label) # sets the window's title
    
        if not defer_display:
            active_figure.show()
    
        # active_pf_2D_figures = [active_figure]            
        
        # # Save the figure out to disk if we need to:
        # should_save_to_disk = enable_saving_to_disk
        # if should_save_to_disk:
        #     _save_displayed_figure_if_needed(active_config.plotting_config, plot_type_name='_display_2d_placefield_result_plot_ratemaps_2D', active_variant_name=plot_variable_name, active_figures=active_pf_2D_figures)

        # return dict(fig=active_figure, ax=ax_pf_1D)
        return MatplotlibRenderPlots(figures=[active_figure], axes=[ax_pf_1D], context=active_display_fn_identifying_ctx)

     
    @function_attributes(short_name='1d_placefield_occupancy', tags=['display', 'placefields', '1D', 'occupancy', 'matplotlib'], input_requires=["computation_result.computed_data['pf1D']"], output_provides=[], uses=['PfND.plot_ratemaps_2D', 'neuropy.plotting.ratemaps.plot_ratemap_1D'], used_by=[], creation_date='2023-06-15 17:24')
    def _display_1d_placefield_occupancy(computation_result, active_config, enable_saving_to_disk=False, active_context=None, plot_pos_bin_axes: bool=True, **kwargs):
        """ displays placefield occupancy in a MATPLOTLIB window 
        """
        assert active_context is not None
        active_display_ctx = active_context.adding_context('display_fn', display_fn_name='plot_occupancy_1D')
        # active_display_ctx_string = active_display_ctx.get_description(separator='|')
        
        display_outputs = computation_result.computed_data['pf1D'].plot_occupancy(active_context=active_display_ctx, **({'plot_pos_bin_axes': plot_pos_bin_axes} | kwargs))
        
        # plot_variable_name = ({'plot_variable': None} | kwargs)
        plot_variable_name = kwargs.get('plot_variable', enumTuningMap2DPlotVariables.OCCUPANCY).name
        active_display_ctx = active_display_ctx.adding_context(None, plot_variable=plot_variable_name)

        active_figure = plt.gcf()
        
        # TODO 2023-06-02 - should drop the: 'computation_epochs', 'speed_thresh', 'frate_thresh', 'time_bin_size'
        active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
        _display_add_computation_param_text_box(active_figure, active_pf_computation_params, subset_excludelist=['computation_epochs', 'speed_thresh', 'frate_thresh', 'time_bin_size','frateThresh'], override_float_precision=2) # Adds the parameters text.
        
        ## Setup the plot title and add the session information:
        session_identifier = computation_result.sess.get_description() # 'sess_bapun_RatN_Day4_2019-10-15_11-30-06'
        fig_label = f'{plot_variable_name} | occupancy_1D | {session_identifier} | {active_figure.number}'
        # print(f'fig_label: {fig_label}')
        active_figure.set_label(fig_label)
        active_figure.canvas.manager.set_window_title(fig_label) # sets the window's title
        
        active_figures_list = [active_figure]
        
        # Save the figure out to disk if we need to:
        should_save_to_disk = enable_saving_to_disk
        if should_save_to_disk:
            _save_displayed_figure_if_needed(active_config.plotting_config, plot_type_name='_display_1d_placefield_occupancy', active_variant_name=plot_variable_name, active_figures=active_figures_list)

        return MatplotlibRenderPlots(figures=active_figures_list, axes=display_outputs[1], graphics=[], context=active_display_ctx)



    @function_attributes(short_name='2d_placefield_result_plot_ratemaps_2D', tags=['display', 'placefields', '2D', 'matplotlib'], input_requires=["computation_result.computed_data['pf2D']"], output_provides=[], uses=['PfND.plot_ratemaps_2D', 'neuropy.plotting.ratemaps.plot_ratemap_2D'], used_by=[], creation_date='2023-04-11 03:05')
    def _display_2d_placefield_result_plot_ratemaps_2D(computation_result, active_config, enable_saving_to_disk=False, active_context=None, **kwargs):
        """ displays 2D placefields in a MATPLOTLIB window 
        
        Internally wraps `PfND.plot_ratemaps_2D` which itself wraps `neuropy.plotting.ratemaps.plot_ratemap_2D`
        
            optionally shows peak firing rates
            
        TODO: plot the information about the source of the data, such as the session information? Or perhaps we could just leave that encoded in the exported file name? It is hard to track the figures though
        
        """
        assert active_context is not None
        active_display_ctx = active_context.adding_context('display_fn', display_fn_name='_display_2d_placefield_result_plot_ratemaps_2D')
        
        display_outputs = computation_result.computed_data['pf2D'].plot_ratemaps_2D(**({'subplots': (None, 3), 'resolution_multiplier': 1.0, 'enable_spike_overlay': False, 'brev_mode': PlotStringBrevityModeEnum.MINIMAL} | kwargs))
        
        # plot_variable_name = ({'plot_variable': None} | kwargs)
        plot_variable_name = kwargs.get('plot_variable', enumTuningMap2DPlotVariables.TUNING_MAPS).name
        active_figure = plt.gcf()
        
        active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
        
        
        _display_add_computation_param_text_box(active_figure, active_pf_computation_params) # Adds the parameters text. #TODO 2023-06-13 11:10: - [ ] Fix how this renders at least for PDF output. It's too big and positioned poorly.
        
        ## Setup the plot title and add the session information:
        session_identifier = computation_result.sess.get_description() # 'sess_bapun_RatN_Day4_2019-10-15_11-30-06'
        fig_label = f'{plot_variable_name} | plot_ratemaps_2D | {session_identifier} | {active_figure.number}'
        # print(f'fig_label: {fig_label}')
        active_figure.set_label(fig_label)
        active_figure.canvas.manager.set_window_title(fig_label) # sets the window's title
        

        # build final context:
        active_display_ctx.adding_context(None, plot_variable=plot_variable_name) # TODO: enable adding other important info like the params and stuff.

        
        active_pf_2D_figures = [active_figure]            
        
        # Save the figure out to disk if we need to:
        should_save_to_disk = enable_saving_to_disk
        if should_save_to_disk:
            _save_displayed_figure_if_needed(active_config.plotting_config, plot_type_name='_display_2d_placefield_result_plot_ratemaps_2D', active_variant_name=plot_variable_name, active_figures=active_pf_2D_figures)
        
        return MatplotlibRenderPlots(figures=active_pf_2D_figures, axes=display_outputs[1], graphics=display_outputs[2], context=active_display_ctx)
    

    @function_attributes(short_name='2d_placefield_occupancy', tags=['display', 'placefields', '2D', 'occupancy', 'matplotlib'], input_requires=["computation_result.computed_data['pf2D']"], output_provides=[], uses=['PfND.plot_ratemaps_2D', 'neuropy.plotting.ratemaps.plot_ratemap_2D'], used_by=[], creation_date='2023-04-11 03:05')
    def _display_2d_placefield_occupancy(computation_result, active_config, enable_saving_to_disk=False, active_context=None, **kwargs):
        """ displays placefield occupancy in a MATPLOTLIB window 
        
        Internally wraps `PfND.plot_ratemaps_2D` which itself wraps `neuropy.plotting.ratemaps.plot_ratemap_2D`
        
            optionally shows peak firing rates
            
        TODO: plot the information about the source of the data, such as the session information? Or perhaps we could just leave that encoded in the exported file name? It is hard to track the figures though
        
        """
        assert active_context is not None
        active_display_ctx = active_context.adding_context('display_fn', display_fn_name='plot_occupancy')
        # active_display_ctx_string = active_display_ctx.get_description(separator='|')
        
        display_outputs = computation_result.computed_data['pf2D'].plot_occupancy(**({} | kwargs))
        
        # plot_variable_name = ({'plot_variable': None} | kwargs)
        plot_variable_name = kwargs.get('plot_variable', enumTuningMap2DPlotVariables.OCCUPANCY).name
        active_display_ctx = active_display_ctx.adding_context(None, plot_variable=plot_variable_name)

        active_figure = plt.gcf()
        
        # TODO 2023-06-02 - should drop the: 'computation_epochs', 'speed_thresh', 'frate_thresh', 'time_bin_size'
        active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
        _display_add_computation_param_text_box(active_figure, active_pf_computation_params, subset_excludelist=['computation_epochs', 'speed_thresh', 'frate_thresh', 'time_bin_size','frateThresh'], override_float_precision=2) # Adds the parameters text.
        
        ## Setup the plot title and add the session information:
        session_identifier = computation_result.sess.get_description() # 'sess_bapun_RatN_Day4_2019-10-15_11-30-06'
        fig_label = f'{plot_variable_name} | occupancy_2D | {session_identifier} | {active_figure.number}'
        # print(f'fig_label: {fig_label}')
        active_figure.set_label(fig_label)
        active_figure.canvas.manager.set_window_title(fig_label) # sets the window's title
        
        active_pf_2D_figures = [active_figure]
        
        # Save the figure out to disk if we need to:
        should_save_to_disk = enable_saving_to_disk
        if should_save_to_disk:
            _save_displayed_figure_if_needed(active_config.plotting_config, plot_type_name='_display_2d_placefield_occupancy', active_variant_name=plot_variable_name, active_figures=active_pf_2D_figures)

        return MatplotlibRenderPlots(figures=active_pf_2D_figures, axes=display_outputs[1], graphics=[], context=active_display_ctx)


    @function_attributes(short_name='normal', tags=['display', 'placefields', '2D', 'matplotlib'], input_requires=["computation_result.computed_data['pf2D']"], output_provides=[], uses=['neuropy.plotting.placemaps.plot_all_placefields', 'neuropy.plotting.ratemaps.plot_ratemap_2D'], used_by=[], creation_date='2023-04-11 03:05')
    def _display_normal(computation_result, active_config, **kwargs):
        """
        
        Internally wraps `neuropy.plotting.placemaps.plot_all_placefields` which itself wraps `PfND.plot_ratemaps_2D` which itself wraps `neuropy.plotting.ratemaps.plot_ratemap_2D`
        
        Usage:
            _display_normal(curr_kdiba_pipeline.computation_results['maze1'], curr_kdiba_pipeline.active_configs['maze1'])
        """
        if active_config.computation_config is None:
            active_config.computation_config = computation_result.computation_config
        ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(None, computation_result.computed_data['pf2D'], active_config, **({'should_save_to_disk': False} | kwargs))

        # return occupancy_fig, active_pf_2D_figures
        return MatplotlibRenderPlots(figures=[occupancy_fig, active_pf_2D_figures])   

    @function_attributes(short_name='placemaps_pyqtplot_2D', tags=['display', 'placefields', '2D', 'pyqtgraph', 'pyqtplot'], conforms_to=['context_returning'], input_requires=["computation_result.computed_data['pf2D']"], output_provides=[], uses=['display_all_pf_2D_pyqtgraph_binned_image_rendering'], used_by=[], creation_date='2023-04-11 03:05')
    def _display_placemaps_pyqtplot_2D(computation_result, active_config, enable_saving_to_disk=False, active_context=None, defer_show:bool=False, **kwargs):
        """  displays 2D placefields in a pyqtgraph window
        """
        #TODO 2023-06-13 19:54: - [ ] Update to conform_to: ['output_registering', 'figure_saving']

        # Get the decoders from the computation result:
        # active_one_step_decoder = computation_result.computed_data['pf2D_Decoder'] # doesn't actually require the Decoder, could just use computation_result.computed_data['pf2D']            
        # # Get flat list of images:
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        # occupancy = active_one_step_decoder.ratemap.occupancy
        # app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy, 
        #                                                                         app=kwargs.get('app',None), parent_root_widget=kwargs.get('parent_root_widget',None), root_render_widget=kwargs.get('root_render_widget',None), max_num_columns=kwargs.get('max_num_columns', 8))
        # display_outputs = DynamicParameters(root_render_widget=root_render_widget, plot_array=plot_array, img_item_array=img_item_array, other_components_array=other_components_array)
        # if not defer_show:
        #     parent_root_widget.show()

        # return PyqtgraphRenderPlots(app=app, parent_root_widget=parent_root_widget, display_outputs=display_outputs)

        ## Post 2022-10-22 display_all_pf_2D_pyqtgraph_binned_image_rendering-based method:
        assert active_context is not None
        active_pf_2D = computation_result.computed_data['pf2D']
        # figure_format_config = {} # empty dict for config
        figure_format_config = kwargs # kwargs as default figure_format_config
        out_all_pf_2D_pyqtgraph_binned_image_fig = display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config) # output is BasicBinnedImageRenderingWindow

        # Set the window title from the context
        out_all_pf_2D_pyqtgraph_binned_image_fig.setWindowTitle(f'{active_context.get_description()}')

        if not defer_show:
            out_all_pf_2D_pyqtgraph_binned_image_fig.show()

        return PyqtgraphRenderPlots(parent_root_widget=out_all_pf_2D_pyqtgraph_binned_image_fig, context=active_context)

    @function_attributes(short_name='recurrsive_latent_placefield_comparisons', tags=['display', 'recurrsive', 'placefields', '2D', 'pyqtplot'], input_requires=["computation_result.computed_data['pf2D_RecursiveLatent']", "computation_result.computed_data['pf2D_Decoder']"], output_provides=[], uses=[], used_by=[], creation_date='2023-04-11 03:05')
    def _display_recurrsive_latent_placefield_comparisons(computation_result, active_config, owning_pipeline_reference=None, enable_saving_to_disk=False, active_context=None, defer_show:bool=False, **kwargs):
            """ Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18) 
            NOTE: Ignores `active_config` because context_nested_docks is for all contexts
            
            Usage:
            
                _out = curr_active_pipeline.display('_display_recurrsive_latent_placefield_comparisons', active_identifying_filtered_session_ctx)
                master_dock_win = _out['master_dock_win']
                curr_out_items = _out['out_items']

            """
            assert active_context is not None
            active_recursive_latent_pf_2Ds = computation_result.computed_data['pf2D_RecursiveLatent']

            ## Build the outer window:
            master_dock_win, app = DockAreaWrapper.build_default_dockAreaWindow(title='recurrsive_latent_placefield_comparisons', defer_show=False)
            master_dock_win.resize(1920, 1024)
            out_items = {}

            ## build plot_ratemaps_2D_kwargs:
            plot_ratemaps_2D_kwargs = kwargs.copy()        

            ## Compute neurons common to all deconding depths (for comparison across depths):
            pf_neurons_either = plot_ratemaps_2D_kwargs.get('included_unit_neuron_IDs', None)
            if pf_neurons_either is None:
                ## If no user-specified set of neurons to plot, find all common to any of the plots:
                active_first_order_2D_decoder = active_recursive_latent_pf_2Ds[0].get('pf2D_Decoder', None)
                active_second_order_2D_decoder = active_recursive_latent_pf_2Ds[1].get('pf2D_Decoder', None)
                active_third_order_2D_decoder = active_recursive_latent_pf_2Ds[2].get('pf2D_Decoder', None)

                pf_neurons_either = _find_any_context_neurons(active_first_order_2D_decoder.ratemap.neuron_ids, active_second_order_2D_decoder.ratemap.neuron_ids, active_third_order_2D_decoder.ratemap.neuron_ids)
                
                plot_ratemaps_2D_kwargs['included_unit_neuron_IDs'] = pf_neurons_either

            ## First Order:
            recursive_depth = 0
            recursive_depth_label = 'first'
            active_first_order_2D_decoder = active_recursive_latent_pf_2Ds[recursive_depth].get('pf2D_Decoder', None)
            active_decoder = active_first_order_2D_decoder
            active_identifying_ctx = active_context.adding_context('display_fn', decoder_order=recursive_depth_label)
            active_dock_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme.red, showCloseButton=True)
            out_items = out_items | _plot_latent_recursive_pfs_depth_level(master_dock_win, active_decoder, active_identifying_ctx, active_dock_config=active_dock_config, plot_ratemaps_2D_kwargs=plot_ratemaps_2D_kwargs)

            ## Second Order:
            recursive_depth = 1
            recursive_depth_label = 'second'
            active_identifying_ctx = active_context.adding_context('display_fn', decoder_order=recursive_depth_label)
            active_dock_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme.green, showCloseButton=True)
            active_second_order_2D_decoder = active_recursive_latent_pf_2Ds[recursive_depth].get('pf2D_Decoder', None)
            active_decoder = active_second_order_2D_decoder
            out_items = out_items | _plot_latent_recursive_pfs_depth_level(master_dock_win, active_decoder, active_identifying_ctx, active_dock_config=active_dock_config, plot_ratemaps_2D_kwargs=plot_ratemaps_2D_kwargs) 
            
            ## Third Order:
            recursive_depth = 2
            recursive_depth_label = 'third'
            active_third_order_2D_decoder = active_recursive_latent_pf_2Ds[recursive_depth].get('pf2D_Decoder', None)
            active_decoder = active_third_order_2D_decoder
            active_identifying_ctx = active_context.adding_context('display_fn', decoder_order=recursive_depth_label)
            active_dock_config = CustomCyclicColorsDockDisplayConfig(named_color_scheme=NamedColorScheme.blue, showCloseButton=True)
            out_items = out_items | _plot_latent_recursive_pfs_depth_level(master_dock_win, active_decoder, active_identifying_ctx, active_dock_config=active_dock_config, plot_ratemaps_2D_kwargs=plot_ratemaps_2D_kwargs)
            
            ## Layout all panels as we desire them: 
            desired_restore_state, backup_state, dock_keys_dict = _layout_latent_recursive_pfs_docks(master_dock_win, debug_print=False)

            # return master_dock_win, app, out_items
            return {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}




def _plot_latent_recursive_pfs_depth_level(master_dock_win, active_decoder, active_identifying_ctx, active_dock_config=None, plot_ratemaps_2D_kwargs=None, defer_render=False): # , recursive_depth = 1, recursive_depth_label = 'second'
    """ Used by _display_recurrsive_latent_placefield_comparisons to plot a single depth level of the latent recursive pf analysis.

    active_second_order_2D_decoder = active_recursive_latent_pf_2Ds[recursive_depth].get('pf2D_Decoder', None)
    _plot_latent_recursive_pfs_depth_level(active_second_order_2D_decoder, active_context=active_identifying_filtered_session_ctx, recursive_depth = 1, recursive_depth_label = 'second')
    
    """
    ## Nth Order:
    curr_out_items = {}
    ## Add the occupancy:
    active_display_ctx = active_identifying_ctx.adding_context('display_fn', display_fn_name='plot_occupancy')
    active_display_ctx_string = active_display_ctx.get_description(separator='|')
    mw = CustomMatplotlibWidget(size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=False) # , scrollAreaContents_MinimumHeight=params.all_plots_height
    subplot = mw.getFigure().add_subplot(111)
    active_decoder.pf.plot_occupancy(fig=mw.getFigure(), ax=subplot)
    if not defer_render:
        mw.show()
    ## Install the matplotlib-widget in the dock
    # dockAddLocationOpts = ['right', _last_dock_item] # position relative to the _last_dock_outer_nested_item for this figure
    dockAddLocationOpts = ['bottom']
    _last_widget, _last_dock_item = master_dock_win.add_display_dock(identifier=active_display_ctx_string, widget=mw, display_config=active_dock_config, dockSize=(10, 100), dockAddLocationOpts=dockAddLocationOpts)
    curr_out_items[active_display_ctx] = (mw, mw.getFigure(), subplot)
    # mw.draw()

    ## Add the placemaps:
    active_display_ctx = active_identifying_ctx.adding_context('display_fn', display_fn_name='plot_ratemaps_2D')
    active_display_ctx_string = active_display_ctx.get_description(separator='|')
    mw = CustomMatplotlibWidget(size=(15,15*4), dpi=72, scrollable_figure=False) # , constrained_layout=True, scrollAreaContents_MinimumHeight=params.all_plots_height
    if plot_ratemaps_2D_kwargs is None:
        plot_ratemaps_2D_kwargs = {} 
    # 'included_unit_neuron_IDs': [2, 3, 4, 5, 8, 10, 11, 13, 14, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 36, 37, 41, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 73, 74, 75, 76, 78, 81, 82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 96, 98, 100, 102, 105, 108, 109]
    # _out_plot_ratemaps = active_decoder.pf.plot_ratemaps_2D(fig=mw.getFigure(), **overriding_dict_with(lhs_dict={'subplots': (None, 3), 'brev_mode': PlotStringBrevityModeEnum.NONE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS, 'bg_rendering_mode': 'NONE'}, **plot_ratemaps_2D_kwargs)) # 5
    _out_plot_ratemaps = active_decoder.pf.plot_ratemaps_2D(fig=mw.getFigure(), **({'subplots': (None, 3), 'brev_mode': PlotStringBrevityModeEnum.NONE, 'plot_variable': enumTuningMap2DPlotVariables.TUNING_MAPS, 'bg_rendering_mode': BackgroundRenderingOptions.EMPTY} | plot_ratemaps_2D_kwargs)) # 5

    mw.getFigure().tight_layout() ## This actually fixes the tiny figure in the middle
    if not defer_render:
        mw.show()
    ## Install the matplotlib-widget in the dock
    # dockAddLocationOpts = ['bottom', _last_dock_item] # position relative to the _last_dock_outer_nested_item for this figure
    dockAddLocationOpts = ['bottom']
    _last_widget, _last_dock_item = master_dock_win.add_display_dock(identifier=active_display_ctx_string, widget=mw, display_config=active_dock_config, dockSize=(10, 400), dockAddLocationOpts=dockAddLocationOpts)
    curr_out_items[active_display_ctx] = (mw, mw.getFigure(), *_out_plot_ratemaps)
    mw.draw()
    return curr_out_items


def _layout_latent_recursive_pfs_docks(master_dock_win, debug_print=False):
    """Layout all panels as we desire them."""
    dock_all_keys = list(master_dock_win.dynamic_display_dict.keys())
    # ['kdiba|2006-6-08_14-26-15|maze1_PYR|first|plot_occupancy',
    #  'kdiba|2006-6-08_14-26-15|maze1_PYR|first|plot_ratemaps_2D',
    #  'kdiba|2006-6-08_14-26-15|maze1_PYR|second|plot_occupancy',
    #  'kdiba|2006-6-08_14-26-15|maze1_PYR|second|plot_ratemaps_2D',
    #  'kdiba|2006-6-08_14-26-15|maze1_PYR|third|plot_occupancy',
    #  'kdiba|2006-6-08_14-26-15|maze1_PYR|third|plot_ratemaps_2D']

    # Get occupancy plots:
    dock_occupancy_keys = [a_key for a_key in dock_all_keys if a_key.endswith('plot_occupancy')]
    dock_ratemaps_keys = [a_key for a_key in dock_all_keys if a_key.endswith('plot_ratemaps_2D')]
    # dock_keys_dict is returned in case want it later
    dock_keys_dict = {'all': dock_all_keys, 'occupancy': dock_occupancy_keys, 'dock_ratemaps_keys': dock_ratemaps_keys}

    if debug_print:
        print(f'dock_occupancy_keys: {dock_occupancy_keys}\ndock_ratemaps_keys: {dock_ratemaps_keys}')
    desired_restore_state = {'main': ('vertical',
    [('horizontal',
        [('dock', a_key, {}) for a_key in dock_occupancy_keys],
        {'sizes': np.repeat(1144, len(dock_occupancy_keys))}),
    ('horizontal',
        [('dock', a_key, {}) for a_key in dock_ratemaps_keys],
        {'sizes': np.repeat(1144, len(dock_occupancy_keys))})],
    {'sizes': [274, 1098]}),
    'float': []}

    if debug_print:
        print(f'desired_restore_state: {desired_restore_state}')
    # backup the current state before trying to restore:
    backup_state = deepcopy(master_dock_win.displayDockArea.saveState())
    if debug_print:   
        print(f'backup_state: {backup_state}')
    ## Perfrom the restore (set up layout):
    master_dock_win.displayDockArea.restoreState(state=desired_restore_state)
    return desired_restore_state, backup_state, dock_keys_dict

